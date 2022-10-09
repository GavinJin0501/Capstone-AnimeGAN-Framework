import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vgg import Vgg19
from utils.image_processing import rgb_to_yuv, gram

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LossSummary:
    def __init__(self):
        self.loss_g_adv = []
        self.loss_content = []
        self.loss_gram = []
        self.loss_color = []
        self.loss_d_adv = []

        self.reset()

    def reset(self):
        self.loss_g_adv = []
        self.loss_content = []
        self.loss_gram = []
        self.loss_color = []
        self.loss_d_adv = []

    def update_generator_loss(self, adv, gram, color, content):
        """
        :param adv: adversarial loss that affects the animation transformation process
        :param gram: grayscale style loss that makes the generated images have the right generated(real) styles
        :param color: color reconstruction loss that makes the generated images have the color of the input(anime)
        :param content: content loss that helps the generated image retain the input(anime) content
        :return: None
        """
        self.loss_g_adv.append(adv.cpu().detach().numpy())
        self.loss_gram.append(gram.cpu().detach().numpy())
        self.loss_color.append(color.cpu().detach().numpy())
        self.loss_content.append(content.cpu().detach().numpy())

    def update_discriminator_loss(self, loss):
        self.loss_d_adv.append(loss.cpu().detach().numpy())

    def get_avg_generator_loss(self):
        return (
            self._avg(self.loss_g_adv),
            self._avg(self.loss_gram),
            self._avg(self.loss_color),
            self._avg(self.loss_content),
        )

    def get_avg_discriminator_loss(self):
        return self._avg(self.loss_d_adv)

    @staticmethod
    def _avg(losses):
        return sum(losses) / len(losses)


class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.huber = nn.SmoothL1Loss()

    def forward(self, ori_img, gen_img):
        ori_img = rgb_to_yuv(ori_img)
        gen_img = rgb_to_yuv(gen_img)

        # After convert to yuv, both images have channel last
        return (self.l1(ori_img[:, :, :, 0], gen_img[:, :, :, 0]) +
                self.huber(ori_img[:, :, :, 1], gen_img[:, :, :, 1]) +
                self.huber(ori_img[:, :, :, 2], gen_img[:, :, :, 2]))


class AnimeGanLoss:
    def __init__(self, args):
        # define different losses
        self.content_loss = nn.L1Loss().to(DEVICE)
        self.gram_loss = nn.L1Loss().to(DEVICE)
        self.color_loss = ColorLoss().to(DEVICE)
        self.bce_loss = nn.BCELoss()

        # weights for different losses
        self.wadvg = args.wadvg
        self.wadvd = args.wadvd
        self.wcon = args.wcon
        self.wgra = args.wgra
        self.wcol = args.wcol

        self.vgg19 = Vgg19().to(DEVICE).eval()
        self.adv_type = args.gan_loss

    def compute_generator_loss(self, fake_img, img, fake_logit, gray_style_img):
        """
        Compute loss for Generator

        @Arguments:
            :param fake_img: generated image
            :param img: original image
            :param fake_logit: output of Discriminator given the fake image
            :param gray_style_img: grayscale of style image

        :return: loss
        """
        fake_feat = self.vgg19(fake_img)
        style_feat = self.vgg19(gray_style_img)
        img_feat = self.vgg19(img).detach()

        return [
            self.wadvg * self._generator_adv_loss(fake_logit),
            self.wcon * self.content_loss(img_feat, fake_feat),
            self.wgra * self.gram_loss(gram(style_feat), gram(fake_feat)),
            self.wcol * self.color_loss(img, fake_img)
        ]

    def compute_discriminator_loss(self, fake_logit, style_logit, gray_style_logit, smooth_gray_style_logit):
        return self.wadvd * (
            self._discriminator_style_adv_loss(style_logit) +
            self._discriminator_fake_adv_loss(fake_logit) +
            self._discriminator_fake_adv_loss(gray_style_logit) +
            0.2 * self._discriminator_fake_adv_loss(smooth_gray_style_logit)
        )

    def compute_vgg_content_loss(self, image, reconstruction):
        feat = self.vgg19(image)
        re_feat = self.vgg19(reconstruction)

        return self.content_loss(feat, re_feat)

    def _generator_adv_loss(self, pred):
        if self.adv_type == "hinge":
            return -torch.mean(pred)
        elif self.adv_type == "lsgan":
            return torch.mean(torch.square(pred - 1.0))
        elif self.adv_type == "normal":
            return self.bce_loss(pred, torch.zeros_like(pred))

        raise ValueError("Do not support type: %s" % self.adv_type)

    def _discriminator_style_adv_loss(self, pred):
        if self.adv_type == 'hinge':
            return torch.mean(F.relu(1.0 - pred))
        elif self.adv_type == 'lsgan':
            return torch.mean(torch.square(pred - 1.0))
        elif self.adv_type == 'normal':
            return self.bce_loss(pred, torch.ones_like(pred))

        raise ValueError("Do not support type: %s" % self.adv_type)

    def _discriminator_fake_adv_loss(self, pred):
        if self.adv_type == 'hinge':
            return torch.mean(F.relu(1.0 + pred))
        elif self.adv_type == 'lsgan':
            return torch.mean(torch.square(pred))
        elif self.adv_type == 'normal':
            return self.bce_loss(pred, torch.zeros_like(pred))

        raise ValueError("Do not support type: %s" % self.adv_type)

