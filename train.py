import torch
import numpy as np
import cv2 as cv

import os
import argparse

from utils.image_processing import denormalize_input

# Global Variables
GAUSSIAN_MEAN = torch.tensor(0.0)
GAUSSIAN_STD = torch.tensor(0.1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Hayao")
    parser.add_argument("--data-dir", type=str, default="/content/dataset")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--init-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--checkpoint-dir", type=int, default="/content/checkpoints")
    parser.add_argument("--save-image-dir", type=int, default="/content/images")
    parser.add_argument("--gan-loss", type=str, default="lsgan", help="lsgan/hinge/bce")
    parser.add_argument('--resume', type=str, default='False')
    parser.add_argument('--use_sn', action='store_true')
    parser.add_argument('--save-interval', type=int, default=1)
    parser.add_argument('--debug-samples', type=int, default=0)
    parser.add_argument('--lr-g', type=float, default=2e-4)
    parser.add_argument('--lr-d', type=float, default=4e-4)
    parser.add_argument('--init-lr', type=float, default=1e-3)
    parser.add_argument('--wadvg', type=float, default=10.0, help='Adversarial loss weight for G')
    parser.add_argument('--wadvd', type=float, default=10.0, help='Adversarial loss weight for D')
    parser.add_argument('--wcon', type=float, default=1.5, help='Content loss weight')
    parser.add_argument('--wgra', type=float, default=3.0, help='Gram loss weight')
    parser.add_argument('--wcol', type=float, default=30.0, help='Color loss weight')
    parser.add_argument('--d-layers', type=int, default=3, help='Discriminator conv layers')
    parser.add_argument('--d-noise', action='store_true')

    return parser.parse_args()


def collate_fn(batch):
    img, anime, anime_gray, anime_smt_gray = zip(*batch)
    return (
        torch.stack(img, 0),
        torch.stack(anime, 0),
        torch.stack(anime_gray, 0),
        torch.stack(anime_smt_gray, 0)
    )


def check_params(args):
    data_path = os.path.join(args.data_dir, args.dataset)

    if not os.path.exists(data_path):
        raise FileNotFoundError("Dataset not found: %s" % data_path)

    if not os.path.exists(args.save_image_dir):
        print("%s does not exist, creating..." % args.save_image_dir)
        os.mkdir(args.save_image_dir)

    if not os.path.exists(args.checkpoint_dir):
        print("%s does not exist, creating..." % args.checkpoint_dir)
        os.mkdir(args.checkpoint_dir)

    assert args.gan_loss in ["lsgan", "hinge", "bce"], f'{args.gan_loss} is not supported'


def save_examples(generator, loader, args, max_imgs=2, subname="gen"):
    generator.eval()

    max_iter = (max_imgs // args.batch_size) + 1
    fake_imgs = []

    for i, (img, *_) in enumerate(loader):
        with torch.no_grad():
            fake_img = generator(img.cuda())
            fake_img = fake_img.detach().cpu().numpy()
            # Channel first -> channel last
            fake_img = fake_img.transpose(0, 2, 3, 1)
            fake_imgs.append(denormalize_input(fake_img, dtype=np.int16))

        if i + 1 == max_iter:
            break

    fake_imgs = np.concatenate(fake_imgs, axis=0)

    for i, img in enumerate(fake_imgs):
        save_path = os.path.join(args.save_image_dir, "%s_%d.jpg" % (subname, i))
        cv.imwrite(save_path, img[..., ::-1])


def gaussian_noise():
    return torch.normal(GAUSSIAN_MEAN, GAUSSIAN_STD)


def main(args):
    check_params(args)

    print("Init models...")




if __name__ == '__main__':
    pass
