import torch
import numpy as np
import cv2 as cv

import os
import argparse
from multiprocessing import cpu_count

from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.common import load_checkpoint, set_lr, save_checkpoint
from utils.dataset import AnimeDataSet
from utils.image_processing import denormalize_input
from models.anime_GAN import Generator, Discriminator
from models.losses import LossSummary, AnimeGanLoss

# Global Variables
GAUSSIAN_MEAN = torch.tensor(0.0)
GAUSSIAN_STD = torch.tensor(0.1)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Hayao")
    parser.add_argument("--data-dir", type=str, default="dataset")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--init-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--checkpoint-dir", type=str, default="content/checkpoints")
    parser.add_argument("--save-image-dir", type=str, default="content/images")
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
            fake_img = generator(img.to(DEVICE))
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

    G = Generator(dataset=args.dataset).to(DEVICE)
    D = Discriminator(args).to(DEVICE)
    return 0
    loss_tracker = LossSummary()

    loss_fn = AnimeGanLoss(args)

    # Create DataLoader
    data_loader = DataLoader(
        AnimeDataSet(args),
        batch_size=args.batch_size,
        num_workers=cpu_count(),
        pin_memory=True,
        shuffle=True,
        collate_fn=collate_fn
    )

    optimizer_g = optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    start_epoch = 0
    if args.resume == "GD":
        # Load G and D
        try:
            start_epoch = load_checkpoint(G, args.checkpoint_dir)
            print("G weight loaded")
            load_checkpoint(D, args.checkpoint_dir)
            print("D weight loaded")
        except Exception as e:
            print("Could not load checkpoint, train from scratch", e)
    elif args.resume == "G":
        try:
            start_epoch = load_checkpoint(G, args.checkpoint_dir, postfix="_init")
        except Exception as e:
            print("Could not load G init checkpoint, train form scratch", e)

    for e in range(start_epoch, args.epochs):
        print("Epoch %d/%d" % (e, args.epochs))
        bar = tqdm(data_loader)
        G.train()

        init_losses = []

        if e < args.init_epochs:
            # Train with content loss only
            set_lr(optimizer_g, args.init_lr)

            for img, *_ in bar:
                img = img.to(DEVICE)

                optimizer_g.zero_grad()

                fake_img = G(img)
                loss = loss_fn.compute_vgg_content_loss(img, fake_img)
                loss.backward()
                optimizer_g.step()

                init_losses.append(loss.cpu().detach().numpy())
                avg_content_loss = sum(init_losses) / len(init_losses)
                bar.set_description("[Init Training G] content loss: %.2f" % avg_content_loss)

            set_lr(optimizer_g, args.lr_g)
            save_checkpoint(G, optimizer_g, e, args, postfix="_init")
            save_examples(G, data_loader, args, subname="initg")
            continue

        loss_tracker.reset()
        for img, style, style_gray, style_smt_gray in bar:
            img = img.to(DEVICE)
            style = style.to(DEVICE)
            style_gray = style_gray.to(DEVICE)
            style_smt_gray = style_smt_gray.to(DEVICE)

            # ---------------- TRAIN D ---------------- #
            optimizer_d.zero_grad()
            fake_img = G(img).detach()

            # Add some Gaussian noise to images before feeding to D
            if args.d_noise:
                fake_img += gaussian_noise()
                style += gaussian_noise()
                style_gray += gaussian_noise()
                style_smt_gray += gaussian_noise()

            fake_d = D(fake_img)
            style_d = D(style)
            style_gray_d = D(style_gray)
            style_smt_gray_d = D(style_smt_gray)

            loss_d = loss_fn.compute_discriminator_loss(fake_d, style_d, style_gray_d, style_smt_gray_d)
            loss_d.backward()
            optimizer_d.step()

            loss_tracker.update_discriminator_loss(loss_d)

            # ---------------- TRAIN G ---------------- #
            optimizer_g.zero_grad()

            fake_img = G(img)
            fake_d = D(fake_img)

            adv_loss, con_loss, gra_loss, col_loss = loss_fn.compute_generator_loss(fake_img, img, fake_d, style_gray)
            loss_g = adv_loss + col_loss + gra_loss + con_loss
            loss_g.backward()
            optimizer_g.step()

            loss_tracker.update_generator_loss(adv_loss, gra_loss, col_loss, con_loss)

            avg_adv, avg_gram, avg_color, avg_content = loss_tracker.get_avg_generator_loss()
            avg_adv_d = loss_tracker.get_avg_discriminator_loss()
            bar.set_description("loss G: adv %.2f, con %.2f, gram %.2f, color %.2f / loss D: %.2f" % (avg_adv, avg_content, avg_gram, avg_color, avg_adv_d))

        if e % args.save_interval == 0:
            save_checkpoint(G, optimizer_g, e, args)
            save_checkpoint(D, optimizer_d, e, args)
            save_examples(G, data_loader, args)


if __name__ == "__main__":
    args = parse_args()

    print("# ==== Train Config ==== #")
    for arg in vars(args):
        print(arg, getattr(args, arg))
    print("==========================")

    main(args)


