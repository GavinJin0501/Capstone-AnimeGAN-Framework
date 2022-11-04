import os
import argparse
from inference import Transformer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='/content/checkpoints')
    parser.add_argument('--src', type=str, default='/content/checkpoints', help='source dir, contain real images')
    parser.add_argument('--dest', type=str, default='/content/results', help='destination dir to save generated images')
    parser.add_argument('--weight', type=str, default='/content/checkpoints/generator_Hayao_init.pth')

    return parser.parse_args()


def main(args):
    transformer = Transformer(args.checkpoint, weight=args.weight)

    if os.path.exists(args.src) and not os.path.isfile(args.src):
        transformer.transform_in_dir(args.src, args.dest)
    else:
        transformer.transform_file(args, args.dest)


if __name__ == "__main__":
    args = parse_args()
    main(args)