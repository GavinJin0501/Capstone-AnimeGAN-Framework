import os

import torch
import numpy as np
import cv2 as cv
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_rgb_to_yuv_kernel = torch.tensor([
    [0.299, -0.14714119, 0.61497538],
    [0.587, -0.28886916, -0.51496512],
    [0.114, 0.43601035, -0.10001026]
]).float().to(DEVICE)


def normalize_input(images):
    """
    [0, 255] -> [-1, 1]
    """
    return images / 127.5 - 1.0


def denormalize_input(images, dtype=None):
    """
    [-1, 1] -> [0, 255]
    """
    images = images * 127.5 + 127.5

    if dtype is not None:
        if isinstance(images, torch.Tensor):  # Tensor
            images = images.type(dtype)
        else:  # Numpy
            images = images.astype(dtype)

    return images


def rgb_to_yuv(image):
    """
    https://en.wikipedia.org/wiki/YUV
    output: Image of shape (H, W, C) (channel last)
    """
    # -1 1 -> 0 1
    image = (image + 1.0) / 2.0

    yuv_img = torch.tensordot(
        image,
        _rgb_to_yuv_kernel,
        dims=([image.ndim - 3], [0]))

    return yuv_img


def gram(matrix):
    """
    Calculate Gram Matrix
    https://pytorch.org/tutorials/advanced/neural_style_tutorial.html#style-loss
    """
    b, c, w, h = matrix.size()

    x = matrix.view(b * c, w * h)

    gram_matrix = torch.mm(x, x.T)

    # normalize by total elements
    return gram_matrix.div(b * c * w * h)


def compute_data_mean(data_folder):
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f'Folder {data_folder} does not exits')

    image_files = os.listdir(data_folder)
    total = np.zeros(3)

    print(f"Compute mean (R, G, B) from {len(image_files)} images")

    for img_file in tqdm(image_files):
        path = os.path.join(data_folder, img_file)
        image = cv.imread(path)
        total += image.mean(axis=(0, 1))

    channel_mean = total / len(image_files)
    mean = np.mean(channel_mean)

    return mean - channel_mean[..., ::-1]  # Convert to BGR for training
