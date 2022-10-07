import torch


def denormalize_input(images, dtype=None):
    """
    [-1, 1] -> [0, 255]
    :param images:
    :param dtype:
    :return:
    """
    images = images * 127.5 + 127.5

    if dtype is not None:
        if isinstance(images, torch.Tensor):  # Tensor
            images = images.type(dtype)
        else:  # Numpy
            images = images.astype(dtype)

    return images
