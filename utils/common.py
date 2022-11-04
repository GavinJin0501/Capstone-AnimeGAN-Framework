import os
import gc
import urllib.request

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

HTTP_PREFIXES = [
    'http',
    'data:image/jpeg',
]

SUPPORT_WEIGHTS = {
    'hayao',
    'shinkai',
}

ASSET_HOST = 'https://github.com/ptran1203/pytorch-animeGAN/releases/download/v1.0'


def set_lr(optimizer: optim.Optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch, args, postfix=""):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch
    }
    path = os.path.join(args.checkpoint_dir, "%s%s.pth" % (model.name, postfix))
    torch.save(checkpoint, path)


def load_checkpoint(model, checkpoint_dir, postfix=""):
    path = os.path.join(checkpoint_dir, "%s%s.pth" % (model.name, postfix))
    return load_weights(model, path)


def load_weights(model: nn.Module, weight):
    if weight.lower() in SUPPORT_WEIGHTS:
        weight = _download_weight(weight)

    checkpoint = torch.load(weight, map_location="cuda:0") if torch.cuda.is_available() else torch.load(weight, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    epoch = checkpoint["epoch"]
    del checkpoint
    torch.cuda.empty_cache()
    gc.collect()

    return epoch


class DownloadProgressBar(tqdm):
    """
    https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads
    """

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def _download_weight(weight):
    """
    Download weight and save to local file
    """
    filename = f'generator_{weight.lower()}.pth'
    os.makedirs('.cache', exist_ok=True)
    url = f'{ASSET_HOST}/{filename}'
    save_path = f'.cache/{filename}'
    print("save_path:", save_path)
    if os.path.isfile(save_path):
        return save_path

    desc = f'Downloading {url} to {save_path}'
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
        urllib.request.urlretrieve(url, save_path, reporthook=t.update_to)

    return save_path
