import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class Generator(nn.Module):
    def __int__(self, dataset=""):
        super(Generator, self).__init__()
        self.name = "generator_%s" % dataset
        bias = False

        self.encoding_blocks = nn.Sequential(

        )