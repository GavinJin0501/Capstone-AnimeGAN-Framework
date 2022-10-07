import torch.nn as nn
import torch.nn.functional as F
from utils.model_init import initialize_weights


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=bias)
        self.ins_norm = nn.InstanceNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2, True)

        initialize_weights(self)

    def forward(self, x):
        x = self.conv(x)
        x = self.ins_norm(x)
        x = self.activation(x)

        return x


class SeparableConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, bias=False):
        super(SeparableConv2D, self).__init__()

        self.depth_wise = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                   stride=stride, padding=1, groups=in_channels, bias=bias)
        self.point_wise = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=1, stride=1, bias=bias)

        self.ins_norm1 = nn.InstanceNorm2d(in_channels)
        self.activation1 = nn.LeakyReLU(0.2, True)
        self.ins_norm2 = nn.InstanceNorm2d(out_channels)
        self.activation2 = nn.LeakyReLU(0.2, True)

        initialize_weights(self)

    def forward(self, x):
        x = self.depth_wise(x)
        x = self.ins_norm1(x)
        x = self.activation1(x)

        x = self.point_wise(x)
        x = self.ins_norm2(x)
        x = self.activation2(x)

        return x


class DownConv(nn.Module):
    """
    Downsampling module to reduce the resolution of the feature maps.
    The input is resized to half the size.
    The output is the sum of the output of DSConv with stride 1 and 2
    """
    def __init__(self, channels, bias=False):
        super(DownConv, self).__init__()

        self.conv1 = SeparableConv2D(channels, channels, stride=2, bias=bias)
        self.conv2 = SeparableConv2D(channels, channels, stride=1, bias=bias)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = F.interpolate(x, scale_factor=0.5, mode="bilinear")
        out2 = self.conv2(out2)

        return out1 + out2


class UpConv(nn.Module):
    """
        Upsampling module to increase the resolution of the feature maps.
        The input is resized to 2 times the size.
        Used instead of fractionally strided convolutional layer with stride 1/2, because that method can cause the checkerboard artifacts in the synthesized images and affect the quality of the images.
    """
    def __init__(self, channels, bias=False):
        super(UpConv, self).__init__()

        self.conv = SeparableConv2D(channels, channels, stride=1, bias=bias)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2.0, mode="bilinear")
        out = self.conv(out)

        return out


class InvertedResBlock(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, expand_ratio=2, bias=False):
        super(InvertedResBlock, self).__init__()

        bottleneck_dim = round(expand_ratio * in_channels)
        self.conv_block = ConvBlock(in_channels, bottleneck_dim, kernel_size=1,
                                    stride=1, padding=0, bias=bias)
        self.depth_wise = nn.Conv2d(bottleneck_dim, bottleneck_dim, kernel_size=3,
                                    groups=bottleneck_dim, stride=1, padding=1, bias=bias)
        self.conv = nn.Conv2d(bottleneck_dim, out_channels, kernel_size=1,
                              stride=1, bias=bias)

        self.ins_norm1 = nn.InstanceNorm2d(out_channels)
        self.ins_norm2 = nn.InstanceNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2, True)

        initialize_weights(self)

    def forward(self, x):
        out = self.conv_block(x)
        out = self.depth_wise(out)
        out = self.ins_norm1(out)
        out = self.activation(out)
        out = self.conv(out)
        out = self.ins_norm2(out)

        return out + x

