from torch.nn.utils import spectral_norm

from models.base_blocks import *


class Generator(nn.Module):
    def __init__(self, dataset=""):
        super(Generator, self).__init__()

        self.name = "generator_%s" % dataset
        bias = False

        self.encoding_blocks = nn.Sequential(
            ConvBlock(3, 64, bias=bias),
            ConvBlock(64, 128, bias=bias),
            DownConv(128, bias=bias),
            ConvBlock(128, 128, bias=bias),
            SeparableConv2D(128, 256, bias=bias),
            DownConv(256, bias=bias),
            ConvBlock(256, 256, bias=bias),
        )

        self.res_blocks = nn.Sequential(
            InvertedResBlock(256, 256, bias=bias),
            InvertedResBlock(256, 256, bias=bias),
            InvertedResBlock(256, 256, bias=bias),
            InvertedResBlock(256, 256, bias=bias),
            InvertedResBlock(256, 256, bias=bias),
            InvertedResBlock(256, 256, bias=bias),
            InvertedResBlock(256, 256, bias=bias),
            InvertedResBlock(256, 256, bias=bias)
        )

        self.decoding_blocks = nn.Sequential(
            ConvBlock(256, 128, bias=bias),
            UpConv(128, bias=bias),
            SeparableConv2D(128, 128, bias=bias),
            ConvBlock(128, 128, bias=bias),
            UpConv(128, bias=bias),
            ConvBlock(128, 64, bias=bias),
            ConvBlock(64, 64, bias=bias),
            nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.Tanh()
        )

        initialize_weights(self)

    def forward(self, x):
        x = self.encoding_blocks(x)
        x = self.res_blocks(x)
        x = self.decoding_blocks(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()

        self.name = "discriminator_%s" % args.dataset
        self.bias = False
        self.channels = 32
        self.layers = [
            nn.Conv2d(3, self.channels, kernel_size=3, stride=1, padding=1, bias=self.bias),
            nn.LeakyReLU(0.2, True)
        ]

        for i in range(args.d_layers):
            self.layers += [
                nn.Conv2d(self.channels, self.channels * 2, kernel_size=3, stride=2, padding=1, bias=self.bias),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(self.channels * 2, self.channels * 4, kernel_size=3, stride=1, padding=1, bias=self.bias),
                nn.InstanceNorm2d(self.channels * 4),
                nn.LeakyReLU(0.2, True)
            ]
            self.channels *= 4

        self.layers += [
            nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, bias=self.bias),
            nn.InstanceNorm2d(self.channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(self.channels, 1, kernel_size=3, stride=1, padding=1, bias=self.bias)
        ]

        if args.use_sn:
            for i in range(len(self.layers)):
                if isinstance(self.layers[i], nn.Conv2d):
                    self.layers[i] = spectral_norm(self.layers[i])

        self.discriminator = nn.Sequential(*self.layers)

        initialize_weights(self)

    def forward(self, x):
        return self.discriminator(x)