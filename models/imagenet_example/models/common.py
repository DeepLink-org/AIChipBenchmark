import torch.nn as nn
import math

__all__ = [
    'round_channels', 'conv1x1', 'ConvBlock', 'conv1x1_block', 'conv7x7_block',
    'SEBlock', 'ResNeXtBottleneck', 'ResInitBlock'
]


def round_channels(channels, divisor=8):
    rounded_channels = max(
        int(channels + divisor / 2.0) // divisor * divisor, divisor)
    if float(rounded_channels) < 0.9 * channels:
        rounded_channels += divisor
    return rounded_channels


def conv1x1(in_channels, out_channels, stride=1, groups=1, bias=False):
    return nn.Conv2d(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=1,
                     stride=stride,
                     groups=groups,
                     bias=bias)


class ConvBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 groups=1,
                 bias=False,
                 use_bn=True,
                 bn_eps=1e-5):
        super(ConvBlock, self).__init__()
        self.use_bn = use_bn

        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              groups=groups,
                              bias=bias)
        if self.use_bn:
            self.bn = nn.BatchNorm2d(num_features=out_channels, eps=bn_eps)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.activ(x)
        return x


def conv1x1_block(in_channels,
                  out_channels,
                  stride=1,
                  padding=0,
                  groups=1,
                  bias=False,
                  use_bn=True,
                  bn_eps=1e-5):
    return ConvBlock(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=1,
                     stride=stride,
                     padding=padding,
                     groups=groups,
                     bias=bias,
                     use_bn=use_bn,
                     bn_eps=bn_eps)


def conv3x3_block(in_channels,
                  out_channels,
                  stride=1,
                  padding=1,
                  dilation=1,
                  groups=1,
                  bias=False,
                  use_bn=True,
                  bn_eps=1e-5):
    return ConvBlock(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=3,
                     stride=stride,
                     padding=padding,
                     dilation=dilation,
                     groups=groups,
                     bias=bias,
                     use_bn=use_bn,
                     bn_eps=bn_eps)


def conv7x7_block(in_channels,
                  out_channels,
                  stride=1,
                  padding=3,
                  bias=False,
                  use_bn=True):
    return ConvBlock(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=7,
                     stride=stride,
                     padding=padding,
                     bias=bias,
                     use_bn=use_bn)


class SEBlock(nn.Module):

    def __init__(self, channels, reduction=16, round_mid=False):
        super(SEBlock, self).__init__()
        mid_channels = channels // reduction if not round_mid else round_channels(
            float(channels) / reduction)

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv1 = conv1x1(in_channels=channels,
                             out_channels=mid_channels,
                             bias=True)
        self.activ = nn.ReLU(inplace=True)
        self.conv2 = conv1x1(in_channels=mid_channels,
                             out_channels=channels,
                             bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.pool(x)
        w = self.conv1(w)
        w = self.activ(w)
        w = self.conv2(w)
        w = self.sigmoid(w)
        x = x * w
        return x


class ResInitBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResInitBlock, self).__init__()
        self.conv = conv7x7_block(in_channels=in_channels,
                                  out_channels=out_channels,
                                  stride=2)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class ResNeXtBottleneck(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 cardinality,
                 bottleneck_width,
                 bottleneck_factor=4):
        super(ResNeXtBottleneck, self).__init__()
        mid_channels = out_channels // bottleneck_factor
        D = int(math.floor(mid_channels * (bottleneck_width / 64.0)))
        group_width = cardinality * D

        self.conv1 = conv1x1_block(in_channels=in_channels,
                                   out_channels=group_width)
        self.conv2 = conv3x3_block(in_channels=group_width,
                                   out_channels=group_width,
                                   stride=stride,
                                   groups=cardinality)
        self.conv3 = conv1x1_block(in_channels=group_width,
                                   out_channels=out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x
