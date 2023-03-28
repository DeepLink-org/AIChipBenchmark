import torch
import torch.nn as nn

__all__ = ['InceptionV2', 'inception_v2']

# modified according to
# https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/bninception.py
# batch normalization & 3×3*2 & delete maxpool in 3c and 4e


def inception_v2(**kwargs):
    return InceptionV2(**kwargs)


class Inception_2(nn.Module):

    def __init__(self,
                 in_planes,
                 n1x1,
                 n3x3red,
                 n3x3,
                 n5x5red,
                 n5x5,
                 pool_planes,
                 pool_type='avg'):
        super(Inception_2, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1, affine=True),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red, affine=True),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3, affine=True),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red, affine=True),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5, affine=True),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5, affine=True),
            nn.ReLU(True),
        )

        # 3x3 pool
        if pool_type == 'avg':
            self.b4 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1), )
        else:
            self.b4 = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1), )

        # 1x1 conv branch
        self.b5 = nn.Sequential(
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes, affine=True),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4_pool = self.b4(x)
        y4 = self.b5(y4_pool)
        return torch.cat([y1, y2, y3, y4], 1)


class Inception_through(nn.Module):

    def __init__(self, in_planes, n3x3red, n3x3, n3x3red_double, n3x3_double):
        super(Inception_through, self).__init__()

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red, affine=True),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(n3x3, affine=True),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red_double, kernel_size=1),
            nn.BatchNorm2d(n3x3red_double, affine=True),
            nn.ReLU(True),
            nn.Conv2d(n3x3red_double, n3x3_double, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3_double, affine=True),
            nn.ReLU(True),
            nn.Conv2d(n3x3_double,
                      n3x3_double,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            nn.BatchNorm2d(n3x3_double, affine=True),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(nn.MaxPool2d(3, stride=2, padding=1), )

    def forward(self, x):
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y2, y3, y4], 1)


class InceptionV2(nn.Module):

    def __init__(self, num_classes=1000):
        super(InceptionV2, self).__init__()
        self.c1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(True),
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1),
            nn.ReLU(True),
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
        )

        self.a3 = Inception_2(192, 64, 64, 64, 64, 96, 32)
        self.b3 = Inception_2(256, 64, 64, 96, 64, 96, 64)
        self.c3 = Inception_through(320, 128, 160, 64, 96)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.lrn = nn.LocalResponseNorm(2)
        self.a4 = Inception_2(576, 224, 64, 96, 96, 128, 128)
        self.b4 = Inception_2(576, 192, 96, 128, 96, 128, 128)
        self.c4 = Inception_2(576, 160, 128, 160, 128, 160, 128)
        self.d4 = Inception_2(608, 96, 128, 192, 160, 192, 128)
        self.e4 = Inception_through(608, 128, 192, 192, 256)

        self.a5 = Inception_2(1056, 352, 192, 320, 160, 224, 128)
        self.b5 = Inception_2(1024, 352, 192, 320, 192, 224, 128, 'max')

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.linear = nn.Linear(1024, num_classes)

    def forward(self, x):
        out = self.c1(x)
        out = self.maxpool(out)
        out = self.lrn(out)
        out = self.c2(out)
        out = self.lrn(out)
        out = self.maxpool(out)
        out = self.a3(out)
        out = self.b3(out)
        out = self.c3(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
