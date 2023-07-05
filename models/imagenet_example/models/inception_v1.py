import torch
import torch.nn as nn

__all__ = ['InceptionV1', 'inception_v1']

# modified according to https://github.com/minghao-wu/DeepLearningFromScratch/blob/master/GoogLeNet/GoogLeNet.py
# aux_classifier and dropout


def inception_v1(**kwargs):
    return InceptionV1(**kwargs)


class Inception(nn.Module):

    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5,
                 pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=5, padding=2),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], 1)


class AuxClassifier(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(AuxClassifier, self).__init__()
        self.pool1 = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=128,
                      kernel_size=1), nn.ReLU(inplace=True))

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=4 * 4 * 128, out_features=1024),
            nn.ReLU(inplace=True))
        self.drop = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(in_features=1024, out_features=num_classes)

    def forward(self, x):
        x = self.pool1(x)
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        return (x)


class InceptionV1(nn.Module):

    def __init__(self, num_classes=1000, aux_classifier=True):
        super(InceptionV1, self).__init__()
        self.aux_classifier = aux_classifier
        self.c1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(True),
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1),
            nn.ReLU(True),
        )
        self.c3 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
        )
        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.lrn = nn.LocalResponseNorm(2)
        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        if aux_classifier:
            self.aux0 = AuxClassifier(in_channels=512, num_classes=num_classes)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        if aux_classifier:
            self.aux1 = AuxClassifier(in_channels=528, num_classes=num_classes)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.drop = nn.Dropout(p=0.4)
        self.linear = nn.Linear(1024, num_classes)

    def forward(self, x):
        out = self.c1(x)
        out = self.maxpool(out)
        out = self.lrn(out)
        out = self.c2(out)
        out = self.c3(out)
        out = self.lrn(out)
        out = self.maxpool(out)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        if self.training and self.aux_classifier:
            output0 = self.aux0(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        if self.training and self.aux_classifier:
            output1 = self.aux1(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.drop(out)
        out = self.linear(out)
        if self.training and self.aux_classifier:
            out += (output0 + output1) * 0.3
        return out
