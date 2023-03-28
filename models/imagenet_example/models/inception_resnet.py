import torch
import torch.nn as nn

__all__ = ['inception_resnet_v1', 'inception_resnet_v2']


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes,
                              out_planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001,
                                 momentum=0.1,
                                 affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Stem_tail_v1(nn.Module):

    def __init__(self):
        super(Stem_tail_v1, self).__init__()
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.conv2d_4b = BasicConv2d(192, 256, kernel_size=3, stride=2)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)
        return x


class Stem_tail_v2(nn.Module):

    def __init__(self):
        super(Stem_tail_v2, self).__init__()
        self.branch0_0 = nn.MaxPool2d(3, stride=2)
        self.branch0_1 = BasicConv2d(64, 96, kernel_size=3, stride=2)

        self.branch1_0 = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1))

        self.branch1_1 = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 64, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(64, 64, kernel_size=(7, 1), stride=1, padding=(3, 0)),
            BasicConv2d(64, 96, kernel_size=(3, 3), stride=1))

        self.branch2_0 = BasicConv2d(192, 128, kernel_size=3, stride=2)
        self.branch2_1 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0_0 = self.branch0_0(x)
        x0_1 = self.branch0_1(x)
        x0 = torch.cat((x0_0, x0_1), 1)
        x1_0 = self.branch1_0(x0)
        x1_1 = self.branch1_1(x0)
        x1 = torch.cat((x1_0, x1_1), 1)
        x2_0 = self.branch2_0(x1)
        x2_1 = self.branch2_1(x1)
        out = torch.cat((x2_0, x2_1), 1)
        return out


class Block35(nn.Module):

    def __init__(self, scale, input_channels):
        super(Block35, self).__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(input_channels, 32, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(input_channels, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1))

        if input_channels == 256:
            conv3_1 = 32
            conv3_2 = 32
        else:
            conv3_1 = 48
            conv3_2 = 64
        self.branch2 = nn.Sequential(
            BasicConv2d(input_channels, 32, kernel_size=1, stride=1),
            BasicConv2d(32, conv3_1, kernel_size=3, stride=1, padding=1),
            BasicConv2d(conv3_1, conv3_2, kernel_size=3, stride=1, padding=1))

        self.conv2d = nn.Conv2d(conv3_2 + 64,
                                input_channels,
                                kernel_size=1,
                                stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_6a(nn.Module):

    def __init__(self, input_channels, k, ll, m, n):
        super(Mixed_6a, self).__init__()

        self.branch0 = BasicConv2d(input_channels, n, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv2d(input_channels, k, kernel_size=1, stride=1),
            BasicConv2d(k, ll, kernel_size=3, stride=1, padding=1),
            BasicConv2d(ll, m, kernel_size=3, stride=2))

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Block17(nn.Module):

    def __init__(self, scale, input_channels, m):
        super(Block17, self).__init__()

        self.scale = scale
        self.branch0 = BasicConv2d(input_channels,
                                   m // 2,
                                   kernel_size=1,
                                   stride=1)

        step1 = (m // 2 + 128) // 2

        self.branch1 = nn.Sequential(
            BasicConv2d(input_channels, 128, kernel_size=1, stride=1),
            BasicConv2d(128,
                        step1,
                        kernel_size=(1, 7),
                        stride=1,
                        padding=(0, 3)),
            BasicConv2d(step1,
                        m // 2,
                        kernel_size=(7, 1),
                        stride=1,
                        padding=(3, 0)))

        self.conv2d = nn.Conv2d(m, input_channels, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_7a(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(Mixed_7a, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(input_channels, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2))

        channels_middle = (output_channels + 256) // 2
        self.branch1 = nn.Sequential(
            BasicConv2d(input_channels, 256, kernel_size=1, stride=1),
            BasicConv2d(256, channels_middle, kernel_size=3, stride=2))

        self.branch2 = nn.Sequential(
            BasicConv2d(input_channels, 256, kernel_size=1, stride=1),
            BasicConv2d(256,
                        channels_middle,
                        kernel_size=3,
                        stride=1,
                        padding=1),
            BasicConv2d(channels_middle,
                        output_channels,
                        kernel_size=3,
                        stride=2))

        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block8(nn.Module):

    def __init__(self, scale, input_channels, ll, noReLU=False):
        super(Block8, self).__init__()

        self.scale = scale
        self.noReLU = noReLU

        self.branch0 = BasicConv2d(input_channels,
                                   192,
                                   kernel_size=1,
                                   stride=1)

        step1 = (ll + 192) // 2
        self.branch1 = nn.Sequential(
            BasicConv2d(input_channels, 192, kernel_size=1, stride=1),
            BasicConv2d(192,
                        step1,
                        kernel_size=(1, 3),
                        stride=1,
                        padding=(0, 1)),
            BasicConv2d(step1,
                        ll,
                        kernel_size=(3, 1),
                        stride=1,
                        padding=(1, 0)))

        self.conv2d = nn.Conv2d(192 + ll,
                                input_channels,
                                kernel_size=1,
                                stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out


class InceptionResNet(nn.Module):

    def __init__(self,
                 channel35,
                 k,
                 l_index,
                 m,
                 n,
                 num_classes=1000,
                 num_repeat=[10, 20, 9],
                 scale=[0.17, 0.10, 0.20]):
        super(InceptionResNet, self).__init__()
        # Special attributs
        self.input_space = None
        self.input_size = (299, 299, 3)
        self.mean = None
        self.std = None
        # Modules
        self.stem_head = nn.Sequential(
            BasicConv2d(3, 32, kernel_size=3, stride=2),
            BasicConv2d(32, 32, kernel_size=3, stride=1),
            BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1),
        )
        if channel35 == 256:  # inception_resnet_v1
            self.stem_tail = Stem_tail_v1()
        else:  # inception_resnet_v2  320
            self.stem_tail = Stem_tail_v2()

        sequence_list0 = []
        for j in range(num_repeat[0]):
            sequence_list0.append(
                Block35(scale=scale[0], input_channels=channel35))
        self.repeat0 = nn.Sequential(*sequence_list0)

        self.mixed_6a = Mixed_6a(input_channels=channel35,
                                 k=k,
                                 ll=l_index,
                                 m=m,
                                 n=n)

        repeat1_input = channel35 + m + n
        sequence_list1 = []
        for j in range(num_repeat[1]):
            sequence_list1.append(
                Block17(scale=scale[1], input_channels=repeat1_input, m=m))
        self.repeat1 = nn.Sequential(*sequence_list1)

        self.mixed_7a = Mixed_7a(input_channels=repeat1_input,
                                 output_channels=channel35)

        repeat2_input = repeat1_input + channel35 * 3 // 2 + 512
        sequence_list2 = []
        for j in range(num_repeat[2]):
            sequence_list2.append(
                Block8(scale=scale[2],
                       input_channels=repeat2_input,
                       ll=l_index))
        self.repeat2 = nn.Sequential(*sequence_list2)

        self.block8 = Block8(scale=scale[2],
                             input_channels=repeat2_input,
                             ll=l_index,
                             noReLU=True)
        self.conv2d_7b = BasicConv2d(repeat2_input,
                                     1536,
                                     kernel_size=1,
                                     stride=1)
        self.avgpool_1a = nn.AvgPool2d(8)
        self.drop = nn.Dropout(p=0.2)
        self.last_linear = nn.Linear(1536, num_classes)

    def features(self, input):
        x = self.stem_head(input)
        x = self.stem_tail(x)
        x = self.repeat0(x)
        x = self.mixed_6a(x)
        x = self.repeat1(x)
        x = self.mixed_7a(x)
        x = self.repeat2(x)
        x = self.block8(x)
        x = self.conv2d_7b(x)
        return x

    def logits(self, features):
        x = self.avgpool_1a(features)
        self.drop(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


def inception_resnet_v1(**kwargs):
    model = InceptionResNet(channel35=256,
                            k=192,
                            l_index=192,
                            m=256,
                            n=384,
                            **kwargs)
    return model


def inception_resnet_v2(**kwargs):
    model = InceptionResNet(channel35=320,
                            k=256,
                            l_index=256,
                            m=384,
                            n=384,
                            **kwargs)
    return model
