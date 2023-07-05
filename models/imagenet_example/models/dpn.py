import torch as torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['DPN', 'dpn68', 'dpn68b', 'dpn92', 'dpn98', 'dpn131', 'dpn107']


def dpn68(**kwargs):
    model = DPN(small=True,
                num_init_features=10,
                k_r=128,
                groups=32,
                k_sec=(3, 4, 12, 3),
                inc_sec=(16, 32, 32, 64),
                test_time_pool=True,
                **kwargs)
    return model


def dpn68b(**kwargs):
    model = DPN(small=True,
                num_init_features=10,
                k_r=128,
                groups=32,
                b=True,
                k_sec=(3, 4, 12, 3),
                inc_sec=(16, 32, 32, 64),
                test_time_pool=True,
                **kwargs)
    return model


def dpn92(**kwargs):
    model = DPN(num_init_features=64,
                k_r=96,
                groups=32,
                k_sec=(3, 4, 20, 3),
                inc_sec=(16, 32, 24, 128),
                test_time_pool=True,
                **kwargs)
    return model


def dpn98(**kwargs):
    model = DPN(num_init_features=96,
                k_r=160,
                groups=40,
                k_sec=(3, 6, 20, 3),
                inc_sec=(16, 32, 32, 128),
                test_time_pool=True,
                **kwargs)
    return model


def dpn131(**kwargs):
    model = DPN(num_init_features=128,
                k_r=160,
                groups=40,
                k_sec=(4, 8, 28, 3),
                inc_sec=(16, 32, 32, 128),
                test_time_pool=True,
                **kwargs)
    return model


def dpn107(pretrained='imagenet+5k', **kwargs):
    model = DPN(num_init_features=128,
                k_r=200,
                groups=50,
                k_sec=(4, 8, 20, 3),
                inc_sec=(20, 64, 64, 128),
                test_time_pool=True,
                **kwargs)
    return model


class CatBnAct(nn.Module):

    def __init__(self, in_chs, activation_fn=nn.ReLU(inplace=True)):
        super(CatBnAct, self).__init__()
        self.bn = nn.BatchNorm2d(in_chs, eps=0.001)
        self.act = activation_fn

    def forward(self, x):
        x = torch.cat(x, dim=1) if isinstance(x, tuple) else x
        return self.act(self.bn(x))


class BnActConv2d(nn.Module):

    def __init__(self,
                 in_chs,
                 out_chs,
                 kernel_size,
                 stride,
                 padding=0,
                 groups=1,
                 activation_fn=nn.ReLU(inplace=True)):
        super(BnActConv2d, self).__init__()
        self.bn = nn.BatchNorm2d(in_chs, eps=0.001)
        self.act = activation_fn
        self.conv = nn.Conv2d(in_chs,
                              out_chs,
                              kernel_size,
                              stride,
                              padding,
                              groups=groups,
                              bias=False)

    def forward(self, x):
        return self.conv(self.act(self.bn(x)))


class InputBlock(nn.Module):

    def __init__(self,
                 num_init_features,
                 kernel_size=7,
                 padding=3,
                 activation_fn=nn.ReLU(inplace=True)):
        super(InputBlock, self).__init__()
        self.conv = nn.Conv2d(3,
                              num_init_features,
                              kernel_size=kernel_size,
                              stride=2,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(num_init_features, eps=0.001)
        self.act = activation_fn
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        return x


class DualPathBlock(nn.Module):

    def __init__(self,
                 in_chs,
                 num_1x1_a,
                 num_3x3_b,
                 num_1x1_c,
                 inc,
                 groups,
                 block_type='normal',
                 b=False):
        super(DualPathBlock, self).__init__()
        self.num_1x1_c = num_1x1_c
        self.inc = inc
        self.b = b
        if block_type == 'proj':
            self.key_stride = 1
            self.has_proj = True
        elif block_type == 'down':
            self.key_stride = 2
            self.has_proj = True
        else:
            assert block_type == 'normal'
            self.key_stride = 1
            self.has_proj = False

        if self.has_proj:
            if self.key_stride == 2:
                self.c1x1_w_s2 = BnActConv2d(in_chs=in_chs,
                                             out_chs=num_1x1_c + 2 * inc,
                                             kernel_size=1,
                                             stride=2)
            else:
                self.c1x1_w_s1 = BnActConv2d(in_chs=in_chs,
                                             out_chs=num_1x1_c + 2 * inc,
                                             kernel_size=1,
                                             stride=1)
        self.c1x1_a = BnActConv2d(in_chs=in_chs,
                                  out_chs=num_1x1_a,
                                  kernel_size=1,
                                  stride=1)
        self.c3x3_b = BnActConv2d(in_chs=num_1x1_a,
                                  out_chs=num_3x3_b,
                                  kernel_size=3,
                                  stride=self.key_stride,
                                  padding=1,
                                  groups=groups)
        if b:
            self.c1x1_c = CatBnAct(in_chs=num_3x3_b)
            self.c1x1_c1 = nn.Conv2d(num_3x3_b,
                                     num_1x1_c,
                                     kernel_size=1,
                                     bias=False)
            self.c1x1_c2 = nn.Conv2d(num_3x3_b, inc, kernel_size=1, bias=False)
        else:
            self.c1x1_c = BnActConv2d(in_chs=num_3x3_b,
                                      out_chs=num_1x1_c + inc,
                                      kernel_size=1,
                                      stride=1)

    def forward(self, x):
        x_in = torch.cat(x, dim=1) if isinstance(x, tuple) else x
        if self.has_proj:
            if self.key_stride == 2:
                x_s = self.c1x1_w_s2(x_in)
            else:
                x_s = self.c1x1_w_s1(x_in)
            x_s1 = x_s[:, :self.num_1x1_c, :, :]
            x_s2 = x_s[:, self.num_1x1_c:, :, :]
        else:
            x_s1 = x[0]
            x_s2 = x[1]
        x_in = self.c1x1_a(x_in)
        x_in = self.c3x3_b(x_in)
        if self.b:
            x_in = self.c1x1_c(x_in)
            out1 = self.c1x1_c1(x_in)
            out2 = self.c1x1_c2(x_in)
        else:
            x_in = self.c1x1_c(x_in)
            out1 = x_in[:, :self.num_1x1_c, :, :]
            out2 = x_in[:, self.num_1x1_c:, :, :]
        resid = x_s1 + out1
        dense = torch.cat([x_s2, out2], dim=1)
        return resid, dense


class DPN(nn.Module):

    def __init__(self,
                 small=False,
                 num_init_features=64,
                 k_r=96,
                 groups=32,
                 b=False,
                 k_sec=(3, 4, 20, 3),
                 inc_sec=(16, 32, 24, 128),
                 num_classes=1000,
                 test_time_pool=False):
        super(DPN, self).__init__()
        self.test_time_pool = test_time_pool
        self.b = b
        bw_factor = 1 if small else 4

        blocks = []

        # conv1
        if small:
            blocks.append(
                InputBlock(num_init_features, kernel_size=3, padding=1))
        else:
            blocks.append(
                InputBlock(num_init_features, kernel_size=7, padding=3))

        # conv2
        bw = 64 * bw_factor
        inc = inc_sec[0]
        r = (k_r * bw) // (64 * bw_factor)
        blocks.append(
            DualPathBlock(num_init_features, r, r, bw, inc, groups, 'proj', b))
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[0] + 1):
            blocks.append(
                DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b))
            in_chs += inc

        # conv3
        bw = 128 * bw_factor
        inc = inc_sec[1]
        r = (k_r * bw) // (64 * bw_factor)
        blocks.append(DualPathBlock(in_chs, r, r, bw, inc, groups, 'down', b))
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[1] + 1):
            blocks.append(
                DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b))
            in_chs += inc

        # conv4
        bw = 256 * bw_factor
        inc = inc_sec[2]
        r = (k_r * bw) // (64 * bw_factor)
        blocks.append(DualPathBlock(in_chs, r, r, bw, inc, groups, 'down', b))
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[2] + 1):
            blocks.append(
                DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b))
            in_chs += inc

        # conv5
        bw = 512 * bw_factor
        inc = inc_sec[3]
        r = (k_r * bw) // (64 * bw_factor)
        blocks.append(DualPathBlock(in_chs, r, r, bw, inc, groups, 'down', b))
        in_chs = bw + 3 * inc
        for i in range(2, k_sec[3] + 1):
            blocks.append(
                DualPathBlock(in_chs, r, r, bw, inc, groups, 'normal', b))
            in_chs += inc
        blocks.append(CatBnAct(in_chs))
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Conv2d(in_chs,
                                    num_classes,
                                    kernel_size=1,
                                    bias=True)

    def logits(self, features):
        if not self.training and self.test_time_pool:
            x = F.avg_pool2d(features, kernel_size=7, stride=1)
            out = self.classifier(x)
            out = adaptive_avgmax_pool2d(out, pool_type='avgmax')
        else:
            x = adaptive_avgmax_pool2d(features, pool_type='avg')
            out = self.classifier(x)
        return out.view(out.size(0), -1)

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


def pooling_factor(pool_type='avg'):
    return 2 if pool_type == 'avgmaxc' else 1


def adaptive_avgmax_pool2d(x, pool_type='avg', padding=0):
    if pool_type == 'avgmaxc':
        x = torch.cat([
            F.avg_pool2d(
                x, kernel_size=(x.size(2), x.size(3)), padding=padding),
            F.max_pool2d(
                x, kernel_size=(x.size(2), x.size(3)), padding=padding)
        ],
                      dim=1)
    elif pool_type == 'avgmax':
        x_avg = F.avg_pool2d(x,
                             kernel_size=(x.size(2), x.size(3)),
                             padding=padding)
        x_max = F.max_pool2d(x,
                             kernel_size=(x.size(2), x.size(3)),
                             padding=padding)
        x = 0.5 * (x_avg + x_max)
    elif pool_type == 'max':
        x = F.max_pool2d(x,
                         kernel_size=(x.size(2), x.size(3)),
                         padding=padding)
    else:
        if pool_type != 'avg':
            print(
                'Invalid pool type %s specified. Defaulting to average pooling.'
                % pool_type)
        x = F.avg_pool2d(x,
                         kernel_size=(x.size(2), x.size(3)),
                         padding=padding)
    return x


class AdaptiveAvgMaxPool2d(torch.nn.Module):

    def __init__(self, output_size=1, pool_type='avg'):
        super(AdaptiveAvgMaxPool2d, self).__init__()
        self.output_size = output_size
        self.pool_type = pool_type
        if pool_type == 'avgmaxc' or pool_type == 'avgmax':
            self.pool = nn.ModuleList([
                nn.AdaptiveAvgPool2d(output_size),
                nn.AdaptiveMaxPool2d(output_size)
            ])
        elif pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d(output_size)
        else:
            if pool_type != 'avg':
                print(
                    'Invalid pool type %s specified. Defaulting to average pooling.'
                    % pool_type)
            self.pool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x):
        if self.pool_type == 'avgmaxc':
            x = torch.cat([p(x) for p in self.pool], dim=1)
        elif self.pool_type == 'avgmax':
            x = 0.5 * torch.sum(torch.stack([p(x) for p in self.pool]),
                                0).squeeze(dim=0)
        else:
            x = self.pool(x)
        return x

    def factor(self):
        return pooling_factor(self.pool_type)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + 'output_size=' + str(self.output_size) \
               + ', pool_type=' + self.pool_type + ')'
