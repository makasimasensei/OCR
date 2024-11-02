import math

import cv2
import torch
from matplotlib import pyplot as plt
from torch import nn
import torch.nn.functional as F


class MaxPool2dStaticSamePadding(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.stride = self.pool.stride
        self.kernel_size = self.pool.kernel_size

        self.stride = [self.stride] * 2
        self.kernel_size = [self.kernel_size] * 2

    def forward(self, x):
        h, w = x.shape[-2:]

        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.pool(x)
        return x


class SeparableConvBlock(nn.Module):
    def __init__(self):
        super(SeparableConvBlock, self).__init__()

        self.depthwise_conv = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, groups=256, bias=False)
        self.pointwise_conv = nn.Conv2d(256, 256, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(num_features=256, momentum=0.01, eps=1e-3)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x


class BMFPN(nn.Module):
    def __init__(self, embed_dims=[64, 128, 256, 512], ):
        super(BMFPN, self).__init__()

        self.in5 = nn.Conv2d(embed_dims[-1], 256, 1, bias=False)
        self.in4 = nn.Conv2d(embed_dims[-2], 256, 1, bias=False)
        self.in3 = nn.Conv2d(embed_dims[-3], 256, 1, bias=False)
        self.in2 = nn.Conv2d(embed_dims[-4], 256, 1, bias=False)

        self.up5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')

        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()
        self.p2_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p2_w1_relu = nn.ReLU()

        self.conv4_up = SeparableConvBlock()
        self.conv3_up = SeparableConvBlock()
        self.conv2_up = SeparableConvBlock()

        self.conv3_down = SeparableConvBlock()
        self.conv4_down = SeparableConvBlock()
        self.conv5_down = SeparableConvBlock()

        self.p3_downsample = MaxPool2dStaticSamePadding()
        self.p4_downsample = MaxPool2dStaticSamePadding()
        self.p5_downsample = MaxPool2dStaticSamePadding()

        self.p3_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p3_w2_relu = nn.ReLU()
        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()

        self.out5 = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1, bias=False),
            nn.Upsample(scale_factor=8, mode='nearest'))
        self.out4 = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1, bias=False),
            nn.Upsample(scale_factor=4, mode='nearest'))
        self.out3 = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1, bias=False),
            nn.Upsample(scale_factor=2, mode='nearest'))
        self.out2 = nn.Conv2d(256, 64, 3, padding=1, bias=False)

        self.binarize = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, 2, 2),
            nn.Sigmoid())

        self.thresh = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, 2, 2),
            nn.Sigmoid())

        self.thresh.apply(self.weights_init)
        self.binarize.apply(self.weights_init)

        self.in5.apply(self.weights_init)
        self.in4.apply(self.weights_init)
        self.in3.apply(self.weights_init)
        self.in2.apply(self.weights_init)
        self.out5.apply(self.weights_init)
        self.out4.apply(self.weights_init)
        self.out3.apply(self.weights_init)
        self.out2.apply(self.weights_init)

    @staticmethod
    def weights_init(m):
        class_name = m.__class__.__name__
        if class_name.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif class_name.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    @staticmethod
    def step_function(x, y):
        return torch.reciprocal(1 + torch.exp(-50 * (x - y)))

    def forward(self, features):
        c2, c3, c4, c5 = features
        in5 = self.in5(c5)
        in4 = self.in4(c4)
        in3 = self.in3(c3)
        in2 = self.in2(c2)

        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + 1e-3)
        p4_up = self.conv4_up(weight[0] * in4 + weight[1] * self.up5(in5))

        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + 1e-3)
        p3_up = self.conv3_up(weight[0] * in3 + weight[1] * self.up4(p4_up))

        p2_w1 = self.p2_w1_relu(self.p2_w1)
        weight = p2_w1 / (torch.sum(p2_w1, dim=0) + 1e-3)
        p2_up = self.conv2_up(weight[0] * in2 + weight[1] * self.up3(p3_up))

        p3_w2 = self.p3_w2_relu(self.p3_w2)
        weight = p3_w2 / (torch.sum(p3_w2, dim=0) + 1e-3)
        p3_out = self.conv3_down(weight[0] * in3 + weight[1] * p3_up + weight[2] * self.p3_downsample(p2_up))

        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + 1e-3)
        p4_out = self.conv4_down(weight[0] * in4 + weight[1] * p4_up + weight[2] * self.p4_downsample(p3_out))

        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + 1e-3)
        p5_out = self.conv5_down(weight[0] * in5 + weight[1] * self.p5_downsample(p4_out))

        p5 = self.out5(p5_out)
        p4 = self.out4(p4_out)
        p3 = self.out3(p3_out)
        p2 = self.out2(p2_up)

        fuse = torch.cat((p5, p4, p3, p2), 1)
        binary = self.binarize(fuse)
        thresh = self.thresh(fuse)
        thresh_binary = self.step_function(binary, thresh)

        result = torch.cat((binary, thresh, thresh_binary), 1)
        return result


if __name__ == '__main__':
    x1 = torch.randn(1, 512, 20, 20)
    x2 = torch.randn(1, 256, 40, 40)
    x3 = torch.randn(1, 128, 80, 80)
    x4 = torch.randn(1, 64, 160, 160)
    input_x = (x4, x3, x2, x1)
    bmfpn = BMFPN()
    total = sum([param.nelement() for param in bmfpn.parameters()])
    print("Number of parameter: %.2fM" % (total/1e6))
    out = bmfpn(input_x)
    print(out.shape)
