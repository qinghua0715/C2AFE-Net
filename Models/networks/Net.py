# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 09:57:49 2019

@author: Fsl
"""
from scipy import ndimage
import torch
from torchvision import models
import torch.nn as nn
# from .resnet import resnet34
# from resnet import resnet34
# import resnet
from torch.nn import functional as F
# import torchsummary
from torch.nn import init
import numpy as np
from functools import partial
from thop import profile
from Models.networks.SegNeXt import MSCAN

up_kwargs = {'mode': 'bilinear', 'align_corners': True}
BatchNorm2d = nn.BatchNorm2d
nonlinearity = partial(F.relu, inplace=True)


class SpatialAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttentionBlock, self).__init__()
        self.query = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True)
        )
        self.key = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True)
        )
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: input( BxCxHxW )
        :return: affinity value + x
        """
        B, C, H, W = x.size()
        # compress x: [B,C,H,W]-->[B,H*W,C], make a matrix transpose
        proj_query = self.query(x).view(B, -1, W * H).permute(0, 2, 1)
        proj_key = self.key(x).view(B, -1, W * H)
        affinity = torch.matmul(proj_query, proj_key)
        affinity = self.softmax(affinity)
        proj_value = self.value(x).view(B, -1, H * W)
        weights = torch.matmul(proj_value, affinity.permute(0, 2, 1))
        weights = weights.view(B, C, H, W)
        out = self.gamma * weights + x
        return out


class ChannelAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttentionBlock, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: input( BxCxHxW )
        :return: affinity value + x
        """
        B, C, H, W = x.size()
        proj_query = x.view(B, C, -1)
        proj_key = x.view(B, C, -1).permute(0, 2, 1)
        affinity = torch.matmul(proj_query, proj_key)
        affinity_new = torch.max(affinity, -1, keepdim=True)[0].expand_as(affinity) - affinity
        affinity_new = self.softmax(affinity_new)
        proj_value = x.view(B, C, -1)
        weights = torch.matmul(affinity_new, proj_value)
        weights = weights.view(B, C, H, W)
        out = self.gamma * weights + x
        return out


class AffinityAttention2(nn.Module):
    """ Affinity attention module """

    def __init__(self, in_channels):
        super(AffinityAttention2, self).__init__()
        self.sab = SpatialAttentionBlock(in_channels)
        self.cab = ChannelAttentionBlock(in_channels)
        # self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, x):
        """
        sab: spatial attention block
        cab: channel attention block
        :param x: input tensor
        :return: sab + cab
        """
        sab = self.sab(x)
        cab = self.cab(sab)
        out = sab + cab
        return out



class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = nn.BatchNorm2d(out_planes)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Up_1(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, h, w, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:

            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.att_gate = Attention_Gate(out_channels, h, w)
            self.conv = DoubleConv(2 * out_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x1 = self.up(x1)


        x2 = self.att_gate(x1, x2)

        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)



class CBAM_Module2(nn.Module):
    def __init__(self, channels=512, reduction=2):
        super(CBAM_Module2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid_channel = nn.Sigmoid()
        self.conv_after_concat = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Channel Attention module
        module_input = x
        avg = self.avg_pool(x)
        mx = self.max_pool(x)
        avg = self.fc1(avg)
        mx = self.fc1(mx)
        avg = self.relu(avg)
        mx = self.relu(mx)
        avg = self.fc2(avg)
        mx = self.fc2(mx)
        x = avg + mx
        x = self.sigmoid_channel(x)
        # Spatial Attention module
        x = module_input * x + module_input
        module_input = x
        avg = torch.mean(x, 1, True)
        mx, _ = torch.max(x, 1, True)
        x = torch.cat((avg, mx), 1)
        x = self.conv_after_concat(x)
        x = self.sigmoid_spatial(x)
        x = module_input * x + module_input
        return x


class Attention_Gate(nn.Module):

    def __init__(self, channel, h, w):
        super(Attention_Gate, self).__init__()
        # self.h = h
        # self.w = w
        #
        # self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
        # self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))
        #
        # self.block = nn.Sequential(
        #     nn.Conv2d(h + w, h + w, kernel_size=(7, 1), padding=(3, 0), bias=False),
        # )

        # # self.softmax = nn.Softmax(dim=1)

        self.W_x = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(channel),
            nn.GELU()
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_x = nn.Conv1d(1, 1, kernel_size=7, padding=(7 - 1) // 2, bias=False)

        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.Sigmoid()
        )

        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, g, x):
        shortcut = x

        g_hw_avg = torch.mean(g, 1, True)
        # g_hw_avg = self.W_g(g)
        # x_hw_max, _ =torch.max(x, 1, True)

        x = self.W_x(x)

        x_avg = self.avg_pool(x)
        x_avg = self.conv_x(x_avg.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        x = x * g_hw_avg.expand_as(x) * x_avg.expand_as(x)

        x = self.conv(x)
        out = shortcut + self.gamma * x

        return out


class net(nn.Module):
    def __init__(self, channels=3, classes=2, ccm=True, norm_layer=nn.BatchNorm2d, is_training=True, expansion=2,
                 base_channel=32):
        super(net, self).__init__()
        self.encoder = MSCAN()
        # self.backbone = models.resnet34(weights=None)

        # self.up1 = Up(768, 512 // 2, bilinear=True)
        # self.up2 = Up(384, 256 // 2, bilinear=True)
        # self.up3 = Up(192, 64, bilinear=True)

        self.up1 = Up_1(512, 256, 14, 14, bilinear=True)
        self.up2 = Up_1(256, 128, 28, 28, bilinear=True)
        self.up3 = Up_1(128, 64, 56, 56, bilinear=True)
        # self.up4 = Up_1(64, 64, 112, 112, bilinear=True)

        self.affinity_attention = AffinityAttention2(512)
        self.cbam = CBAM_Module2()
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.gamma3 = nn.Parameter(torch.zeros(1))

        self.final = nn.Conv2d(64, classes, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        c2, c3, c4, c5 = self.encoder(x)
        # c2:B, 64, 56, 56  c3:B, 128, 28, 28  c4:B, 256, 14, 14   c5:B, 512, 7, 7

        # x = self.backbone.conv1(x)
        # x = self.backbone.bn1(x)
        # c1 = self.backbone.relu(x)  # 1/2  64
        # x = self.backbone.maxpool(c1)
        # c2 = self.backbone.layer1(x)  # 1/4   64
        # c3 = self.backbone.layer2(c2)  # 1/8   128
        # c4 = self.backbone.layer3(c3)  # 1/16   256
        # c5 = self.backbone.layer4(c4)  # 1/32   512

        # m2, m3, m4 = self.bridge(c2, c3, c4)

        attention = self.affinity_attention(c5)
        cbam_attn = self.cbam(c5)
        c5 = self.gamma1 * attention + self.gamma2 * cbam_attn + self.gamma3 * c5  # 多种并行方式， 用不用bn relu, 用不用scale aware

        d4 = self.up1(c5, c4)
        d3 = self.up2(d4, c3)
        d2 = self.up3(d3, c2)
        # d1 = self.up4(d2, c1)

        # d4 = self.up1(c5, c4)    # 256 14 14
        # d3 = self.up2(d4, c3)    # 128 28 28
        # d2 = self.up3(d3, c2)    # 64 56 56

        out = F.interpolate(d2, scale_factor=4, mode='bilinear', align_corners=True)  # B, 64, 112, 112
        out = self.final(out)
        out = torch.sigmoid(out)

        return out


if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)
    net = net(3, 1)
    out = net(x)
    print(out.shape)
