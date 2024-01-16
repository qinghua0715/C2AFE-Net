import torch.nn as nn
import torch
from torch.nn import functional as F


# 定义CConv类，将Unet内多次用到的几个相同步骤组合在一起成一个网络，避免重复代码太多
class CConv(nn.Module):
    # 定义网络结构
    def __init__(self, in_ch, out_ch):
        super(CConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

    # 重写父类forward方法
    def forward(self, t):
        t = self.conv1(t)
        t = self.bn1(t)
        t = F.relu(t)
        t = self.conv2(t)
        t = self.bn2(t)
        output = F.relu(t)
        return output


class SE(nn.Module):
    def __init__(self, in_channels, se_ratio=0.25):
        super(SE, self).__init__()
        squeezed_channels = max(int(in_channels * se_ratio), 1)
        self.reduce = nn.Conv2d(in_channels, squeezed_channels, kernel_size=1, bias=True)
        self.expand = nn.Conv2d(squeezed_channels, in_channels, kernel_size=1, bias=True)

    def forward(self, t):
        t_se = F.adaptive_avg_pool2d(t, 1)  # adaptive_avg_pool2d(input， output_size)
        t_se = self.reduce(t_se)
        t_se = F.mish(t_se)  # F.mish是一种类似relu的激活函数
        t_se = self.expand(t_se)
        t_se = torch.sigmoid(t_se)     # [B, 4*C, H/2, W/2]
        out = t * t_se
        return out


class SE2(nn.Module):
    def __init__(self, in_channels, se_ratio=0.25):
        super(SE2, self).__init__()
        squeezed_channels = max(int(in_channels*4 * se_ratio), 1)
        self.reduce = nn.Conv2d(in_channels*4, squeezed_channels, kernel_size=1, bias=True)
        self.expand = nn.Conv2d(squeezed_channels, in_channels*4, kernel_size=1, bias=True)

    def forward(self, t):
        B, C, H, W = t.shape
        out = torch.ones_like(t)
        out1 = torch.zeros_like(t)
        x00 = t[:, :, 0:H//2, 0:W//2]  # [B, C, H/2, W/2]    # 即batch、通道维度不变，高宽方向以2为间隔采样
        x01 = t[:, :, 0:H//2, W//2:]  # [B, C, H/2, W/2]
        x10 = t[:, :, H//2:, 0:W//2]  # [B, C, H/2, W/2]
        x11 = t[:, :, H//2:, W//2:]  # [B, C, H/2, W/2]
        t_se = torch.cat([x00, x01, x10, x11], 1)  # [B, 4*C, H/2, W/2]
        t_se1 = F.adaptive_avg_pool2d(t_se, 1)  # adaptive_avg_pool2d(input， output_size)
        t_se2 = self.reduce(t_se1)
        t_se3 = F.mish(t_se2)  # F.mish是一种类似relu的激活函数
        t_se4 = self.expand(t_se3)
        t_se5 = torch.sigmoid(t_se4)     # [B, 4*C, H/2, W/2]
        t_se6 = torch.chunk(t_se5, chunks=4, dim=1)
        out1[:, :, 0:H//2, 0:W//2] = torch.mul(out[:, :, 0:H//2, 0:W//2], t_se6[0])
        out1[:, :, 0:H//2, W//2:] = torch.mul(out[:, :, 0:H//2, W//2:], t_se6[1])
        out1[:, :, H//2:, 0:W//2] = torch.mul(out[:, :, H//2:, 0:W//2], t_se6[2])
        out1[:, :, H//2:, W//2:] = torch.mul(out[:, :, H//2:, W//2:], t_se6[3])
        out2 = t * out1
        return out2


class SE4(nn.Module):
    def __init__(self, in_channels, se_ratio=0.25):
        super(SE4, self).__init__()
        squeezed_channels = max(int(in_channels*16 * se_ratio), 1)
        self.reduce = nn.Conv2d(in_channels*16, squeezed_channels, kernel_size=1, bias=True)
        self.expand = nn.Conv2d(squeezed_channels, in_channels*16, kernel_size=1, bias=True)

    def forward(self, t):
        B, C, H, W = t.shape
        # 如果输入feature map的H，W不是4的整数倍，需要进行padding
        pad_input = (H % 4 != 0) or (W % 4 != 0)
        if pad_input:
            # to pad the last 3 dimensions, starting from the last dimension and moving forward. 填充最后三个维度，且倒着开始，即从C开始填充
            # (C_front, C_back, W_left, W_right, H_top, H_bottom)
            # 注意这里的Tensor通道是[B, H, W, C]，所以会和官方文档有些不同
            t = F.pad(t, (0, 0, 0, H % 4, 0, W % 4))
        B, C, H1, W1 = t.shape
        out = torch.ones_like(t)
        out1 = torch.zeros_like(t)
        grid_h = H1//4
        grid_w = W1//4
        x00 = t[:, :, 0:grid_h, 0:grid_w]  # [B, C, H/4, W/4]    # 即batch、通道维度不变，高宽方向以2为间隔采样
        x01 = t[:, :, 0:grid_h, grid_w:2*grid_w]  # [B, C, H/4, W/4]
        x02 = t[:, :, 0:grid_h, 2*grid_w:3*grid_w]
        x03 = t[:, :, 0:grid_h, 3*grid_w:]

        x10 = t[:, :, grid_h:2*grid_h, 0:grid_w]
        x11 = t[:, :, grid_h:2*grid_h, grid_w:2*grid_w]
        x12 = t[:, :, grid_h:2*grid_h, 2*grid_w:3*grid_w]
        x13 = t[:, :, grid_h:2*grid_h, 3*grid_w:]

        x20 = t[:, :, 2*grid_h:3*grid_h, 0:grid_w]
        x21 = t[:, :, 2*grid_h:3*grid_h, grid_w:2*grid_w]
        x22 = t[:, :, 2*grid_h:3*grid_h, 2*grid_w:3*grid_w]
        x23 = t[:, :, 2*grid_h:3*grid_h, 3*grid_w:]

        x30 = t[:, :, 3*grid_h:, 0:grid_w]
        x31 = t[:, :, 3*grid_h:, grid_w:2*grid_w]
        x32 = t[:, :, 3*grid_h:, 2*grid_w:3*grid_w]
        x33 = t[:, :, 3*grid_h:, 3*grid_w:]
        # [B, 16*C, H//4, W//4]
        t_se = torch.cat([x00, x01, x02, x03, x10, x11, x12, x13, x20, x21, x22, x23, x30, x31, x32, x33], 1)
        t_se1 = F.adaptive_avg_pool2d(t_se, 1)   # adaptive_avg_pool2d(input， output_size)
        t_se2 = self.reduce(t_se1)
        t_se3 = F.mish(t_se2)  # F.mish是一种类似relu的激活函数
        t_se4 = self.expand(t_se3)
        t_se5 = torch.sigmoid(t_se4)     # [B, 16*C, H//4, W//4]
        t_se6 = torch.chunk(t_se5, chunks=16, dim=1)
        out1[:, :, 0:grid_h, 0:grid_w] = torch.mul(out[:, :, 0:grid_h, 0:grid_w], t_se6[0])
        out1[:, :, 0:grid_h, grid_w:2*grid_w] = torch.mul(out[:, :, 0:grid_h, grid_w:2*grid_w], t_se6[1])
        out1[:, :, 0:grid_h, 2*grid_w:3*grid_w] = torch.mul(out[:, :, 0:grid_h, 2*grid_w:3*grid_w], t_se6[2])
        out1[:, :, 0:grid_h, 3*grid_w:] = torch.mul(out[:, :, 0:grid_h, 3*grid_w:], t_se6[3])

        out1[:, :, grid_h:2*grid_h, 0:grid_w] = torch.mul(out[:, :, grid_h:2*grid_h, 0:grid_w], t_se6[4])
        out1[:, :, grid_h:2*grid_h, grid_w:2*grid_w] = torch.mul(out[:, :, grid_h:2*grid_h, grid_w:2*grid_w], t_se6[5])
        out1[:, :, grid_h:2*grid_h, 2*grid_w:3*grid_w] = torch.mul(out[:, :, grid_h:2*grid_h, 2*grid_w:3*grid_w], t_se6[6])
        out1[:, :, grid_h:2*grid_h, 3*grid_w:] = torch.mul(out[:, :, grid_h:2*grid_h, 3*grid_w:], t_se6[7])

        out1[:, :, 2*grid_h:3*grid_h, 0:grid_w] = torch.mul(out[:, :, 2*grid_h:3*grid_h, 0:grid_w], t_se6[8])
        out1[:, :, 2*grid_h:3*grid_h, grid_w:2*grid_w] = torch.mul(out[:, :, 2*grid_h:3*grid_h, grid_w:2*grid_w], t_se6[9])
        out1[:, :, 2*grid_h:3*grid_h, 2*grid_w:3*grid_w] = torch.mul(out[:, :, 2*grid_h:3*grid_h, 2*grid_w:3*grid_w], t_se6[10])
        out1[:, :, 2*grid_h:3*grid_h, 3*grid_w:] = torch.mul(out[:, :, 2*grid_h:3*grid_h, 3*grid_w:], t_se6[11])

        out1[:, :, 3*grid_h:, 0:grid_w] = torch.mul(out[:, :, 3*grid_h:, 0:grid_w], t_se6[12])
        out1[:, :, 3*grid_h:, grid_w:2*grid_w] = torch.mul(out[:, :, 3*grid_h:, grid_w:2*grid_w], t_se6[13])
        out1[:, :, 3*grid_h:, 2*grid_w:3*grid_w] = torch.mul(out[:, :, 3*grid_h:, 2*grid_w:3*grid_w], t_se6[14])
        out1[:, :, 3*grid_h:, 3*grid_w:] = torch.mul(out[:, :, 3*grid_h:, 3*grid_w:], t_se6[15])
        out2 = t * out1
        out2 = out2[:, :, 0:H, 0:W]
        return out2


# 用单跳连接和分支结构指导的上采样
class Merging(nn.Module):


    def __init__(self, skip_dim, dim, norm_layer=nn.BatchNorm2d):
        super().__init__()
        # self.dim = dim
        # self.norm = norm_layer(4 * dim)
        self.conv32 = nn.Conv2d(skip_dim + dim, skip_dim, kernel_size=(3,3), stride=(1,1), padding=1)
        self.conv1 = nn.Conv2d(4*skip_dim, skip_dim, kernel_size=(1,1))
        self.norm = norm_layer(skip_dim)

    def forward(self, x_skip, x):
        """
        x: B, C, H, W
        """
        x_skip_00 = x_skip[:, :, 0::2, 0::2]  # B C H/2 W/2
        x_skip_01 = x_skip[:, :, 1::2, 0::2]  # B C H/2 W/2
        x_skip_10 = x_skip[:, :, 0::2, 1::2]  # B C H/2 W/2
        x_skip_11 = x_skip[:, :, 1::2, 1::2]  # B C H/2 W/2

        x_00 = torch.cat([x, x_skip_00], 1)    # B 2C H/2 W/2
        x_01 = torch.cat([x, x_skip_01], 1)    # B 2C H/2 W/2
        x_10 = torch.cat([x, x_skip_10], 1)    # B 2C H/2 W/2
        x_11 = torch.cat([x, x_skip_11], 1)    # B 2C H/2 W/2
        x_00 = insert(x_00)  # B 2*C H W
        x_01 = insert(x_01)  # B 2*C H W
        x_10 = insert(x_10)  # B 2*C H W
        x_11 = insert(x_11)  # B 2*C H W

        x_00 = self.conv32(x_00)    # B C H W
        x_01 = self.conv32(x_01)     # B C H W
        x_10 = self.conv32(x_10)     # B C H W
        x_11 = self.conv32(x_11)     # B C H W
        x = torch.cat([x_00, x_01, x_10, x_11], 1)  # B 4*C H W

        x = self.conv1(x)    # B C H W
        x = self.norm(x)    # B C H W

        return x


# 定义的用于扩展填0的函数
def insert(x):
    B, C, H, W = x.shape
    x = x.reshape(-1)
    x = x.unsqueeze(1)
    a = torch.zeros_like(x)
    x = torch.cat((x, a), 1)
    x = x.reshape(B, C, H, 2*W)

    x = x.permute(0, 1, 3, 2)
    x = x.reshape(-1)
    x = x.unsqueeze(1)
    b = torch.zeros_like(x)
    x = torch.cat((x, b), 1)
    x = x.reshape(B, C, 2*W, 2*H)
    x = x.permute(0, 1, 3, 2)
    return x


# AG门控
class Attention_block(nn.Module):

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g,
                      F_int,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
            nn.BatchNorm2d(F_int))

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l,
                      F_int,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


# patchmerging下采样
class Merging1(nn.Module):

    def __init__(self, dim, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.reduction = nn.Conv2d(4 * dim, 2 * dim, kernel_size=1, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, C, H, W
        """
        x_00 = x[:, :, 0::2, 0::2]  # B C H/2 W/2
        x_01 = x[:, :, 1::2, 0::2]  # B C H/2 W/2
        x_10 = x[:, :, 0::2, 1::2]  # B C H/2 W/2
        x_11 = x[:, :, 1::2, 1::2]  # B C H/2 W/2

        x = torch.cat([x_00, x_01, x_10, x_11], 1)    # B 4C H/2 W/2

        x = self.norm(x)
        x = self.reduction(x)      # B 2C H/2 W/2

        return x


class CA_Block(nn.Module):
    def __init__(self, channel, h, w, reduction=4):
        super(CA_Block, self).__init__()

        self.h = h
        self.w = w

        # 全局平均池化
        self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_h = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv2d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)
        x_w = self.avg_pool_y(x)

        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_h, x_w), dim=3))))
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([self.h, self.w], dim=3)
        s_h=self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))

        out_put = x * s_h.expand_as(x) * s_w.expand_as(x)

        return out_put


# 以通道维度拼接，执行liner，实现空间关系建立
class CAS_Block(nn.Module):
    def __init__(self, channel, h, w):
        super(CAS_Block, self).__init__()

        self.h = h
        self.w = w

        # 全局平均池化
        self.avg_pool_x = nn.AdaptiveMaxPool2d((h, 1))
        self.avg_pool_y = nn.AdaptiveMaxPool2d((1, w))
        self.block = nn.Sequential(
            nn.Linear(in_features=h + w, out_features=h + w, bias=False),
            nn.ReLU(),
        )
        self.bn_xh = nn.BatchNorm2d(channel)
        self.bn_xw = nn.BatchNorm2d(channel)

    def forward(self, x):
        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)
        x_w = self.avg_pool_y(x)
        x_cat = torch.cat((x_h, x_w), dim=3)
        x_cat = self.block(x_cat)
        x_h, x_w = x_cat.split([self.h, self.w], dim=3)
        s_h = torch.sigmoid(self.bn_xh(x_h.permute(0, 1, 3, 2)))
        s_w = torch.sigmoid(self.bn_xw(x_w))

        out_put = x * s_h.expand_as(x) * s_w.expand_as(x)

        return out_put


# 以高度维度拼接，执行conv7，实现通道关系建立
class CAC_Block(nn.Module):
    def __init__(self, channel, h, w):
        super(CAC_Block, self).__init__()

        self.h = h
        self.w = w

        # 全局平均池化
        self.avg_pool_x = nn.AdaptiveMaxPool2d((h, 1))
        self.avg_pool_y = nn.AdaptiveMaxPool2d((1, w))
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=(3, 7), padding=(1, 3), bias=False),
            nn.ReLU(),
        )
        self.bn_xh = nn.BatchNorm2d(channel)
        self.bn_xw = nn.BatchNorm2d(channel)

    def forward(self, x):
        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)
        x_w = self.avg_pool_y(x)
        x_cat = torch.cat((x_h, x_w), dim=2)
        x_cat = self.block(x_cat)
        x_h, x_w = x_cat.split([1, 1], dim=2)
        s_h = torch.sigmoid(self.bn_xh(x_h.permute(0, 1, 3, 2)))
        s_w = torch.sigmoid(self.bn_xw(x_w))

        out_put = x + x * s_h.expand_as(x) * s_w.expand_as(x)

        return out_put


# 以高度维度拼接，转换通道维度与宽度维度，执行conv7，实现通道关系建立
class CAC_Block1(nn.Module):
    def __init__(self, channel, h, w):
        super(CAC_Block1, self).__init__()

        self.h = h
        self.w = w

        # 全局平均池化
        self.avg_pool_x = nn.AdaptiveMaxPool2d((h, 1))
        self.avg_pool_y = nn.AdaptiveMaxPool2d((1, w))
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=h, out_channels=h, kernel_size=7, padding=3, bias=False),
            nn.ReLU(),
        )
        self.bn_xh = nn.BatchNorm2d(channel)
        self.bn_xw = nn.BatchNorm2d(channel)

    def forward(self, x):
        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)
        x_w = self.avg_pool_y(x)
        x_cat = torch.cat((x_h, x_w), dim=2).permute(0, 3, 2, 1)
        x_cat = self.block(x_cat).permute(0, 3, 2, 1)
        x_h, x_w = x_cat.split([1, 1], dim=2)
        s_h = torch.sigmoid(self.bn_xh(x_h.permute(0, 1, 3, 2)))
        s_w = torch.sigmoid(self.bn_xw(x_w))

        out_put = x + x * s_h.expand_as(x) * s_w.expand_as(x)

        return out_put


class CAS_Block1(nn.Module):
    def __init__(self, channel, h, w):
        super(CAS_Block1, self).__init__()

        self.h = h
        self.w = w

        # 全局平均池化
        self.avg_pool_x = nn.AdaptiveMaxPool2d((h, 1))
        self.avg_pool_y = nn.AdaptiveMaxPool2d((1, w))
        self.block = nn.Sequential(
            nn.Conv2d(h+w, h+w, kernel_size=1),
            nn.Conv2d(h+w, h+w, kernel_size=7, padding=(7 - 1) // 2, bias=False),
        )
        self.bn_xh = nn.BatchNorm2d(channel)
        self.bn_xw = nn.BatchNorm2d(channel)

    def forward(self, x):
        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)
        x_w = self.avg_pool_y(x)
        x_cat = torch.cat((x_h, x_w), dim=3).permute(0, 3, 2, 1)
        x_cat = self.block(x_cat).permute(0, 3, 2, 1)
        x_h, x_w = x_cat.split([self.h, self.w], dim=3)
        s_h = torch.sigmoid(self.bn_xh(x_h.permute(0, 1, 3, 2)))
        s_w = torch.sigmoid(self.bn_xw(x_w))

        out_put = x + x * s_h.expand_as(x) * s_w.expand_as(x)

        return out_put


class SpatialAttentionBlock(nn.Module):
    def __init__(self):
        super(SpatialAttentionBlock, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: input( BxCxHxW )
        :return: affinity value + x
        """
        mx, _ = torch.max(x, 1, True)
        B, C, H, W = mx.size()
        # compress x: [B,C,H,W]-->[B,H*W,C], make a matrix transpose
        proj_query = mx.view(B, -1, W * H).permute(0, 2, 1)
        proj_key = mx.view(B, -1, W * H)
        affinity = torch.matmul(proj_query, proj_key)
        affinity = self.softmax(affinity)
        proj_value = mx.view(B, -1, H * W)
        weights = torch.matmul(proj_value, affinity.permute(0, 2, 1))
        weights = weights.view(B, C, H, W)
        out = self.gamma * weights * x + x
        return out


class ChannelAttentionBlock(nn.Module):
    def __init__(self, h=56, w=56):
        super(ChannelAttentionBlock, self).__init__()
        self.max_pool_x = nn.AdaptiveMaxPool2d((h, 1))
        self.max_pool_y = nn.AdaptiveMaxPool2d((1, w))
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: input( BxCxHxW )
        :return: affinity value + x
        """

        x_h = self.max_pool_x(x)
        B, C, H, W = x_h.size()
        proj_query = x_h.view(B, C, -1)
        proj_key = x_h.view(B, C, -1).permute(0, 2, 1)
        affinity = torch.matmul(proj_query, proj_key)
        affinity_new = torch.max(affinity, -1, keepdim=True)[0].expand_as(affinity) - affinity
        affinity_new = self.softmax(affinity_new)
        proj_value = x_h.view(B, C, -1)
        weights1 = torch.matmul(affinity_new, proj_value)
        weights1 = weights1.view(B, C, H, W)

        x_w = self.max_pool_y(x)
        B, C, H, W = x_w.size()
        proj_query = x_w.view(B, C, -1)
        proj_key = x_w.view(B, C, -1).permute(0, 2, 1)
        affinity = torch.matmul(proj_query, proj_key)
        affinity_new = torch.max(affinity, -1, keepdim=True)[0].expand_as(affinity) - affinity
        affinity_new = self.softmax(affinity_new)
        proj_value = x_w.view(B, C, -1)
        weights2 = torch.matmul(affinity_new, proj_value)
        weights2 = weights2.view(B, C, H, W)
        out = self.gamma * weights1 * x * weights2 + x

        return out


class Cross_Attention(nn.Module):
    def __init__(self, channel, h, w):
        super(Cross_Attention, self).__init__()
        self.max_pool_x = nn.AdaptiveMaxPool2d((h, 1))
        self.max_pool_y = nn.AdaptiveMaxPool2d((1, w))
        self.bn1 = nn.BatchNorm2d(1)
        self.bn_cha = nn.BatchNorm2d(channel)

    def forward(self, x):
        B, C, H, W = x.size()
        x_hw, _ = torch.max(x, 1, True)
        x_hw = x_hw.squeeze()    # B, 1, H, W -> # B, H, W
        x_ch = self.max_pool_x(x).squeeze()    # B, C, H, 1 -> # B, C, H
        x_cw = self.max_pool_y(x).squeeze()    # B, C, 1, W -> # B, C, W
        if B == 1:
            x_hw = x_hw.unsqueeze(0)
            x_cw = x_cw.unsqueeze(0)
            x_ch = x_ch.unsqueeze(0)

        q3, k3, v3 = x_cw.clone(), x_cw.clone(), x_cw.clone()    # B, C, W

        q1, k1, v1 = x_hw.clone(),  x_hw.clone(), x_hw.clone()   # H, W
        q2, k2, v2 = x_ch.clone().permute(0, 2, 1), x_ch.clone().permute(0, 2, 1), x_ch.clone().permute(0, 2, 1)  # H, C

        sim12 = torch.matmul(q1.permute(0, 2, 1), k2)    # W, C
        # sim21= sim12.permute(0, 2, 1)    # C, W
        attn12 = F.softmax(sim12.view(B, -1), dim=-1).view(B, W, C)
        attn21 = attn12.permute(0, 2, 1)   # C, W
        y12 = torch.matmul(attn12, v2.permute(0, 2, 1)).permute(0, 2, 1).contiguous()   # H, W
        y21 = torch.matmul(attn21, v1.permute(0, 2, 1))    # C, H


        q1, k1, v1 = x_hw.clone().permute(0, 2, 1), x_hw.clone().permute(0, 2, 1), x_hw.clone().permute(0, 2, 1)  # W, H
        q3, k3, v3 = x_cw.clone().permute(0, 2, 1), x_cw.clone().permute(0, 2, 1), x_cw.clone().permute(0, 2, 1)  # W, C

        sim13 = torch.matmul(q1.permute(0, 2, 1), k3)  # H, C
        # sim31 = sim13.permute(0, 2, 1)  # C, H
        attn13 = F.softmax(sim13.view(B, -1), dim=-1).view(B, H, C)
        attn31 = attn13.permute(0, 2, 1)   # C, H
        y13 = torch.matmul(attn13, v3.permute(0, 2, 1))    # H, W
        y31 = torch.matmul(attn31, v1.permute(0, 2, 1))    # C, W


        q2, k2, v2 = x_ch.clone(), x_ch.clone(), x_ch.clone()  # C, H
        q3, k3, v3 = x_cw.clone(), x_cw.clone(), x_cw.clone()  # C, W

        sim23 = torch.matmul(q2.permute(0, 2, 1), k3)  # H, W
        # sim32 = sim13.permute(0, 2, 1)  # W, H
        attn23 = F.softmax(sim23.view(B, -1), dim=-1).view(B, H, W)
        attn32 = attn23.permute(0, 2, 1)  # W, H
        y23 = torch.matmul(attn23, v3.permute(0, 2, 1)).permute(0, 2, 1).contiguous()  # C, H
        y32 = torch.matmul(attn32, v2.permute(0, 2, 1)).permute(0, 2, 1).contiguous()  # C, W

        y12, y13 = y12.unsqueeze(1), y13.unsqueeze(1)
        y21, y23 = y21.unsqueeze(-1), y23.unsqueeze(-1)
        y31, y32 = y31.unsqueeze(2), y32.unsqueeze(2)

        y12 = torch.sigmoid(self.bn1(y12))
        y13 = torch.sigmoid(self.bn1(y13))
        y21 = torch.sigmoid(self.bn_cha(y21))
        y23 = torch.sigmoid(self.bn_cha(y23))
        y31 = torch.sigmoid(self.bn_cha(y31))
        y32 = torch.sigmoid(self.bn_cha(y32))

        out = x*y12*y13*y21*y23*y31*y32 + x

        return out


class Attention_block1(nn.Module):
    def __init__(self, channel, h, w):
        super(Attention_block1, self).__init__()
        self.channel_att = ChannelAttentionBlock(h, w)
        self.spatial_att = SpatialAttentionBlock()
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, t):
        shortcut = t
        t = self.channel_att(t)
        t = self.spatial_att(t)
        out = self.gamma*t*shortcut + shortcut
        return out


class Attention_Gate(nn.Module):

    def __init__(self, channel, h, w):
        super(Attention_Gate, self).__init__()
        self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))

        self.W_x = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=7, stride=1, padding=3, bias=True),
            nn.GELU()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(channel),
            nn.Sigmoid()
        )

        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, g, x):
        shortcut = x
        g_hw_avg = torch.mean(g, 1, True)
        g_ch_avg = self.avg_pool_x(g)
        g_cw_avg = self.avg_pool_y(g)

        x = self.W_x(x)
        x = x * g_hw_avg * g_ch_avg * g_cw_avg

        x = self.conv(x)
        out = shortcut + self.gamma*x

        return out


# 用多尺度SE实现空间注意力的Unet
class Unet_1(nn.Module):
    # 定义网络结构共9层，四次下采样，四次上采样
    def __init__(self, input_channels, output_channels):
        super(Unet_1, self).__init__()
        self.conv1 = CConv(in_ch=input_channels, out_ch=64)
        self.down1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = CConv(in_ch=64, out_ch=128)
        self.down2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = CConv(in_ch=128, out_ch=256)
        self.down3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = CConv(in_ch=256, out_ch=512)
        self.down4 = nn.MaxPool2d(kernel_size=2)
        self.conv5 = CConv(in_ch=512, out_ch=1024)
        self.se11 = SE(in_channels=1024)
        self.se12 = SE2(in_channels=1024)
        self.bn1 = nn.BatchNorm2d(num_features=1024)
        self.up1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.conv6 = CConv(in_ch=1024, out_ch=512)
        self.se21 = SE(in_channels=512)
        self.se22 = SE2(in_channels=512)
        self.bn2 = nn.BatchNorm2d(num_features=512)
        self.up2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.conv7 = CConv(in_ch=512, out_ch=256)
        self.se31 = SE(in_channels=256)
        self.se32 = SE2(in_channels=256)
        self.bn3 = nn.BatchNorm2d(num_features=256)
        self.up3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.conv8 = CConv(in_ch=256, out_ch=128)
        self.se41 = SE(in_channels=128)
        self.se42 = SE2(in_channels=128)
        self.se44 = SE4(in_channels=128)
        self.bn4 = nn.BatchNorm2d(num_features=128)
        self.up4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.conv9 = CConv(in_ch=128, out_ch=64)
        self.se51 = SE(in_channels=64)
        self.se52 = SE2(in_channels=64)
        self.se54 = SE4(in_channels=64)
        self.bn5 = nn.BatchNorm2d(num_features=64)
        self.conv10 = nn.Conv2d(in_channels=64, out_channels=output_channels, kernel_size=1)


    # 重写父类forward方法
    def forward(self, t):
        # layer1
        c1 = self.conv1(t)
        t = self.down1(c1)

        # layer2
        c2 = self.conv2(t)
        t = self.down2(c2)

        # layer3
        c3 = self.conv3(t)
        t = self.down3(c3)

        # layer4
        c4 = self.conv4(t)
        t = self.down4(c4)

        # layer5
        c5 = self.conv5(t)
        # SE_module
        se11 = self.se11(c5)
        se12 = self.se12(c5)
        se = se11 + se12
        se = self.bn1(se)
        t = self.up1(se)

        # layer6
        t = torch.cat([t, c4], dim=1)
        t = self.conv6(t)
        se21 = self.se21(t)
        se22 = self.se22(t)
        se = se21 + se22
        se = self.bn2(se)
        t = self.up2(se)

        # layer7
        t = torch.cat([t, c3], dim=1)
        t = self.conv7(t)
        se31 = self.se31(t)
        se32 = self.se32(t)
        se = se31 + se32
        se = self.bn3(se)
        t = self.up3(se)

        # layer8
        t = torch.cat([t, c2], dim=1)
        t = self.conv8(t)
        se41 = self.se41(t)
        se42 = self.se42(t)
        se44 = self.se44(t)
        se = se41 + se42 + se44
        se = self.bn4(se)
        t = self.up4(se)

        # layer9
        t = torch.cat([t, c1], dim=1)
        t = self.conv9(t)
        se51 = self.se51(t)
        se52 = self.se52(t)
        se54 = self.se54(t)
        se = se51 + se52 + se54
        se = self.bn5(se)
        t = self.conv10(se)
        out = torch.sigmoid(t)
        return out


# 纯Unet
class Unet(nn.Module):
    # 定义网络结构共9层，四次下采样，四次上采样
    def __init__(self, input_channels, output_channels):
        super(Unet, self).__init__()
        self.conv1 = CConv(in_ch=input_channels, out_ch=64)
        self.down1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = CConv(in_ch=64, out_ch=128)
        self.down2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = CConv(in_ch=128, out_ch=256)
        self.down3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = CConv(in_ch=256, out_ch=512)
        self.down4 = nn.MaxPool2d(kernel_size=2)
        self.conv5 = CConv(in_ch=512, out_ch=1024)

        self.bn1 = nn.BatchNorm2d(num_features=1024)
        self.up1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.conv6 = CConv(in_ch=1024, out_ch=512)

        self.bn2 = nn.BatchNorm2d(num_features=512)
        self.up2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.conv7 = CConv(in_ch=512, out_ch=256)

        self.up3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.conv8 = CConv(in_ch=256, out_ch=128)

        self.up4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.conv9 = CConv(in_ch=128, out_ch=64)

        self.conv10 = nn.Conv2d(in_channels=64, out_channels=output_channels, kernel_size=1)


    # 重写父类forward方法
    def forward(self, t):
        # layer1
        c1 = self.conv1(t)
        t = self.down1(c1)

        # layer2
        c2 = self.conv2(t)
        t = self.down2(c2)

        # layer3
        c3 = self.conv3(t)
        t = self.down3(c3)

        # layer4
        c4 = self.conv4(t)
        t = self.down4(c4)

        # layer5
        c5 = self.conv5(t)

        d1 = self.up1(c5)

        # layer6
        t = torch.cat([d1, c4], dim=1)
        t = self.conv6(t)
        d2 = self.up2(t)

        # layer7
        t = torch.cat([d2, c3], dim=1)
        t = self.conv7(t)
        d3 = self.up3(t)

        # layer8
        t = torch.cat([d3, c2], dim=1)
        t = self.conv8(t)
        d4 = self.up4(t)

        # layer9
        t = torch.cat([d4, c1], dim=1)
        t = self.conv9(t)
        t = self.conv10(t)
        out = torch.sigmoid(t)
        return out


# 极简Unet
class Unet_S(nn.Module):
    def __init__(self, input_channels, output_channels, c_list=[8, 16, 32, 64, 128, 256]):
        super(Unet_S, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(c_list[0])
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(c_list[0], c_list[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(c_list[1])
        )

        self.encoder3 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(c_list[2])
        )

        self.encoder4 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(c_list[3])
        )


        self.encoder5 = nn.Sequential(
            nn.Conv2d(c_list[3], c_list[4], kernel_size=3, padding=1),
            nn.BatchNorm2d(c_list[4])
        )


        self.bottle_neck = nn.Sequential(
            nn.Conv2d(c_list[4], c_list[5], kernel_size=3, padding=1),
            nn.Conv2d(c_list[5], c_list[5], kernel_size=3, padding=1),
            nn.Conv2d(c_list[5], c_list[4], kernel_size=3, padding=1)
        )

        self.ca = Cross_Attention(channel=c_list[4], h=14, w=14)

        # self.skip1 = nn.Conv2d(c_list[0], c_list[0], kernel_size=3, padding=1)
        # self.skip2 = nn.Conv2d(c_list[1], c_list[1], kernel_size=3, padding=1)
        # self.skip3 = nn.Conv2d(c_list[2], c_list[2], kernel_size=3, padding=1)
        # self.skip4 = nn.Conv2d(c_list[3], c_list[3], kernel_size=3, padding=1)
        # self.skip5 = nn.Conv2d(c_list[4], c_list[4], kernel_size=3, padding=1)

        self.decoder5 = nn.Sequential(
            nn.Conv2d(c_list[4], c_list[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(c_list[3])
            )

        self.decoder4 = nn.Sequential(
            nn.Conv2d(2*c_list[3], c_list[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(c_list[2])
            )

        self.decoder3 = nn.Sequential(
            nn.Conv2d(2*c_list[2], c_list[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(c_list[1])
        )

        self.decoder2 = nn.Sequential(
            nn.Conv2d(2*c_list[1], c_list[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(c_list[0])
        )

        self.decoder1 = nn.Sequential(
            nn.Conv2d(2*c_list[0], c_list[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(c_list[0])
        )

        self.skip_aug4 = Attention_Gate(64, 14, 14)
        self.skip_aug3 = Attention_Gate(32, 28, 28)
        self.skip_aug2 = Attention_Gate(16, 56, 56)
        self.skip_aug1 = Attention_Gate(8, 112, 112)

        self.final = nn.Conv2d(in_channels=c_list[0], out_channels=output_channels, kernel_size=1)


    # 重写父类forward方法
    def forward(self, x):
        # layer1
        e1 = F.gelu(F.max_pool2d(self.encoder1(x), 2, 2))     # B 8 H/2 W/2
        # s1 = self.skip1(e1)

        # layer2
        e2 = F.gelu(F.max_pool2d(self.encoder2(e1), 2, 2))     # B 16 H/4 W/4
        # s2 = self.skip2(e2)

        # layer3
        e3 = F.gelu(F.max_pool2d(self.encoder3(e2), 2, 2))    # B 32 H/8 W/8
        # s3 = self.skip3(e3)

        # layer4
        e4 = F.gelu(F.max_pool2d(self.encoder4(e3), 2, 2))    # B 64 H/16 W/16
        # s4 = self.skip4(e4)

        # layer5
        e5 = F.gelu(self.encoder5(e4))  # B 128 H/16 W/16
        # s5 = self.skip5(e5)

        d5 = F.gelu(self.bottle_neck(e5))    # B 128 H/16 W/16
        d5 = self.ca(d5)

        # layer6
        # d5_ = torch.cat([d5, e5], dim=1)    # B 256 H/16 W/16
        d4 = F.gelu(self.decoder5(d5))   # B 64 H/16 W/16

        # layer7
        s4 = self.skip_aug4(d4, e4)
        d4_ = torch.cat([d4, s4], dim=1)    # B 128 H/16 W/16
        d3 = F.gelu(F.interpolate(self.decoder4(d4_), scale_factor=(2, 2), mode='bilinear',
                                  align_corners=True))   # B 32 H/8 W/8

        # layer8
        s3 = self.skip_aug3(d3, e3)
        d3_ = torch.cat([d3, s3], dim=1)    # B 64 H/8 W/8
        d2 = F.gelu(F.interpolate(self.decoder3(d3_), scale_factor=(2, 2), mode='bilinear',
                                  align_corners=True))  # B 16 H/4 W/4

        # layer9
        s2 = self.skip_aug2(d2, e2)
        d2_ = torch.cat([d2, s2], dim=1)    # B 32 H/4 W/4
        d1 = F.gelu(F.interpolate(self.decoder2(d2_), scale_factor=(2, 2), mode='bilinear',
                                  align_corners=True))  # B 8 H/2 W/2

        s1 = self.skip_aug1(d1, e1)
        d1_ = torch.cat([d1, s1], dim=1)  # B 16 H/2 W/2
        out = F.gelu(F.interpolate(self.decoder1(d1_), scale_factor=(2, 2), mode='bilinear',
                                  align_corners=True))  # B 8 H W
        out = self.final(out)    # B 1 H W
        out = torch.sigmoid(out)
        return out


# 改变下采样方式
class Unet_d(nn.Module):
    def __init__(self, input_channels, output_channels, c_list=[8, 16, 32, 64, 128, 256]):
        super(Unet_d, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(c_list[0]),
            Merging1(c_list[0])
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(c_list[1]),
            Merging1(c_list[1])
        )

        self.encoder3 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(c_list[2]),
            Merging1(c_list[2])
        )

        self.encoder4 = nn.Sequential(
            nn.Conv2d(c_list[3], c_list[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(c_list[3]),
            Merging1(c_list[3])
        )


        self.encoder5 = nn.Sequential(
            nn.Conv2d(c_list[4], c_list[4], kernel_size=3, padding=1),
            nn.BatchNorm2d(c_list[4])
        )


        self.bottle_neck = nn.Sequential(
            nn.Conv2d(c_list[4], c_list[5], kernel_size=3, padding=1),
            nn.Conv2d(c_list[5], c_list[5], kernel_size=3, padding=1),
            nn.Conv2d(c_list[5], c_list[4], kernel_size=3, padding=1)
        )

        # self.skip1 = nn.Conv2d(c_list[0], c_list[0], kernel_size=3, padding=1)
        # self.skip2 = nn.Conv2d(c_list[1], c_list[1], kernel_size=3, padding=1)
        # self.skip3 = nn.Conv2d(c_list[2], c_list[2], kernel_size=3, padding=1)
        # self.skip4 = nn.Conv2d(c_list[3], c_list[3], kernel_size=3, padding=1)
        # self.skip5 = nn.Conv2d(c_list[4], c_list[4], kernel_size=3, padding=1)

        self.decoder5 = nn.Sequential(
            nn.Conv2d(2*c_list[4], c_list[4], kernel_size=3, padding=1),
            nn.BatchNorm2d(c_list[4])
            )

        self.decoder4 = nn.Sequential(
            nn.Conv2d(2*c_list[4], c_list[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(c_list[3])
            )

        self.decoder3 = nn.Sequential(
            nn.Conv2d(2*c_list[3], c_list[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(c_list[2])
        )

        self.decoder2 = nn.Sequential(
            nn.Conv2d(2*c_list[2], c_list[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(c_list[1])
        )

        self.decoder1 = nn.Sequential(
            nn.Conv2d(2*c_list[1], c_list[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(c_list[0])
        )

        self.final = nn.Conv2d(in_channels=c_list[0], out_channels=output_channels, kernel_size=1)


    # 重写父类forward方法
    def forward(self, x):
        # layer1
        e1 = F.gelu(self.encoder1(x))     # B 8 H/2 W/2
        # s1 = self.skip1(e1)

        # layer2
        e2 = F.gelu(self.encoder2(e1))     # B 16 H/4 W/4
        # s2 = self.skip2(e2)

        # layer3
        e3 = F.gelu(self.encoder3(e2))    # B 24 H/8 W/8
        # s3 = self.skip3(e3)

        # layer4
        e4 = F.gelu(self.encoder4(e3))    # B 32 H/16 W/16
        # s4 = self.skip4(e4)

        # layer5
        e5 = F.gelu(self.encoder5(e4))  # B 48 H/16 W/16
        # s5 = self.skip5(e5)

        d5 = F.gelu(self.bottle_neck(e5))    # B 48 H/16 W/16

        # layer6
        d5_ = torch.cat([d5, e5], dim=1)    # B 96 H/16 W/16
        d4 = F.gelu(self.decoder5(d5_))   # B 32 H/16 W/16

        # layer7
        d4_ = torch.cat([d4, e4], dim=1)    # B 64 H/16 W/16
        d3 = F.gelu(F.interpolate(self.decoder4(d4_), scale_factor=(2, 2), mode='bilinear',
                                  align_corners=True))   # B 24 H/8 W/8

        # layer8
        d3_ = torch.cat([d3, e3], dim=1)    # B 48 H/8 W/8
        d2 = F.gelu(F.interpolate(self.decoder3(d3_), scale_factor=(2, 2), mode='bilinear',
                                  align_corners=True))  # B 16 H/4 W/4

        # layer9
        d2_ = torch.cat([d2, e2], dim=1)    # B 32 H/4 W/4
        d1 = F.gelu(F.interpolate(self.decoder2(d2_), scale_factor=(2, 2), mode='bilinear',
                                  align_corners=True))  # B 8 H/2 W/2

        d1_ = torch.cat([d1, e1], dim=1)  # B 16 H/2 W/2
        out = F.gelu(F.interpolate(self.decoder1(d1_), scale_factor=(2, 2), mode='bilinear',
                                  align_corners=True))  # B 8 H W
        out = self.final(out)    # B 1 H W
        out = torch.sigmoid(out)
        return out



# 用指导上采样替换普通反卷积的Unet
class Unet_2(nn.Module):
    # 定义网络结构共9层，四次下采样，四次上采样
    def __init__(self, input_channels, output_channels):
        super(Unet_2, self).__init__()
        self.conv1 = CConv(in_ch=input_channels, out_ch=64)
        self.down1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = CConv(in_ch=64, out_ch=128)
        self.down2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = CConv(in_ch=128, out_ch=256)
        self.down3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = CConv(in_ch=256, out_ch=512)
        self.down4 = nn.MaxPool2d(kernel_size=2)
        self.conv5 = CConv(in_ch=512, out_ch=1024)

        self.conv31 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv33 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv34 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.up1 = Merging(512, 1024)
        self.up2 = Merging(256, 512)
        self.up3 = Merging(128, 256)
        self.up4 = Merging(64, 128)

        self.bn1 = nn.BatchNorm2d(num_features=1024)
        # self.up1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.conv6 = CConv(in_ch=1024, out_ch=512)

        self.bn2 = nn.BatchNorm2d(num_features=512)
        # self.up2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.conv7 = CConv(in_ch=512, out_ch=256)

        # self.up3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.conv8 = CConv(in_ch=256, out_ch=128)

        # self.up4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.conv9 = CConv(in_ch=128, out_ch=64)

        self.conv10 = nn.Conv2d(in_channels=64, out_channels=output_channels, kernel_size=1)


    # 重写父类forward方法
    def forward(self, t):
        # layer1
        c1 = self.conv1(t)
        t = self.down1(c1)
        c1 = self.conv34(c1)


        # layer2
        c2 = self.conv2(t)
        t = self.down2(c2)
        c2 = self.conv33(c2)

        # layer3
        c3 = self.conv3(t)
        t = self.down3(c3)
        c3 = self.conv32(c3)

        # layer4
        c4 = self.conv4(t)
        t = self.down4(c4)
        c4 = self.conv31(c4)

        # layer5
        c5 = self.conv5(t)
        d1 = self.up1(c4, c5)

        # layer6
        t = torch.cat([d1, c4], dim=1)
        t = self.conv6(t)
        d2 = self.up2(c3, t)

        # layer7
        t = torch.cat([d2, c3], dim=1)
        t = self.conv7(t)
        d3 = self.up3(c2, t)

        # layer8
        t = torch.cat([d3, c2], dim=1)
        t = self.conv8(t)
        d4 = self.up4(c1, t)

        # layer9
        t = torch.cat([d4, c1], dim=1)
        t = self.conv9(t)
        t = self.conv10(t)
        out = torch.sigmoid(t)
        return out


# 从全噪声图恢复图像
class Unet_3(nn.Module):
    # 定义网络结构共9层，四次下采样，四次上采样
    def __init__(self, input_channels, output_channels):
        super(Unet_3, self).__init__()
        self.conv1 = CConv(in_ch=input_channels, out_ch=64)
        self.down1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = CConv(in_ch=64, out_ch=128)
        self.down2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = CConv(in_ch=128, out_ch=256)
        self.down3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = CConv(in_ch=256, out_ch=512)
        self.down4 = nn.MaxPool2d(kernel_size=2)
        self.conv5 = CConv(in_ch=512, out_ch=1024)

        self.Att5 = Attention_block(F_g=1024,
                                    F_l=1024,
                                    F_int=512)
        self.Att4 = Attention_block(F_g=512,
                                    F_l=512,
                                    F_int=256)
        self.Att3 = Attention_block(F_g=256,
                                    F_l=256,
                                    F_int=128)
        self.Att2 = Attention_block(F_g=128,
                                    F_l=128,
                                    F_int=64)
        self.Att1 = Attention_block(F_g=64,
                                    F_l=64,
                                    F_int=1)

        self.up4 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)

        self.up3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)

        self.up2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)

        self.up1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)

        self.conv10 = nn.Conv2d(in_channels=64, out_channels=output_channels, kernel_size=1)


    # 重写父类forward方法
    def forward(self, t):
        B, _, H, W = t.shape
        noise5 = torch.ones([B, 1024, int(H/16), int(W/16)]).cuda()
        # layer1
        c1 = self.conv1(t)
        t = self.down1(c1)

        # layer2
        c2 = self.conv2(t)
        t = self.down2(c2)

        # layer3
        c3 = self.conv3(t)
        t = self.down3(c3)

        # layer4
        c4 = self.conv4(t)
        t = self.down4(c4)

        # layer5
        c5 = self.conv5(t)     # B，1024，H/16, H/16
        noise4 = self.Att5(c5, noise5)   # B，1024，H/16, H/16
        t = self.up4(noise4)

        noise3 = self.Att4(c4, t)
        t = self.up3(noise3)

        noise2 = self.Att3(c3, t)
        t = self.up2(noise2)

        noise1 = self.Att2(c2, t)
        t = self.up1(noise1)

        noise0 = self.Att1(c1, t)

        t = self.conv10(noise0)
        out = torch.sigmoid(t)
        return out


# 测试SE模块代码
if __name__ == '__main__':

    x = torch.rand(16, 3, 224, 224)
    model = Unet_S(3, 1)
    output = model(x)
    print(output.shape)
