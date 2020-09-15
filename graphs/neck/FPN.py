"""
@author: yangqiang
@contact: whuhit2020@gmail.com
@file: FPN.py
@time: 2020/9/14 14:02
"""
from torch import nn
from torch.nn import functional as F


class FPN(nn.Module):
    def __init__(self, in_channels=[512, 1024, 2048], channel_out=256):
        """
        FPN,特征金字塔
        :param in_channels: 输入层的通道数 C3, C4, C5
        :param channel_out:
        """
        super(FPN, self).__init__()
        self.in_channels = in_channels

        # 横向层 改变维度，不改变特征图大小
        self.lateral_conv1 = nn.Conv2d(self.in_channels[2], channel_out, kernel_size=1, stride=1, padding=0)
        self.lateral_conv2 = nn.Conv2d(self.in_channels[1], channel_out, kernel_size=1, stride=1, padding=0)
        self.lateral_conv3 = nn.Conv2d(self.in_channels[0], channel_out, kernel_size=1, stride=1, padding=0)

        # 3*3卷积层, 特征平滑， 不改变特征图大小和通道数
        self.smooth_layer_conv = nn.Conv2d(channel_out, channel_out, kernel_size=3, stride=1, padding=1)

        # 下采样层，特征图大小减半，通道不变，p6 p7
        self.down_layer = nn.Conv2d(channel_out, channel_out, kernel_size=3, stride=2, padding=1)

    def forward(self, features):
        """
        :param features:
        :return:
        """
        c3, c4, c5 = features

        p5 = self.lateral_conv1(c5)
        p5 = self.smooth_layer_conv(p5)

        p4 = self.lateral_conv2(c4)
        p4 = F.interpolate(input=p5, size=(p4.size(2), p4.size(3)), mode="bilinear", align_corners=False) + p4
        p4 = self.smooth_layer_conv(p4)

        p3 = self.lateral_conv3(c3)
        p3 = F.interpolate(input=p4, size=(p3.size(2), p3.size(3)), mode="bilinear", align_corners=False) + p3
        p3 = self.smooth_layer_conv(p3)

        p6 = self.down_layer(p5)
        p7 = self.down_layer(p6)

        outs = [p3, p4, p5, p6, p7]

        # relu 可加可不加，看到很多地方都没加。
        # for out in outs:
        #     F.relu(out, inplace=True)

        return outs
