"""
@author: yangqiang
@contact: whuhit2020@gmail.com
@file: demo.py
@time: 2020/9/10 10:22
"""
from graphs.backbone.resnet import resnet18_backbone
from graphs.backbone.FPN import FPN
import torch
from graphs.models.efficientnet_pytorch import EfficientNet


# net = EfficientNet.from_name("efficientnet-b0")
# input_ = torch.rand(1, 3, 550, 550)
# # features = net.extract_features(input_)  # 1/32
# points = net.extract_endpoints(input_)

# print(features.shape)
# print(points['reduction_1'].shape)
# print(points['reduction_2'].shape)
# print(points['reduction_3'].shape)
# print(points['reduction_4'].shape)
# print(points['reduction_5'].shape)

# net = resnet18_backbone()
# c2, c3, c4, c5 = net(input_)
#
# fpn = FPN(in_channels=(128, 256, 512), channel_out=256)
#
# outs = fpn([c3, c4, c5])
#
# for out in outs:
#     print(out.shape)

import cv2

img = cv2.imread("/Users/yang/Downloads/face.jpeg")

cv2.imshow("", img)
cv2.waitKey()
