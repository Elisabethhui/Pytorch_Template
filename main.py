from graphs.backbone.resnet import resnext50_32x4d
import torch
from torchvision.models import densenet


if __name__ == '__main__':
    img = torch.rand(1, 3, 550, 550)
    net = resnext50_32x4d()
    outs = net.extract_endpoints(img)
    for out in outs:
        print(out.shape)