from graphs.backbone.resnet import resnet18
import torch
from torchvision.models import densenet


if __name__ == '__main__':
    img1 = torch.zeros(1, 3, 550, 550)
    img2 = torch.ones(1, 3, 550, 550)
    net = resnet18(num_classes=5)
    out1 = net(img1)
    out2 = net(img2)
    print(out1)
    print(out2)