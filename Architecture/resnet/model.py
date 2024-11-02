import torch
from matplotlib import pyplot as plt
from torch import nn

from Architecture.resnet.fpn import FeaturePyramid
from Architecture.resnet.resnet import resnet50


class My_ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet50(in_channels=3, pretrained=True)
        self.fpn = FeaturePyramid()

    def forward(self, x):
        x = self.resnet(x)
        # for j in range(10):
        #     temp = x[3].cpu().detach().numpy()
        #     plt.imshow(temp[0, j])
        #     plt.savefig(f"C:/Users/14485/Pictures/Screenshots/ResNet_3{j}.png")
        #     plt.show()
        x = self.fpn(x)
        return x


if __name__ == "__main__":
    net = My_ResNet()
    x = torch.zeros(2, 3, 640, 640)
    out = net(x)
    print(out.shape)
