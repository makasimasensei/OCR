import torch
from torch import nn


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


if __name__ == '__main__':
    input_x = torch.randn(1, 3136, 768)
    dwc = DWConv()
    out = dwc(input_x, 56, 56)
    print(out.shape)

