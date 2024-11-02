import torch
import torch.nn as nn


class ASF(nn.Module):
    def __init__(self, in_channels=None):
        super(ASF, self).__init__()
        if in_channels is None:
            in_channels = 768
        self.in1 = nn.Conv2d(in_channels, 256, kernel_size=1, bias=False)
        self.in1.apply(self.weights_init)

        self.concat_attention = ScaleFeatureSelection(256, 256,
                                                      attention_type='scale_channel_spatial')

        self.thresh = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, 2, 2),
            nn.Sigmoid())
        self.binarize = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, 2, 2),
            nn.Sigmoid())

        self.binarize.apply(self.weights_init)
        self.thresh.apply(self.weights_init)

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

    def forward(self, x):
        # 提取基础特征
        in1 = self.in1(x)

        fuse = self.concat_attention(in1)

        thresh_output = self.thresh(fuse)
        binarize_output = self.binarize(fuse)
        thresh_binary = self.step_function(binarize_output, thresh_output)
        fuse = torch.cat((binarize_output, thresh_output, thresh_binary), dim=1)
        return fuse


class ScaleChannelSpatialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, num_features, init_weight=True):
        super(ScaleChannelSpatialAttention, self).__init__()
        self.channel_wise = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, out_planes, 1, bias=False),
            # nn.BatchNorm2d(out_planes),
            nn.ReLU(),
            nn.Conv2d(out_planes, in_planes, 1, bias=False)
        )
        self.spatial_wise = nn.Sequential(
            # Nx1xHxW
            nn.Conv2d(1, 1, 3, bias=False, padding=1),
            nn.ReLU(),
            nn.Conv2d(1, 1, 1, bias=False),
            nn.Sigmoid()
        )
        self.attention_wise = nn.Sequential(
            nn.Conv2d(in_planes, num_features, 1, bias=False),
            nn.Sigmoid()
        )
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # global_x = self.avgpool(x)
        # shape Nx4x1x1
        global_x = self.channel_wise(x).sigmoid()
        # shape: NxCxHxW
        global_x = global_x + x
        # shape:Nx1xHxW
        x = torch.mean(global_x, dim=1, keepdim=True)
        global_x = self.spatial_wise(x) + global_x
        global_x = self.attention_wise(global_x)
        return global_x


class ScaleFeatureSelection(nn.Module):
    def __init__(self, in_channels, inter_channels, out_features_num=4, attention_type='scale_spatial'):
        super(ScaleFeatureSelection, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.out_features_num = out_features_num
        self.conv = nn.Conv2d(in_channels, inter_channels, 3, padding=1)
        self.type = attention_type
        self.enhanced_attention = ScaleChannelSpatialAttention(inter_channels, inter_channels // 2, out_features_num)

    def _initialize_weights(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def forward(self, concat_x):
        concat_x = self.conv(concat_x)
        score = self.enhanced_attention(concat_x)
        x = score[:, 0:1] * concat_x
        return x


if __name__ == "__main__":
    # 创建特征金字塔模型实例
    fpn1 = ASF()

    # 使用示例输入进行测试
    input5 = torch.randn(1, 768, 160, 160)

    outputs = fpn1(input5)
    p = outputs

    # 输出特征金字塔的各个层级特征形状
    print("P shape:", p.shape)
