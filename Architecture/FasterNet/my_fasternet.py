# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from functools import partial
from typing import List
from torch import Tensor


class Partial_conv3(nn.Module):

    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: Tensor) -> Tensor:
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x: Tensor) -> Tensor:
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        return x


class MLPBlock(nn.Module):

    def __init__(self,
                 num,
                 dim,
                 n_div,
                 mlp_ratio,
                 drop_path,
                 layer_scale_init_value,
                 act_layer,
                 norm_layer,
                 pconv_fw_type
                 ):

        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div

        mlp_hidden_dim = [16, 24, 56, 480]

        mlp_layer: List[nn.Module] = [
            nn.Conv2d(dim, mlp_hidden_dim[num], 1, bias=False),
            norm_layer(mlp_hidden_dim[num]),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim[num], dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        self.spatial_mixing = Partial_conv3(
            dim,
            n_div,
            pconv_fw_type
        )

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x

    def forward_layer_scale(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


class BasicStage(nn.Module):

    def __init__(self,
                 num,
                 dim,
                 depth,
                 n_div,
                 mlp_ratio,
                 drop_path,
                 layer_scale_init_value,
                 norm_layer,
                 act_layer,
                 pconv_fw_type
                 ):

        super().__init__()

        blocks_list = [
            MLPBlock(
                num=num,
                dim=dim,
                n_div=n_div,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path[i],
                layer_scale_init_value=layer_scale_init_value,
                norm_layer=norm_layer,
                act_layer=act_layer,
                pconv_fw_type=pconv_fw_type
            )
            for i in range(depth)
        ]

        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        return x


class PatchEmbed(nn.Module):

    def __init__(self, patch_size, patch_stride, in_chans, embed_dim, norm_layer):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_stride, bias=False)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(self.proj(x))
        return x


class PatchMerging(nn.Module):

    def __init__(self, num, patch_size2, patch_stride2, dim, norm_layer):
        super().__init__()
        self.reduction = nn.Conv2d(dim, num, kernel_size=patch_size2, stride=patch_stride2, bias=False)
        if norm_layer is not None:
            self.norm = norm_layer(num)
        else:
            self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(self.reduction(x))
        return x


class FasterNet(nn.Module):

    def __init__(self,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=16,
                 depths=(1, 2, 8, 2),
                 mlp_ratio=2.,
                 n_div=4,
                 patch_size=4,
                 patch_stride=4,
                 patch_size2=2,  # for subsequent layers
                 patch_stride2=2,
                 patch_norm=True,
                 feature_dim=1280,
                 drop_path_rate=0.1,
                 layer_scale_init_value=0,
                 norm_layer='BN',
                 act_layer='RELU',
                 fork_feat=False,
                 init_cfg=None,
                 pretrained=None,
                 pconv_fw_type='split_cat',
                 **kwargs):
        super().__init__()

        if norm_layer == 'BN':
            norm_layer = nn.BatchNorm2d
        else:
            raise NotImplementedError

        if act_layer == 'GELU':
            act_layer = nn.GELU
        elif act_layer == 'RELU':
            act_layer = partial(nn.ReLU, inplace=True)
        else:
            raise NotImplementedError

        if not fork_feat:
            self.num_classes = num_classes
        self.num_stages = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_stages - 1))
        self.mlp_ratio = mlp_ratio
        self.depths = depths

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            patch_stride=patch_stride,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None
        )

        # stochastic depth decay rule
        dpr = [x.item()
               for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        stages_list1 = []
        stage = BasicStage(num=1,
                           dim=int(embed_dim * 2 ** 0),
                           n_div=n_div,
                           depth=depths[0],
                           mlp_ratio=self.mlp_ratio,
                           drop_path=dpr[sum(depths[:0]):sum(depths[:0 + 1])],
                           layer_scale_init_value=layer_scale_init_value,
                           norm_layer=norm_layer,
                           act_layer=act_layer,
                           pconv_fw_type=pconv_fw_type
                           )
        stages_list1.append(stage)
        stages_list1.append(
            PatchMerging(num=24,
                         patch_size2=patch_size2,
                         patch_stride2=patch_stride2,
                         dim=int(embed_dim * 2 ** 0),
                         norm_layer=norm_layer)
        )
        self.stage1 = nn.Sequential(*stages_list1)

        stages_list2 = []
        stage = BasicStage(num=2,
                           dim=int(embed_dim * 1.5),
                           n_div=n_div,
                           depth=depths[1],
                           mlp_ratio=self.mlp_ratio,
                           drop_path=dpr[sum(depths[:1]):sum(depths[:1 + 1])],
                           layer_scale_init_value=layer_scale_init_value,
                           norm_layer=norm_layer,
                           act_layer=act_layer,
                           pconv_fw_type=pconv_fw_type
                           )
        stages_list2.append(stage)
        stages_list2.append(
            PatchMerging(num=56,
                         patch_size2=patch_size2,
                         patch_stride2=patch_stride2,
                         dim=int(embed_dim * 1.5),
                         norm_layer=norm_layer)
        )
        self.stage2 = nn.Sequential(*stages_list2)

        stages_list3 = []
        stage = BasicStage(num=3,
                           dim=int(embed_dim * 3.5),
                           n_div=n_div,
                           depth=depths[2],
                           mlp_ratio=self.mlp_ratio,
                           drop_path=dpr[sum(depths[:2]):sum(depths[:2 + 1])],
                           layer_scale_init_value=layer_scale_init_value,
                           norm_layer=norm_layer,
                           act_layer=act_layer,
                           pconv_fw_type=pconv_fw_type
                           )
        stages_list3.append(stage)
        stages_list3.append(
            PatchMerging(num=480,
                         patch_size2=patch_size2,
                         patch_stride2=patch_stride2,
                         dim=int(embed_dim * 3.5),
                         norm_layer=norm_layer)
        )
        stage_another = BasicStage(num=2,
                                   dim=int(embed_dim * 30),
                                   n_div=n_div,
                                   depth=depths[3],
                                   mlp_ratio=self.mlp_ratio,
                                   drop_path=dpr[sum(depths[:2]):sum(depths[:3 + 1])],
                                   layer_scale_init_value=layer_scale_init_value,
                                   norm_layer=norm_layer,
                                   act_layer=act_layer,
                                   pconv_fw_type=pconv_fw_type
                                   )
        stages_list3.append(stage_another)
        self.stage3 = nn.Sequential(*stages_list3)

    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # output only the features of last layer for image classification
        x = self.patch_embed(x)
        x1 = x
        x2 = self.stage1(x1)
        x3 = self.stage2(x2)
        x4 = self.stage3(x3)
        return x1, x2, x3, x4


if __name__ == '__main__':
    input = torch.randn(1, 3, 640, 640)
    fun = FasterNet()
    print(fun)
    output = fun(input)
    total_params = sum(param.nelement() for param in fun.parameters())
    print(total_params)
    # for i in range(4):
    #     print(output[i].shape)
