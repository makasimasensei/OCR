from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import nn
from Loss.det_basic_loss import BalanceLoss, MaskL1Loss, DiceLoss


class DBLoss(nn.Module):
    """
    Differentiable Binarization (DB) Loss Function
    args:
        param (dict): the super paramter for DB Loss
    """

    def __init__(self,
                 balance_loss=True,
                 main_loss_type='DiceLoss',
                 alpha=1,
                 beta=10,
                 ohem_ratio=3,
                 eps=1e-6, ):
        super(DBLoss, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.alpha = alpha
        self.beta = beta
        self.dice_loss = DiceLoss(eps=eps)
        self.l1_loss = MaskL1Loss(eps=eps)
        self.bce_loss = BalanceLoss(
            balance_loss=balance_loss,
            main_loss_type=main_loss_type,
            negative_ratio=ohem_ratio)

    def forward(self, local_data, predicts):
        local_loss_all = 0.0
        for i in range(predicts.size(0)):
            img = local_data['image'][i]
            label_threshold_map = local_data['threshold_map'][i].to(self.device)
            label_threshold_mask = local_data['threshold_mask'][i].to(self.device)
            label_shrink_map = local_data['shrink_map'][i].to(self.device)
            label_shrink_mask = local_data['shrink_mask'][i].to(self.device)
            height, width, _ = img.shape

            predict_maps = F.interpolate(predicts[i, :, :, :].unsqueeze(0), size=(height, width), mode='nearest')
            # plt.subplot(2, 2, 1)
            # img1 = predicts[i, 0, :, :]
            # plt.imshow(img1.cpu().detach().numpy())
            # plt.subplot(2, 2, 2)
            # img1 = predict_maps[0, 0, :, :]
            # plt.imshow(img1.cpu().detach().numpy())
            # plt.show()

            shrink_maps = predict_maps[0, 0, :, :]
            threshold_maps = predict_maps[0, 1, :, :]
            binary_maps = predict_maps[0, 2, :, :]

            loss_shrink_maps = self.bce_loss(shrink_maps, label_shrink_map,
                                             label_shrink_mask)
            loss_threshold_maps = self.l1_loss(threshold_maps, label_threshold_map,
                                               label_threshold_mask)
            loss_binary_maps = self.dice_loss(binary_maps, label_shrink_map,
                                              label_shrink_mask)
            loss_shrink_maps = self.alpha * loss_shrink_maps
            loss_threshold_maps = self.beta * loss_threshold_maps

            loss_all = loss_shrink_maps + loss_threshold_maps + loss_binary_maps
            local_loss_all += loss_all

        return local_loss_all
