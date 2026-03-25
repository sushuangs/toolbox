import math
import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F
from utils.config import namespace_to_dict

from losses.vgg_arch import VGGFeatureExtractor
from losses.patch_loss_c import PatchesKernel3D
import numpy as np


class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 perceptual_patch_weight=1.0,
                 style_weight=0.,
                 criterion='patch',
                 perceptual_kernels=[4,8],
                 use_std_to_force=True):

        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.patch_weights = perceptual_patch_weight
        self.style_weight = style_weight
        self.layer_weights = namespace_to_dict(layer_weights)
        self.perceptual_kernels = perceptual_kernels
        self.use_std_to_force = use_std_to_force
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(namespace_to_dict(layer_weights).keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.L2loss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        elif self.criterion_type == 'patch':
            self.criterion = torch.nn.L1Loss()
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                elif self.criterion_type == 'patch':
                    if self.patch_weights == 0:
                        percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
                    else:
                        percep_loss += self.patch(x_features[k], gt_features[k], self.use_std_to_force) * self.layer_weights[k] * self.patch_weights + self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram

    def patch(self, x, gt, use_std_to_force):
        loss = 0.
        for _kernel in self.perceptual_kernels:
            _patchkernel3d = PatchesKernel3D(_kernel, _kernel//2).to('cuda')   # create instance
            x_trans = _patchkernel3d(x)
            gt_trans = _patchkernel3d(gt)
            x_trans = x_trans.reshape(-1, x_trans.shape[-1])
            gt_trans = gt_trans.reshape(-1, gt_trans.shape[-1])
            dot_x_y = torch.einsum('ik,ik->i', x_trans, gt_trans)
            if use_std_to_force == False:
                cosine0_x_y = torch.div(torch.div(dot_x_y, torch.sqrt(torch.sum(x_trans ** 2, dim=1))), torch.sqrt(torch.sum(gt_trans ** 2, dim=1)))
                loss = loss + torch.mean(1-cosine0_x_y) # y = 1-x
            else:
                dy = torch.std(gt_trans, dim=1)
                cosine_x_y = torch.div(torch.div(dot_x_y, torch.sqrt(torch.sum(x_trans ** 2, dim=1))), torch.sqrt(torch.sum(gt_trans ** 2, dim=1)))
                cosine_x_y_d = torch.mul((1-cosine_x_y), dy) # y = (1-x)dy
                loss = loss + torch.mean(cosine_x_y_d)
        return loss