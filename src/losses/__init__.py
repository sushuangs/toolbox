import os
from importlib import import_module
from utils.config import namespace_to_dict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import lpips


class Loss(nn.modules.loss._Loss):
    def __init__(self, args, writer):
        super(Loss, self).__init__()
        print('Preparing loss function:')

        self.n_GPUs = args.n_GPUs
        self.loss = []
        self.loss_module = nn.ModuleList()
        for loss in args.losses:
            loss_type = loss.type
            weight = loss.loss_weight
            if loss_type == 'MSE':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type.find('VGG') >= 0:
                module = import_module('losses.perceptual_loss')
                loss_function = getattr(module, 'PerceptualLoss')(
                    **namespace_to_dict(loss.params)
                )
            elif loss_type.find('LPIPS') >= 0:
                loss_function = lpips.LPIPS(net='vgg').cuda().eval()
            elif loss_type.find('GAN') >= 0:
                module = import_module('loss.adversarial')
                loss_function = getattr(module, 'Adversarial')(
                    args,
                    loss_type
                )
            elif loss_type.find('CML') >= 0:
                module = import_module('losses.complexity_loss')
                loss_function = getattr(module, 'MultiClassLoss')(
                    writer,
                    **namespace_to_dict(loss.params)
                )
            elif loss_type.find('PLC') >= 0:
                module = import_module('losses.patch_loss_c')
                loss_function = getattr(module, 'patchLoss3DXD')(
                    **namespace_to_dict(loss.params)
                )
            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )
            if loss_type.find('GAN') >= 0:
                self.loss.append({'type': 'DIS', 'weight': 1, 'function': None})

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])


        device = torch.device('cpu' if args.cpu else 'cuda')
        self.loss_module.to(device)
        if args.precision == 'half': self.loss_module.half()
        if not args.cpu and args.n_GPUs > 1:
            self.loss_module = nn.DataParallel(
                self.loss_module, range(args.n_GPUs)
            )

    def forward(self, sr, hr, **args):
        losses = []
        metric_list = {}
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                if l['type'].find('LPIPS') >= 0:
                    loss = l['function'](sr, hr, **args).mean()
                else:
                    loss = l['function'](sr, hr, **args)
                effective_loss = l['weight'] * loss
                metric_list[l['type']] = loss.item()
                losses.append(effective_loss)
        loss_sum = sum(losses)
        metric_list['loss_total'] = loss_sum.item()

        return loss_sum, metric_list
    

__all__ = ['Loss']