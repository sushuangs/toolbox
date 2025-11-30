import os
from importlib import import_module

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utility import Tensor2np

class Loss(nn.modules.loss._Loss):
    def __init__(self, args):
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
                module = import_module('loss.vgg')
                loss_function = getattr(module, 'VGG')(
                    loss_type[3:],
                    rgb_range=args.rgb_range
                )
            elif loss_type.find('GAN') >= 0:
                module = import_module('loss.adversarial')
                loss_function = getattr(module, 'Adversarial')(
                    args,
                    loss_type
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

    def forward(self, sr, hr):
        losses = []
        metric_list = {}
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                loss = l['function'](sr, hr)
                effective_loss = l['weight'] * loss
                metric_list[l['type']] = loss.item()
                losses.append(effective_loss)
        loss_sum = sum(losses)
        metric_list['loss_total'] = loss_sum.item()

        return loss_sum, metric_list


from metrics.psnr_ssim import calculate_psnr, calculate_ssim
from metrics.lpips import calculate_lpips
from metrics.niqe import calculate_niqe

SUPPORTED_METRICS = {
    "psnr": calculate_psnr,
    "ssim": calculate_ssim,
    "lpips": calculate_lpips,
    "niqe": calculate_niqe,
}


class MetricCalculator:
    def __init__(self, metric_configs):
        self.metric_configs = self._validate_configs(metric_configs)

    def _validate_configs(self, configs):
        validated = {}
        for cfg in configs:
            name = cfg.name
            metric_type = cfg.type
            params = cfg.params

            if not name:
                continue
            
            validated[name] = {
                "func": SUPPORTED_METRICS[name],
                "params": vars(params)
            }
        return validated

    def calculate(self, pred, hr):
        pred, hr = Tensor2np(pred, hr)
        results = {}
        for name, m_set in self.metric_configs.items():
            try:
                if name == 'niqe':
                    value = m_set["func"](pred, **m_set["params"])
                else:
                    value = m_set["func"](pred, hr, **m_set["params"])
                results[name] = float(value) if hasattr(value, "item") else value
            except Exception as e:
                results[name] = None
                print(e)
        return results

    
__all__ = ['MetricCalculator', 'Loss']

