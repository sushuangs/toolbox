import os
from importlib import import_module
import pyiqa

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utility import Tensor2np
from utils.config import namespace_to_dict

from metrics.psnr_ssim import calculate_psnr, calculate_ssim
from metrics.lpips import calculate_lpips
from metrics.niqe import calculate_niqe


class MetricCalculator:
    def __init__(self, metric_configs, rgb_range):
        self.metric_configs = self._validate_configs(metric_configs)
        self.rgb_range = rgb_range

    def _validate_configs(self, configs):
        validated = {}
        for cfg in configs:
            name = cfg.name
#             params = cfg.params

            if not name:
                continue
            
            validated[name] = {
                "func": pyiqa.create_metric(cfg.name, device='cuda')
#                 "params": vars(params)
            }
        return validated

    def calculate(self, pred, hr):
#         pred, hr = Tensor2np(pred, hr, rgb_range=self.rgb_range)
        pred = torch.clamp(pred, min=0.0, max=1.0)
        hr = torch.clamp(hr, min=0.0, max=1.0)
        results = {}
        for name, m_set in self.metric_configs.items():
            try:
#                 if name == 'niqe':
#                     value = m_set["func"](pred, **m_set["params"])
#                 else:
#                     value = m_set["func"](pred, hr, **m_set["params"])
                value = m_set["func"](pred, hr)
                results[name] = float(value) if hasattr(value, "item") else value
            except Exception as e:
                results[name] = None
                print(e)
        return results

    
__all__ = ['MetricCalculator']

