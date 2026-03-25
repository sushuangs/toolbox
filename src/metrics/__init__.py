import os
from importlib import import_module

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utility import Tensor2np
from utils.config import namespace_to_dict

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
    def __init__(self, metric_configs, rgb_range):
        self.metric_configs = self._validate_configs(metric_configs)
        self.rgb_range = rgb_range

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
        pred, hr = Tensor2np(pred, hr, rgb_range=self.rgb_range)
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

    
__all__ = ['MetricCalculator']

