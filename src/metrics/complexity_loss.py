import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Callable, Tuple
import numpy as np
import math
import matplotlib.pyplot as plt
import warnings
import json


class MultiClassLoss(nn.Module):
    def __init__(
        self,
        writer,
        win_size: Tuple[int, int] = (12, 12),
        stride: Tuple[int, int] = (12, 12),
        class_configs: List[Dict] = None,  
    ):
        super().__init__()
        self.win_size = win_size
        self.stride = stride
        self.writer = writer
        self.global_step = 0
        print(self.win_size, self.stride)

        default_configs = [
            {
                "losses": [self._l1_loss, self._multi_scale_block_loss], 
                "loss_weights": [1.0, 1.0],                          
            }
        ]
#         default_configs = [
#             {
#                 "losses": [self._l1_loss, self._multi_scale_block_loss, self._tv_loss], 
#                 "loss_weights": [1.0, 1.0, 0.001],                          
#             }
#         ]
        self.class_configs = class_configs if class_configs is not None else default_configs
        self.num_classes = len(self.class_configs)
        
        self._write_configs_to_tensorboard(default_configs)
        
        self.scales = [1, 2, 4]

        self.scale_weights = self._get_weight(1.50, [1,2,4], norm=True)
#         self.scale_weights = [0.5,0.35,0.15]
        self.b_weights = self._get_weight(2.50, [1,2,4])
        self.b_con = 0.30

#         self.scale_cos_weight = self._get_weight(3.00, self.scales)
        
        print(self.b_weights)
        print(self.scale_weights)

#         self.scales = [1, 2, 4]
#         self.scale_weights = [0.6, 0.3, 0.1]
        
        self.gauss_kernels = self._generate_multi_scale_gaussian_kernels()
        
        self.gray_weights = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32)
        self.sobel_x_kernel = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y_kernel = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32).view(1, 1, 3, 3)

    def _generate_multi_scale_gaussian_kernels(self):
        gauss_kernels = {}
        for scale in self.scales:
            if scale == 1:
                kernel_size = 3
                sigma = 1.0
                x = torch.linspace(-(kernel_size-1)/2, (kernel_size-1)/2, kernel_size)
                y = torch.linspace(-(kernel_size-1)/2, (kernel_size-1)/2, kernel_size)
                xx, yy = torch.meshgrid(x, y)
                gauss = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
                gauss = gauss / gauss.sum()
                kernel = gauss
                kernel = kernel.unsqueeze(0).unsqueeze(0)
            else:
                kernel_size = 2 * scale + 1
                sigma = scale / 2.0
                x = torch.linspace(-(kernel_size-1)/2, (kernel_size-1)/2, kernel_size)
                y = torch.linspace(-(kernel_size-1)/2, (kernel_size-1)/2, kernel_size)
                xx, yy = torch.meshgrid(x, y)
                gauss = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
                gauss = gauss / gauss.sum()
                kernel = gauss
                kernel = kernel.unsqueeze(0).unsqueeze(0)
                kernel = kernel.repeat(3, 1, 1, 1)
            gauss_kernels[scale] = kernel
        return gauss_kernels

    def _write_configs_to_tensorboard(self, configs):
        if self.writer is None:
            return
        serializable_configs = []
        
        for cfg in configs:
            serializable_cfg = cfg.copy()
            if "losses" in serializable_cfg:
                serializable_cfg["losses"] = [
                    f"{loss.__name__}" if callable(loss) and hasattr(loss, "__name__")
                    else f"{loss.__class__.__name__}(reduction={loss.reduction})"
                    for loss in serializable_cfg["losses"]
                ]
            serializable_configs.append(serializable_cfg)
        
        configs_str = json.dumps(serializable_configs, indent=2)
        
        self.writer.add_text(
            tag="LossConfig/DefaultConfigs",
            text_string=configs_str,
            global_step=self.global_step
        )

    def _sliding_window_unfold(self, x: torch.Tensor, h_win, w_win, h_stride, w_stride):
        B, C, H, W = x.shape
        
        if H <= h_win and W <= w_win:
            x_blocks = x.unsqueeze(-1)
            return x_blocks, 1

        assert (H - h_win) % h_stride == 0, "Height not divisible by stride with window size"
        assert (W - w_win) % w_stride == 0, "Width not divisible by stride with window size"

        unfold = nn.Unfold(kernel_size=(h_win, w_win), stride=(h_stride, w_stride)).to(x.device)
        x_unfold = unfold(x)  # (B, C*h_win*w_win, num_blocks)
        
        num_blocks = x_unfold.shape[2]

        x_blocks = x_unfold.transpose(1, 2).reshape(B, num_blocks, C, h_win, w_win).permute(0, 2, 3, 4, 1)
        
        return x_blocks, num_blocks
    
    def _rgb2gray(self, x, norm=False):
        B, C, H, W = x.shape
        if C == 3:
            weights = self.gray_weights.to(x.device).type(x.dtype)
        else:
            weights = torch.ones(C, dtype=x.dtype, device=x.device) / C
        gray = torch.sum(x * weights.view(1, C, 1, 1), dim=1, keepdim=True)
        return gray

    def _get_weight(self, k, scales, norm=False):
        weights = []
        for scale in range(0, len(scales)):
            weight = math.exp(-k * scale / len(scales))
            weights.append(weight)
        if norm:
            weights_norm = []
            for weight in weights:
                weight = weight / sum(weights)
                weights_norm.append(weight)
            return weights_norm
        return weights

    def _tv_loss(self, pred, target, reduction="mean"):
        tv_h = torch.mean(torch.abs(pred[..., 1:, :] - pred[..., :-1, :]))
        tv_w = torch.mean(torch.abs(pred[..., :, 1:] - pred[..., :, :-1]))
        return tv_h + tv_w
    
    def _add_pixel_noise(
        self,
        pred: torch.Tensor,
        noise_amp: int = 2,
    ) -> torch.Tensor:
        device = pred.device
        noise = torch.randn_like(pred) * (noise_amp / 3)
        noise = torch.clamp(noise, -noise_amp, noise_amp)
        pred_noisy = torch.clamp(pred + noise, 0.0, 255.0)
        return pred_noisy

    def _l1_loss(self, pred, target, reduction: str = "none"):
#         pred_noisy = self._add_pixel_noise(pred)
        loss = F.l1_loss(pred, target, reduction=reduction)

        return loss

    def _scale_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        reduction: str = "none",
        pred_blurred: torch.Tensor = None,
        target_blurred: torch.Tensor = None,
    ) -> torch.Tensor:
        eps = 1e-8
        B, C, H, W = pred.shape
        device = pred.device
        dtype = pred.dtype

        grad_x_kernel = self.sobel_x_kernel.to(device).type(dtype)
        grad_y_kernel = self.sobel_y_kernel.to(device).type(dtype)
        if C==1:            
            pred_grad_x = F.conv2d(pred, grad_x_kernel, padding=1)
            pred_grad_y = F.conv2d(pred, grad_y_kernel, padding=1)
            target_grad_x = F.conv2d(target, grad_x_kernel, padding=1)
            target_grad_y = F.conv2d(target, grad_y_kernel, padding=1)
        else:
            grad_x_kernel = grad_x_kernel.repeat(3, 1, 1, 1)
            grad_y_kernel = grad_y_kernel.repeat(3, 1, 1, 1)
            pred_grad_x = F.conv2d(pred, grad_x_kernel, padding=1, groups=3)
            pred_grad_y = F.conv2d(pred, grad_y_kernel, padding=1, groups=3)
            target_grad_x = F.conv2d(target, grad_x_kernel, padding=1, groups=3)
            target_grad_y = F.conv2d(target, grad_y_kernel, padding=1, groups=3)

        target_grad_amp = torch.sqrt(target_grad_x**2 + target_grad_y**2 + eps)
#         pred_grad_amp = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + eps)

        p = 1.0
        target_grad_amp_exp = target_grad_amp ** p
        target_grad_amp_sum = torch.sum(target_grad_amp_exp, dim=(2,3), keepdim=True)
        target_grad_amp_weight = target_grad_amp_exp / target_grad_amp_sum
        
#         amp_loss = F.l1_loss(pred_grad_amp, target_grad_amp, reduction='none')

        def cosine_dir_loss(pred_x, pred_y, target_x, target_y):
            pred_vec = torch.cat([pred_x.unsqueeze(-1), pred_y.unsqueeze(-1)], dim=-1)
            target_vec = torch.cat([target_x.unsqueeze(-1), target_y.unsqueeze(-1)], dim=-1)
            pred_vec = F.normalize(pred_vec, p=2, dim=-1, eps=eps)
            target_vec = F.normalize(target_vec, p=2, dim=-1, eps=eps)
            cos_sim = torch.sum(pred_vec * target_vec, dim=-1)
            dir_loss = 1 - cos_sim
            return dir_loss

        dir_loss = cosine_dir_loss(pred_grad_x, pred_grad_y, target_grad_x, target_grad_y)
#         total_grad_loss = (0.8*dir_loss + 0.2*amp_loss) * target_grad_amp_weight
#         total_grad_loss = dir_loss * target_grad_amp_weight
#         total_grad_loss = torch.sum(total_grad_loss, dim=(2,3))
        total_grad_loss = dir_loss 
        
        if reduction == "mean":
            total_grad_loss = torch.mean(total_grad_loss)
        elif reduction == "sum":
            total_grad_loss = torch.sum(total_grad_loss)

        return total_grad_loss
    
    def _cosine_similarity_loss(
        self,
        pred_blocks: torch.Tensor,
        target_blocks: torch.Tensor,
        eps: float = 1e-8
    ) -> torch.Tensor:
        B,C,h,w,N = pred_blocks.shape
        pred_vec = pred_blocks.reshape(B,C,h*w,N)
        target_vec = target_blocks.reshape(B,C,h*w,N)
        
        pred_vec = F.normalize(pred_vec, p=2, dim=2, eps=eps)
        target_vec = F.normalize(target_vec, p=2, dim=2, eps=eps)
        cos_sim = torch.sum(pred_vec * target_vec, dim=2)
        cos_loss = 1 - cos_sim
        return cos_loss
    
    def _compute_histogram_soft(self, x, bins=32, a=2, min_val=-0.50, max_val=1.50, eps=1e-8):
        device = x.device
        dtype = x.dtype
        bin_width = (max_val - min_val) / bins
        bin_centers = torch.linspace(min_val + bin_width/2, max_val - bin_width/2, bins, device=device, dtype=dtype)
        x_expanded = x.unsqueeze(-1)
        centers_expanded = bin_centers.unsqueeze(0).unsqueeze(0)
        sigma = bin_width / a
        gaussian_weights = torch.exp(-((x_expanded - centers_expanded) ** 2) / (2 * sigma ** 2))
        gaussian_weights = gaussian_weights / (gaussian_weights.sum(dim=-1, keepdim=True) + eps)
        hist = gaussian_weights.sum(dim=1)
        hist_sum = hist.sum(dim=1, keepdim=True) + eps
        hist = hist / hist_sum
        hist = torch.clamp(hist, min=eps, max=1.0 - eps)
        return hist

    def _block_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        bins,
        scale,
        a,
        reduction: str = "none",
    ) -> torch.Tensor:
        eps = 1e-8
        B, C, h_win, w_win, num_blocks = pred.shape
        
#         cos_loss_per_block = self._cosine_similarity_loss(pred, target, eps=eps)
        
        pred_flat = pred.permute(0, 1, 4, 2, 3).reshape(-1, h_win * w_win)
        target_flat = target.permute(0, 1, 4, 2, 3).reshape(-1, h_win * w_win)

        pred_hist = self._compute_histogram_soft(pred_flat, bins=bins, a=a, eps=eps)
        target_hist = self._compute_histogram_soft(target_flat, bins=bins, a=a, eps=eps)

        kl_loss_per_block = F.kl_div(
            torch.log(pred_hist),
            target_hist,
            reduction="none"
        ).sum(dim=1)

        kl_loss_per_block = kl_loss_per_block.reshape(B, C, num_blocks)
        
#         total_loss_per_block = self.scale_cos_weight[scale]*cos_loss_per_block

        total_loss_per_block = kl_loss_per_block

        if reduction == "mean":
            total_loss = total_loss_per_block.mean()
        elif reduction == "sum":
            total_loss = total_loss_per_block.sum()
        elif reduction == "none":
            total_loss = total_loss_per_block

        return total_loss
    
    def _block_weight(self, target_blocks, reduction="none"):
        target_block_std = torch.std(target_blocks, dim=(2, 3))

        std_sum = torch.sum(target_block_std, dim=-1, keepdim=True)+1e-8
        std_weight = target_block_std / std_sum

        std_weight = std_weight.detach()
        
        if reduction == "mean":
            std_weight = std_weight.mean()
        elif reduction == "sum":
            std_weight = std_weight.sum()
            
        return std_weight
    
    def _get_down_scale(self, img, scale):
        B, C, H, W = img.shape
        if scale == 1:
            img_scaled = img
        else:
            gauss_kernel = self.gauss_kernels[scale].to(img.device).type(img.dtype)
            padding = gauss_kernel.shape[-1] // 2

            img_blur = F.conv2d(img, gauss_kernel, padding=padding, groups=C)
            img_scaled = F.avg_pool2d(img_blur, kernel_size=scale, stride=scale)
        return img_scaled

    def _multi_scale_block_loss(self, pred: torch.Tensor, target: torch.Tensor, reduction="mean"):
        assert pred.shape == target.shape, "pred and target must have same shape"
        total_loss = 0.0
        B, C, H, W = pred.shape
        h_win, w_win = self.win_size
        h_stride, w_stride = self.stride
        bins = 16
        a = 2.0
        
        for scale_idx, scale in enumerate(self.scales):
            pred_scaled = self._get_down_scale(pred, scale)
            target_scaled = self._get_down_scale(target, scale)

            pred_gray = self._rgb2gray(pred_scaled)
            target_gray = self._rgb2gray(target_scaled)

            pred_blocks, _ = self._sliding_window_unfold(pred_scaled, h_win, w_win, h_stride, w_stride)
            target_blocks, _ = self._sliding_window_unfold(target_scaled, h_win, w_win, h_stride, w_stride)
#             pred_blocks, _ = self._sliding_window_unfold(pred_gray, h_win, w_win, h_stride, w_stride)
#             target_blocks, _ = self._sliding_window_unfold(target_gray, h_win, w_win, h_stride, w_stride)

            pixel_scale_loss = self._scale_loss(pred_gray, target_gray, reduction="mean")
#             pixel_scale_loss = self._scale_loss(pred_scaled, target_scaled, reduction="mean")
            pixel_block_loss = self._block_loss(pred_blocks, target_blocks, int(bins), scale_idx, a=a, reduction="mean")
            scale_loss = pixel_block_loss*(1-self.b_con*self.b_weights[scale_idx]) + pixel_scale_loss*self.b_con*self.b_weights[scale_idx]

            total_loss += self.scale_weights[scale_idx]*scale_loss
            if scale_idx%2>0:
                bins = bins / 2

        if reduction == "mean":
            total_loss = total_loss.mean()
        elif reduction == "sum":
            total_loss = total_loss.sum()

        return total_loss
    
    def _compute_class_loss(self, pred: torch.Tensor, target: torch.Tensor):
        cfg = self.class_configs[0]

        class_loss_total = 0.0
        for loss_idx, (loss_fn, loss_weight) in enumerate(zip(cfg["losses"], cfg["loss_weights"])):
            pixel_loss = loss_fn(pred, target, reduction="mean")
            class_loss_total += pixel_loss * loss_weight

        return class_loss_total

    def forward(self, pred, target, epoch, is_plot):
        total_loss = self._compute_class_loss(
            pred, target
        )

        return total_loss