import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Callable, Tuple


class MultiClassMultiLossWithVariance(nn.Module):
    def __init__(
        self,
        win_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (8, 8),
        class_configs: List[Dict] = None,  
        return_details: bool = False       
    ):
        super().__init__()
        self.win_size = win_size
        self.stride = stride
        self.return_details = return_details

        default_configs = [
            {
                "threshold_range": (0.0, 0.3),  
                "losses": [F.huber_loss, F.l1_loss], 
                "loss_weights": [0.05, 0.03],          
                "class_weight": 0.1                   
            },
            {
                "threshold_range": (0.3, 0.7),
                "losses": [F.smooth_l1_loss, F.mse_loss],
                "loss_weights": [0.2, 0.15],
                "class_weight": 0.4
            },
            {
                "threshold_range": (0.7, 1.0),
                "losses": [F.mse_loss, self.ssim_loss],
                "loss_weights": [1.0, 0.5],
                "class_weight": 1.0
            }
        ]
        self.class_configs = class_configs if class_configs is not None else default_configs
        self.num_classes = len(self.class_configs)


    def _sliding_window_unfold(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        B, C, H, W = x.shape
        h_win, w_win = self.win_size
        h_stride, w_stride = self.stride

        unfold = nn.Unfold(kernel_size=self.win_size, stride=self.stride).to(x.device)
        x_unfold = unfold(x)  # (B, C*h_win*w_win, num_blocks)

        num_h = (H - h_win) // h_stride + 1
        num_w = (W - w_win) // w_stride + 1
        num_blocks = num_h * num_w

        # (B, C, h_win, w_win, num_blocks)
        x_blocks = x_unfold.reshape(B, C, h_win, w_win, num_blocks)
        return x_blocks, num_blocks

    def _compute_multi_class_mask(self, x_blocks: torch.Tensor, num_blocks: int) -> torch.Tensor:
        B, C = x_blocks.shape[:2]

        rgb2gray_weights = torch.tensor([0.299, 0.587, 0.114], dtype=x_blocks.dtype, device=x_blocks.device)
        rgb2gray_weights = rgb2gray_weights.reshape(1, 3, 1, 1, 1)

        gray_blocks = torch.sum(x_blocks * rgb2gray_weights, dim=1, keepdim=True)

        win_var = torch.var(gray_blocks, dim=(2, 3), unbiased=False)
        win_complexity = win_var.squeeze(1)

        per_img_min = win_complexity.min(dim=1, keepdim=True)[0]  # (B, 1)
        per_img_max = win_complexity.max(dim=1, keepdim=True)[0]  # (B, 1)
        norm_complexity = (win_complexity - per_img_min) / (per_img_max - per_img_min + 1e-8)  # (B, num_blocks)
        mask = torch.zeros(B, self.num_classes, num_blocks, device=x_blocks.device)
        for i, cfg in enumerate(self.class_configs):
            min_thr, max_thr = cfg["threshold_range"]
            if i == self.num_classes - 1:
                class_mask = (norm_complexity >= min_thr) & (norm_complexity <= max_thr)
            else:
                class_mask = (norm_complexity >= min_thr) & (norm_complexity < max_thr)
            mask[:, i, :] = class_mask.float()

        return mask, norm_complexity

    @staticmethod
    def ssim_loss(pred: torch.Tensor, target: torch.Tensor, reduction: str = "none") -> torch.Tensor:
        B, C, h, w, num_blocks = pred.shape
        pred_reshaped = pred.permute(0, 4, 1, 2, 3).reshape(B*num_blocks, C, h, w)
        target_reshaped = target.permute(0, 4, 1, 2, 3).reshape(B*num_blocks, C, h, w)

        ssim_val = F.l1_loss(pred_reshaped, target_reshaped, reduction=reduction)
        ssim_loss = 1 - ssim_val

        ssim_loss = ssim_loss.reshape(B, num_blocks, C, h, w).mean(dim=(2,3,4))
        return ssim_loss

    def _compute_class_loss(self, pred_blocks: torch.Tensor, target_blocks: torch.Tensor, mask: torch.Tensor, class_idx: int) -> Tuple[torch.Tensor, Dict]:
        cfg = self.class_configs[class_idx]
        B, num_blocks = mask.shape[0], mask.shape[2]
        class_mask = mask[:, class_idx, :]
        class_weight = cfg["class_weight"]

        class_loss_total = 0.0
        loss_details = {}

        for loss_idx, (loss_fn, loss_weight) in enumerate(zip(cfg["losses"], cfg["loss_weights"])):
            loss_name = f"class_{class_idx}_loss_{loss_idx}_{loss_fn.__name__}"

            if loss_fn == self.ssim_loss:
                win_loss = loss_fn(pred_blocks, target_blocks)
            else:
                pixel_loss = loss_fn(pred_blocks, target_blocks, reduction="none")
                win_loss = torch.mean(pixel_loss, dim=(1, 2, 3))

            masked_loss_sum = (win_loss * class_mask).sum(dim=1)
            valid_win_count = class_mask.sum(dim=1) + 1e-8
            per_img_loss = (masked_loss_sum / valid_win_count) * loss_weight * class_weight

            class_loss_total += per_img_loss.mean()

            loss_details[loss_name] = {
                "loss_value": per_img_loss.mean().item(),
                "valid_win_count": class_mask.sum(dim=1).int().tolist()
            }

        return class_loss_total, loss_details

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor | Tuple[torch.Tensor, Dict]:
        pred_blocks, num_blocks = self._sliding_window_unfold(pred)
        target_blocks, _ = self._sliding_window_unfold(target)

        multi_class_mask, norm_complexity = self._compute_multi_class_mask(target_blocks, num_blocks)

        total_loss = 0.0
        all_loss_details = {
            "norm_complexity_range": (norm_complexity.min().item(), norm_complexity.max().item()),
            "class_valid_wins": [multi_class_mask[:, i, :].sum(dim=1).int().tolist() for i in range(self.num_classes)]
        }

        for class_idx in range(self.num_classes):
            class_loss, class_loss_details = self._compute_class_loss(
                pred_blocks, target_blocks, multi_class_mask, class_idx
            )
            total_loss += class_loss
            all_loss_details.update(class_loss_details)

        if self.return_details:
            all_loss_details["total_loss"] = total_loss.item()
            return total_loss, all_loss_details
        return total_loss
