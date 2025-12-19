import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Callable, Tuple
import numpy as np
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
                "quantile_range": (0.0, 0.7),
                "losses": [F.mse_loss, self._block_norm_l1_loss],
                "loss_weights": [0.065, 1.0],
                "class_weight": 1.0
            },
            {
                "quantile_range": (0.7, 1.0),  
                "losses": [F.mse_loss, self._block_norm_l1_loss], 
                "loss_weights": [0.065, 1.0],          
                "class_weight": 1.0                   
            }
        ]
        self.class_configs = class_configs if class_configs is not None else default_configs
        self.num_classes = len(self.class_configs)
        
        self._write_configs_to_tensorboard(default_configs)

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

    def _sliding_window_unfold(self, x: torch.Tensor):
        B, C, H, W = x.shape
        h_win, w_win = self.win_size
        h_stride, w_stride = self.stride

        assert (H - h_win) % h_stride == 0, "Height not divisible by stride with window size"
        assert (W - w_win) % w_stride == 0, "Width not divisible by stride with window size"

        unfold = nn.Unfold(kernel_size=self.win_size, stride=self.stride).to(x.device)
        x_unfold = unfold(x)  # (B, C*h_win*w_win, num_blocks)
        
        num_blocks = x_unfold.shape[2]

        x_blocks = x_unfold.transpose(1, 2).reshape(B, num_blocks, C, h_win, w_win).permute(0, 2, 3, 4, 1)
        
        return x_blocks, num_blocks
    
    def _block_norm_l1_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        reduction: str = "none",
        eps: float = 1e-8
    ) -> torch.Tensor:
        # 1. 计算分子：逐元素L1损失
        l1_loss = F.l1_loss(pred, target, reduction="none")  # (B, C, H, W, num_blocks)

        # 2. 计算分母：每个区块的L2模长（在H/W维度计算）
        # 计算pred和target的区块模长，取均值作为分母
        pred_norm = torch.norm(pred, p=2, dim=(2, 3))  # (B, C, num_blocks)
        target_norm = torch.norm(target, p=2, dim=(2, 3))  # (B, C, num_blocks)
        block_norm = (pred_norm + target_norm) / 2  # (B, C, num_blocks)

        # 扩展维度以匹配L1损失形状 (B, C, 1, 1, num_blocks)
        block_norm = block_norm.unsqueeze(2).unsqueeze(3)

        # 3. 计算最终损失（逐元素）
        loss = l1_loss / (block_norm + eps)

        # 4. 归约处理（兼容PyTorch损失接口）
        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()

        return loss
    
    def _plot_sorted_variance(self, x_blocks, current_epoch):
        B, C = x_blocks.shape[:2]

        rgb2gray_weights = torch.tensor([0.299, 0.587, 0.114], dtype=x_blocks.dtype, device=x_blocks.device)
        rgb2gray_weights = rgb2gray_weights.reshape(1, 3, 1, 1, 1)

        gray_blocks = torch.sum(x_blocks * rgb2gray_weights, dim=1, keepdim=True)

        win_var = torch.var(gray_blocks, dim=(2, 3), unbiased=False)
        win_complexity = win_var.squeeze(1)

        sorted_complexity, _ = torch.sort(win_complexity, dim=1)

        sorted_np = sorted_complexity.detach().cpu().numpy()
        B, num_blocks = sorted_np.shape
        batch_mean_sorted = np.mean(sorted_np, axis=0)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(num_blocks), batch_mean_sorted, color='#E74C3C', linewidth=2.5,
                label=f'Epoch {current_epoch} - First Batch (Samples: {B})')
        ax.set_xlabel('Block Index (Sorted by Variance, Ascending)')
        ax.set_ylabel('Block Variance (Texture Complexity)')
        ax.set_title(f'Epoch {current_epoch} - Single Batch Block Variance')
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        self.writer.add_figure(
            tag="Single_Batch/Block_Variance_Sorted",
            figure=fig,
            global_step=current_epoch
        )
        plt.close(fig)

        stats_text = f"""
        === Epoch {current_epoch} - First Single Batch Statistics ===
        Batch samples: {B}
        Batch mean variance min: {batch_mean_sorted.min():.4f}
        Batch mean variance max: {batch_mean_sorted.max():.4f}
        Batch mean variance avg: {batch_mean_sorted.mean():.4f}
        """
        self.writer.add_text(
            tag="Single_Batch/Variance_Statistics",
            text_string=stats_text,
            global_step=current_epoch
        )

#     def _compute_multi_class_mask(self, x_blocks: torch.Tensor, num_blocks: int):
#         B, C = x_blocks.shape[:2]

#         rgb2gray_weights = torch.tensor([0.299, 0.587, 0.114], dtype=x_blocks.dtype, device=x_blocks.device)
#         rgb2gray_weights = rgb2gray_weights.reshape(1, 3, 1, 1, 1)

#         gray_blocks = torch.sum(x_blocks * rgb2gray_weights, dim=1, keepdim=True)

#         win_var = torch.var(gray_blocks, dim=(2, 3), unbiased=False)
#         win_complexity = win_var.squeeze(1)

#         per_img_min = win_complexity.min(dim=1, keepdim=True)[0] # (B, 1)
#         per_img_max = win_complexity.max(dim=1, keepdim=True)[0] + 1e-8 # (B, 1)
#         norm_complexity = (win_complexity - per_img_min) / (per_img_max - per_img_min + 1e-8) # (B, num_blocks)
#         mask = torch.zeros(B, self.num_classes, num_blocks, device=x_blocks.device)
#         for i, cfg in enumerate(self.class_configs):
#             min_thr, max_thr = cfg["threshold_range"]
#             if i == self.num_classes - 1:
#                 class_mask = (norm_complexity >= (min_thr - 1e-8)) & (norm_complexity <= max_thr + 1e-8)
#             else:
#                 class_mask = (norm_complexity >= min_thr - 1e-8) & (norm_complexity < (max_thr + 1e-8))
#             mask[:, i, :] = class_mask.float()

#         return mask, norm_complexity

    def _compute_multi_class_mask(self, x_blocks: torch.Tensor, num_blocks: int):
        B, C = x_blocks.shape[:2]
        device = x_blocks.device

        # 1. 原有灰度转换和方差计算（完全保留）
        rgb2gray_weights = torch.tensor([0.299, 0.587, 0.114], dtype=x_blocks.dtype, device=device)
        rgb2gray_weights = rgb2gray_weights.reshape(1, 3, 1, 1, 1)
        gray_blocks = torch.sum(x_blocks * rgb2gray_weights, dim=1, keepdim=True)

        win_var = torch.var(gray_blocks, dim=(2, 3), unbiased=False)
        win_complexity = win_var.squeeze(1)  # (B, num_blocks)

        # 2. 归一化（安全除零，保留原逻辑）
        per_img_min = win_complexity.min(dim=1, keepdim=True)[0]  # (B, 1)
        per_img_max = win_complexity.max(dim=1, keepdim=True)[0] + 1e-8  # (B, 1)
        denominator = per_img_max - per_img_min + 1e-8  # 避免除零
        norm_complexity = (win_complexity - per_img_min) / denominator  # (B, num_blocks)

        # 3. 初始化mask（维度保留原逻辑：B, num_classes, num_blocks）
        mask = torch.zeros(B, self.num_classes, num_blocks, device=device)

        # 4. 核心修改：按分位数生成每个类别的动态阈值
        for i, cfg in enumerate(self.class_configs):
            # 读取当前类别的分位数区间（如[0.0, 0.2]表示0-20分位）
            min_q, max_q = cfg["quantile_range"]

            # 计算每个样本的norm_complexity在该分位数下的数值（逐样本计算，dim=1）
            # torch.quantile的q参数范围是0-1，dim=1表示对每个样本的num_blocks维度计算
            min_thr = torch.quantile(norm_complexity, q=min_q, dim=1, keepdim=True)  # (B, 1)
            max_thr = torch.quantile(norm_complexity, q=max_q, dim=1, keepdim=True)  # (B, 1)

            # 边界处理：最后一个类别包含上限，其他类别包含下限不包含上限（保留原浮点偏移）
            if i == self.num_classes - 1:
                # 最后一类：>= 低分位数 - 1e-8 且 <= 高分位数 + 1e-8
                class_mask = (norm_complexity >= (min_thr - 1e-8)) & (norm_complexity <= (max_thr + 1e-8))
            else:
                # 非最后一类：>= 低分位数 - 1e-8 且 < 高分位数 + 1e-8
                class_mask = (norm_complexity >= (min_thr - 1e-8)) & (norm_complexity < max_thr)

            # 赋值到对应类别的mask位置
            mask[:, i, :] = class_mask.float()

        return mask, norm_complexity
    
    def _compute_class_loss(self, pred_blocks: torch.Tensor, target_blocks: torch.Tensor, mask: torch.Tensor, class_idx: int):
        cfg = self.class_configs[class_idx]
        B, C, H, W, num_blocks = pred_blocks.shape
        class_mask = mask[:, class_idx, :]  # (B,1,1,1,num_blocks)
        class_mask = class_mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        class_weight = cfg["class_weight"]

        class_loss_total = 0.0
        for loss_idx, (loss_fn, loss_weight) in enumerate(zip(cfg["losses"], cfg["loss_weights"])):
            pixel_loss = loss_fn(pred_blocks, target_blocks, reduction="none")  # (B,C,H,W,num_blocks)
            pixel_loss_flat = pixel_loss.flatten()

            class_mask_expanded = class_mask.expand(B, C, H, W, num_blocks)
            class_mask_flat = class_mask_expanded.flatten()

            pixel_loss_masked = pixel_loss_flat * class_mask_flat
            masked_loss_mean = torch.mean(pixel_loss_masked)
            class_loss_total += masked_loss_mean * loss_weight * class_weight

        return class_loss_total 
    
#     def _compute_class_loss(self, pred_blocks: torch.Tensor, target_blocks: torch.Tensor, mask: torch.Tensor, class_idx: int):
#         cfg = self.class_configs[class_idx]
#         class_weight = cfg["class_weight"]

#         # 定义差异过大的阈值（根据你的场景调整，1e-8是正常浮点误差分界）
#         WARN_THRESHOLD = 1e-8  

#         class_loss_total = 0.0
#         for loss_idx, (loss_fn, loss_weight) in enumerate(zip(cfg["losses"], cfg["loss_weights"])):
#             # 步骤1：计算逐元素损失
#             pixel_loss = loss_fn(pred_blocks, target_blocks)

#             # 步骤2：手动mean + 原生mean对比（无权重）
#             manual_mean = pixel_loss.mean()
#             native_mean = nn.L1Loss(reduction='mean')(pred_blocks, target_blocks)

#             # 计算无权重差异
#             diff_none = abs(manual_mean.item() - native_mean.item())

#             # 打印核心对比（无权重）
#             print(f"=== 无权重对比 ===")
#             print(f"手动mean值: {manual_mean.item():.15f}")
#             print(f"原生mean值: {native_mean.item():.15f}")
#             print(f"无权重差异: {diff_none:.20f}")

#             # 差异过大时触发醒目提醒 + TensorBoard add_text记录（无权重）
#             if diff_none > WARN_THRESHOLD:
#                 # 控制台彩色打印警告
#                 warn_msg = f"⚠️ 警告：无权重损失差异过大！可能存在计算逻辑错误！"
#                 diff_msg = f"差异值: {diff_none:.6f} > 阈值: {WARN_THRESHOLD}"
#                 print("\033[91m" + warn_msg + "\033[0m")
#                 print(f"\033[91m" + diff_msg + "\033[0m")

#                 # 构造TensorBoard文本内容（包含完整上下文）
#                 tb_text = f"""
#                 === 损失计算警告（无权重） ===
#                 类别索引: {class_idx}
#                 损失索引: {loss_idx}
#                 手动mean值（15位精度）: {manual_mean.item():.15f}
#                 原生mean值（15位精度）: {native_mean.item():.15f}
#                 差异值: {diff_none:.20f}
#                 警告阈值: {WARN_THRESHOLD:.20f}
#                 结论: 差异超过阈值，可能存在计算逻辑错误！
#                 """
#                 # 写入TensorBoard（标签要清晰，方便检索）
#                 self.writer.add_text(
#                     tag=f"loss_warning/class_{class_idx}/loss_{loss_idx}/no_weight",
#                     text_string=tb_text,
#                     global_step=self.global_step
#                 )
#                 writer.flush()  # 强制刷新，确保日志立即写入

#             # 步骤3：加权重后的对比
#             manual_with_weight = manual_mean * loss_weight * class_weight
#             native_with_weight = native_mean * loss_weight * class_weight

#             # 计算加权重差异
#             diff_with_weight = abs(manual_with_weight.item() - native_with_weight.item())

#             print(f"\n=== 加权重对比 ===")
#             print(f"手动加权重: {manual_with_weight.item():.15f}")
#             print(f"原生加权重: {native_with_weight.item():.15f}")
#             print(f"加权重差异: {diff_with_weight:.20f}")

#             # 差异过大时触发醒目提醒 + TensorBoard add_text记录（加权重）
#             if diff_with_weight > WARN_THRESHOLD:
#                 # 控制台彩色打印警告
#                 warn_msg = f"⚠️ 警告：加权重后损失差异过大！可能存在权重计算错误！"
#                 diff_msg = f"差异值: {diff_with_weight:.6f} > 阈值: {WARN_THRESHOLD}"
#                 print("\033[91m" + warn_msg + "\033[0m")
#                 print(f"\033[91m" + diff_msg + "\033[0m")

#                 # 构造TensorBoard文本内容（包含完整上下文）
#                 tb_text = f"""
#                 === 损失计算警告（加权重） ===
#                 类别索引: {class_idx}
#                 损失索引: {loss_idx}
#                 损失权重: {loss_weight}
#                 类别权重: {class_weight}
#                 手动加权值（15位精度）: {manual_with_weight.item():.15f}
#                 原生加权值（15位精度）: {native_with_weight.item():.15f}
#                 差异值: {diff_with_weight:.20f}
#                 警告阈值: {WARN_THRESHOLD:.20f}
#                 结论: 差异超过阈值，可能存在权重计算错误！
#                 """
#                 # 写入TensorBoard（标签要清晰，方便检索）
#                 self.writer.add_text(
#                     tag=f"loss_warning/class_{class_idx}/loss_{loss_idx}/with_weight",
#                     text_string=tb_text,
#                     global_step=self.global_step
#                 )
#                 writer.flush()  # 强制刷新，确保日志立即写入

#             class_loss_total += manual_with_weight
#         self.global_step += 1

#         return class_loss_total

    def forward(self, pred, target, epoch, is_plot):
        pred_blocks, num_blocks = self._sliding_window_unfold(pred)
        target_blocks, _ = self._sliding_window_unfold(target)

        multi_class_mask, norm_complexity = self._compute_multi_class_mask(target_blocks, num_blocks)

        if is_plot and (epoch+1)%5==0:
            self._plot_sorted_variance(target_blocks, epoch)

        total_loss = 0.0
        
        for class_idx in range(self.num_classes):
            class_loss = self._compute_class_loss(
                pred_blocks, target_blocks, multi_class_mask, class_idx
            )
            total_loss += class_loss

        return total_loss
