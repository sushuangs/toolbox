import json
import os
from .utility import tensor_to_np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional

class ExperimentVisualizer:
    def __init__(self, results_dir: str = "../experiment/results"):
        self.results_dir = results_dir
        self.experiment_data = {}

    def _load_experiment(self, exp_name: str) -> Dict[str, Any]:
        file_path = os.path.join(self.results_dir, f"{exp_name}.json")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{exp_name} not exist:{file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.experiment_data[exp_name] = data
        return data

    def _extract_metric(self, exp_name: str, metric_name: str) -> Dict[int, float]:
        if exp_name not in self.experiment_data:
            self._load_experiment(exp_name)
        metrics = self.experiment_data[exp_name]["metrics"]
        return {
            item["epoch"]: item[metric_name]
            for item in metrics
            if metric_name in item
        }

    def plot_metric_curve(
        self,
        exp_names: List[str],
        metric_name: str,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ):
        plt.figure(figsize=(10, 6))
        
        for exp_name in exp_names:
            epoch_values = self._extract_metric(exp_name, metric_name)
            if not epoch_values:
                print(f"警告：实验 {exp_name} 未记录 {metric_name} 指标")
                continue
            epochs = sorted(epoch_values.keys())
            values = [epoch_values[epoch] for epoch in epochs]
            plt.plot(epochs, values, marker="o", label=exp_name, alpha=0.8, linewidth=2)
        
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel(metric_name, fontsize=12)
        plt.title(title or f"{metric_name} Comparison", fontsize=14, pad=20)
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"折线图已保存至：{save_path}")
        else:
            plt.show()


class EpochVisualizer:
    def __init__(
        self,
        exp_name: str,  
        metric_name: str,  
        save_dir: str = "./experiment/results",  
        xlabel: str = "Epoch",  
        ylabel: Optional[str] = None, 
        title: Optional[str] = None
    ):
        self.exp_name = exp_name
        self.metric_name = metric_name
        self.save_dir = os.path.join(save_dir, exp_name)
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.epochs: List[int] = []
        self.values: List[float] = []
        
        self.xlabel = xlabel
        self.ylabel = ylabel or metric_name
        self.title = title or f"{exp_name} - {metric_name} Convergence Curve"
        
        self.fig, self.ax = plt.subplots(figsize=(8, 5))

    def update(self, epoch: int, value: float, save: bool = True) -> str:
        self.epochs.append(epoch)
        self.values.append(value)
        
        self.ax.clear()
        self.ax.plot(self.epochs, self.values, marker="o", color="tab:blue", alpha=0.8, linewidth=2)
        
        if self.epochs:
            last_epoch = self.epochs[-1]
            last_value = self.values[-1]
            self.ax.annotate(
                f"{last_value:.4f}",
                xy=(last_epoch, last_value),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.5)
            )
        
        self.ax.set_xlabel(self.xlabel, fontsize=11)
        self.ax.set_ylabel(self.ylabel, fontsize=11)
        self.ax.set_title(self.title, fontsize=13, pad=15)
        self.ax.grid(alpha=0.3)
        self.fig.tight_layout()
        
        save_path = None
        if save:
            save_path = os.path.join(self.save_dir, f"{self.metric_name}_curve.png")
            self.fig.savefig(save_path, dpi=300, bbox_inches="tight")
        
        plt.close(self.fig)
        return save_path
    
    def save_lr_hr_pred_comparison(
        self,
        lr_tensors,
        hr_tensors,
        pred_tensors,
        epoch,
        num_samples = 1
    ) -> str:
        save_path = os.path.join(self.save_dir, f"comparison_epoch_{epoch}.png")

        num_samples = min(num_samples, len(lr_tensors), len(hr_tensors), len(pred_tensors))
        if num_samples == 0:
            raise ValueError

        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
        if num_samples == 1:
            axes = [axes]

        for i in range(num_samples):
            lr_img = tensor_to_np(lr_tensors[i])
            hr_img = tensor_to_np(hr_tensors[i])
            pred_img = tensor_to_np(pred_tensors[i])

            axes[i][0].imshow(lr_img)
            axes[i][0].set_title(f"LR")
            axes[i][0].axis("off")

            axes[i][1].imshow(pred_img)
            axes[i][1].set_title(f"Pred")
            axes[i][1].axis("off")

            axes[i][2].imshow(hr_img)
            axes[i][2].set_title(f"HR")
            axes[i][2].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return save_path