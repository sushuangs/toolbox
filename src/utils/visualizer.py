import json
import os
from utils.utility import Tensor2np
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
