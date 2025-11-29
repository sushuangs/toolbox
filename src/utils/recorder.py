import os
import json
import imageio
from datetime import datetime
import matplotlib.pyplot as plt
from utils.utility import Tensor2np
from typing import Dict, Any, Optional


class ExperimentRecorder:
    def __init__(
        self,
        exp_name: str,
        config: Dict[str, Any],
        logger,
        save_dir: str = "./experiment/result"
    ):
        self.exp_name = exp_name
        self.config = config
        self.logger = logger
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.file_path = os.path.join(save_dir, f"{exp_name}.json")
        self._init_img_path()
        self._init_results()

        if os.path.exists(self.file_path):
            self._load_existing_results()
    
    def _init_results(self):
        self.results = {
            "meta": {
                "exp_name": self.exp_name,
                "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": None,
                "config": self.config
            },
            "loss": [],
            "metrics": {},  # [{"epoch": 0, "PSNR": 30.2, "SSIM": 0.91}, ...]
            "best_metric": {}
        }

        for d in self.config.data_test:
            if d not in self.results['metrics']:
                self.results['metrics'][d] = []
            if d not in self.results['best_metric']:
                self.results['best_metric'][d] = {}

            for m in self.config.metrics:
                better = m.better if m.better is not None else 'higher'
                init_val = float('-inf') if better == 'higher' else float('inf')
                init_best_metric = dict(better=better, val=init_val, iter=-1)
                self.results['best_metric'][d].update({m.name:init_best_metric})

    def _init_img_path(self):
        self.img_path_dict = {}

        for name in self.config.data_test:
            img_base_path = os.path.join(self.save_dir, name)
            img_path_lr = os.path.join(img_base_path, 'lr')
            img_path_hr = os.path.join(img_base_path, 'hr')
            self.img_path_dict[name] = (img_path_lr, img_path_hr)

    def _load_existing_results(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        self.results = existing
    
    def record_img_pair(self, lr, hr, dataset_name, index, data_range=255):
        lr_np, hr_np = Tensor2np(lr.squeeze(), hr.squeeze(), data_range)
        img_path_lr, img_path_hr = self.img_path_dict[dataset_name]
        img_path_lr_result = os.path.join(img_path_lr, f"lr_{index}.png")
        img_path_hr_result = os.path.join(img_path_hr, f"hr_{index}.png")
        imageio.imwrite(img_path_lr_result, lr_np)
        imageio.imwrite(img_path_hr_result, hr_np)
    
    def plot(self):
        self.plot_loss('all_losses', ylabel=None)
        for dataset_name in self.config.data_test:
            self.plot_metric('all_metrics', dataset_name, ylabel=None)

    def plot_loss(self, name, xlabel: str = "epoch", ylabel: Optional[str] = None):
        if not self.results["loss"]:
            self.logger.info("No loss data to plot!")
            return

        first_loss_item = self.results["loss"][0]
        all_metrics = [key for key in first_loss_item.keys() if key != xlabel]
        
        target_metrics = all_metrics if ylabel is None else [ylabel]

        for metric in target_metrics:
            values = [d.get(metric, 0.0) for d in self.results["loss"]] 
            index = [d[xlabel] for d in self.results["loss"]]
            if not values:
                self.logger.info(f"No data for loss metric: {metric}")
                continue

            plt.figure(figsize=(10, 6))
            plt.plot(index, values, marker="o", alpha=0.8, linewidth=2, label=metric)
            
            plt.xlabel(xlabel, fontsize=12)
            plt.ylabel(metric, fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(alpha=0.3)
            plt.tight_layout()

            metric_save_path = os.path.join(self.save_dir, 'plot_loss')
            os.makedirs(metric_save_path, exist_ok=True)
            save_filename = f"{name}_{metric}.png" if ylabel is None else f"{name}.png"
            plt.savefig(os.path.join(metric_save_path, save_filename), dpi=300, bbox_inches="tight")
            plt.close()

    def plot_metric(self, name, dataset_name, xlabel: str = "epoch", ylabel: Optional[str] = None):
        if dataset_name not in self.results["metrics"]:
            self.logger.warning(f"Dataset {dataset_name} not found in metrics!")
            return
        metric_data = self.results["metrics"][dataset_name]
        if not metric_data:
            self.logger.info(f"No metric data for dataset: {dataset_name}")
            return

        first_metric_item = metric_data[0]
        all_metrics = [key for key in first_metric_item.keys() if key != xlabel]
        
        target_metrics = all_metrics if ylabel is None else [ylabel]

        for metric in target_metrics:
            values = [d.get(metric, 0.0) for d in metric_data]
            index = [d[xlabel] for d in metric_data]
            if not values:
                self.logger.info(f"No data for metric {metric} (dataset: {dataset_name})")
                continue

            plt.figure(figsize=(10, 6))
            plt.plot(index, values, marker="o", alpha=0.8, linewidth=2, label=metric)
            
            plt.xlabel(xlabel, fontsize=12)
            plt.ylabel(metric, fontsize=12)
            plt.legend(fontsize=10)
            plt.grid(alpha=0.3)
            plt.tight_layout()

            metric_save_path = os.path.join(self.save_dir, f'plot_{dataset_name}')
            os.makedirs(metric_save_path, exist_ok=True)
            save_filename = f"{name}_{metric}.png" if ylabel is None else f"{name}.png"
            plt.savefig(os.path.join(metric_save_path, save_filename), dpi=300, bbox_inches="tight")
            plt.close()

    def record_metric_iter(self, index: int, index_name, dataset_name, metrics: Dict[str, float], save_file=False, writer_mode="default"):
        epoch_data = {f"{index_name}": index}
        epoch_data.update(metrics)

        log_prefix = f"[VAL] {dataset_name} | {index_name}={index}"
        
        metric_logs = []
        for key, value in metrics.items():
            if key == index_name:
                continue
            
            if key not in self.results["best_metric"].get(dataset_name, {}):
                self.logger.warning(f"{log_prefix} | Metric {key} not in best_metric config, skip update")
                metric_logs.append(f"{key}: {value:.4f} (no config)")
                continue

            best_info = self.results["best_metric"][dataset_name][key]
            better_mode = best_info['better']
            current_best_val = best_info['val']
            is_updated = False

            if better_mode == 'higher':
                if value > current_best_val:
                    best_info['val'] = value
                    best_info['iter'] = index
                    is_updated = True
            else:  # lower is better (e.g., loss)
                if value < current_best_val:
                    best_info['val'] = value
                    best_info['iter'] = index
                    is_updated = True

            if is_updated:
                metric_logs.append(
                    f"{key}: {value:.4f} (BEST, prev: {current_best_val:.4f})"
                )
            else:
                metric_logs.append(
                    f"{key}: {value:.4f} (best: {current_best_val:.4f} @ {index_name}={best_info['iter']})"
                )

        self.logger.info(f"{log_prefix} | Metrics:")
        for log in metric_logs:
            self.logger.info(f"  - {log}")

        if writer_mode.find("+") > 0:
            for i, item in enumerate(self.results["metrics"][dataset_name]):
                if item[index_name] == index:
                    self.results["metrics"][dataset_name][i] = epoch_data
                    break
        else:
            self.results["metrics"][dataset_name].append(epoch_data)

        if save_file:
            self.save_to_file()


    def record_loss_iter(self, index: int, index_name, losses: Dict[str, float], save_file=False, writer_mode="default"):
        epoch_data = {f"{index_name}": index}
        epoch_data.update(losses)  # {"epoch":0, "PSNR":30.2, "train_loss":0.5}

        log_prefix = f"[TRAIN] | {index_name}={index}"
        
        total_loss_key = 'total_loss' if 'total_loss' in losses else next(iter(losses.keys()), None)
        if total_loss_key is None:
            self.logger.warning(f"{log_prefix} | No loss data to record")
            return
        
        loss_logs = [f"Total Loss: {losses[total_loss_key]:.6f}"]
        for key, value in losses.items():
            if key != total_loss_key and key != index_name:
                loss_logs.append(f"{key}: {value:.6f}")
        
        self.logger.info(f"{log_prefix} | Loss:")
        for log in loss_logs:
            self.logger.info(f"  - {log}")
        
        if writer_mode.find("+") > 0:
            for i, item in enumerate(self.results["loss"]):
                if item[index_name] == index:
                    self.results["loss"][i] = epoch_data
                    break
        else:
            self.results["loss"].append(epoch_data)
        
        if  save_file:
            self.save_to_file()

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
            lr_img = Tensor2np(lr_tensors[i])
            hr_img = Tensor2np(hr_tensors[i])
            pred_img = Tensor2np(pred_tensors[i])

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

    def save_to_file(self):
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

    def finish(self):
        self.results["meta"]["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.save_to_file()