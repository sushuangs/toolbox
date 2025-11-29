import json
import os
from datetime import datetime
from typing import Dict, Any, Optional

class ExperimentRecorder:
    def __init__(
        self,
        exp_name: str,
        mode,
        config: Dict[str, Any],
        save_dir: str = "./experiment/result"
    ):
        self.exp_name = exp_name
        self.config = config
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.file_path = os.path.join(save_dir, f"{exp_name}_{mode}.json")
        
        self.results = {
            "meta": {
                "exp_name": exp_name,
                "mode": mode,
                "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": None,
                "config": config
            },
            "metrics": []  # [{"epoch": 0, "PSNR": 30.2, "SSIM": 0.91}, ...]
        }
        
        if os.path.exists(self.file_path):
            self._load_existing_results()

    def _load_existing_results(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        self.results = existing

    def record_epoch(self, index: int, index_name, metrics: Dict[str, float], writer_mode="default"):
        epoch_data = {f"{index_name}": index}
        epoch_data.update(metrics)  # {"epoch":0, "PSNR":30.2, "train_loss":0.5}
        
        if writer_mode.find("+") > 0:
            for i, item in enumerate(self.results["metrics"]):
                if item[index_name] == index:
                    self.results["metrics"][i] = epoch_data
                    break
        else:
            self.results["metrics"].append(epoch_data)
        
        self._save_to_file()

    def _save_to_file(self):
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

    def finish(self):
        self.results["meta"]["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._save_to_file()