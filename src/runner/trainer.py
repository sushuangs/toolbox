# runner/trainer.py
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from data import Data
from models import Model
from utils import ExperimentRecorder, load_network, MetricStats
from metrics import MetricCalculator

class Trainer:
    def __init__(self, config, logger, save_dir):
        self.config = config
        self.model = Model(config, logger)
        self.data = Data(config)
        self.save_dir = save_dir
        self.total_epoch = config.total_iter
        self.writer = SummaryWriter(log_dir='/root/tf-logs')
        self.device = torch.device('cpu' if config.cpu else 'cuda')
        self.train_loader = self.data.loader_train
        self.test_loader = self.data.loader_test
        self.recorder = ExperimentRecorder(config.exp_name, config, logger, self.writer, self.save_dir)
        self.metric_caculator = MetricCalculator(config.metrics, config.rgb_range)
        self.TMS = MetricStats()
        self.VMS = MetricStats()
        self.logger = logger
        self.save_dir = save_dir

    def _train_one_epoch(self, epoch: int):
        self.logger.info(f"\n{'='*20} EPOCH {epoch+1}/{self.total_epoch} - TRAIN START {'='*20}")
        self.TMS.reset()

        for batch_idx, batch in enumerate(self.train_loader):
            lr, hr = batch[0].to(self.device), batch[1].to(self.device)
            loss, batch_metrics = self.model.train(lr, hr)
            self.TMS.update(batch_metrics)

        mean_result = self.TMS.get_mean()
        self.recorder.record_loss_iter(epoch, 'epoch', mean_result)
        self.logger.info(f"{'='*20} EPOCH {epoch+1} - TRAIN DONE {'='*20}\n")

    def _val_one_epoch(self, epoch: int):
        self.logger.info(f"\n{'='*20} EPOCH {epoch+1} - VALIDATION START {'='*20}")
        for datasetloadr in self.test_loader:
            self.VMS.reset()
            dataset_name = datasetloadr['name']
            loader = datasetloadr['loader']

            val_pbar = tqdm(
                loader, 
                desc=f"[Val] {dataset_name}", 
                unit="batch", 
                colour="yellow",
                leave=False
            )

            self.logger.info(f"[{dataset_name}] Validation")
            for batch in val_pbar:
                lr, hr = batch[0].to(self.device), batch[1].to(self.device)
                pred = self.model.validation(lr)
                batch_metrics = self.metric_caculator.calculate(pred, hr)
                self.VMS.update(batch_metrics)

            val_pbar.close()

            mean_result = self.VMS.get_mean()
            self.recorder.record_metric_iter(epoch, 'epoch', dataset_name, mean_result)

        self.logger.info(f"{'='*20} EPOCH {epoch+1} - VALIDATION DONE {'='*20}\n")
        
    def _get_stage_freq(self, current_epoch, stage_freq_list):
        """
        分段获取当前Epoch对应的保存/验证频率
        :param current_epoch: 当前实际轮数 (从1开始)
        :param stage_freq_list: 分段配置 [(截止Epoch, 频率), ...]
        :return: 当前阶段的频率
        """
        for end_epoch, freq in stage_freq_list:
            if current_epoch <= end_epoch:
                return freq

        return stage_freq_list[-1][1]

    def run(self):
        epoch_pbar = tqdm(
            range(0, self.total_epoch),
            desc="Total Training",
            unit="epoch", 
            colour="blue"
        )
        for epoch_idx in epoch_pbar:
            self.data.set_epoch(epoch_idx)
            self._train_one_epoch(epoch_idx)
            
            current_epoch = epoch_idx + 1

            val_freq = self._get_stage_freq(current_epoch, self.config.val_freq)
            recoder_freq = self._get_stage_freq(current_epoch, self.config.recoder_freq)
            save_freq = self._get_stage_freq(current_epoch, self.config.save_freq)

            if current_epoch % val_freq == 0:
                self._val_one_epoch(epoch_idx)
            if current_epoch % recoder_freq == 0:
                self.recorder.save_checkpoint()
            if current_epoch % save_freq == 0:
                model.save()   
            epoch_pbar.set_postfix({
                "Epoch": f"{current_epoch}/{self.total_epoch}",
                "Progress": f"{(current_epoch)/self.total_epoch*100:.1f}%"
            })
        self.recorder.finish()
        epoch_pbar.close()
        self.logger.info("Training finished, all results saved")
