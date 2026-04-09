# runner/trainer.py
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils import make_optimizer, ExperimentRecorder, get_bare_model, save_network, load_network, MetricStats
from metrics import MetricCalculator
from losses import Loss

class Trainer:
    def __init__(self, config, model, data, logger, save_dir):
        self.config = config
        self.model = model
        self.data = data
        self.save_dir = save_dir
        self.total_epoch = config.total_iter
        self.writer = SummaryWriter(log_dir='/root/tf-logs')
        self.device = torch.device('cpu' if config.cpu else 'cuda')
        self.train_loader = data.loader_train
        self.test_loader = data.loader_test
        self.optimizer = make_optimizer(config.optim_args, self.model)
        self.recorder = ExperimentRecorder(config.exp_name, config, logger, self.writer, self.save_dir)
        self.loss_fn = Loss(config, self.writer)
        self.metric_caculator = MetricCalculator(config.metrics, config.rgb_range)
        self.TMS = MetricStats()
        self.VMS = MetricStats()
        self.logger = logger
        self.save_dir = save_dir

    def _train_one_epoch(self, epoch: int):
        self.logger.info(f"\n{'='*20} EPOCH {epoch+1}/{self.total_epoch} - TRAIN START {'='*20}")
        self.model.train()
        self.TMS.reset()

        batch_losses = []
        first_batch = True
        for batch_idx, batch in enumerate(self.train_loader):
            lr, hr = batch[0].to(self.device), batch[1].to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(lr)
            loss, batch_metrics = self.loss_fn(pred, hr)
            
            loss.backward()
            self.optimizer.step()
            self.optimizer.schedule()
            
            self.TMS.update(batch_metrics)
            batch_losses.append(loss.item())
            first_batch = False

        mean_result = self.TMS.get_mean()
        self.recorder.record_loss_iter(epoch, 'epoch', mean_result)
        self.logger.info(f"{'='*20} EPOCH {epoch+1} - TRAIN DONE {'='*20}\n")

    def _val_one_epoch(self, epoch: int):
        self.logger.info(f"\n{'='*20} EPOCH {epoch+1} - VALIDATION START {'='*20}")
        self.model.eval()
        with torch.no_grad():
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
                    pred = self.model(lr)
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
                model = get_bare_model(self.model)
                save_network(model, self.config.exp_name, epoch_idx, self.logger, self.save_dir)
                self.optimizer.save(self.save_dir, epoch_idx)
            epoch_pbar.set_postfix({
                "Epoch": f"{current_epoch}/{self.total_epoch}",
                "Progress": f"{(current_epoch)/self.total_epoch*100:.1f}%"
            })
        self.recorder.finish()
        epoch_pbar.close()
        self.logger.info("Training finished, all results saved")
