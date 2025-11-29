# runner/trainer.py
import torch
import numpy as np
from tqdm import tqdm
from utils import make_optimizer, ExperimentRecorder, get_bare_model, save_network, load_network, MetricStats
from metrics import Loss, MetricCaculator

class Trainer:
    def __init__(self, config, model, data, logger, save_dir):
        self.config = config
        self.model = model
        self.data = data
        self.device = torch.device('cpu' if config.cpu else 'cuda')
        self.train_loader = data.train_loader
        self.test_loader = data.test_loader
        self.optimizer = make_optimizer(config.optim_args, self.model)
        self.recorder = ExperimentRecorder(config.exp_name, config, logger, self.save_dir)
        self.loss_fn = Loss(config)
        self.metric_caculator = MetricCaculator(config)
        self.TMS = MetricStats()
        self.VMS = MetricStats()
        self.logger = logger
        self.save_dir = save_dir

    def _train_one_epoch(self, epoch: int):
        self.logger.info(f"\n{'='*20} EPOCH {epoch+1}/{self.config.epoch} - TRAIN START {'='*20}")
        self.model.train()
        self.TMS.reset()

        train_pbar = tqdm(
            self.train_loader, 
            desc=f"[Train] Epoch {epoch+1}", 
            unit="batch", 
            colour="green",
            leave=False
        )
        batch_losses = []
        for batch_idx, batch in enumerate(train_pbar):
            lr, hr = batch[0].to(self.device), batch[1].to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(lr)
            loss, batch_metrics = self.loss_fn(pred, hr)
            self.TMS.update(batch_metrics)
            batch_losses.append(loss.item())
            
            loss.backward()
            self.optimizer.step()

            train_pbar.set_postfix({
                "Batch Loss": f"{loss.item():.6f}",
                "Avg Loss": f"{np.mean(batch_losses):.6f}"
            })

        train_pbar.close()

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

    def run(self):
        total_epoch = self.config.epoch
        epoch_pbar = tqdm(
            range(0, total_epoch),
            desc="Total Training",
            unit="epoch", 
            colour="blue"
        )
        for epoch_idx in epoch_pbar:
            self.data.set_epoch(epoch_idx)
            self._train_one_epoch(epoch_idx)
            self.optimizer.schedule()
            if (epoch_idx+1) % self.config.val_freq == 0:
                self._val_one_epoch(epoch_idx)
            if (epoch_idx+1) % self.config.recoder_freq == 0:
                self.recorder.plot()
                self.recorder.finish()
            if (epoch_idx+1) % self.config.save_freq == 0:
                model = get_bare_model(self.model)
                save_network(model, self.config.exp_name, epoch_idx, self.logger, self.save_dir)
                self.optimizer.save(self.save_dir, epoch_idx)
            epoch_pbar.set_postfix({
                "Epoch": f"{epoch_idx+1}/{total_epoch}",
                "Progress": f"{(epoch_idx+1)/total_epoch*100:.1f}%"
            })
        self.recorder.finish()
        epoch_pbar.close()
        self.logger.info("Training finished, all results saved")
