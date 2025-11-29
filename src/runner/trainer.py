# runner/trainer.py
import torch

class Trainer():
    def __init__(self, config, model, data):
        super().__init__(config, model, data)
        self.val_loader = data.val_loader
        # self.optimizer
        # self.loss_fn
        self.stop_flag = False

    def _run_one_epoch(self, epoch: int):
        self.model.train()
        self.epoch_metrics = {"train_loss": 0.0, "val_psnr": 0.0}

        for batch_idx, batch in enumerate(self.data_loader):
            self.call_hooks("before_batch", epoch=epoch, batch_idx=batch_idx)
            lr, hr = batch[0].to(self.device), batch[1].to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(lr)
            loss = self.loss_fn(pred, hr)
            loss.backward()
            self.optimizer.step()

            batch_metrics = {"loss": loss.item()}
            self.epoch_metrics["train_loss"] += batch_metrics["loss"]
            self.call_hooks("after_batch", epoch=epoch, batch_idx=batch_idx, metrics=batch_metrics)

        self.epoch_metrics["train_loss"] /= len(self.data_loader)

        self.model.eval()
        with torch.no_grad():
            total_psnr = 0.0
            for batch in self.val_loader:
                lr, hr = batch[0].to(self.device), batch[1].to(self.device)
                pred = self.model(lr)
                total_psnr += calculate_psnr(pred, hr)
            self.epoch_metrics["val_psnr"] = total_psnr / len(self.val_loader)