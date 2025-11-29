# runner/tester.py
import torch

class Tester():
    def __init__(self, config, model, test_loader, hooks=None):
        super().__init__(config, model, test_loader, hooks)

    def _run_one_epoch(self, epoch: int = 0):
        self.model.eval()
        self.epoch_metrics = {"test_psnr": 0.0, "test_ssim": 0.0}

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.data_loader):
                self.call_hooks("before_batch", epoch=epoch, batch_idx=batch_idx)
                lr, hr = batch[0].to(self.device), batch[1].to(self.device)
                pred = self.model(lr)

                self.epoch_metrics["test_psnr"] += calculate_psnr(pred, hr)
                self.epoch_metrics["test_ssim"] += calculate_ssim(pred, hr)
                self.call_hooks("after_batch", epoch=epoch, batch_idx=batch_idx, metrics={
                    "psnr": calculate_psnr(pred, hr),
                    "ssim": calculate_ssim(pred, hr)
                })

        self.epoch_metrics["test_psnr"] /= len(self.data_loader)
        self.epoch_metrics["test_ssim"] /= len(self.data_loader)