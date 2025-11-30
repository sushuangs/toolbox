import torch


class MetricStats:
    def __init__(self):
        self._stats = {}
        self.reset()

    def reset(self):
        self._stats = {k: (0.0, 0) for k in self._stats.keys()} if self._stats else {}

    def update(self, batch_metrics):
        for name in batch_metrics.keys():
            if name not in self._stats:
                self._stats[name] = (0.0, 0)

        for name, value in batch_metrics.items():
            if value is not None:
                total, count = self._stats[name]
                self._stats[name] = (total + value, count + 1)

    def get_mean(self):
        mean_metrics = {}
        for name, (total, count) in self._stats.items():
            if count == 0:
                mean_metrics[name] = None
            else:
                mean_metrics[name] = total / count
        return mean_metrics