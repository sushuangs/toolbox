import torch
import random
import numpy as np
from functools import partial
from importlib import import_module
from torch.utils.data import dataloader
from torch.utils.data import Sampler


class Data:
    def __init__(self, config):
        self.loader_train = None
        if not config.test_only:
            d = config.data_train
            module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
            m = import_module('data.' + module_name.lower())
            dataset = getattr(m, module_name)(config, name=d)

            self.sampler = DeterministicSampler(dataset)

            self.loader_train = dataloader.DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=False,
                sampler = self.sampler,
                pin_memory=not config.cpu,
                num_workers=config.n_threads,
                worker_init_fn=partial(worker_init_fn, seed=config.manual_seed) if config.manual_seed is not None else None
            )

        self.loader_test = []
        for d in config.data_test:
            if d in ['Set5', 'Set14', 'B100', 'Urban100']:
                m = import_module('data.benchmark')
                testset = getattr(m, 'Benchmark')(config, train=False, name=d)
            else:
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                m = import_module('data.' + module_name.lower())
                testset = getattr(m, module_name)(config, train=False, name=d)

            self.loader_test.append({'name':d,
                'loader':dataloader.DataLoader(
                    testset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=0,
                )
                }
            )

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)


class DeterministicSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.epoch = 0
    
    def set_epoch(self, epoch):
        self.epoch = epoch
    
    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()
        return iter(indices)
    
    def __len__(self):
        return len(self.dataset)

def worker_init_fn(worker_id, seed):
    worker_seed = seed + worker_id
    np.random.seed(worker_seed % (2**32 - 1))
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
