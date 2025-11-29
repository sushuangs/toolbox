from importlib import import_module
from torch.utils.data import dataloader
from torch.utils.data import ConcatDataset


class Data:
    def __init__(self, config):
        self.loader_train = None
        if not config.test_only:
            datasets = []
            for d in config.data_train:
                module_name = d if d.find('DIV2K-Q') < 0 else 'DIV2KJPEG'
                m = import_module('data.' + module_name.lower())
                datasets.append(getattr(m, module_name)(config, name=d))

            self.loader_train = dataloader.DataLoader(
                ConcatDataset(datasets),
                batch_size=config.batch_size,
                shuffle=True,
                pin_memory=not config.cpu,
                num_workers=config.n_threads,
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

            self.loader_test.append(
                dataloader.DataLoader(
                    testset,
                    batch_size=1,
                    shuffle=False,
                    pin_memory=not config.cpu,
                    num_workers=config.n_threads,
                )
            )