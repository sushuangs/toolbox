import os
import numpy as np
import time
import random
from copy import deepcopy
import logging
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel


def get_logger(
    experiment_name: str,
    log_dir: str = "./experiment/logs",
    level: int = logging.INFO,
) -> logging.Logger:

    exp_log_dir = log_dir
    os.makedirs(exp_log_dir, exist_ok=True)

    logger = logging.getLogger(experiment_name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    log_filename = f"{experiment_name}.log"
    log_filepath = os.path.join(exp_log_dir, log_filename)

    file_handler = logging.FileHandler(
        filename=log_filepath,
        mode='a',
        encoding="utf-8"
    )

    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)

    return logger


def Tensor2np(*args, rgb_range=255):
    def _tensor2np(tensor):
        tensor = tensor.squeeze(0)
        np_arr = tensor.detach().cpu().numpy()
        np_arr = np_arr.transpose((1, 2, 0))
        np_arr = np_arr * (255.0 / rgb_range)
        np_arr = np.clip(np_arr, 0, 255)
        return np_arr
    
    return [_tensor2np(a) for a in args]


def np2Tensor(*args, rgb_range=255):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)

        return tensor

    return [_np2Tensor(a) for a in args]

def get_bare_model(net):
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net = net.module
        return net

def save_network(net, net_label, current_iter, logger, save_dir, param_key='params'):
    """Save networks.

    Args:
        net (nn.Module | list[nn.Module]): Network(s) to be saved.
        net_label (str): Network label.
        current_iter (int): Current iter number.
        param_key (str | list[str]): The parameter key(s) to save network.
            Default: 'params'.
    """
    if current_iter == -1:
        current_iter = 'latest'
    save_filename = f'{net_label}_{current_iter}.pth'
    save_path = os.path.join(save_dir, save_filename)

    net = net if isinstance(net, list) else [net]
    param_key = param_key if isinstance(param_key, list) else [param_key]
    assert len(net) == len(param_key), 'The lengths of net and param_key should be the same.'

    save_dict = {}
    for net_, param_key_ in zip(net, param_key):
        net_ = get_bare_model(net_)
        state_dict = net_.state_dict()
        for key, param in state_dict.items():
            if key.startswith('module.'):  # remove unnecessary 'module.'
                key = key[7:]
            state_dict[key] = param.cpu()
        save_dict[param_key_] = state_dict

    # avoid occasional writing errors
    retry = 3
    while retry > 0:
        try:
            torch.save(save_dict, save_path)
        except Exception as e:
            logger.warning(f'Save model error: {e}, remaining retry times: {retry - 1}')
            time.sleep(1)
        else:
            break
        finally:
            retry -= 1
    if retry == 0:
        logger.warning(f'Still cannot save {save_path}. Just ignore it.')
        # raise IOError(f'Cannot save {save_path}.')

def load_network(net, load_path, logger, strict=True, param_key='params'):
    """Load network.

    Args:
        load_path (str): The path of networks to be loaded.
        net (nn.Module): Network.
        strict (bool): Whether strictly loaded.
        param_key (str): The parameter key of loaded network. If set to
            None, use the root 'path'.
            Default: 'params'.
    """
    net = get_bare_model(net)
    load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
    if param_key is not None:
        if param_key not in load_net and 'params' in load_net:
            param_key = 'params'
            logger.info('Loading: params_ema does not exist, use params.')
        load_net = load_net[param_key]
    logger.info(f'Loading {net.__class__.__name__} model from {load_path}, with param key: [{param_key}].')
    # remove unnecessary 'module.'
    for k, v in deepcopy(load_net).items():
        if k.startswith('module.'):
            load_net[k[7:]] = v
            load_net.pop(k)
    net.load_state_dict(load_net, strict=strict)

def make_optimizer(args, target):
    # optimizer
    trainable = filter(lambda x: x.requires_grad, target.parameters())
    kwargs_optimizer = {'lr': args.lr, 'weight_decay': args.weight_decay}

    if args.optimizer == 'SGD':
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = args.momentum
    elif args.optimizer == 'ADAM':
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = args.betas
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'RMSprop':
        optimizer_class = optim.RMSprop
        kwargs_optimizer['eps'] = args.epsilon

    # scheduler
    kwargs_scheduler = {'milestones': args.milestones, 'gamma': args.gamma}
    scheduler_class = lrs.MultiStepLR

    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)

        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs)

        def save(self, save_dir, current_epoch):
            save_path = self.get_dir(save_dir)
            state = {
                'optimizer': self.state_dict(),
                'scheduler': self.scheduler.state_dict() if self.scheduler else None,
                'epoch': current_epoch
            }
            torch.save(state, save_path)

        def load(self, load_dir):
            load_path = self.get_dir(load_dir)
            if not os.path.exists(load_path):
                raise FileNotFoundError(f"Optimizer checkpoint not found: {load_path}")
            
            state = torch.load(load_path, map_location='cpu')
            self.load_state_dict(state['optimizer'])
            if self.scheduler and state['scheduler'] is not None:
                self.scheduler.load_state_dict(state['scheduler'])
            return state['epoch']

        def get_dir(self, dir_path):
            return os.path.join(dir_path, 'optimizer.pt')

        def schedule(self):
            self.scheduler.step()

        def get_lr(self):
            return self.scheduler.get_lr()[0]

        def get_last_epoch(self):
            return self.scheduler.last_epoch
    
    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    return optimizer


def set_seed(seed=42):
    random.seed(seed)
    
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False