import os
from importlib import import_module
import time
import torch
import torch.nn as nn
import torch.nn.parallel as P
from torch.nn import functional as F
from utils import make_optimizer, get_bare_model, save_network
from losses import Loss

class Model():
    def __init__(self, args, logger):
        super(Model, self).__init__()
        print('Making model...')

        self.self_ensemble = args.self_ensemble
        self.precision = args.precision
        self.type = args.model.lower()
        self.cpu = args.cpu
        
        self.logger = logger
        
        self.loss_fn = Loss(args)

        if self.type.find('swinir') >= 0:
            self.window_size = args.network_g.window_size
            self.scale = args.scale
        
        if self.cpu:
            self.device = torch.device('cpu')
        else:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')

        self.n_GPUs = args.n_GPUs

        module = import_module('models.' + args.model.lower())
        self.network_g = module.make_model(args.network_g).to(self.device)

        if args.ema_decay > 0:
            self.ema_decay = args.ema_decay
            self.network_g_ema = module.make_model(args.network_g).to(self.device)
            self.model_ema(0)
            self.network_g_ema.eval()

        self.optimizer = make_optimizer(args.optim_args, self.network_g)

        if args.precision == 'half':
            self.network_g.half()
            if hasattr(self, 'network_g_ema'):
                self.network_g_ema.half()
            
    def model_ema(self, decay=0.999):
        net_g = get_bare_model(self.network_g)

        net_g_params = dict(net_g.named_parameters())
        net_g_ema_params = dict(self.network_g_ema.named_parameters())

        for k in net_g_ema_params.keys():
            net_g_ema_params[k].data.mul_(decay).add_(net_g_params[k].data, alpha=1 - decay)
            
    def train(self, lr, hr):
        self.optimizer.zero_grad()
        self.network_g.train()
        if self.n_GPUs > 1:
            pred =  P.data_parallel(self.network_g, lr, range(self.n_GPUs))
        else:
            pred = self.network_g(lr)
        loss, batch_metrics = self.loss_fn(pred, hr)
        loss.backward()
        self.optimizer.step()
        self.optimizer.schedule()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
        
        return loss, batch_metrics
    
    def validation(self, lr):
        self.network_g.eval()
        model = self.network_g
        if hasattr(self, 'network_g_ema'):
            self.network_g_ema.eval()
            model = self.network_g_ema
        with torch.no_grad():
            if self.type.find('swinir') >= 0:
                mod_pad_h, mod_pad_w = 0, 0
                _, _, h, w = lr.size()
                if h % self.window_size != 0:
                    mod_pad_h = self.window_size - h % self.window_size
                if w % self.window_size != 0:
                    mod_pad_w = self.window_size - w % self.window_size
                img = F.pad(lr, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
                if self.self_ensemble:
                    output = self.forward_x8(img, model=model)
                else:
                    output = model(img)
                _, _, h, w = output.size()
                output = output[:, :, 0:h - mod_pad_h * self.scale, 0:w - mod_pad_w * self.scale]
                return output
            else:
                if self.self_ensemble:
                    output = self.forward_x8(lr, model=model)
                else:
                    output = model(lr)
        self.network_g.train()
        return output  

    def forward_x8(self, *args, model=None):
        def _transform(v, op):
            if self.precision != 'single': v = v.float()

            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            if self.precision == 'half': ret = ret.half()

            return ret

        list_x = []
        for a in args:
            x = [a]
            for tf in 'v', 'h', 't': x.extend([_transform(_x, tf) for _x in x])

            list_x.append(x)

        list_y = []
        for x in zip(*list_x):
            y = model(*x)
            if not isinstance(y, list): y = [y]
            if not list_y:
                list_y = [[_y] for _y in y]
            else:
                for _list_y, _y in zip(list_y, y): _list_y.append(_y)

        for _list_y in list_y:
            for i in range(len(_list_y)):
                if i > 3:
                    _list_y[i] = _transform(_list_y[i], 't')
                if i % 4 > 1:
                    _list_y[i] = _transform(_list_y[i], 'h')
                if (i % 4) % 2 == 1:
                    _list_y[i] = _transform(_list_y[i], 'v')

        y = [torch.cat(_y, dim=0).mean(dim=0, keepdim=True) for _y in list_y]
        if len(y) == 1: y = y[0]

        return y

    def save(self, exp_name, epoch_idx, logger, save_dir):
        if hasattr(self, 'net_g_ema'):
            save_network([self.net_g, self.net_g_ema], exp_name, epoch_idx, logger, save_dir, param_key=['params', 'params_ema'])
        else:
            save_network(self.net_g, 'net_g', current_iter)
        self.optimizer.save(self.save_dir, epoch_idx)