import os
import os.path as osp
import shutil
import random
from collections import OrderedDict

import numpy as np

import torch

from .dist_ops import is_master_process


def save_checkpoint(state, is_best, output_dir, filename='checkpoint.pth.tar'):
    if is_master_process:
        ckpt_dir = osp.join(output_dir, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        filename = osp.join(ckpt_dir, filename)
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, osp.join(ckpt_dir, 'model_best.pth.tar'))
        print(f"saved checkpoint to {filename}, is_best = {is_best}")


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def compute_total_params(model):
    return sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad])


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, total_iters, meter_names, prefix=""):
        self.iter_fmtstr = self._get_iter_fmtstr(total_iters)
        self.meters = OrderedDict({mn: AverageMeter(mn, ':6.3f') 
                                   for mn in meter_names})
        self.prefix = prefix
    
    def update(self, n=1, **kwargs):
        for k, v in kwargs.items():
            self.meters[k].update(v, n=n)

    def display(self, iteration):
        entries = [self.prefix + self.iter_fmtstr.format(iteration)]
        entries += [str(meter) for meter in self.meters.values()]
        print('\t'.join(entries))

    def _get_iter_fmtstr(self, total_iters):
        num_digits = len(str(total_iters // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(total_iters) + ']'  
