import torch
import torch.distributed as dist


class DistributedDataParallel(torch.nn.parallel.DistributedDataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

def is_master_process():
    return get_rank() == 0

def get_rank():
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def get_size():
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def all_reduce(tensor, op=dist.ReduceOp.SUM):
    if not dist.is_initialized():
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=op)
    return tensor


def all_reduce_list(tensor_list, op=dist.ReduceOp.SUM):
    if not dist.is_initialized():
        return tensor_list
    tensor_list = [t.clone() for t in tensor_list]
    handles = [dist.all_reduce(t, op=op, async_op=True)
               for t in tensor_list]
    for h in handles:
        h.wait()
    return tensor_list


def all_reduce_avg(tensor):
    if not dist.is_initialized():
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tensor


def all_reduce_avg_list(tensor_list):
    if not dist.is_initialized():
        return tensor_list
    size = get_size()
    tensor_list = [t.clone() for t in tensor_list]
    handles = [dist.all_reduce(t, op=dist.ReduceOp.SUM,
                               async_op=True)
               for t in tensor_list]
    for h in handles:
        h.wait()
    return [t / size for t in tensor_list]

def all_reduce_avg_dict(tensor_dict):
    if not dist.is_initialized():
        return tensor_dict
    size = get_size()
    tensor_dict = {k: t.clone() for k, t in tensor_dict.items()}
    handles = [dist.all_reduce(t, op=dist.ReduceOp.SUM,
                               async_op=True)
               for t in tensor_dict.values()]
    for h in handles:
        h.wait()
    return {k: t / size for k, t in tensor_dict.items()}


def all_gather(tensor):
    if not dist.is_initialized():
        return tensor 
    size = dist.get_world_size()
    tensor_list = [torch.zeros_like(tensor) for _ in range(size)]
    dist.all_gather(tensor_list, tensor)
    return torch.cat(tensor_list)


def broadcast(tensor, src):
    if not dist.is_initialized():
        return tensor 
    tensor = tensor.clone()
    dist.broadcast(tensor, src)
    return tensor