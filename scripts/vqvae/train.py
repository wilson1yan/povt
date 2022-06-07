import sys
import time
import os
import os.path as osp
import pickle
import yaml

import wandb

import torch
import torch.nn as nn
import torch.distributed as dist
from torchvision.utils import make_grid
import torch.nn.functional as F

from povt.data import Data, cycle
from povt.train_utils import save_checkpoint, seed_all, compute_total_params, \
    ProgressMeter
from povt.dist_ops import is_master_process, DistributedDataParallel, \
    get_size, all_reduce_avg_dict
from povt.models import get_model


def main(save_dir):
    global args
    args = pickle.load(open(osp.join(save_dir, 'args'), 'rb'))
    args_d = vars(args)
    args_d.update(yaml.load(open(args.config, 'r')))
    if is_master_process():
        pickle.dump(args, open(osp.join(save_dir, 'args'), 'wb'))

    rank = int(os.environ['LOCAL_RANK'])
    size = int(os.environ['WORLD_SIZE'])
    args.rank = rank
    args.size = size

    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(rank)
    torch.backends.cudnn.benchmark = True

    seed = args.seed + rank
    seed_all(seed)

    if size > 1:
        dist.init_process_group(backend='nccl',
                                init_method=f'tcp://{os.environ["MASTER_ADDR"]}:{os.environ["MASTER_PORT"]}',
                                world_size=size, rank=rank)

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location='cpu')
        if is_master_process():
            print(f'Loaded ckpt from {args.ckpt} at itr {ckpt["iteration"]} with loss {ckpt["cur_loss"]}')
        ckpt_dir = osp.dirname(osp.dirname(args.ckpt))
    else:
        ckpt_dir = None

    if is_master_process():
        root_dir = os.environ['OCL_DATA_DIR']
        os.makedirs(osp.join(root_dir, 'wandb'), exist_ok=True)

        wandb.init(project='povt', config=args,
                   dir=root_dir)
        wandb.run.name = osp.basename(args.output_dir)
        wandb.run.save()

    data = Data(args)
    train_loader, test_loader = data.train_dataloader(), data.test_dataloader()
    if is_master_process():
        print(f"Dataset {args.data_path}: {len(train_loader.dataset)} (train), {len(test_loader.dataset)} (test)")
     
    model = get_model(args, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                           T_max=args.total_steps)

    if ckpt_dir is not None:
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        best_loss, iteration = ckpt['best_loss'], ckpt['iteration'] + 1
    else:
        best_loss, iteration = float('inf'), 1

    if size > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(model, device_ids=[rank],
                                        broadcast_buffers=False, find_unused_parameters=False)
    scheduler.step()

    if is_master_process():
        total_parameters = compute_total_params(model)
        print(f"model size: params count with grads = {total_parameters}")

    if args.no_output and os.environ.get('DEBUG') != '1':
        sys.stdout = open(os.devnull, 'w')

    train_loader = cycle(train_loader, start_iteration=iteration)
    while iteration <= args.total_steps:
        iteration = train(iteration, model, train_loader, optimizer, scheduler, device)
        if iteration % args.test_interval == 0:
            test_loss = validate(iteration, model, test_loader, device)

            is_best = test_loss < best_loss
            best_loss = min(test_loss, best_loss)
            ckpt_dict = dict(
                iteration=iteration,
                best_loss=best_loss,
                cur_loss=test_loss,
                model=model.module.state_dict() if get_size() > 1 else model.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict()
            )
            if is_master_process():
                save_checkpoint(ckpt_dict, is_best=is_best, output_dir=args.output_dir)
        if iteration % args.eval_interval == 0:
            if is_master_process():
                visualize(iteration, model, test_loader, device)
        
        iteration += 1


def train(iteration, model, train_loader, optimizer, scheduler, device):
    progress = ProgressMeter(
        args.total_steps,
        ['time', 'data'] + model.metrics
    )

    model.train()
    end = time.time()

    while True:
        batch = next(train_loader)
        videos = batch['video'].to(device, non_blocking=True)  # BCHW
        batch_size = videos.shape[0]

        if is_master_process() and args.log_train:
            wandb.log({'lr': optimizer.param_groups[0]['lr']}, step=iteration)
        progress.update(data=time.time() - end)

        return_dict = model(videos)
        optimizer.zero_grad()
        return_dict['loss'].backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
        optimizer.step()
        scheduler.step()

        metrics = {metric: return_dict[metric] for metric in model.metrics} 
        progress.update(n=batch_size, **{k: v.item() for k, v in metrics.items()})

        if is_master_process() and args.log_train:
            wandb.log({f'train/{metric}': val 
                       for metric, val in metrics.items()},
                       step=iteration)

        progress.update(time=time.time() - end)
        end = time.time()

        if is_master_process() and iteration % args.log_interval == 0:
            progress.display(iteration)

        if iteration % args.eval_interval == 0 or \
           iteration % args.test_interval == 0 or \
           iteration >= args.total_steps:
            return iteration

        iteration += 1


def validate(iteration, model, test_loader, device):
    progress = ProgressMeter(
        len(test_loader),
        ['time', 'data'] + model.metrics,
        prefix='\tTest:'
    )

    model.eval()
    end = time.time()

    torch.set_grad_enabled(False)
    for i, batch in enumerate(test_loader):
        videos = batch['video'].to(device, non_blocking=True)
        batch_size = videos.shape[0]
        progress.update(data=time.time() - end)

        return_dict = model(videos)

        metrics = {metric: return_dict[metric] for metric in model.metrics}
        progress.update(n=batch_size, **{k: v.item() for k, v in metrics.items()})

        progress.update(time=time.time() - end)
        end = time.time()

        if is_master_process() and i % args.log_interval == 0:
            progress.display(i)

        if i > 100 or (os.environ.get('DEBUG') == '1' and i > 5):
            break

    if is_master_process():
        progress.display(i)

    metrics = {metric: torch.tensor(progress.meters[metric].avg, device=device)
               for metric in model.metrics}
    metrics = all_reduce_avg_dict(metrics)

    if is_master_process():
        wandb.log({f'val/{metric}': val
                   for metric, val in metrics.items()},
                  step=iteration)
    torch.set_grad_enabled(True)
    return metrics['loss'].item()

    
def visualize(iteration, model, loader, device):
    model.eval()
    torch.set_grad_enabled(False)

    batch = next(iter(loader))
    videos = batch['video'][:32].to(device)

    out = model(videos)

    recon_viz = torch.stack([out['original'], out['reconstruction']], dim=1)
    recon_viz = torch.clamp(recon_viz.cpu(), 0, 1)
    recon_viz = make_grid(recon_viz.flatten(end_dim=1), nrow=8) 
    recon_viz = wandb.Image(recon_viz)

    wandb.log({'eval/recon': recon_viz}, step=iteration)
    torch.set_grad_enabled(True)

if __name__ == '__main__':
    main(sys.argv[1])
