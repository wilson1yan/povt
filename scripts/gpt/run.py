import os
import os.path as osp
import warnings
import argparse
import time
import pickle
import multiprocessing as mp
import torch
import yaml

from train import main as train_main


def worker(rank, size, port,  output_dir):
    cmd = f"MASTER_ADDR=localhost MASTER_PORT={port} WORLD_SIZE={size} NODE_RANK=0 LOCAL_RANK={rank} python scripts/gpt/train.py {output_dir}"
    os.system(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--ckpt', type=str, default=None)
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-p', '--port', type=int, default=23456)
    args = parser.parse_args()

    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        warnings.warn("Warning: This script by default uses all visible GPUs. Limit GPU usage through CUDA_VISIBLE_DEVICES")

    if not osp.isabs(args.output_dir):
        if 'OCL_DATA_DIR' not in os.environ:
            raise Exception('OCL_DATA_DIR environment variable not set')
        root_folder = os.environ['OCL_DATA_DIR']
        args.output_dir = osp.join(root_folder, args.output_dir)

    config = yaml.load(open(args.config, 'r'))
    if os.environ.get('DEBUG') == '1':
        config['eval_interval'] = 10
        config['test_interval'] = 10
        config['log_interval'] = 1
        args.output_dir = osp.join(osp.dirname(args.output_dir), f'DEBUG_{osp.basename(args.output_dir)}')

    args.output_dir = f"{args.output_dir}_{time.time()}"
    print(f"Logging to {args.output_dir}")
    os.makedirs(args.output_dir)

    new_config_path = osp.join(args.output_dir, 'config.yaml')
    yaml.dump(config, open(new_config_path, 'w'))
    args.config = new_config_path
    pickle.dump(args, open(osp.join(args.output_dir, 'args'), 'wb'))

    n = torch.cuda.device_count()
    if n == 1:
        import os
        os.environ['LOCAL_RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        train_main(args.output_dir)
    else:
        procs = [mp.Process(target=worker, args=(i, n, args.port, args.output_dir), daemon=True)
                for i in range(n)]
        [p.start() for p in procs]
        [p.join() for p in procs]
