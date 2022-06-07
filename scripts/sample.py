import torch
import numpy as np
import os.path as osp
import argparse
import pickle
from povt.models import get_model
from povt.data import Data
from povt.train_utils import seed_all
from povt.utils import save_video_grid, bbox_norm_to_image

def draw_bbox(obs, tl_r, tl_c, br_r, br_c, color):
    # in-place operation
    # obs: CHW

    # top border
    obs[0, tl_r, tl_c:br_c] = color[0]
    obs[1, tl_r, tl_c:br_c] = color[1]
    obs[2, tl_r, tl_c:br_c] = color[2]

    # left border
    obs[0, tl_r:br_r, tl_c] = color[0]
    obs[1, tl_r:br_r, tl_c] = color[1]
    obs[2, tl_r:br_r, tl_c] = color[2]

    # right border
    obs[0, tl_r:br_r, br_c - 1] = color[0]
    obs[1, tl_r:br_r, br_c - 1] = color[1]
    obs[2, tl_r:br_r, br_c - 1] = color[2]

    # bottom border
    obs[0, br_r - 1, tl_c:br_c] = color[0]
    obs[1, br_r - 1, tl_c:br_c] = color[1]
    obs[2, br_r - 1, tl_c:br_c] = color[2]

def main(args):
    seed_all(args.seed)

    ckpt = args.ckpt
    root_dir = osp.dirname(osp.dirname(args.ckpt))
    args_model = pickle.load(open(osp.join(root_dir, 'args'), 'rb'))
    args_model.batch_size = args.n_batch
    args_model.object_invariance = True
    data = Data(args_model)
    loader = data.test_dataloader()
    args_model.n_obj = loader.dataset.n_obj
    args_model.n_timesteps_gen = 16
    batch = next(iter(loader))

    device = torch.device('cuda')
    ckpt = torch.load(ckpt, map_location=device)
    model = get_model(args_model, device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    torch.set_grad_enabled(False)

    video = batch['video'].repeat_interleave(args.n_repeat, dim=0).to(device)
    bboxes = batch['bboxes'].repeat_interleave(args.n_repeat, dim=0).to(device)
    valid_bboxes = batch['valid_bboxes'].repeat_interleave(args.n_repeat, dim=0).to(device)

    # BTCHW, BTK, BTK4
    samples, (valid, bboxes) = model.sample(video.shape[0], video[:, :1],
                    bboxes[:, :1], valid_bboxes[:, :1],
                    fast_decode=args.fast_decode,
                    return_bbox=True)
    shift, scale = bboxes.chunk(2, dim=-1)
    shift = model.shift_quantizer.dequantize(shift)
    scale = model.scale_quantizer.dequantize(scale)
    bboxes = torch.cat([shift, scale], dim=-1) # BTK4
    bboxes = bbox_norm_to_image(bboxes, args_model.resolution, args_model.resolution)

    B, T, K = bboxes.shape[:-1]
    samples_box = samples.clone()
    for b in range(B):
        for t in range(T):
            for k in range(K):
                if not valid[b, t, k].item():
                    continue
                bbox = bboxes[b, t, k].tolist()
                draw_bbox(samples_box[b, t], *bbox, (1., 0., 0.))
    viz = torch.stack((samples, samples_box), dim=1).flatten(end_dim=1)

    save_path = osp.join(root_dir, 'samples.gif')
    save_video_grid(viz, save_path, nrow=8)
    print('Saved to', save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ckpt', type=str, required=True)
    parser.add_argument('-n', '--n_batch', type=int, default=32)
    parser.add_argument('-r', '--n_repeat', type=int, default=1)
    parser.add_argument('--fast_decode', action='store_true')
    parser.add_argument('-s', '--seed', type=int, default=0)
    args = parser.parse_args()
    main(args)
