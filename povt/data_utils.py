from functools import reduce
import numpy as np
import torch
import torch.nn.functional as F

from .utils import view_range


def process_stack_obs(obs, n_stack):
    # obs: BCTHW, actions: BT'(action_dim), masks: None or (BCTHW, ...)
    B, C, T, H, W = obs.shape
    assert n_stack <= T - 1, f"{n_stack} > {T} - 1"

    new_T = T - (n_stack - 1)
    obs = torch.movedim(obs, 1, 2) # BCTHW -> BTCHW
    obs_stack = torch.cat([obs[:, i:i+new_T] for i in range(n_stack)], dim=2) # B(new_T)(n_stack * C)HW
    return obs_stack

def process_stack(x, n_stack):
    # x: BTD
    B, T, D = x.shape
    assert n_stack <= T - 1, f"{n_stack} > {T} - 1"

    new_T = T - (n_stack - 1)    
    x = torch.cat([x[:, i:i+new_T] for i in range(n_stack)], dim=-1) # B(new_T)(n_stack * D)
    return x

    
def segment_obs(obs, mode):
    # obs: BCTHW in [-1, 1]
    obs = obs * 0.5 + 0.5 # [-1, 1] -> [0, 1]
    B, C, T, H, W = obs.shape

    if mode == 'rope_simple':
        is_background = torch.abs(obs - obs.mean(dim=1, keepdim=True)) < 1e-4
        is_background = torch.all(is_background, dim=1) # BTHW
        is_background[:, :, :, :6] = 1
        is_background[:, :, :, 58:] = 1

        BACKGROUND = (0.5608, 0.5608, 0.5059)
        GREEN = [(0.20, 0.49, 0.20), (0., 0.31, 0.)]  # rope
        GREEN = [np.mean(GREEN, axis=0).tolist()]
        BLUE = [(0.18, 0.16, 0.53)] # manipulator
        BLUE = [np.mean(BLUE, axis=0).tolist()]

        colors = torch.FloatTensor([BACKGROUND, *GREEN, *BLUE]).to(obs.device) # 3 x n_colors
        obs_flat = torch.movedim(obs, 1, -1).flatten(end_dim=-2) # BCTHW -> (BTHW)C
        dists = torch.norm(obs_flat.unsqueeze(1) - colors, dim=-1) # (BTHW)3
        labels = torch.argmin(dists, dim=-1).view(B, T, H, W) # BTHW

        rope_masks = [labels == i for i in range(1, 1 + len(GREEN))]
        rope_mask = reduce(torch.logical_or, rope_masks)
        rope_mask = torch.logical_and(rope_mask, ~is_background)
        rope_mask = rope_mask.unsqueeze(1).repeat_interleave(3, dim=1)

        manipulator_masks = [labels == i for i in range(1 + len(GREEN), 1 + len(GREEN) + len(BLUE))]
        manipulator_mask = reduce(torch.logical_or, manipulator_masks)
        manipulator_mask = torch.logical_and(manipulator_mask, ~is_background)
        manipulator_mask = manipulator_mask.unsqueeze(1).repeat_interleave(3, dim=1)

        # both masks: BCTHW
        return rope_mask, manipulator_mask
    elif mode == 'pusher_only':
        is_background = torch.abs(obs - obs.mean(dim=1, keepdim=True)) < 1e-2
        is_background = torch.all(is_background, dim=1) # BTHW
        is_background[:, :, :, :6] = 1
        is_background[:, :, :, 58:] = 1

        BACKGROUND = (0.5608, 0.5608, 0.5059)
        BLUE = (0.15686273574829102, 0.09803920984268188, 0.4117647111415863)

        colors = torch.FloatTensor([BACKGROUND, BLUE]).to(obs.device)
        obs_flat = torch.movedim(obs, 1, -1).flatten(end_dim=-2) # BCTHW -> (BTHW)C
        dists = torch.norm(obs_flat.unsqueeze(1) - colors, dim=-1) # (BTHW)2
        labels = torch.argmin(dists, dim=-1).view(B, T, H, W) # BTHW

        mask = torch.logical_and(labels == 1, ~is_background)
        mask = mask.unsqueeze(1).repeat_interleave(3, dim=1)
        return (mask,)
    else:
        raise ValueError(f"Invalid mode: {mode}")


def compute_bounding_boxes(masks, expansion=1.2):
    # masks: a tuple of batched masks (BCTHW, ...)
    # assumes masks across channels are duplicated
    masks = [mask[:, 0] for mask in masks] # BTHW
    H, W = masks[0].shape[2], masks[0].shape[3]

    device = masks[0].device
    rows = torch.arange(H, device=device, dtype=torch.long).view(1, 1, H, 1)
    cols = torch.arange(W, device=device, dtype=torch.long).view(1, 1, 1, W)

    max_int = torch.iinfo(torch.long).max
    min_int = torch.iinfo(torch.long).min

    # (BT, ...)
    bboxes_tl_r = [rows.masked_fill(~mask, max_int).flatten(start_dim=2).min(dim=2)[0]
                   for mask in masks]
    bboxes_tl_c = [cols.masked_fill(~mask, max_int).flatten(start_dim=2).min(dim=2)[0]
                   for mask in masks]
    bboxes_br_r = [rows.masked_fill(~mask, min_int).flatten(start_dim=2).max(dim=2)[0] + 1
                   for mask in masks]
    bboxes_br_c = [cols.masked_fill(~mask, min_int).flatten(start_dim=2).max(dim=2)[0] + 1
                   for mask in masks]

    # (BT4, ...)
    bboxes = [torch.stack((tl_r, tl_c, br_r, br_c), dim=-1)
              for tl_r, tl_c, br_r, br_c 
              in zip(bboxes_tl_r, bboxes_tl_c, bboxes_br_r, bboxes_br_c)]

    # Just expands the the bounding box by some factor
    if expansion > 1:
        # BT
        heights = [bbox[:, :, 2] - bbox[:, :, 0] for bbox in bboxes]
        widths = [bbox[:, :, 3] - bbox[:, :, 1] for bbox in bboxes]

        # BT
        pix_h = [(height * (expansion - 1.) / 2.).long().clamp(min=1) 
                 for height in heights]
        pix_w = [(width * (expansion - 1.) / 2.).long().clamp(min=1)
                 for width in widths]
        
        n_bboxes = len(bboxes)
        for i in range(n_bboxes):
            bboxes[i][:, :, 0] -= pix_h[i]
            bboxes[i][:, :, 1] -= pix_w[i]
            bboxes[i][:, :, 2] += pix_h[i]
            bboxes[i][:, :, 3] += pix_w[i]

            bboxes[i][:, :, [0, 2]] = bboxes[i][:, :, [0, 2]].clamp(0, H)
            bboxes[i][:, :, [1, 3]] = bboxes[i][:, :, [1, 3]].clamp(0, W)

    # Process bbox coords and convert to spatial transformer format
    bboxes = torch.stack(bboxes, dim=2) # BTM4
    
    # shift lies in [-1, 1]
    shift_y = (bboxes[:, :, :, 0] + bboxes[:, :, :, 2]) / 2
    shift_y = (shift_y - H / 2) / (H / 2)
    shift_x = (bboxes[:, :, :, 1] + bboxes[:, :, :, 3]) / 2
    shift_x = (shift_x - W / 2) / (W / 2)
    shift = torch.stack((shift_x, shift_y), dim=-1) # BTM2

    # scale lies in [0, 1]
    height = (bboxes[:, :, :, 2] - bboxes[:, :, :, 0]) / H
    width = (bboxes[:, :, :, 3] - bboxes[:, :, :, 1]) / W
    scale = torch.stack((width, height), dim=-1) # BTM2
              
    return shift, scale

def bbox_image_to_norm(bboxes, H, W):
    # bbox: BTM4
    # shift lies in [-1, 1]
    shift_y = (bboxes[:, :, :, 0] + bboxes[:, :, :, 2]) / 2
    shift_y = (shift_y - H / 2) / (H / 2)
    shift_x = (bboxes[:, :, :, 1] + bboxes[:, :, :, 3]) / 2
    shift_x = (shift_x - W / 2) / (W / 2)
    shift = torch.stack((shift_x, shift_y), dim=-1) # BTM2

    # scale lies in [0, 1]
    height = (bboxes[:, :, :, 2] - bboxes[:, :, :, 0]) / H
    width = (bboxes[:, :, :, 3] - bboxes[:, :, :, 1]) / W
    scale = torch.stack((width, height), dim=-1) # BTM2

    return torch.cat((shift, scale), dim=-1) # BTM4


def bbox_norm_to_image(bbox, H, W):
    # bbox: BTM4

    bbox_shift, bbox_scale = bbox.chunk(2, dim=-1) # BTM2 (x2)

    width = bbox_scale[:, :, :, 0] * W
    height = bbox_scale[:, :, :, 1] * H

    center_r = bbox_shift[:, :, :, 1] * (H / 2) + (H / 2)
    center_c = bbox_shift[:, :, :, 0] * (W / 2) + (W / 2)
    tl_r = torch.round(center_r - height / 2)
    tl_c = torch.round(center_c - width / 2)
    br_r = torch.round(center_r + height / 2)
    br_c = torch.round(center_c + width / 2)

    tl_r = torch.clamp(tl_r, 0, H - 1).long() # BTM
    tl_c = torch.clamp(tl_c, 0, W - 1).long()
    br_r = torch.clamp(br_r, 0, H).long()
    br_c = torch.clamp(br_c, 0, W).long()

    return torch.stack((tl_r, tl_c, br_r, br_c), dim=-1) # BTM4

# https://github.com/zhixuan-lin/SPACE
def spatial_transform(obs, shift, scale, out_shape, inverse=False):
    # obs:  BCHW
    # shift, scale: B2
    theta = torch.zeros(2, 3).repeat(obs.shape[0], 1, 1).to(obs.device)
    theta[:, 0, 0] = scale[:, 0] if not inverse else 1 / (scale[:, 0] + 1e-9)
    theta[:, 1, 1] = scale[:, 0] if not inverse else 1 / (scale[:, 1] + 1e-9)
    theta[:, :, -1] = shift if not inverse else -shift / (scale + 1e-9)

    grid = F.affine_grid(theta, out_shape)
    return F.grid_sample(obs, grid)

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


def bboxes_from_attn(attn, seg_method, threshold=None, return_masks=False):
    # attn: BSHW
    if seg_method == 'threshold': 
        masks = attn >= threshold # BSHW
        is_zero = masks.flatten(start_dim=2).any(dim=2) == 0 # BS
        while is_zero.float().sum() > 0:
            threshold *= 0.5
            is_zero = is_zero.unsqueeze(-1).unsqueeze(-1) # BS11
            update = torch.logical_and(attn >= threshold, is_zero) # BSHW
            masks = torch.logical_or(masks, update) # BSHW
            is_zero = masks.flatten(start_dim=2).any(dim=2) == 0 # BS
    elif seg_method == 'largest_gap':
        vals = attn.flatten(start_dim=2) # BS(HW)
        vals = torch.sort(vals, dim=-1)[0] # BS(HW)
        diff = vals[:, :, 1:] - vals[:, :, :-1] # BS(HW-1)
        max_idx = torch.argmax(diff, dim=-1) + 1 # BS
        thresholds = torch.gather(vals, -1, max_idx.unsqueeze(-1)) # BS1
        thresholds = thresholds.unsqueeze(-1) # BS11
        masks = attn >= thresholds # BSHW
    else:
        raise ValueError(f"Invalid seg_method: {seg_method}")

    masks = masks.unsqueeze(1).repeat_interleave(3, dim=1) # B3SHW
    bbox_shift, bbox_scale = compute_bounding_boxes((masks,)) # BS12 (x2)
    bbox_shift = bbox_shift.flatten(end_dim=2)
    bbox_scale = bbox_scale.flatten(end_dim=2)
    if return_masks:
        return bbox_shift, bbox_scale, masks
    return bbox_shift, bbox_scale

# RLBench Masks
def compute_rlbench_mask(mask, data_file):
    if 'lamp_on' in data_file:
        mask_arm = mask == 35
        mask_bulb = mask ==  84
        mask_button = (mask == 85) | (mask == 89) | (mask == 90)
        
        return (mask_arm, mask_bulb, mask_button)
    else:
        raise ValueError(f"Unsupported dataset', data_file")
