import argparse
from tqdm import tqdm
import torch
from povt.data import Data
from povt.utils import save_video_grid, view_range, extract_glimpses, \
    bbox_norm_to_image, quantize_bboxes


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


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_path', type=str, required=True)
parser.add_argument('-s', '--sequence_length', type=int, default=16)
parser.add_argument('-r', '--resolution', type=int, default=64)
parser.add_argument('-b', '--batch_size', type=int, default=64)
args = parser.parse_args()
args.num_workers = 1
args.object_invariance = True

data = Data(args)
data.test_dataloader(), data.train_dataloader()
loader = data.train_dataloader()
batch = next(iter(loader))
video = batch['video'] # BTCHW
save_video_grid(video, 'videos.gif', nrow=10)

bboxes = batch['bboxes'] # BTK4
bboxes, _ = quantize_bboxes(bboxes)
valid_bboxes = batch['valid_bboxes'] # BTK
B, T = bboxes.shape[:2]
glimpses = extract_glimpses(video.flatten(end_dim=1), 
                            bboxes.flatten(end_dim=1), 
                            valid_bboxes.flatten(end_dim=1), 
                            args.resolution) # (BT)KCHW
glimpses = view_range(glimpses, 0, 1, (B, T)) # BTKCHW
glimpses = glimpses.movedim(2, 1) # BKTCHW

K = glimpses.shape[1]
bboxes = bbox_norm_to_image(bboxes, args.resolution, args.resolution) # BTK4
video_box = video.clone() # BTCHW
for b in range(B):
    for t in range(T):
        for k in range(K):
            if not valid_bboxes[b, t, k].item():
                continue
            bbox = bboxes[b, t, k].tolist()
            draw_bbox(video_box[b, t], *bbox, (1., 0., 0.))

viz = torch.cat((video.unsqueeze(1), video_box.unsqueeze(1), glimpses), dim=1) # B(2+K2)TCHW
save_video_grid(viz.flatten(end_dim=1), 'videos_with_bboxes.gif', viz.shape[1])
