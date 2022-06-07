import torch
import os
import os.path as osp
import argparse
import pickle
from ocl.models import get_model
from ocl.data import Data
from ocl.utils import view_range, save_video_grid


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--ckpt', type=str, required=True)
args = parser.parse_args()
ckpt = args.ckpt

args = pickle.load(open(osp.join(osp.dirname(osp.dirname(args.ckpt)), 'args'), 'rb'))
args.sequence_length = 16
args.object_invariance = False
data = Data(args)
loader = data.test_dataloader()
args.n_obj = loader.dataset.n_obj

device = torch.device('cuda')
ckpt = torch.load(ckpt, map_location=device)
model = get_model(args, device)
model.load_state_dict(ckpt['model'])
model.eval()

torch.set_grad_enabled(False)

batch = next(iter(loader))
video = batch['video'][:32].to(device)
B, T = video.shape[:2]

video = video.flatten(end_dim=1)
encodings = model.encode(video)[1]
recon = model.decode(encodings)
recon = view_range(recon, 0, 1, (B, T))
recon = torch.clamp(recon, 0, 1)
video = view_range(video, 0, 1, (B, T))

viz = torch.stack((video, recon), dim=1).flatten(end_dim=1)
save_video_grid(viz, fname='recon.gif', nrow=8)
