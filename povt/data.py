import os.path as osp
import json
import math
import pickle
import warnings

import h5py
import numpy as np

import torch
import torch.utils.data as data
import torch.distributed as dist
import torch.nn.functional as F
from torchvision.datasets.video_utils import VideoClips

from .dist_ops import get_rank
from .utils import resize_bbox


def cycle(dataloader, start_iteration: int = 0):
    iteration = start_iteration

    while True:
        if isinstance(dataloader.sampler, torch.utils.data.DistributedSampler):
            dataloader.sampler.set_epoch(iteration)

        for batch in dataloader:
            yield batch
            iteration += 1


def preprocess(video, resolution, sequence_length=None, normalize=True,
               return_tranforms=False):
    # video: THWC, {0, ..., 255}
    video = video.permute(0, 3, 1, 2).float() # TCHW
    if normalize:
        video /= 255.
    t, c, h, w = video.shape

    # temporal crop
    if sequence_length is not None:
        assert sequence_length <= t
        video = video[:sequence_length]

    # scale shorter side to resolution
    scale = resolution / min(h, w)
    if h < w:
        target_size = (resolution, math.ceil(w * scale))
    else:
        target_size = (math.ceil(h * scale), resolution)
    video = F.interpolate(video, size=target_size, mode='bilinear',
                          align_corners=False)
    
    scale = tuple([t / r for t, r in zip(target_size, (h, w))])

    # center crop
    t, c, h, w = video.shape
    w_start = (w - resolution) // 2
    h_start = (h - resolution) // 2
    video = video[:, :, h_start:h_start + resolution, w_start:w_start + resolution]

    shift = (h_start, w_start)
 
    if return_tranforms:
        return video, shift, scale
    return video

 
class HDF5Dataset(data.Dataset):
    """ Generic dataset for data stored in h5py as uint8 numpy arrays.
    Reads videos in {0, ..., 255} and returns in range [-0.5, 0.5] """
    def __init__(self, args, train=True):
        """
        Args:
            args.data_path: path to the pickled data file with the
                following format:
                {
                    'train_data': [B, H, W, 3] np.uint8,
                    'train_idx': [B], np.int64 (start indexes for each video)
                    'train_bboxes': [B, K, 4] np.float32 
                        ([x, y] shift in [-1, 1], and [x, y] scale in [0, 1])
                        a bbox of all 0's is invalid
                    'test_data': [B', H, W, 3] np.uint8,
                    'test_idx': [B'], np.int64,
                    'test_bboxes': [B', K, 4] np.float32
                }
            args.sequence_length: length of extracted video sequences
        """
        super().__init__()
        self.train = train
        self.frame_skip = args.frame_skip if hasattr(args, 'frame_skip') else 1
        self.sequence_length = args.sequence_length * self.frame_skip
        self.resolution = args.resolution
        self.hparams = args

        # read in data
        self.data_file = args.data_path
        self.data = h5py.File(self.data_file, 'r')
        self.prefix = 'train' if train else 'test'
        self._images = self.data[f'{self.prefix}_data']
        self._idx = self.data[f'{self.prefix}_idx'][:]
        if f'{self.prefix}_bboxes' in self.data:
            self._bboxes = self.data[f'{self.prefix}_bboxes'][:]
            self.n_obj = self._bboxes.shape[1]
        else:
            self.n_obj = None

        self.size = len(self._idx)

    def __getstate__(self):
        state = self.__dict__
        state['data'].close()
        state['data'] = None
        state['_images'] = None

        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.data = h5py.File(self.data_file, 'r')
        self._images = self.data[f'{self.prefix}_data']

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return_dict = dict()

        start = self._idx[idx]
        end = self._idx[idx + 1] if idx < len(self._idx) - 1 else len(self._images)
        if end - start > self.sequence_length:
            start = start + np.random.randint(low=0, high=end - start - self.sequence_length)
        assert start < start + self.sequence_length <= end, f'{start}, {end}'
        video = torch.tensor(self._images[start:start + self.sequence_length]) 
        video = video[::self.frame_skip]

        if f'{self.prefix}_bboxes' in self.data:
            bboxes = torch.FloatTensor(self._bboxes[start:start + self.sequence_length])
            bboxes = bboxes[::self.frame_skip] # TK4 (x_shift, y_shift, x_scale, y_scale)
            valid = ~torch.all(bboxes == 0, dim=-1) # TK

            perm = np.random.permutation(valid.shape[1])
            bboxes, valid = bboxes[:, perm], valid[:, perm]

            return_dict['bboxes'] = bboxes
            return_dict['valid_bboxes'] = valid
            
        return_dict['video'] = preprocess(video, self.resolution)

        return return_dict


class SomethingSomething(data.Dataset):
    def __init__(self, args, train=True):
        super().__init__()
        self.train = train
        self.sequence_length = args.sequence_length
        self.resolution = args.resolution
        self.root = args.data_path
        self.hparams = args

        split = 'train' if train else 'test'
        video_ids = json.load(open(osp.join(self.root, f'{split}_subset.json'), 'r'))
        to_exclude = json.load(open(osp.join(self.root, 'exclude.json'), 'r'))
        to_exclude = set(to_exclude)
        video_ids = list(filter(lambda vid: vid not in to_exclude, video_ids))

        files = [osp.join(self.root, '20bn-something-something-v2', f'{vid}.webm')
                 for vid in video_ids]

        warnings.filterwarnings('ignore')
        cache_file = osp.join(self.root, f'{split}_metadata_{self.sequence_length}.pkl')
        if not osp.exists(cache_file):
            clips = VideoClips(files, self.sequence_length, num_workers=8)
            if get_rank() == 0:
                pickle.dump(clips.metadata, open(cache_file, 'wb'))
        else:
            metadata = pickle.load(open(cache_file, 'rb'))
            clips = VideoClips(files, self.sequence_length,
                               _precomputed_metadata=metadata)
        self._clips = clips
        self.n_obj = 4
    
    def __len__(self):
        return self._clips.num_clips()
    
    def __getitem__(self, idx):
        return_dict = dict()

        resolution = self.resolution
        video = self._clips.get_clip(idx)[0]
        video, shift, scale = preprocess(video, resolution, return_tranforms=True)

        video_idx, clip_idx = self._clips.get_clip_location(idx)

        video_path = self._clips.video_paths[video_idx]
        clip_pts = self._clips.clips[video_idx][clip_idx].tolist()
        video_pts = self._clips.video_pts[video_idx].tolist()
        start_i = video_pts.index(clip_pts[0])
        end_i = video_pts.index(clip_pts[-1]) + 1
        assert end_i - start_i == self.sequence_length, f'{end_i} - {start_i} != {self.sequence_length}'

        video_id = osp.basename(video_path).split('.')[0]
        txt_path = osp.join(osp.dirname(video_path), f'{video_id}_bbox.txt')

        # idx 0: frame (starts at 1)
        # idx 1: track # of a tracked object
        # idx 2-5: x1, y1, x2, y2 of the object
        bboxes = np.loadtxt(txt_path, delimiter=',') # N10
        bboxes[:, 0] -= 1
        bboxes = bboxes[(bboxes[:, 0] >= start_i) & (bboxes[:, 0] < end_i)] # N10
        N = bboxes.shape[0]
        bboxes[:, 0] -= start_i

        unique = np.unique(bboxes[:, 1]).tolist()
        ids = np.random.permutation(self.n_obj).astype(int)[:len(unique)].tolist()
        track2id = {u: i for u, i in zip(unique, ids)}

        out_bboxes = np.zeros((self.sequence_length, self.n_obj, 4), dtype=np.float32)
        valid_bboxes = np.zeros((self.sequence_length, self.n_obj), dtype=np.bool)
        for i in range(N):
            row = bboxes[i]
            t = int(row[0])
            id = track2id[row[1]]
            bbox = row[2:6]
            bbox[2:] += bbox[:2]
            bbox = resize_bbox(bbox, shift, scale)
            bbox = np.clip(bbox, 0, self.resolution)
            x1, y1, x2, y2 = bbox
            if x2 - x1 <= 0 or y2 - y1 <= 0: # invalid bbox (out of frame)
                continue

            shift_y = (y2 + y1) / 2
            shift_y = (shift_y - self.resolution / 2) / (self.resolution / 2)
            shift_x = (x2 + x1) / 2
            shift_x = (shift_x - self.resolution / 2) / (self.resolution / 2)

            scale_y = (y2 - y1) / self.resolution
            scale_x = (x2 - x1) / self.resolution
            
            out_bboxes[t, id] = [shift_x, shift_y, scale_x, scale_y]
            valid_bboxes[t, id] = True
        
        return_dict['video'] = video
        return_dict['valid_bboxes'] = torch.tensor(valid_bboxes)
        return_dict['bboxes'] = torch.tensor(out_bboxes)
        return return_dict


class Data:
    def __init__(self, args):
        super().__init__()
        self.hparams = args

    def _dataset(self, train):
        if 'something-something' in self.hparams.data_path:
            dataset = SomethingSomething(self.hparams, train)
        elif 'hdf5' in self.hparams.data_path:
            dataset = HDF5Dataset(self.hparams, train)
        else:
            raise ValueError(f'Unknown dataset: {self.hparams.data_path}')
        return dataset

    def _dataloader(self, train):
        dataset = self._dataset(train)
        if dist.is_initialized():
            sampler = data.distributed.DistributedSampler(
                dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank()
            )
        else:
            sampler = None
        dataloader = data.DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            sampler=sampler,
            shuffle=sampler is None,
            drop_last=False
        )
        return dataloader

    def train_dataloader(self):
        return self._dataloader(True)

    def val_dataloader(self):
        return self._dataloader(False)

    def test_dataloader(self):
        return self.val_dataloader()
