import os.path as osp
import h5py
import argparse
import glob
import json
import numpy as np
from PIL import Image
import multiprocessing as mp
from tqdm import tqdm

MAX_OBJECTS = 10

def read(path):
    img_files = glob.glob(osp.join(path, 'rgba_*.png'))
    img_files.sort()
    seg_files = glob.glob(osp.join(path, 'segmentation_*.png'))
    seg_files.sort()

    md = json.load(open(osp.join(path, 'metadata.json')))
    K = len(md['instances'])
    T = len(img_files)
    assert len(img_files) == len(seg_files)
    
    bboxes = np.zeros((T, MAX_OBJECTS, 4), dtype=np.float32)
    for t, seg_f in enumerate(seg_files):
        seg = Image.open(seg_f)
        seg = np.array(seg) # HW
        for k in range(K):
            idxs = np.array(np.where(seg == k+1), dtype=np.float32)
            if idxs.size > 0:
                y_min = float(idxs[0].min() / seg.shape[0])
                x_min = float(idxs[1].min() / seg.shape[1])
                y_max = float((idxs[0].max() + 1) / seg.shape[0])
                x_max = float((idxs[1].max() + 1) / seg.shape[1])

                shift_x = 2 * (x_min + x_max) / 2 - 1
                shift_y = 2 * (y_min + y_max) / 2 - 1
                width = x_max - x_min
                height = y_max - y_min
                bboxes[t, k] = np.array([shift_x, shift_y, width, height])

    imgs = []
    for img_f in img_files:
        img = Image.open(img_f)
        img = np.array(img)[:, :, :-1] # no alpha channel
        imgs.append(img)
    imgs = np.stack(imgs) # THWC

    return imgs, bboxes

def process_split(episode_paths, split):
    pool = mp.Pool(32)
    result = list(tqdm(pool.imap(read, episode_paths), total=len(episode_paths)))
    imgs, bboxes = list(zip(*result))
    idxs = [0]
    for i in range(len(imgs)):
        idxs.append(idxs[-1] + len(imgs[i]))
    idxs = np.array(idxs[:-1])
    assert len(idxs) == len(imgs) == len(bboxes)

    imgs = np.concatenate(imgs, axis=0)
    bboxes = np.concatenate(bboxes, axis=0)
    assert len(imgs) == len(bboxes)
    
    f.create_dataset(f'{split}_data', data=imgs)
    f.create_dataset(f'{split}_bboxes', data=bboxes.astype(np.float32))
    f.create_dataset(f'{split}_idx', data=idxs)

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_path', type=str, required=True)
args = parser.parse_args()

if args.data_path[-1] == '/':
    args.data_path = args.data_path[:-1]

episode_paths = glob.glob(osp.join(args.data_path, 'episode_*'))
print(f'Found {len(episode_paths)} episodes')

f = h5py.File(args.data_path + '.hdf5', 'a')

t = min(500, int(len(episode_paths) * 0.1))
train_paths = episode_paths[:-t]
process_split(train_paths, 'train')

test_paths = episode_paths[-t:]
process_split(test_paths, 'test')
