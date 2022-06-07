import cater_with_masks
import tensorflow as tf
import h5py
import glob
import numpy as np
from tqdm import tqdm
import sys
import os.path as osp

def read_md(split):
    files = glob.glob(osp.join(cater_root, f'cater_with_masks_{split}.tfrecords*'))
    files.sort()
    print(f'Found {len(files)} tfrecords')

    total = 0
    for fpath in tqdm(files):
        dataset = cater_with_masks.dataset(fpath)
        iterator = dataset.make_one_shot_iterator()
        data = iterator.get_next()
        sess = tf.Session()

        while True:
            try:
                d = sess.run(data)
                ep = d['image'] # THWC
                total += ep.shape[0]
            except tf.errors.OutOfRangeError:
                break
    return total



def read(f, split):
    total = read_md(split)
    f.create_dataset(f'{split}_data', (total, 64, 64, 3), dtype=np.uint8)
    f.create_dataset(f'{split}_masks', (total, 64, 64, 11), dtype=np.float32)

    files = glob.glob(osp.join(cater_root, f'cater_with_masks_{split}.tfrecords*'))
    files.sort()
    print(f'Found {len(files)} tfrecords')
    
    idx = [0]
    for fpath in tqdm(files):
        dataset = cater_with_masks.dataset(fpath)
        iterator = dataset.make_one_shot_iterator()
        data = iterator.get_next()
        sess = tf.Session()

        while True:
            try:
                d = sess.run(data)
                ep = d['image'] # THWC
                m = d['mask'] # TKHW1
                m = np.squeeze(m, axis=-1) # TKHW
                m = np.transpose(m, (0, 2, 3, 1)) # THWK
                m = m / 255.
                
                f[f'{split}_data'][idx[-1]:idx[-1] + len(ep)] = ep
                f[f'{split}_masks'][idx[-1]:idx[-1] + len(ep)] = m
                idx.append(idx[-1] + len(ep))
            except tf.errors.OutOfRangeError:
                break
    idx = np.array(idx)[:-1]    
    f.create_dataset(f'{split}_idx', data=idx)

cater_root = sys.argv[1]
f = h5py.File('data/cater_cam.hdf5', 'a')
read(f, 'train')
read(f, 'test')

for k in f.keys():
    print(k, f[k].shape, f[k].dtype)

f.close()
