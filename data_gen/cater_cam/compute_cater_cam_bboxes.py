import argparse
import numpy as np
from tqdm import tqdm
import h5py
import multiprocessing as mp


def mask_to_bboxes(mask):
    # mask: HWK
    H, W = mask.shape[0], mask.shape[1]

    rows = np.tile(np.arange(H).reshape(H, 1).astype(int), (1, W))
    cols = np.tile(np.arange(W).reshape(1, W).astype(int), (H, 1))

    max_int = np.iinfo(int).max
    min_int = np.iinfo(int).min

    areas = mask.sum(axis=(0, 1)) # K
    bboxes = []
    for k in range(len(areas)):
        m = mask[:, :, k]
        if areas[k] < args.min_area:
            bboxes.append([0] * 4)
        else:
            tl_r = np.ma.array(rows, mask=~m).filled(fill_value=max_int).min()
            tl_c = np.ma.array(cols, mask=~m).filled(fill_value=max_int).min()
            br_r = np.ma.array(rows, mask=~m).filled(fill_value=min_int).max() + 1
            br_c = np.ma.array(cols, mask=~m).filled(fill_value=min_int).max() + 1
            
            shift_y = (tl_r + br_r) / 2
            shift_y = (shift_y - H / 2) / (H / 2)
            shift_x = (tl_c + br_c) / 2
            shift_x = (shift_x - W / 2) / (W / 2)
            height = (br_r - tl_r) / H
            width = (br_c - tl_c) / W
            bbox = [shift_x, shift_y, width, height]
            bboxes.append(bbox)
    return np.array(bboxes) 

def process(split):
    assert f'{split}_masks' in f
    masks = f[f'{split}_masks']
    
    all_bboxes = []
    for i in tqdm(list(range(0, masks.shape[0], 32))):
        m = masks[i:i + 32, :, :, 1:] > 0.05
        result = pool.map(mask_to_bboxes, m)
        bboxes = np.stack(result, axis=0)
        all_bboxes.append(bboxes)
    all_bboxes = np.concatenate(all_bboxes, axis=0).astype(np.float32)

    if f'{split}_bboxes' in f:
        del f[f'{split}_bboxes']
    f.create_dataset(f'{split}_bboxes', data=all_bboxes)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_file', type=str, required=True)
    parser.add_argument('-a', '--min_area', type=int, default=5)
    args = parser.parse_args()

    f = h5py.File(args.data_file, 'a')
    pool = mp.Pool(32)

    process('train')
    process('test')
