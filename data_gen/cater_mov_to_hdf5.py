import argparse
from tqdm import tqdm
import os.path as osp
import glob
import h5py
import numpy as np
import multiprocessing as mp
from torchvision.io import read_video

THRESHOLD = 0.8
MAX_N_OBJECTS = 10


def read(video_file):
    try:
        v_id = osp.basename(video_file)[:-4]
        color_file = osp.join(osp.dirname(video_file), f'{v_id}_seg_colors.pkl')
        seg_files = glob.glob(osp.join(osp.dirname(video_file), f'{v_id}_seg_*.avi'))
        segs = np.stack([read_video(sf)[0].numpy() for sf in seg_files], axis=1)
        segs = np.all(segs > THRESHOLD * 255, axis=-1) # TKHW

        video = read_video(video_file)[0].numpy() # THWC
        assert video.shape[0] == segs.shape[0]
        T = video.shape[0]
        K = segs.shape[1]

        bboxes = np.zeros((T, MAX_N_OBJECTS, 4), dtype=np.float32)
        for t in range(T):
            for k in range(K):
                seg = segs[t, k]
                idxs = np.array(np.where(seg), dtype=np.float32)
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
        return video, bboxes, True
    except:
        return None, None, False

def process_split(episode_paths, split):
    pool = mp.Pool(32)
    result = list(tqdm(pool.imap(read, episode_paths), total=len(episode_paths)))
    imgs, bboxes, success = list(zip(*result))
    print(f'Success {sum(success)} / {len(success)}')

    imgs = [img for s, img in zip(success, imgs) if s]
    bboxes = [bbox for s, bbox in zip(success, bboxes) if s]

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str, required=True)
    args = parser.parse_args()

    if args.data_path[-1] == '/':
        args.data_path = args.data_path[:-1]

    video_files = glob.glob(osp.join(args.data_path, 'images', '*.avi'))
    video_files = [v for v in video_files if 'seg' not in v]
    video_files.sort()
    print(f'Found {len(video_files)} videos')

    f = h5py.File(args.data_path + '.hdf5', 'a')

    t = min(500, int(len(video_files) * 0.1))
    train_paths = video_files[:-t]
    process_split(train_paths, 'train')

    test_paths = video_files[-t:]
    process_split(test_paths, 'test')
