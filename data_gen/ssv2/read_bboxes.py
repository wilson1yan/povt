import json

from tqdm import tqdm
import numpy as np
import os.path as osp
import sys
import multiprocessing as mp
import cv2


MAX_N_OBJECTS = 4
MIN_N_OBJECTS = 1

def read_video_info(video_file):
    vcap = cv2.VideoCapture(video_file)
    width = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = vcap.get(cv2.CAP_PROP_FPS)
    assert fps == 12, fps

    return height, width


def read_video_bbox(vid):
    video_file = osp.join(root, '20bn-something-something-v2', f'{vid}.webm')
    H, W = read_video_info(video_file)
    video_info = json_file[vid]
    T = len(video_info)
    
    bboxes = np.zeros((T, MAX_N_OBJECTS, 4), dtype=np.float32)
    for t in range(T):
        for i, obj in enumerate(video_info[t]['labels']):
            obj = obj['box2d']
            x1, y1 = obj['x1'], obj['y1']
            x2, y2 = obj['x2'], obj['y2']

            bbox = np.array([x1, y1, x2, y2])
            bboxes[t, i] = bbox
    np.save(osp.join(root, '20bn-something-something-v2', f'{vid}_bbox.npy'), bboxes)

if __name__ == '__main__':
    root = sys.argv[1]
    json_files = [osp.join(root, 'annotations', f'bounding_box_smthsmth_part{i}.json') for i in range(1, 5)]
    json_files = [json.load(open(f, 'r')) for f in json_files]
    json_file = dict()
    for jf in json_files:
        json_file.update(jf)
    original_size = len(json_file)

    keep = []
    for k in json_file.keys():
        vid_info = json_file[k]
        n_objs = [t['nr_instances'] for t in json_file[k]]
        if min(n_objs) >= MIN_N_OBJECTS and max(n_objs) <= MAX_N_OBJECTS and len(vid_info) >= 16:
            keep.append(k)
    json_file = {k: json_file[k] for k in keep}
    print(f'Filtered to {len(json_file)}/{original_size}')

    video_ids = list(json_file.keys())
    video_ids.sort()

    pool = mp.Pool(32)
    list(tqdm(pool.imap(read_video_bbox, video_ids), total=len(video_ids)))

    video_ids = set(video_ids)
    train_list = json.load(open(osp.join(root, 'something-something-v2-train.json'), 'r'))
    train_list = [t['id'] for t in train_list if t['id'] in video_ids]

    test_list = json.load(open(osp.join(root, 'something-something-v2-validation.json'), 'r'))
    test_list = [t['id'] for t in test_list if t['id'] in video_ids]
    print(len(train_list), len(test_list))

    json.dump(train_list, open(osp.join(root, 'train_subset.json'), 'w'))
    json.dump(test_list, open(osp.join(root, 'test_subset.json'), 'w'))
