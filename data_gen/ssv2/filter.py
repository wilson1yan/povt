import glob
import json
import os.path as osp
import multiprocessing as mp
from tqdm import tqdm
import numpy as np

THRESHOLD = 4

def count_n_tracks(filepath):
    bboxes = np.loadtxt(filepath, delimiter=',')
    tracks = np.unique(bboxes[:, 1])
    assert np.all(tracks >= 0), filepath
    return len(tracks) <= THRESHOLD

pool = mp.Pool(32)
files = glob.glob('data/something-something/20bn-something-something-v2/*.txt')
result = list(tqdm(pool.imap(count_n_tracks, files), total=len(files)))

exclude = [f for f, keep in zip(files, result) if not keep]
exclude = [osp.basename(e).split('.')[0].split('_')[0] for e in exclude]
print(len(exclude), exclude[:10])
json.dump(exclude, open('data/something-something/exclude.json' ,'w'))
