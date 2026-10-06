# -*- coding: utf-8 -*-
"""CNN 입력용 전처리 캐시: [NLM 잡음 제거 이미지, 결정립계 능선 맵]"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, time
from multiprocessing import Pool
from skimage import filters
from features_v3 import denoise
from features import load
def work(path):
    import warnings; warnings.filterwarnings('ignore')
    img = load(path); flat, den, sig = denoise(img)
    ridge = filters.sato(den, sigmas=[1.0, 1.5, 2.0], black_ridges=True)
    return np.stack([den, ridge]).astype(np.float32)
if __name__ == '__main__':
    out = {}
    for d, csv in [('train', '../data/train.csv'), ('test', '../data/sample_submission.csv')]:
        t = time.time(); ids = pd.read_csv(csv).ID.tolist()
        with Pool(2) as p: out[d] = np.stack(p.map(work, [f'../data/{d}/{i}.png' for i in ids], chunksize=10))
        print(d, out[d].shape, '%.0fs' % (time.time() - t), flush=True)
    np.savez('../feats/den_cache.npz', tr=out['train'], te=out['test'])
