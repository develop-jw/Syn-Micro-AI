# -*- coding: utf-8 -*-
"""특징 hist: 기지 밝기 기준 상대 밝기 히스토그램 (밝기 분포 모양 전체)

- 조명 보정 -> NLM 잡음 제거 -> 결정립계 메우기(closing) -> 기지 밝기(최빈값) 기준 차이
- 상대 밝기 구간별 면적 비율 + 누적 비율(CDF)
- 원본(잡음 제거 전) 버전도 함께: 잡음 제거 효과 비교용
"""
import numpy as np
from scipy import ndimage as ndi
from skimage import morphology, restoration
from features import load, noise_sigma

EDGES = np.arange(-120, 44, 4)   # 상대 밝기 구간 (기지 = 0)


def mode_of(a):
    h, e = np.histogram(a, bins=np.arange(0, 257, 1))
    h = ndi.gaussian_filter1d(h.astype(float), 2)
    return float(e[np.argmax(h)] + 0.5)


def extract_hist(path):
    img = load(path)
    bg = ndi.gaussian_filter(img, 40)
    flat = img - bg + bg.mean()
    sig = noise_sigma(flat)
    den = restoration.denoise_nl_means(flat, h=0.8 * sig, sigma=sig, patch_size=5, patch_distance=6, fast_mode=True)
    filled = morphology.closing(den, morphology.disk(2))
    f = {}
    for name, a in (('hn', filled), ('hd', den)):
        m0 = mode_of(a)
        rel = a - m0
        h, _ = np.histogram(rel, bins=EDGES)
        h = h / rel.size
        cdf = np.cumsum(h)
        for k, (lo, v, c) in enumerate(zip(EDGES[:-1], h, cdf)):
            f[f'{name}_b{lo}'] = float(v)
            f[f'{name}_c{lo}'] = float(c)
        f[f'{name}_mode'] = m0
        f[f'{name}_below_tot'] = float((rel < -6).mean())
    f['hn_sigma'] = sig
    return f
