# -*- coding: utf-8 -*-
"""특징 v2: v1의 약점(흐린 이미지, 정렬 측정) 보완

- ph2_ : 어두운 상 덩어리의 개수/크기/신장 (흐려도 대비가 커서 안정적)
- pr2_ : 기공을 '상보다 더 어두운 둥근 덩어리'로 재정의
- al_  : 정렬도 - 다중 스케일 structure tensor coherence (전역)
- lv_  : 밝기 수준 분리 (기지/상/기공 각 밝기와 대비)
"""
import numpy as np
from scipy import ndimage as ndi
from skimage import measure, morphology, filters
from features import load, preprocess


def blob_stats(mask, prefix, min_area=10):
    f = {}
    lab = measure.label(mask)
    props = [p for p in measure.regionprops(lab) if p.area >= min_area]
    n = len(props)
    f[f'{prefix}_n'] = n
    f[f'{prefix}_frac'] = float(sum(p.area for p in props) / mask.size)
    if n:
        a = np.array([p.area for p in props], float)
        maj = np.array([p.axis_major_length for p in props]); mnr = np.array([p.axis_minor_length for p in props]) + 1e-3
        asp = maj / mnr
        w = a / a.sum()
        f[f'{prefix}_area_mean'] = float(a.mean())
        f[f'{prefix}_area_med'] = float(np.median(a))
        f[f'{prefix}_eqd_wmean'] = float((np.sqrt(4 * a / np.pi) * w).sum())
        f[f'{prefix}_asp_mean'] = float(asp.mean())
        f[f'{prefix}_asp_wmean'] = float((asp * w).sum())
        f[f'{prefix}_minor_wmean'] = float((mnr * w).sum())
        f[f'{prefix}_major_wmean'] = float((maj * w).sum())
        # 덩어리 방향 정렬 정도 (2배각 평균 벡터 길이)
        th = np.array([p.orientation for p in props])
        ww = w * (asp - 1)
        if ww.sum() > 0:
            f[f'{prefix}_ori_R'] = float(np.hypot((ww * np.cos(2 * th)).sum(), (ww * np.sin(2 * th)).sum()) / ww.sum())
        else:
            f[f'{prefix}_ori_R'] = 0.0
        circ = np.array([4 * np.pi * p.area / (p.perimeter ** 2 + 1e-6) for p in props])
        f[f'{prefix}_circ_mean'] = float(circ.mean())
    else:
        for k in ['area_mean', 'area_med', 'eqd_wmean', 'asp_mean', 'asp_wmean', 'minor_wmean',
                  'major_wmean', 'ori_R', 'circ_mean']:
            f[f'{prefix}_{k}'] = 0.0
    return f


def extract_v2(path):
    img = load(path)
    flat, den, bg, sig, s = preprocess(img)
    f = {}
    # 결정립계 메우기: 얇은 어두운 선 제거
    filled = morphology.closing(den, morphology.disk(3))
    filled = ndi.median_filter(filled, 5)
    hist, edges = np.histogram(filled, bins=128)
    mode = edges[np.argmax(ndi.gaussian_filter1d(hist.astype(float), 2))]  # 기지(가장 흔한 밝기)
    mat = filled[np.abs(filled - mode) < 10]
    f['lv_matrix'] = float(mode)
    f['lv_matrix_spread'] = float(mat.std()) if mat.size else 0.0
    # 상: 기지보다 d 이상 어두운 영역 (여러 기준)
    for d in (10, 18, 28):
        m = filled < mode - d
        m = morphology.opening(m, morphology.disk(1))
        f.update(blob_stats(m, f'ph2_d{d}'))
        f[f'lv_ph_d{d}_contrast'] = float(mode - filled[m].mean()) if m.any() else 0.0
    # 기공: 상보다 더 어두운 핵심부를 가진 둥근 덩어리
    sm = ndi.gaussian_filter(den, 1.0)
    for d in (45, 65):
        m = sm < mode - d
        m = morphology.remove_small_objects(m, max_size=6)
        f.update(blob_stats(m, f'pr2_d{d}', min_area=6))
    # 정렬도: 다중 스케일 전역 coherence (상 마스크 기반 + 이미지 기반)
    for sc in (2.0, 6.0, 12.0):
        gx = ndi.gaussian_filter(filled, sc, order=(0, 1)); gy = ndi.gaussian_filter(filled, sc, order=(1, 0))
        Jxx, Jyy, Jxy = (gx * gx).mean(), (gy * gy).mean(), (gx * gy).mean()
        f[f'al_coh_s{sc}'] = float(np.hypot(Jxx - Jyy, 2 * Jxy) / (Jxx + Jyy + 1e-9))
    # 정렬도: 그래디언트 방향 히스토그램의 2배각 집중도
    gx = ndi.sobel(den, 1); gy = ndi.sobel(den, 0); mag = np.hypot(gx, gy)
    sel = mag > np.percentile(mag, 80)
    th = np.arctan2(gy[sel], gx[sel])
    f['al_grad_R'] = float(np.hypot(np.cos(2 * th).mean(), np.sin(2 * th).mean()))
    # 품질 보정용
    f['q2_noise'] = sig
    f['q2_smooth'] = s
    f['q2_edge_contrast'] = float(np.percentile(mag, 99) / (sig + 1))
    return f
