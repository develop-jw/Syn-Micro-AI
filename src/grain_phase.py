# -*- coding: utf-8 -*-
"""결정립 단위 상 분류 (v4 후보)

1) NLM 잡음 제거 + 능선 경계 골격 -> 결정립 영역 라벨
2) 결정립별 평균 밝기, 면적, 형상
3) 결정립 밝기 분포를 2-성분 가우시안 혼합으로 나눠 기지/어두운 상 분류
4) 상 면적 비율, 상 결정립 수/크기, 기지와 상의 밝기 차이 등
"""
import numpy as np
from scipy import ndimage as ndi
from skimage import measure, morphology
from sklearn.mixture import GaussianMixture
from features_v3 import denoise, boundary_mask
from features import load


def grain_table(den, sk):
    lab = measure.label(~morphology.dilation(sk, morphology.disk(1)), connectivity=1)
    props = measure.regionprops(lab, intensity_image=den)
    rows = []
    for p in props:
        if p.area < 15:
            continue
        rows.append((p.area, p.intensity_mean, p.axis_major_length / (p.axis_minor_length + 1e-3),
                     4 * np.pi * p.area / (p.perimeter ** 2 + 1e-6)))
    return np.array(rows, float) if rows else np.zeros((0, 4)), lab


def extract_gp(path):
    img = load(path)
    flat, den, sig = denoise(img)
    ridge, out = boundary_mask(den, sig)
    m, sk = out[1.0]
    T, lab = grain_table(den, sk)
    f = {}
    if len(T) < 8:
        return {k: np.nan for k in ['gp_n', 'gp_phase_frac', 'gp_phase_nfrac', 'gp_gap', 'gp_gap_rel',
                                    'gp_mat_mean', 'gp_phase_mean', 'gp_mean_std', 'gp_phase_area_med',
                                    'gp_mat_area_med', 'gp_phase_asp', 'gp_mat_asp', 'gp_phase_frac_gm',
                                    'gp_dark2_frac']}
    area, mean, asp, circ = T.T
    w = area / area.sum()
    f['gp_n'] = len(T)
    f['gp_mean_std'] = float(np.sqrt(np.cov(mean, aweights=area)))
    # 2-성분 GMM으로 기지/상 분리
    gm = GaussianMixture(2, random_state=0).fit(mean[:, None])
    lo = int(np.argmin(gm.means_.ravel()))
    post = gm.predict_proba(mean[:, None])[:, lo]
    is_ph = post > 0.5
    mat_mean = float(np.average(mean[~is_ph], weights=area[~is_ph])) if (~is_ph).any() else float(mean.max())
    ph_mean = float(np.average(mean[is_ph], weights=area[is_ph])) if is_ph.any() else float(mean.min())
    f['gp_phase_frac_gm'] = float((post * area).sum() / area.sum())   # 확률 가중 상 면적 비
    f['gp_phase_frac'] = float(area[is_ph].sum() / area.sum())
    f['gp_phase_nfrac'] = float(is_ph.mean())
    f['gp_mat_mean'] = mat_mean
    f['gp_phase_mean'] = ph_mean
    f['gp_gap'] = mat_mean - ph_mean
    f['gp_gap_rel'] = f['gp_gap'] / (np.sqrt(gm.covariances_.ravel()).mean() + 1e-6)
    f['gp_phase_area_med'] = float(np.median(area[is_ph])) if is_ph.any() else 0.0
    f['gp_mat_area_med'] = float(np.median(area[~is_ph])) if (~is_ph).any() else 0.0
    f['gp_phase_asp'] = float(np.average(asp[is_ph], weights=area[is_ph])) if is_ph.any() else 1.0
    f['gp_mat_asp'] = float(np.average(asp[~is_ph], weights=area[~is_ph])) if (~is_ph).any() else 1.0
    # 기준 밝기(면적 가중 중앙값)보다 일정 이상 어두운 결정립 면적 비
    order = np.argsort(mean); cw = np.cumsum(w[order])
    med = mean[order][np.searchsorted(cw, 0.5)]
    f['gp_dark2_frac'] = float(w[mean < med - 2 * f['gp_mean_std'] * 0.5].sum())
    return f
