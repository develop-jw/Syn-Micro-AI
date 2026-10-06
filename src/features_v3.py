# -*- coding: utf-8 -*-
"""특징 v3: 결정립 크기 정밀 측정

- 잡음 제거: Non-Local Means (경계 보존형)
- 경계 검출: Hessian 기반 어두운 능선(ridge) 필터, 임계값은 이미지별 잡음 수준에 맞춤
- 결정립 크기: 선형 절편법 (ASTM E112) - 4방향 직선이 경계를 지나는 횟수
- 신장도: 방향별 절편 길이의 최대/최소 비
- 결정립 수: 경계 골격으로 닫힌 영역 개수
"""
import numpy as np
from scipy import ndimage as ndi
from skimage import restoration, filters, morphology, measure
from features import load, noise_sigma


# ---- 전처리 ----
def denoise(img):
    bg = ndi.gaussian_filter(img, 40)
    flat = img - bg + bg.mean()
    sig = noise_sigma(flat)
    den = restoration.denoise_nl_means(flat, h=0.8 * sig, sigma=sig, patch_size=5,
                                       patch_distance=6, fast_mode=True)
    return flat, den, sig


# ---- 경계 맵 ----
def boundary_mask(den, sig):
    # 어두운 얇은 선 강조 (sigma 1~2 다중 스케일)
    ridge = filters.sato(den, sigmas=[1.0, 1.5, 2.0], black_ridges=True)
    # 임계값: 능선 반응 분포를 둘(배경/경계)로 나누는 Otsu + 보정 배수
    t0 = filters.threshold_otsu(ridge)
    out = {}
    for k in (1.0, 1.5):
        m = ridge > t0 * k
        m = morphology.remove_small_objects(m, max_size=15)
        sk = morphology.skeletonize(m)
        out[k] = (m, sk)
    return ridge, out


# ---- 선형 절편법 ----
def intercepts(sk, step=4):
    """4방향 직선에서 평균 절편 길이(경계 사이 평균 거리)"""
    res = {}
    H, W = sk.shape
    # 0도(가로), 90도(세로)
    for name, arr in (('h', sk), ('v', sk.T)):
        rows = arr[::step]
        hits = np.sum(np.diff(rows.astype(np.int8), axis=1) == 1)  # 경계 진입 횟수
        res[name] = rows.shape[0] * rows.shape[1] / (hits + 1)
    # 45도, 135도: 대각선
    for name, arr in (('d1', sk), ('d2', np.fliplr(sk))):
        tot_len = 0; hits = 0
        for off in range(-H + 20, W - 20, step):
            d = np.diagonal(arr, offset=off).astype(np.int8)
            tot_len += len(d) * np.sqrt(2); hits += np.sum(np.diff(d) == 1)
        res[name] = tot_len / (hits + 1)
    return res


def extract_v3(path):
    img = load(path)
    flat, den, sig = denoise(img)
    f = {'g3_sigma': float(sig)}
    ridge, out = boundary_mask(den, sig)
    for k, (m, sk) in out.items():
        L = intercepts(sk)
        vals = np.array(list(L.values()))
        f[f'g3_k{k}_L_gmean'] = float(np.exp(np.log(vals).mean()))   # 평균 결정립 크기
        f[f'g3_k{k}_L_max'] = float(vals.max())
        f[f'g3_k{k}_L_min'] = float(vals.min())
        f[f'g3_k{k}_elong'] = float(vals.max() / vals.min())          # 신장도
        f[f'g3_k{k}_skel_density'] = float(sk.mean())                  # 경계 길이 밀도
        f[f'g3_k{k}_inv_sqrt_L'] = float(1 / np.sqrt(f[f'g3_k{k}_L_gmean']))
        # 닫힌 영역(결정립) 수
        lab = measure.label(~morphology.dilation(sk, morphology.disk(1)), connectivity=1)
        areas = np.bincount(lab.ravel())[1:]
        areas = areas[areas > 20]
        f[f'g3_k{k}_n_grains'] = int(len(areas))
        f[f'g3_k{k}_area_med'] = float(np.median(areas)) if len(areas) else 0.0
    f['g3_ridge_mean'] = float(ridge.mean())
    f['g3_ridge_p95'] = float(np.percentile(ridge, 95))
    f['g3_ridge_otsu'] = float(filters.threshold_otsu(ridge))
    # ---- 흐린 이미지용 보조 측정: 밝기 계단(step edge) 밀도 ----
    from skimage import feature as skf
    for sg in (1.5, 3.0):
        e = skf.canny(den, sigma=sg, low_threshold=None, high_threshold=None,
                      use_quantiles=False)
        f[f'g3_canny_s{sg}'] = float(e.mean())
        L = intercepts(e)
        f[f'g3_canny_s{sg}_L'] = float(np.exp(np.log(np.array(list(L.values()))).mean()))
    # ---- 밝기 패턴 상관 길이 (상 덩어리 제외 위해 상위 밝기만 클리핑) ----
    a = np.clip(den, np.percentile(den, 20), None)
    a = a - a.mean()
    P = np.abs(np.fft.fft2(a)) ** 2
    ac = np.fft.fftshift(np.real(np.fft.ifft2(P))); ac /= ac.max()
    c = 128
    prof = [ac[c, c + r] + ac[c, c - r] + ac[c + r, c] + ac[c - r, c] for r in range(1, 60)]
    prof = np.array(prof) / 4
    for lvl in (0.5, 0.3):
        idx = np.where(prof < lvl)[0]
        f[f'g3_aclen{lvl}'] = float(idx[0] + 1) if len(idx) else 60.0
    # ---- 측정 신뢰도 지표: 경계선 대비 / 잡음 ----
    f['g3_line_snr'] = float(np.percentile(ridge, 97) / (np.median(ridge) + 1e-6))
    return f, den, out
