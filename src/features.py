# -*- coding: utf-8 -*-
"""미세구조 이미지 -> 수치 특징 추출

특징 그룹
- q_  : 영상 품질 (잡음, 선명도, 조명 불균일)
- h_  : 밝기 히스토그램
- pore_ : 기공 후보 (작고 둥글고 매우 어두운 덩어리)
- ph_ : 어두운 상 (주변보다 어두운 결정립 영역) 비율
- gb_ : 결정립계 (얇은 어두운 선) 밀도 -> 결정립 크기의 대용
- gs_ : 결정립 크기 (watershed 분할 기반)
- or_ : 정렬/신장 (structure tensor, 자기상관 이방성)
- psd_ : 방사형 파워 스펙트럼 (길이 척도 분포)
"""
import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation, feature


# ---- 기본 전처리 ----
def load(path):
    from PIL import Image
    return np.asarray(Image.open(path), dtype=np.float32)


def noise_sigma(img):
    """Immerkaer 방식 잡음 표준편차 추정"""
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    r = ndi.convolve(img, k)
    h, w = img.shape
    return float(np.sum(np.abs(r[1:-1, 1:-1])) * np.sqrt(0.5 * np.pi) / (6 * (w - 2) * (h - 2)))


def preprocess(img):
    """조명 보정 + 잡음 제거"""
    bg = ndi.gaussian_filter(img, 40)
    flat = img - bg + bg.mean()
    sig = noise_sigma(img)
    # 잡음이 클수록 더 강하게 평활화
    s = float(np.clip(0.6 + sig / 12.0, 0.8, 3.0))
    den = ndi.gaussian_filter(flat, s)
    return flat, den, bg, sig, s


# ---- 품질 특징 ----
def quality_feats(img, den, bg, sig):
    f = {}
    f['q_noise'] = sig
    gx = ndi.sobel(den, 1); gy = ndi.sobel(den, 0)
    gm = np.hypot(gx, gy)
    f['q_grad_mean'] = float(gm.mean())
    f['q_grad_p95'] = float(np.percentile(gm, 95))
    lap = ndi.laplace(ndi.gaussian_filter(img, 1.0))
    f['q_lap_var'] = float(lap.var())
    # 선명도 / 잡음 비
    f['q_sharp_noise'] = f['q_grad_p95'] / (sig + 1e-3)
    f['q_bg_range'] = float(np.percentile(bg, 99) - np.percentile(bg, 1))
    f['q_bg_std'] = float(bg.std())
    return f


def hist_feats(img, den):
    f = {}
    for name, a in [('raw', img), ('den', den)]:
        p = np.percentile(a, [1, 5, 10, 25, 50, 75, 90, 95, 99])
        for q, v in zip([1, 5, 10, 25, 50, 75, 90, 95, 99], p):
            f[f'h_{name}_p{q}'] = float(v)
        f[f'h_{name}_mean'] = float(a.mean())
        f[f'h_{name}_std'] = float(a.std())
    f['h_den_iqr'] = f['h_den_p75'] - f['h_den_p25']
    f['h_den_lowtail'] = f['h_den_p50'] - f['h_den_p5']
    return f


# ---- 기공 ----
def pore_feats(den, sig):
    f = {}
    med = np.median(den)
    mad = np.median(np.abs(den - med)) * 1.4826 + 1e-3
    sm = ndi.gaussian_filter(den, 1.5)
    out = {}
    for k in (3.0, 4.5):
        mask = sm < med - k * max(mad, 4.0)
        mask = morphology.remove_small_objects(mask, 12)
        lab = measure.label(mask)
        props = measure.regionprops(lab)
        # 둥근 덩어리만 (원형도, 신장도 기준)
        keep = [p for p in props if p.area >= 12 and p.eccentricity < 0.85
                and 4 * np.pi * p.area / (p.perimeter ** 2 + 1e-6) > 0.55]
        areas = np.array([p.area for p in keep]) if keep else np.zeros(1)
        f[f'pore_k{k}_n'] = len(keep)
        f[f'pore_k{k}_frac'] = float(areas.sum() / den.size) if keep else 0.0
        f[f'pore_k{k}_meanr'] = float(np.sqrt(areas / np.pi).mean()) if keep else 0.0
        f[f'dark_k{k}_frac'] = float(mask.mean())
        out[k] = (mask, keep, lab)
    # LoG 블롭 검출
    inv = (med - den) / mad
    blobs = feature.blob_log(np.clip(inv, 0, None) / 10.0, min_sigma=2, max_sigma=7,
                             num_sigma=6, threshold=0.25)
    f['pore_log_n'] = len(blobs)
    f['pore_log_meanr'] = float(blobs[:, 2].mean() * np.sqrt(2)) if len(blobs) else 0.0
    return f, out


# ---- 결정립계 / 상 ----
def boundary_map(den, s):
    """얇은 어두운 선 강조: black top-hat"""
    th = morphology.black_tophat(den, morphology.disk(3))
    return th


def phase_boundary_feats(den, sig, s):
    f = {}
    th = boundary_map(den, s)
    thn = th / (np.percentile(th, 99) + 1e-3)
    for t in (0.2, 0.35, 0.5):
        f[f'gb_frac_t{t}'] = float((thn > t).mean())
    f['gb_th_mean'] = float(th.mean())
    f['gb_th_p90'] = float(np.percentile(th, 90))
    # 결정립계 제거 후 영역 밝기 (closing으로 선 메우기)
    filled = morphology.closing(den, morphology.disk(3))
    filled = ndi.median_filter(filled, 5)
    med = np.median(filled)
    mad = np.median(np.abs(filled - med)) * 1.4826 + 1e-3
    for d in (8, 15, 25):
        f[f'ph_frac_abs{d}'] = float((filled < med - d).mean())
    for k in (1.0, 2.0, 3.0):
        f[f'ph_frac_mad{k}'] = float((filled < med - k * mad).mean())
    try:
        o = filters.threshold_otsu(filled)
        f['ph_otsu_frac'] = float((filled < o).mean())
        f['ph_otsu_gap'] = float(filled[filled >= o].mean() - filled[filled < o].mean())
    except Exception:
        f['ph_otsu_frac'] = 0.0; f['ph_otsu_gap'] = 0.0
    f['ph_filled_std'] = float(filled.std())
    f['ph_mad'] = float(mad)
    return f, th, filled


# ---- 결정립 크기 (watershed) ----
def grain_feats(den, th):
    f = {}
    edge = ndi.gaussian_filter(th, 1.0)
    # 결정립 내부 = 경계 반응 낮은 곳
    t = np.percentile(edge, 70)
    interior = edge < t
    interior = morphology.binary_opening(interior, morphology.disk(1))
    dist = ndi.distance_transform_edt(interior)
    coords = feature.peak_local_max(dist, min_distance=4, labels=measure.label(interior))
    markers = np.zeros_like(dist, dtype=np.int32)
    markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
    lab = segmentation.watershed(edge, markers)
    props = measure.regionprops(lab)
    areas = np.array([p.area for p in props if p.area > 15], dtype=np.float64)
    if len(areas) == 0:
        areas = np.array([den.size], dtype=np.float64)
    f['gs_n'] = len(areas)
    f['gs_mean_area'] = float(areas.mean())
    f['gs_med_area'] = float(np.median(areas))
    f['gs_cv_area'] = float(areas.std() / areas.mean())
    f['gs_mean_eqd'] = float(np.sqrt(4 * areas / np.pi).mean())
    ecc = [p.major_axis_length / (p.minor_axis_length + 1e-3) for p in props if p.area > 30]
    f['gs_aspect_mean'] = float(np.mean(ecc)) if ecc else 1.0
    f['gs_aspect_med'] = float(np.median(ecc)) if ecc else 1.0
    return f, lab


# ---- 정렬 / 신장 ----
def orientation_feats(den, th):
    f = {}
    for sc in (1.5, 4.0):
        Axx, Axy, Ayy = feature.structure_tensor(den, sigma=sc, order='rc')
        # 이미지 전체 평균 텐서
        J = np.array([[Axx.mean(), Axy.mean()], [Axy.mean(), Ayy.mean()]])
        ev = np.linalg.eigvalsh(J)
        f[f'or_coh_s{sc}'] = float((ev[1] - ev[0]) / (ev[1] + ev[0] + 1e-6))
        # 국소 coherence 평균
        tr = Axx + Ayy
        dif = np.sqrt((Axx - Ayy) ** 2 + 4 * Axy ** 2)
        f[f'or_lcoh_s{sc}'] = float((dif / (tr + 1e-6)).mean())
    # 자기상관 기반 이방성 (경계 맵)
    a = th - th.mean()
    F = np.fft.fft2(a)
    ac = np.fft.fftshift(np.real(np.fft.ifft2(F * np.conj(F))))
    ac /= ac.max()
    c = np.array(ac.shape) // 2
    win = ac[c[0] - 40:c[0] + 41, c[1] - 40:c[1] + 41]
    yy, xx = np.mgrid[-40:41, -40:41]
    for lvl in (0.2, 0.4):
        m = win > lvl
        w = win * m
        Ixx = (w * xx ** 2).sum(); Iyy = (w * yy ** 2).sum(); Ixy = (w * xx * yy).sum()
        ev = np.linalg.eigvalsh(np.array([[Ixx, Ixy], [Ixy, Iyy]]))
        f[f'or_ac_aniso{lvl}'] = float(np.sqrt(ev[1] / (ev[0] + 1e-6)))
        f[f'or_ac_area{lvl}'] = float(m.sum())  # 상관 길이 ~ 결정립 크기
    # 원본(평활) 이미지 자기상관
    a2 = den - den.mean()
    F2 = np.fft.fft2(a2)
    ac2 = np.fft.fftshift(np.real(np.fft.ifft2(F2 * np.conj(F2))))
    ac2 /= ac2.max()
    win2 = ac2[c[0] - 40:c[0] + 41, c[1] - 40:c[1] + 41]
    for lvl in (0.3, 0.5):
        m = win2 > lvl; w = win2 * m
        Ixx = (w * xx ** 2).sum(); Iyy = (w * yy ** 2).sum(); Ixy = (w * xx * yy).sum()
        ev = np.linalg.eigvalsh(np.array([[Ixx, Ixy], [Ixy, Iyy]]))
        f[f'or_ac2_aniso{lvl}'] = float(np.sqrt(ev[1] / (ev[0] + 1e-6)))
        f[f'or_ac2_area{lvl}'] = float(m.sum())
    return f


# ---- 방사형 파워 스펙트럼 ----
def psd_feats(den, prefix='psd'):
    f = {}
    a = den - den.mean()
    P = np.abs(np.fft.fftshift(np.fft.fft2(a))) ** 2
    h, w = P.shape
    yy, xx = np.mgrid[:h, :w]
    r = np.hypot(yy - h // 2, xx - w // 2)
    edges = np.unique(np.round(np.geomspace(1, 128, 13)).astype(int))
    tot = P[(r >= 1) & (r < 128)].sum() + 1e-6
    for lo, hi in zip(edges[:-1], edges[1:]):
        f[f'{prefix}_{lo}_{hi}'] = float(np.log(P[(r >= lo) & (r < hi)].sum() / tot + 1e-12))
    return f


def extract(path):
    img = load(path)
    flat, den, bg, sig, s = preprocess(img)
    f = {}
    f.update(quality_feats(img, den, bg, sig))
    f.update(hist_feats(flat, den))
    pf, _ = pore_feats(den, sig)
    f.update(pf)
    pbf, th, filled = phase_boundary_feats(den, sig, s)
    f.update(pbf)
    gf, _ = grain_feats(den, th)
    f.update(gf)
    f.update(orientation_feats(den, th))
    f.update(psd_feats(den))
    f.update(psd_feats(th, 'psdgb'))
    return f
