# -*- coding: utf-8 -*-
"""특징 v6: 재료 정보학 표준 기술자

1) 2점 상관 통계 (2-point statistics, MKS 방식)
   - 상 지시함수 χ(x)에 대해 S2(r) = P[χ(x)=1, χ(x+r)=1]
   - FFT로 계산, 비주기 경계 보정(제로패딩 + 겹침 정규화)
   - 방향 평균(회전 불변) + 방향 이방성
   - 상(여러 임계값), 결정립계, 기공, 상-기공 교차 상관, 연속 밝기 상관
2) 민코프스키 함수 곡선
   - 상대 밝기 임계값별 면적 비율, 둘레 밀도, 오일러 수 밀도
"""
import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure

RBINS = np.unique(np.round(np.geomspace(1, 64, 14)).astype(int))   # 거리 구간 (픽셀)


def _fft_corr(a, b, mask_norm):
    """비주기 상관: 제로패딩 FFT 후 겹치는 픽셀 수로 나눔"""
    H, W = a.shape
    Fa = np.fft.rfft2(a, s=(2 * H, 2 * W)); Fb = np.fft.rfft2(b, s=(2 * H, 2 * W))
    c = np.fft.irfft2(np.conj(Fa) * Fb, s=(2 * H, 2 * W))
    c = np.fft.fftshift(c) / mask_norm
    return c


_NORM = None
_RGRID = None


def _setup(H=256, W=256):
    global _NORM, _RGRID
    if _NORM is None:
        one = np.ones((H, W))
        F1 = np.fft.rfft2(one, s=(2 * H, 2 * W))
        n = np.fft.fftshift(np.fft.irfft2(np.conj(F1) * F1, s=(2 * H, 2 * W)))
        _NORM = np.maximum(n, 1.0)
        yy, xx = np.mgrid[-H:H, -W:W]
        _RGRID = (np.hypot(yy, xx), np.arctan2(yy, xx))


def radial_profile(c, prefix, sub=0.0):
    """방향 평균 S2(r) - sub, 그리고 거리별 방향 이방성(2배각 성분 크기)"""
    r, th = _RGRID
    f = {}
    for lo, hi in zip(RBINS[:-1], RBINS[1:]):
        m = (r >= lo) & (r < hi)
        v = c[m]
        f[f'{prefix}_r{lo}'] = float(v.mean() - sub)
        w = v - v.mean()
        f[f'{prefix}_an{lo}'] = float(np.hypot((w * np.cos(2 * th[m])).mean(), (w * np.sin(2 * th[m])).mean()) / (abs(v.mean() - sub) + 1e-6))
    return f


def two_point(den, ridge):
    _setup(*den.shape)
    f = {}
    h, e = np.histogram(den, bins=256, range=(0, 256))
    mode = e[np.argmax(ndi.gaussian_filter1d(h.astype(float), 2))]
    rel = den - mode
    inds = {
        'ph10': (rel < -10).astype(float),
        'ph18': (rel < -18).astype(float),
        'ph28': (rel < -28).astype(float),
        'pore': (rel < -55).astype(float),
        'gb': (ridge > filters.threshold_otsu(ridge)).astype(float),
    }
    for k, chi in inds.items():
        vf = chi.mean()
        f[f'tp_{k}_vf'] = float(vf)
        c = _fft_corr(chi, chi, _NORM)
        prof = radial_profile(c, f'tp_{k}', sub=vf ** 2)
        # 부피 분율로 정규화한 상관 (모양/크기 정보만 남김)
        for kk in list(prof):
            if '_r' in kk:
                f[kk + 'n'] = prof[kk] / (vf * (1 - vf) + 1e-9)
        f.update(prof)
    # 상-기공 교차 상관 (기공이 상 근처에 있는지)
    c = _fft_corr(inds['ph18'], inds['pore'], _NORM)
    f.update(radial_profile(c, 'tp_x_ph_pore', sub=inds['ph18'].mean() * inds['pore'].mean()))
    # 연속 밝기 상관 (분할 없이)
    a = (den - den.mean()) / (den.std() + 1e-6)
    c = _fft_corr(a, a, _NORM)
    f.update(radial_profile(c, 'tp_gray'))
    return f


def minkowski(den):
    f = {}
    h, e = np.histogram(den, bins=256, range=(0, 256))
    mode = e[np.argmax(ndi.gaussian_filter1d(h.astype(float), 2))]
    rel = den - mode
    N = den.size
    for t in range(-60, 15, 5):
        m = rel < t
        area = m.mean()
        per = (m ^ ndi.binary_erosion(m)).sum() / N
        eu = measure.euler_number(m, connectivity=2) / N * 1e4
        f[f'mk_t{t}_A'] = float(area)
        f[f'mk_t{t}_P'] = float(per)
        f[f'mk_t{t}_X'] = float(eu)
    return f


def extract_v6(den, ridge):
    f = two_point(den, ridge)
    f.update(minkowski(den))
    return f
