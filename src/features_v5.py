# -*- coding: utf-8 -*-
"""특징 v5: 크기 분포 균일성 + 공간적 불균일성 (공식 참고자료 힌트 기반)

입력: den_cache.npz (NLM 잡음 제거 이미지, 능선 맵)
- tl_ : 타일별 경계 밀도 / 능선 반응 / 상 비율 / 밝기 편차의 이미지 내 변동
- gd_ : 결정립 면적 분포의 폭, 왜도, 상하위 비, 큰 결정립 면적 비중
"""
import numpy as np
from scipy import stats
from skimage import filters, morphology, measure


def tile_stats(a, t, prefix):
    """a를 t×t 타일로 나눠 타일 평균의 변동 통계"""
    H = a.shape[0] // t
    m = a[:H * t, :H * t].reshape(H, t, H, t).mean((1, 3)).ravel()
    mu = m.mean() + 1e-9
    return {
        f'{prefix}_t{t}_cv': float(m.std() / mu),
        f'{prefix}_t{t}_std': float(m.std()),
        f'{prefix}_t{t}_rng': float((np.percentile(m, 90) - np.percentile(m, 10)) / mu),
    }


def extract_v5(den, ridge):
    f = {}
    t0 = filters.threshold_otsu(ridge)
    bmask = ridge > t0
    # 기지 밝기(최빈값)
    h, e = np.histogram(den, bins=256, range=(0, 256))
    from scipy.ndimage import gaussian_filter1d
    mode = e[np.argmax(gaussian_filter1d(h.astype(float), 2))]
    phase = den < mode - 14
    for t in (32, 64):
        f.update(tile_stats(bmask.astype(float), t, 'tl_bnd'))
        f.update(tile_stats(ridge, t, 'tl_rdg'))
        f.update(tile_stats(phase.astype(float), t, 'tl_ph'))
        # 타일 내부 밝기 표준편차의 변동
        H = 256 // t
        sd = den.reshape(H, t, H, t).std((1, 3)).ravel()
        f[f'tl_sd_t{t}_cv'] = float(sd.std() / (sd.mean() + 1e-9))
    # 결정립 면적 분포
    m = morphology.remove_small_objects(bmask, max_size=15)
    sk = morphology.skeletonize(m)
    lab = measure.label(~morphology.dilation(sk, morphology.disk(1)), connectivity=1)
    areas = np.bincount(lab.ravel())[1:].astype(float)
    areas = areas[(areas > 20) & (areas < 0.25 * den.size)]
    if len(areas) >= 10:
        la = np.log(areas)
        med = np.median(areas)
        f['gd_n'] = len(areas)
        f['gd_logA_std'] = float(la.std())
        f['gd_logA_iqr'] = float(np.subtract(*np.percentile(la, [75, 25])))
        f['gd_logA_skew'] = float(stats.skew(la))
        f['gd_cv'] = float(areas.std() / areas.mean())
        f['gd_p90_p10'] = float(np.percentile(areas, 90) / np.percentile(areas, 10))
        f['gd_big_areafrac'] = float(areas[areas > 3 * med].sum() / areas.sum())   # 큰 결정립이 차지하는 면적
        f['gd_small_nfrac'] = float((areas < med / 3).mean())
        f['gd_max_frac'] = float(areas.max() / areas.sum())
        # 면적 가중 평균 / 개수 평균 (분포 폭에 민감)
        f['gd_wmean_over_mean'] = float((areas ** 2).sum() / areas.sum() / areas.mean())
    else:
        for k in ['gd_n', 'gd_logA_std', 'gd_logA_iqr', 'gd_logA_skew', 'gd_cv', 'gd_p90_p10',
                  'gd_big_areafrac', 'gd_small_nfrac', 'gd_max_frac', 'gd_wmean_over_mean']:
            f[k] = np.nan
    f['gd_valid'] = float(len(areas) >= 10)
    return f
