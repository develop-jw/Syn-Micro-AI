# -*- coding: utf-8 -*-
"""웨이블릿 산란 특징 (학습 없는 고정 필터). 회전 불변이 되도록 방향 축을 평균/표준편차로 요약."""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, torch, time
from kymatio.scattering2d.frontend.torch_frontend import ScatteringTorch2D as Scattering2D
from features import load, preprocess
torch.set_num_threads(2)
J, L = 5, 8
S = Scattering2D(J=J, shape=(256, 256), L=L, max_order=2)
def feats(path):
    img = load(path); flat, den, bg, sig, s = preprocess(img)
    x = (flat - flat.mean()) / (flat.std() + 1e-6)  # 이미지별 표준화 (조명 보정 후)
    with torch.no_grad():
        Sx = S(torch.from_numpy(x.astype(np.float32))[None])[0]  # (C, 8, 8)
    m = Sx.mean((-1, -2)).numpy()  # 공간 평균
    # 계수 구조: order0 1개, order1 J*L, order2 L^2*J(J-1)/2
    o0 = m[:1]; o1 = m[1:1 + J * L].reshape(J, L)
    o2 = m[1 + J * L:]
    f = [np.log(o0 + 1e-6)]
    f += [np.log(o1.mean(1) + 1e-6), o1.std(1) / (o1.mean(1) + 1e-6)]  # 스케일별 강도, 방향 이방성
    # order2: (j1,l1,j2,l2) -> 방향 차이(l2-l1)로 묶어 회전 불변화
    idx = []
    for j1 in range(J):
        for l1 in range(L):
            for j2 in range(j1 + 1, J):
                for l2 in range(L):
                    idx.append((j1, l1, j2, l2))
    # kymatio order2 순서: j1, l1, j2, l2  (문서 기준)
    d = {}
    for v, (j1, l1, j2, l2) in zip(o2, idx):
        d.setdefault((j1, j2, (l2 - l1) % L), []).append(v)
    o2r = np.array([np.mean(d[k]) for k in sorted(d)])
    o1m = o1.mean(1)
    # order1로 정규화한 order2 (질감 모양 정보)
    o2n = np.array([np.mean(d[k]) / (o1m[k[0]] + 1e-6) for k in sorted(d)])
    f += [np.log(o2r + 1e-6), np.log(o2n + 1e-6)]
    return np.concatenate(f)
if __name__ == '__main__':
    for d, csv in [('train', '../data/train.csv'), ('test', '../data/sample_submission.csv')]:
        t = time.time(); ids = pd.read_csv(csv).ID.tolist()
        F = np.stack([feats(f'../data/{d}/{i}.png') for i in ids])
        pd.DataFrame(F, index=ids, columns=[f'sc_{k}' for k in range(F.shape[1])]).to_csv(f'../feats/{d}_scat.csv')
        print(d, F.shape, '%.0fs' % (time.time() - t), flush=True)
