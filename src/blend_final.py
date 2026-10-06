# -*- coding: utf-8 -*-
"""최종 블렌드 (BL5) + 제출 파일 생성

- 입력: ../oof/{모델}_oof.npy, ../oof/{모델}_test.npy  (각 실험 스크립트가 저장)
- 가중치: NNLS(음수 없는 최소제곱)로 OOF에 맞춤, 합이 1이 되도록 정규화
- 평가: 가중치를 fold 밖에서 고르는 중첩 검증으로 낙관 편향 점검
- 출력: ../submissions/sub_BL5_mks_select_blend.csv

사용: python3 blend_final.py
"""
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from cv import rmse, make_folds

# ---- 설정 ----
NAMES = ['V1256_qridge', 'V1256_et', 'V1256_lgb', 'V1256_qsvr',
         'SEL150_qridge', 'SEL40_qridge', 'SEL150_lgb',
         'C03_cnn_w16', 'C04_denridge_w16']
OUT = '../submissions/sub_BL5_mks_select_blend.csv'
NESTED_SEEDS = [11, 22, 33, 44, 55]

tr = pd.read_csv('../data/train.csv'); y = tr.hardness.values
ss = pd.read_csv('../data/sample_submission.csv')
O = np.stack([np.load(f'../oof/{n}_oof.npy') for n in NAMES], 1)
T = np.stack([np.load(f'../oof/{n}_test.npy') for n in NAMES], 1)

# ---- 평가 ----
print('equal-weight blend OOF RMSE %.4f' % rmse(y, O.mean(1)))
res = []
for s in NESTED_SEEDS:
    P = np.zeros(len(y))
    for a, b in make_folds(y, s):
        w, _ = nnls(O[a], y[a]); w /= w.sum(); P[b] = O[b] @ w
    res.append(rmse(y, P))
print('nested NNLS blend OOF RMSE %.4f ± %.4f' % (np.mean(res), np.std(res)))

# ---- 최종 가중치 + 제출 ----
w, _ = nnls(O, y); w /= w.sum()
print('weights:', {n: round(float(x), 3) for n, x in zip(NAMES, w) if x > 0})
sub = ss.copy(); sub['hardness'] = T @ w
sub.to_csv(OUT, index=False, encoding='utf-8')

# ---- 제출 파일 검증 ----
s = pd.read_csv(OUT)
assert list(s.columns) == ['ID', 'hardness'] and len(s) == len(ss)
assert (s.ID == ss.ID).all() and s.ID.is_unique and np.isfinite(s.hardness).all()
print('saved', OUT)
