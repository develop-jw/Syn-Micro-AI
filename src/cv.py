# -*- coding: utf-8 -*-
"""공통 검증 도구: 동일 fold, 동일 지표로 모든 실험을 비교한다."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# ---- 설정 ----
N_FOLDS = 5
SEEDS = [42, 7, 2026]
N_BINS = 10


def rmse(y, p):
    """대회 공식 지표: sqrt(mean((pred - target)^2))"""
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


def make_folds(y, seed):
    """타깃을 구간으로 나눠 각 fold에 고르게 분포시키는 분할"""
    bins = pd.qcut(y, N_BINS, labels=False, duplicates='drop')
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), bins))


def run_cv(fit_predict, X, y, X_test=None, seeds=SEEDS, verbose=True):
    """fit_predict(X_tr, y_tr, X_va, X_te) -> (pred_va, pred_te)

    반환: seed별 OOF RMSE, 평균 OOF 예측, 평균 test 예측
    """
    y = np.asarray(y, dtype=np.float64)
    oofs, scores, tests = [], [], []
    for s in seeds:
        oof = np.zeros(len(y))
        te = np.zeros(len(X_test)) if X_test is not None else None
        for tr, va in make_folds(y, s):
            Xtr = X.iloc[tr] if hasattr(X, 'iloc') else X[tr]
            Xva = X.iloc[va] if hasattr(X, 'iloc') else X[va]
            pv, pt = fit_predict(Xtr, y[tr], Xva, X_test)
            oof[va] = pv
            if pt is not None:
                te += pt / N_FOLDS
        sc = rmse(y, oof)
        scores.append(sc); oofs.append(oof)
        if te is not None:
            tests.append(te)
        if verbose:
            print(f'  seed {s}: OOF RMSE {sc:.4f}', flush=True)
    res = {
        'scores': scores,
        'mean': float(np.mean(scores)),
        'std': float(np.std(scores)),
        'oof': np.mean(oofs, 0),
        'test': np.mean(tests, 0) if tests else None,
    }
    if verbose:
        print(f'  => mean {res["mean"]:.4f} ± {res["std"]:.4f}', flush=True)
    return res
