# -*- coding: utf-8 -*-
"""fold 내부 특징 선택: 학습 fold에서만 LightGBM 중요도로 상위 K개 선택 후 학습"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, sys
import lightgbm as lgb
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import RidgeCV
from cv import run_cv, rmse
from exp_feats import sets, y
P=dict(n_estimators=800,learning_rate=0.015,num_leaves=8,min_child_samples=15,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=3.0,verbose=-1,n_jobs=2,random_state=0)
def sel_fit(K, final):
    def f(Xtr,ytr,Xva,Xte):
        imp=np.zeros(Xtr.shape[1])
        for s in range(3):   # 선택 안정화: 여러 seed 중요도 평균
            m=lgb.LGBMRegressor(**{**P,'random_state':s,'importance_type':'gain'}).fit(Xtr,ytr); imp+=m.feature_importances_
        cols=Xtr.columns[np.argsort(-imp)[:K]]
        if final=='lgb': m=lgb.LGBMRegressor(**P)
        else: m=make_pipeline(SimpleImputer(strategy='median'),QuantileTransformer(n_quantiles=200,output_distribution='normal',random_state=0),RidgeCV(alphas=np.logspace(-1,4,30)))
        m.fit(Xtr[cols],ytr); return m.predict(Xva[cols]), m.predict(Xte[cols])
    return f
if __name__=='__main__':
    X,Xt=sets(['v1','v2','v5','v6'])
    for K in [40,80,150]:
        for fin in ['lgb','qridge']:
            print(f'K={K} {fin}',flush=True); r=run_cv(sel_fit(K,fin),X,y,Xt,verbose=False); print(f'  => {r["mean"]:.4f} ± {r["std"]:.4f}',flush=True)
            np.save(f'../oof/SEL{K}_{fin}_oof.npy',r['oof']); np.save(f'../oof/SEL{K}_{fin}_test.npy',r['test'])
