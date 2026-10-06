# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, sys
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
import lightgbm as lgb
from cv import run_cv, rmse
from exp_feats import sets, y
def sk(fn):
    def f(a,b,c,d): m=fn(); m.fit(a,b); return m.predict(c), m.predict(d)
    return f
qt=lambda: QuantileTransformer(n_quantiles=200,output_distribution='normal',random_state=0)
imp=lambda: SimpleImputer(strategy='median')   # 학습 fold에서만 fit
M={
 'qridge': lambda: make_pipeline(imp(),qt(),RidgeCV(alphas=np.logspace(-1,4,30))),
 'qsvr': lambda: make_pipeline(imp(),qt(),StandardScaler(),SVR(C=20,epsilon=2.0,gamma=0.003)),
 'et': lambda: make_pipeline(imp(),ExtraTreesRegressor(600,min_samples_leaf=3,max_features=0.4,n_jobs=2,random_state=0)),
 'lgb': lambda: lgb.LGBMRegressor(n_estimators=800,learning_rate=0.015,num_leaves=8,min_child_samples=15,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=3.0,verbose=-1,n_jobs=2,random_state=0),
}
if __name__=='__main__':
    tag=sys.argv[1]; names=sys.argv[2].split(',')
    X,Xt=sets(names); print(tag,X.shape,flush=True); O=[];T=[]
    for k,m in M.items():
        print(' ',k,flush=True); r=run_cv(sk(m),X,y,Xt); O.append(r['oof']); T.append(r['test'])
        np.save(f'../oof/{tag}_{k}_oof.npy',r['oof']); np.save(f'../oof/{tag}_{k}_test.npy',r['test'])
    O=np.mean(O,0); T=np.mean(T,0); np.save(f'../oof/{tag}_blend_oof.npy',O); np.save(f'../oof/{tag}_blend_test.npy',T)
    print(f'[{tag}] equal blend OOF RMSE {rmse(y,O):.4f}')
