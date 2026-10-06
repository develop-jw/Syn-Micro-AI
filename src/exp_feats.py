# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, sys, json
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
import lightgbm as lgb
from cv import run_cv
tr=pd.read_csv('../data/train.csv'); te=pd.read_csv('../data/sample_submission.csv'); y=tr.hardness.values
def load(v): return pd.read_csv(f'../feats/train_{v}.csv',index_col=0).loc[tr.ID], pd.read_csv(f'../feats/test_{v}.csv',index_col=0).loc[te.ID]
def sets(names):
    a=[load(n) for n in names]; return pd.concat([x[0] for x in a],axis=1), pd.concat([x[1] for x in a],axis=1)
def sk(fn):
    def f(a,b,c,d): m=fn(); m.fit(a,b); return m.predict(c), m.predict(d)
    return f
ridge=lambda: make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-2,4,30)))
lgbm=lambda: lgb.LGBMRegressor(n_estimators=600,learning_rate=0.02,num_leaves=15,min_child_samples=10,subsample=0.8,subsample_freq=1,colsample_bytree=0.6,reg_lambda=1.0,verbose=-1,n_jobs=2,random_state=0)
if __name__=='__main__':
    for tag,names in [('v2',['v2']),('v1v2',['v1','v2'])]:
        X,Xt=sets(names); print(tag,X.shape)
        for mn,m in [('ridge',ridge),('lgb',lgbm)]:
            print(' ',mn); r=run_cv(sk(m),X,y,Xt)
            np.save(f'../oof/F_{tag}_{mn}_oof.npy',r['oof']); np.save(f'../oof/F_{tag}_{mn}_test.npy',r['test'])
