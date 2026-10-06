# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, time, json
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
import lightgbm as lgb
from cv import run_cv
X=pd.read_csv('../feats/train_v1.csv',index_col=0); Xt=pd.read_csv('../feats/test_v1.csv',index_col=0)
y=pd.read_csv('../data/train.csv').set_index('ID').loc[X.index,'hardness'].values

def sk(model_fn):
    def f(Xtr,ytr,Xva,Xte):
        m=model_fn(); m.fit(Xtr,ytr); return m.predict(Xva), m.predict(Xte)
    return f
models={
 'E00_mean': lambda Xtr,ytr,Xva,Xte:(np.full(len(Xva),ytr.mean()),np.full(len(Xte),ytr.mean())),
 'E01_ridge': sk(lambda: make_pipeline(StandardScaler(),RidgeCV(alphas=np.logspace(-2,3,20)))),
 'E02_svr': sk(lambda: make_pipeline(StandardScaler(),SVR(C=30,epsilon=1.0,gamma='scale'))),
 'E03_et': sk(lambda: ExtraTreesRegressor(500,min_samples_leaf=2,max_features=0.5,n_jobs=2,random_state=0)),
 'E04_lgb': sk(lambda: lgb.LGBMRegressor(n_estimators=600,learning_rate=0.02,num_leaves=15,min_child_samples=10,subsample=0.8,subsample_freq=1,colsample_bytree=0.6,reg_lambda=1.0,verbose=-1,n_jobs=2,random_state=0)),
}
out={}
for k,fn in models.items():
    t=time.time(); print(k); r=run_cv(fn,X,y,Xt); out[k]=r
    print(f'  time {time.time()-t:.0f}s')
    np.save(f'../oof/{k}_oof.npy',r['oof']); np.save(f'../oof/{k}_test.npy',r['test'])
json.dump({k:{'mean':v['mean'],'std':v['std'],'scores':v['scores']} for k,v in out.items()},open('../oof/baseline_scores.json','w'),indent=1)
