# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, json
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import SVR
from scipy.optimize import nnls
import lightgbm as lgb
from cv import run_cv, rmse
from exp_feats import sets, sk, y
X,Xt=sets(['v1','v2'])
qt=lambda: QuantileTransformer(n_quantiles=200,output_distribution='normal',random_state=0)  # 학습 fold에서만 fit (파이프라인)
models={
 'B01_qridge': lambda: make_pipeline(qt(),RidgeCV(alphas=np.logspace(-1,4,30))),
 'B02_qsvr': lambda: make_pipeline(qt(),StandardScaler(),SVR(C=20,epsilon=2.0,gamma=0.003)),
 'B03_et': lambda: ExtraTreesRegressor(600,min_samples_leaf=3,max_features=0.4,n_jobs=2,random_state=0),
 'B04_lgb': lambda: lgb.LGBMRegressor(n_estimators=800,learning_rate=0.015,num_leaves=8,min_child_samples=15,subsample=0.8,subsample_freq=1,colsample_bytree=0.5,reg_lambda=3.0,verbose=-1,n_jobs=2,random_state=0),
}
if __name__=='__main__':
  R={}
  for k,m in models.items():
      print(k); R[k]=run_cv(sk(m),X,y,Xt); np.save(f'../oof/{k}_oof.npy',R[k]['oof']); np.save(f'../oof/{k}_test.npy',R[k]['test'])
  O=np.stack([R[k]['oof'] for k in R],1); T=np.stack([R[k]['test'] for k in R],1)
  w,_=nnls(O,y); w=w/w.sum(); print('nnls weights',dict(zip(R,np.round(w,3))))
  print('blend(nnls) OOF RMSE %.4f | equal %.4f'%(rmse(y,O@w),rmse(y,O.mean(1))))
  json.dump({'scores':{k:R[k]['mean'] for k in R},'w':dict(zip(R,w.tolist()))},open('../oof/blend_B.json','w'),indent=1)
  np.save('../oof/BLEND_B_test.npy',T@w); np.save('../oof/BLEND_B_oof.npy',O@w)
