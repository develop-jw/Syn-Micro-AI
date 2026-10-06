# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, time
from features_v5 import extract_v5
z=np.load('../feats/den_cache.npz')
for d,key,csv in [('train','tr','../data/train.csv'),('test','te','../data/sample_submission.csv')]:
    t=time.time(); ids=pd.read_csv(csv).ID.tolist(); A=z[key]
    rows=[extract_v5(A[i,0],A[i,1]) for i in range(len(ids))]
    pd.DataFrame(rows,index=ids).to_csv(f'../feats/{d}_v5.csv'); print(d,len(rows),'%.0fs'%(time.time()-t),flush=True)
