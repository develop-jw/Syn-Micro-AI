# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, sys, time
from multiprocessing import Pool
from features import extract
def work(a):
    import warnings; warnings.filterwarnings('ignore')
    d,i=a; f=extract(f'../data/{d}/{i}.png'); f['ID']=i; return f
if __name__=='__main__':
    for d,csv in [('train','../data/train.csv'),('test','../data/sample_submission.csv')]:
        t=time.time(); ids=pd.read_csv(csv).ID.tolist()
        with Pool(2) as p: rows=p.map(work,[(d,i) for i in ids],chunksize=10)
        df=pd.DataFrame(rows).set_index('ID').loc[ids]; df.to_csv(f'../feats/{d}_v1.csv')
        print(d,df.shape,'%.0fs'%(time.time()-t),flush=True)
