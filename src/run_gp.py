# -*- coding: utf-8 -*-
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, time
from multiprocessing import Pool
from grain_phase import extract_gp
def work(a):
    import warnings; warnings.filterwarnings('ignore')
    d,i=a; f=extract_gp(f'../data/{d}/{i}.png'); f['ID']=i; return f
if __name__=='__main__':
    for d,csv in [('train','../data/train.csv'),('test','../data/sample_submission.csv')]:
        t=time.time(); ids=pd.read_csv(csv).ID.tolist()
        with Pool(2) as p: rows=p.map(work,[(d,i) for i in ids],chunksize=10)
        pd.DataFrame(rows).set_index('ID').loc[ids].to_csv(f'../feats/{d}_gp.csv'); print(d,len(rows),'%.0fs'%(time.time()-t),flush=True)
