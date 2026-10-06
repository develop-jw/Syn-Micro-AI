# -*- coding: utf-8 -*-
"""사전학습 CNN 미세조정 (회귀)

사용 예: python3 finetune.py --exp T01 --arch resnet18 --weights ../weights/r18.pth --folds 0 --epochs 12
"""
import warnings; warnings.filterwarnings('ignore')
import argparse, json, os, time
import numpy as np, pandas as pd, torch, torch.nn as nn, timm
from PIL import Image
from cv import make_folds, rmse, N_FOLDS

# ---- 설정 ----
p = argparse.ArgumentParser()
p.add_argument('--exp', default='T01')
p.add_argument('--arch', default='resnet18')
p.add_argument('--weights', default='../weights/r18.pth')
p.add_argument('--seeds', default='42')
p.add_argument('--folds', default='all')
p.add_argument('--epochs', type=int, default=12)
p.add_argument('--bs', type=int, default=16)
p.add_argument('--lr', type=float, default=5e-4)
p.add_argument('--crop', type=int, default=224)
p.add_argument('--tta', type=int, default=4)       # 예측 시 회전 수 (1, 4, 8)
p.add_argument('--threads', type=int, default=2)
p.add_argument('--size', type=int, default=256)   # 입력 해상도 (256 또는 128: 2x2 평균 축소)
args = p.parse_args()
torch.set_num_threads(args.threads)
ROOT = '..'

# ---- 데이터 ----
tr = pd.read_csv(f'{ROOT}/data/train.csv'); te = pd.read_csv(f'{ROOT}/data/sample_submission.csv')
def load_all(d, ids):
    return np.stack([np.asarray(Image.open(f'{ROOT}/data/{d}/{i}.png'), np.float32) for i in ids])
cache = f'{ROOT}/feats/raw_cache.npz'
if os.path.exists(cache):
    z = np.load(cache); Xtr, Xte = z['tr'], z['te']
else:
    Xtr, Xte = load_all('train', tr.ID), load_all('test', te.ID); np.savez(cache, tr=Xtr, te=Xte)
Xtr = (Xtr / 255. - 0.45) / 0.225; Xte = (Xte / 255. - 0.45) / 0.225   # 고정 상수 정규화 (이미지별 정규화 안 함)
Xtr = Xtr.astype(np.float32); Xte = Xte.astype(np.float32)
if args.size == 128:   # 2x2 평균 축소 (잡음도 함께 감소)
    Xtr = Xtr.reshape(-1, 128, 2, 128, 2).mean((2, 4)); Xte = Xte.reshape(-1, 128, 2, 128, 2).mean((2, 4))
S = Xtr.shape[-1]
y = tr.hardness.values.astype(np.float64)


def dihedral(x, k):
    if k >= 4: x = torch.flip(x, dims=[-1])
    return torch.rot90(x, k % 4, dims=[-2, -1])


def augment(xb):
    c = args.crop; out = []
    for x in xb:
        i = np.random.randint(0, S - c + 1); j = np.random.randint(0, S - c + 1)
        out.append(dihedral(x[..., i:i + c, j:j + c], np.random.randint(8)))
    return torch.stack(out)


def build():
    m = timm.create_model(args.arch, pretrained=False, num_classes=0, in_chans=1)
    sd = torch.load(args.weights, map_location='cpu')
    sd = {k: v for k, v in sd.items() if not k.startswith(('fc.', 'head', 'classifier'))}
    # 3채널 입력 가중치 -> 1채널 (채널 합)
    k0 = [k for k in sd if sd[k].ndim == 4][0]
    sd[k0] = sd[k0].sum(1, keepdim=True)
    print(' load:', m.load_state_dict(sd, strict=False))
    head = nn.Sequential(nn.Dropout(0.2), nn.Linear(m.num_features, 1))
    return nn.Sequential(m, head)


def predict(model, X):
    model.eval(); outs = []
    with torch.no_grad():
        for k in range(args.tta):
            ps = [model(dihedral(torch.from_numpy(X[i:i + 32])[:, None], k)).squeeze(1).numpy() for i in range(0, len(X), 32)]
            outs.append(np.concatenate(ps))
    return np.mean(outs, 0)


def train_fold(tri, vai, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    mu, sd = y[tri].mean(), y[tri].std()
    model = build()
    bb, hd = model[0].parameters(), model[1].parameters()
    opt = torch.optim.AdamW([{'params': bb, 'lr': args.lr}, {'params': hd, 'lr': args.lr * 5}], weight_decay=1e-4)
    steps = args.epochs * int(np.ceil(len(tri) / args.bs))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr, args.lr * 5], total_steps=steps, pct_start=0.1)
    Xt = torch.from_numpy(Xtr[tri])[:, None]; yt = torch.from_numpy(((y[tri] - mu) / sd).astype(np.float32))
    for ep in range(args.epochs):
        model.train(); perm = torch.randperm(len(tri)); tl = 0; t = time.time()
        for i in range(0, len(tri), args.bs):
            b = perm[i:i + args.bs]
            if len(b) < 2: continue
            loss = nn.functional.mse_loss(model(augment(Xt[b])).squeeze(1), yt[b])
            opt.zero_grad(); loss.backward(); opt.step(); sch.step(); tl += loss.item() * len(b)
        if ep % 3 == 2 or ep == args.epochs - 1:
            pv = predict(model, Xtr[vai]) * sd + mu
            print(f'   ep {ep+1}: train_loss {tl/len(tri):.3f} val RMSE {rmse(y[vai], pv):.3f} ({time.time()-t:.0f}s/ep)', flush=True)
    return predict(model, Xtr[vai]) * sd + mu, predict(model, Xte) * sd + mu


if __name__ == '__main__':
    print(f'[{args.exp}] {vars(args)}', flush=True)
    seeds = [int(s) for s in args.seeds.split(',')]
    oofs, tests, scores = [], [], []
    for s in seeds:
        folds = make_folds(y, s); oof = np.zeros(len(y)); tp = np.zeros(len(te))
        use = range(N_FOLDS) if args.folds == 'all' else [int(k) for k in args.folds.split(',')]
        for k in use:
            t = time.time(); tri, vai = folds[k]
            pv, pt = train_fold(tri, vai, s * 100 + k)
            oof[vai] = pv; tp += pt / len(use)
            print(f'  seed {s} fold {k}: RMSE {rmse(y[vai], pv):.4f} {time.time()-t:.0f}s', flush=True)
            np.save(f'{ROOT}/oof/{args.exp}_s{s}_f{k}_va.npy', pv); np.save(f'{ROOT}/oof/{args.exp}_s{s}_f{k}_te.npy', pt)
        if args.folds == 'all':
            scores.append(rmse(y, oof)); oofs.append(oof); tests.append(tp)
            print(f'  seed {s}: OOF RMSE {scores[-1]:.4f}', flush=True)
    if oofs:
        np.save(f'{ROOT}/oof/{args.exp}_oof.npy', np.mean(oofs, 0)); np.save(f'{ROOT}/oof/{args.exp}_test.npy', np.mean(tests, 0))
        json.dump({'args': vars(args), 'scores': scores}, open(f'{ROOT}/oof/{args.exp}_scores.json', 'w'))
        print(f'[{args.exp}] mean OOF RMSE {np.mean(scores):.4f}')
