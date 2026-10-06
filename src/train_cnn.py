# -*- coding: utf-8 -*-
"""자체 설계 경량 CNN (MicroTexNet) - CPU 학습용

사용 예: python3 train_cnn.py --exp C01 --seeds 42 --epochs 40
"""
import warnings; warnings.filterwarnings('ignore')
import argparse, json, time, os
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from scipy import ndimage as ndi
from PIL import Image
from cv import make_folds, rmse, N_FOLDS

# ---- 설정 ----
p = argparse.ArgumentParser()
p.add_argument('--exp', default='C01')
p.add_argument('--seeds', default='42')
p.add_argument('--folds', default='all')
p.add_argument('--epochs', type=int, default=40)
p.add_argument('--bs', type=int, default=16)
p.add_argument('--lr', type=float, default=2e-3)
p.add_argument('--wd', type=float, default=1e-2)
p.add_argument('--width', type=int, default=24)
p.add_argument('--crop', type=int, default=192)
p.add_argument('--feats', default='')       # 하이브리드: 수치 특징 CSV 접두사 (예: v1)
p.add_argument('--loss', default='mse')
p.add_argument('--threads', type=int, default=2)
p.add_argument('--input', default='raw')        # raw | den | denridge
args = p.parse_args()
torch.set_num_threads(args.threads)
ROOT = '..'


# ---- 데이터 로딩 ----
def load_flat(path):
    img = np.asarray(Image.open(path), dtype=np.float32)
    bg = ndi.gaussian_filter(img, 40)
    flat = img - bg + bg.mean()            # 조명 그라데이션 제거 (절대 대비는 유지)
    return (flat - 128.0) / 40.0           # 전역 상수로 정규화


tr = pd.read_csv(f'{ROOT}/data/train.csv')
te = pd.read_csv(f'{ROOT}/data/sample_submission.csv')
if args.input == 'raw':
    cache = f'{ROOT}/feats/img_cache.npz'
    if os.path.exists(cache):
        z = np.load(cache); Xtr, Xte = z['tr'], z['te']
    else:
        Xtr = np.stack([load_flat(f'{ROOT}/data/train/{i}.png') for i in tr.ID]).astype(np.float32)
        Xte = np.stack([load_flat(f'{ROOT}/data/test/{i}.png') for i in te.ID]).astype(np.float32)
        np.savez(cache, tr=Xtr, te=Xte)
    Xtr, Xte = Xtr[:, None], Xte[:, None]
else:
    z = np.load(f'{ROOT}/feats/den_cache.npz'); Xtr, Xte = z['tr'], z['te']
    # 채널별 고정 상수 정규화 (이미지별 정규화 안 함: 절대 밝기 차이 보존)
    Xtr[:, 0] = (Xtr[:, 0] - 128.) / 40.; Xte[:, 0] = (Xte[:, 0] - 128.) / 40.
    rs = np.percentile(Xtr[:, 1], 99)   # train 전체 상수 (라벨 정보 미사용 스케일)
    Xtr[:, 1] /= rs; Xte[:, 1] /= rs
    if args.input == 'den':
        Xtr, Xte = Xtr[:, :1], Xte[:, :1]
IN_CH = Xtr.shape[1]
y = tr.hardness.values.astype(np.float64)

Ftr = Fte = None
if args.feats:
    Ftr = pd.read_csv(f'{ROOT}/feats/train_{args.feats}.csv', index_col=0).loc[tr.ID].values.astype(np.float32)
    Fte = pd.read_csv(f'{ROOT}/feats/test_{args.feats}.csv', index_col=0).loc[te.ID].values.astype(np.float32)


# ---- 증강 ----
def dihedral(x, k):
    """x: (..., H, W), k in 0..7 : 회전 4 x 반전 2"""
    if k >= 4:
        x = torch.flip(x, dims=[-1])
    return torch.rot90(x, k % 4, dims=[-2, -1])


def augment(xb):
    B = xb.shape[0]; out = []
    c = args.crop
    for b in range(B):
        x = xb[b]
        i = np.random.randint(0, 256 - c + 1); j = np.random.randint(0, 256 - c + 1)
        x = x[..., i:i + c, j:j + c]
        x = dihedral(x, np.random.randint(8))
        x = x * np.random.uniform(0.92, 1.08) + np.random.uniform(-0.08, 0.08)  # 약한 밝기/대비
        out.append(x)
    return torch.stack(out)


# ---- 모델 ----
class Block(nn.Module):
    def __init__(s, ci, co, stride):
        super().__init__()
        s.c1 = nn.Conv2d(ci, co, 3, stride, 1, bias=False); s.b1 = nn.BatchNorm2d(co)
        s.c2 = nn.Conv2d(co, co, 3, 1, 1, bias=False); s.b2 = nn.BatchNorm2d(co)
        s.sk = None if (ci == co and stride == 1) else nn.Sequential(
            nn.Conv2d(ci, co, 1, stride, bias=False), nn.BatchNorm2d(co))

    def forward(s, x):
        h = F.gelu(s.b1(s.c1(x)))
        h = s.b2(s.c2(h))
        return F.gelu(h + (x if s.sk is None else s.sk(x)))


class MicroTexNet(nn.Module):
    """다중 스케일 통계 풀링: 각 단계 특징맵의 채널별 평균+표준편차를 모아 회귀"""
    def __init__(s, w=24, n_feat=0):
        super().__init__()
        s.stem = nn.Sequential(nn.Conv2d(IN_CH, w, 3, 1, 1, bias=False), nn.BatchNorm2d(w), nn.GELU(),
                               nn.Conv2d(w, w, 3, 2, 1, bias=False), nn.BatchNorm2d(w), nn.GELU())
        chs = [w, w * 2, w * 4, w * 4]
        s.stages = nn.ModuleList()
        ci = w
        for co in chs:
            s.stages.append(nn.Sequential(Block(ci, co, 2), Block(co, co, 1)))
            ci = co
        d = sum(2 * c for c in chs)
        s.n_feat = n_feat
        s.fproj = nn.Sequential(nn.Linear(n_feat, 64), nn.GELU()) if n_feat else None
        s.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(d + (64 if n_feat else 0), 64), nn.GELU(),
                               nn.Linear(64, 1))

    def forward(s, x, f=None):
        h = s.stem(x); pooled = []
        for st in s.stages:
            h = st(h)
            pooled += [h.mean((-1, -2)), h.std((-1, -2))]
        z = torch.cat(pooled, 1)
        if s.fproj is not None:
            z = torch.cat([z, s.fproj(f)], 1)
        return s.head(z).squeeze(1)


# ---- 학습 / 추론 ----
def predict(model, X, Fm=None, tta=True):
    model.eval(); outs = []
    ks = range(8) if tta else [0]
    with torch.no_grad():
        for k in ks:
            ps = []
            for i in range(0, len(X), 32):
                xb = dihedral(torch.from_numpy(X[i:i + 32]), k)
                fb = torch.from_numpy(Fm[i:i + 32]) if Fm is not None else None
                ps.append(model(xb, fb).numpy())
            outs.append(np.concatenate(ps))
    return np.mean(outs, 0), outs[0]


def train_fold(tri, vai, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    mu, sd = y[tri].mean(), y[tri].std()                         # 타깃 정규화: 학습 fold에서만
    Fm_tr = Fm_va = Fm_te = None
    if Ftr is not None:
        fm, fs = Ftr[tri].mean(0), Ftr[tri].std(0) + 1e-6        # 특징 표준화: 학습 fold에서만
        Fm_tr = np.clip((Ftr[tri] - fm) / fs, -5, 5).astype(np.float32)
        Fm_va = np.clip((Ftr[vai] - fm) / fs, -5, 5).astype(np.float32)
        Fm_te = np.clip((Fte - fm) / fs, -5, 5).astype(np.float32)
    model = MicroTexNet(args.width, 0 if Ftr is None else Ftr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * int(np.ceil(len(tri) / args.bs))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.15)
    Xt = torch.from_numpy(Xtr[tri]); yt = torch.from_numpy(((y[tri] - mu) / sd).astype(np.float32))
    Ft = torch.from_numpy(Fm_tr) if Fm_tr is not None else None
    lossf = nn.MSELoss() if args.loss == 'mse' else nn.HuberLoss(delta=1.0)
    for ep in range(args.epochs):
        model.train(); perm = torch.randperm(len(tri))
        for i in range(0, len(tri), args.bs):
            b = perm[i:i + args.bs]
            if len(b) < 2: continue
            xb = augment(Xt[b])
            out = model(xb, Ft[b] if Ft is not None else None)
            loss = lossf(out, yt[b])
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    pv, pv_notta = predict(model, Xtr[vai], Fm_va)
    pt, _ = predict(model, Xte, Fm_te)
    return pv * sd + mu, pv_notta * sd + mu, pt * sd + mu


if __name__ == '__main__':
    os.makedirs(f'{ROOT}/oof', exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(',')]
    n_params = sum(p.numel() for p in MicroTexNet(args.width, 0 if Ftr is None else Ftr.shape[1]).parameters())
    print(f'[{args.exp}] params={n_params:,} args={vars(args)}', flush=True)
    res = {'args': vars(args), 'scores': [], 'scores_notta': []}
    oofs, tests = [], []
    for s in seeds:
        oof = np.zeros(len(y)); oof_n = np.zeros(len(y)); tp = np.zeros(len(te))
        folds = make_folds(y, s)
        use = range(N_FOLDS) if args.folds == 'all' else [int(k) for k in args.folds.split(',')]
        for k in use:
            t = time.time(); tri, vai = folds[k]
            pv, pvn, pt = train_fold(tri, vai, s * 100 + k)
            oof[vai] = pv; oof_n[vai] = pvn; tp += pt / len(use)
            print(f'  seed {s} fold {k}: RMSE {rmse(y[vai], pv):.4f} (noTTA {rmse(y[vai], pvn):.4f}) {time.time()-t:.0f}s', flush=True)
        if args.folds == 'all':
            sc, scn = rmse(y, oof), rmse(y, oof_n)
            res['scores'].append(sc); res['scores_notta'].append(scn)
            print(f'  seed {s}: OOF RMSE {sc:.4f} (noTTA {scn:.4f})', flush=True)
            oofs.append(oof); tests.append(tp)
    if oofs:
        res['mean'] = float(np.mean(res['scores']))
        np.save(f'{ROOT}/oof/{args.exp}_oof.npy', np.mean(oofs, 0))
        np.save(f'{ROOT}/oof/{args.exp}_test.npy', np.mean(tests, 0))
        json.dump(res, open(f'{ROOT}/oof/{args.exp}_scores.json', 'w'), indent=1)
        print(f'[{args.exp}] mean OOF RMSE {res["mean"]:.4f}', flush=True)
