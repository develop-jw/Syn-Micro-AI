# -*- coding: utf-8 -*-
"""사전학습 CNN(고정) 중간층 특징 추출: 각 단계 특징맵의 채널별 평균+표준편차
사용: python3 feat_pretrained.py resnet18 ../weights/r18.pth r18
"""
import warnings; warnings.filterwarnings('ignore')
import sys, time, numpy as np, pandas as pd, torch, timm
from PIL import Image
torch.set_num_threads(2)
arch, wpath, tag = sys.argv[1], sys.argv[2], sys.argv[3]
m = timm.create_model(arch, pretrained=False, features_only=True)
sd = torch.load(wpath, map_location='cpu'); sd = {k: v for k, v in sd.items() if not k.startswith(('fc.', 'head', 'classifier'))}
print(m.load_state_dict(sd, strict=False))
m.eval()
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]; STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
def feat(paths):
    x = torch.stack([torch.from_numpy(np.asarray(Image.open(p), np.float32) / 255.)[None].repeat(3, 1, 1) for p in paths])
    x = (x - MEAN) / STD
    out = []
    with torch.no_grad():
        for k in range(4):  # 회전 TTA로 방향 불변성 확보
            fs = m(torch.rot90(x, k, dims=[2, 3]))
            out.append(torch.cat([torch.cat([f.mean((2, 3)), f.std((2, 3))], 1) for f in fs], 1))
    return torch.stack(out).mean(0).numpy()
for d, csv in [('train', '../data/train.csv'), ('test', '../data/sample_submission.csv')]:
    t = time.time(); ids = pd.read_csv(csv).ID.tolist()
    F = np.concatenate([feat([f'../data/{d}/{i}.png' for i in ids[j:j + 25]]) for j in range(0, len(ids), 25)])
    pd.DataFrame(F, index=ids).add_prefix(f'{tag}_').to_csv(f'../feats/{d}_{tag}.csv'); print(d, F.shape, '%.0fs' % (time.time() - t), flush=True)
