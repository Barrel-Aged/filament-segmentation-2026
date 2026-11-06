"""Train one dense U-Net fold model (cnx, cnxL or R34) on the soft annotator-mean target. CUDA only (bf16 autocast).

Not run for the release: the shipped weights ``weights/<name>/ema.pt`` are the ones the submission used. This file
documents and reproduces the recipe.

usage (repository root, after ``python -m training.prep_train``):
  python -m training.train_dense --config cnx-f0            # every argument from configs/cnx-f0.json
  python -m training.train_dense --name cnx-f0 --fold 0 --arch unet --encoder tu-convnext_tiny --steps 24000 --workers 5

Recorded arguments (``configs/<name>.json``, byte copies of the runs' ``args.json``; all with bs 8, crop 512,
warmup 500, lr_enc 2e-4, lr_dec 1e-3, seed 0):
  cnx-f0..3   --arch unet --encoder tu-convnext_tiny --steps 24000 --workers 5   (3,943-5,146 s per fold, RTX 4070 SUPER)
  cnxL-f0..3  --arch unet --encoder tu-convnext_tiny --steps 48000 --workers 5   (8,891-9,221 s per fold)
  r34-f0..3   --arch unet --encoder resnet34          --steps 30000 --workers 6   (4,316-4,613 s per fold)
The original launch lines went through ``scripts/queue_dense.sh <prefix> unet <encoder> <steps> [--workers 5]``, which
for k = 0..3 ran ``dense.py train --name <prefix>-f<k> --fold <k> ...`` and then the fold-k out-of-fold map; the cnx and
r34 launch lines are unrecorded (reconstructed from the args files), cnxL's is ``scripts/start_cnxL.sh`` L5.

Recipe: fold model k trains on the development images of folds other than k (fold 4, the holdout, is never used).
Each sample is a 512 px crop: 60 % centred on a random instance box centre (jitter SD 77 px), 40 % anywhere on the disk;
rotation (half arbitrary, half multiples of 90 degrees), scale 0.9-1.1, horizontal flip, blur 30 %, JPEG q60-95 30 %,
contrast x0.85-1.15 and offset +-0.15 after normalisation. Loss: BCE + 0.5 soft Dice on the mask, + 0.3 BCE on the
spine. AdamW (weight decay 1e-2), 500 warm-up steps then cosine to 1 %, bf16 autocast, gradient-norm clip 1. An EMA of
the weights (decay 0.999) is saved every 5,000 steps and at the end; the final ``ema.pt`` is the model (no validation,
no checkpoint selection). Training is not bit-reproducible (no deterministic kernels were requested).

Changes from the original run
-----------------------------
The code is ``src/cfil/dense.py`` L29-120 (``Crops``, ``dice_loss``, ``train``) and the ``train`` argument parser
(L172-178), with ``build_model`` and ``normalize`` imported from ``filseg.dense``.
- No host guard (``from cfil import guard``, L4) and no ``sys.path`` insert (L3).
- Paths: ``--cache`` (default ``<FILSEG_WORK>/train_cache``, written by ``training/prep_train.py``) and ``--runs``
  (default ``<FILSEG_WORK>/runs``) replace ``ROOT``/``CACHE`` (L8-9, default: the training host's project folder).
- ``--config <name>`` (new) reads every argument from ``configs/<name>.json``.
- Files are opened with ``with`` blocks; the log file is closed at the end.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from filseg import paths
from filseg.dense import build_model, normalize

CACHE = str(paths.WORK / 'train_cache')


class Crops(torch.utils.data.Dataset):
    def __init__(self, idx, meta, crop, length, seed, cache=None):
        self.idx, self.meta, self.crop, self.length, self.seed = list(idx), meta, crop, length, seed
        self.cache = cache or CACHE
        with open(os.path.join(self.cache, 'centers.json'), encoding='utf-8') as fh:
            ann = json.load(fh)
        self.centers = [(i, c) for i in self.idx for c in ann[str(i)]]
        self.mm = None

    def __len__(self):
        return self.length

    def _open(self):
        self.mm = {k: np.load(os.path.join(self.cache, f'train_{k}.npy'), mmap_mode='r') for k in ('img', 'soft', 'spine')}

    def __getitem__(self, k):
        if self.mm is None:
            self._open()
        rng = np.random.default_rng((self.seed, k))
        if rng.random() < 0.6:
            i, (cx, cy) = self.centers[rng.integers(len(self.centers))]
            cx += rng.normal(0, self.crop * 0.15); cy += rng.normal(0, self.crop * 0.15)
        else:
            i = self.idx[rng.integers(len(self.idx))]
            r = 965 * math.sqrt(rng.random()); t = rng.random() * 2 * math.pi
            cx, cy = 1024 + r * math.cos(t), 1024 + r * math.sin(t)
        m = self.meta[i]
        ang = rng.uniform(0, 360) if rng.random() < 0.5 else 90 * rng.integers(4)
        sc = rng.uniform(0.9, 1.1)
        C = self.crop
        half = int(C * 0.75 / sc) + 4
        x0, y0 = int(round(cx)) - half, int(round(cy)) - half
        x0 = min(max(x0, 0), 2048 - 2 * half); y0 = min(max(y0, 0), 2048 - 2 * half)
        sl = (i, slice(y0, y0 + 2 * half), slice(x0, x0 + 2 * half))
        img, soft, sp = (np.asarray(self.mm[k][sl]) for k in ('img', 'soft', 'spine'))
        M = cv2.getRotationMatrix2D((half, half), ang, sc)
        M[:, 2] += C / 2 - half
        warp = lambda a: cv2.warpAffine(a, M, (C, C), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)  # noqa: E731
        img, soft, sp = warp(img), warp(soft), warp(sp)
        if rng.random() < 0.5:
            img, soft, sp = img[:, ::-1], soft[:, ::-1], sp[:, ::-1]
        if rng.random() < 0.3:
            img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.2))
        if rng.random() < 0.3:
            ok, enc = cv2.imencode('.jpg', np.ascontiguousarray(img), [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(60, 96))])
            img = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
        x = normalize(img, m['disk_mean'], m['disk_std'])
        x = x * rng.uniform(0.85, 1.15) + rng.uniform(-0.15, 0.15)
        y = np.stack([soft, sp]).astype(np.float32) / 255
        return torch.from_numpy(np.ascontiguousarray(x[None])), torch.from_numpy(np.ascontiguousarray(y))


def dice_loss(logit, target):
    p = torch.sigmoid(logit)
    inter = (p * target).sum((1, 2)); den = (p + target).sum((1, 2))
    return (1 - (2 * inter + 1) / (den + 1)).mean()


def train(a):
    with open(os.path.join(a.cache, 'meta.json'), encoding='utf-8') as fh:
        meta = json.load(fh)['train']
    idx = [i for i, m in enumerate(meta) if m['fold'] != 4 and m['fold'] != a.fold]
    out = os.path.join(a.runs, a.name); os.makedirs(out, exist_ok=True)
    rec = {k: v for k, v in vars(a).items() if k not in ('cache', 'runs', 'config')}
    with open(os.path.join(out, 'args.json'), 'w', encoding='utf-8') as fh:
        json.dump(rec | {'n_train_images': len(idx)}, fh, indent=1)
    torch.manual_seed(a.seed)
    dev = 'cuda'
    model = build_model(a.arch, a.encoder).to(dev).to(memory_format=torch.channels_last)
    ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
    enc = [p for n, p in model.named_parameters() if n.startswith('encoder.')]
    dec = [p for n, p in model.named_parameters() if not n.startswith('encoder.')]
    opt = torch.optim.AdamW([{'params': enc, 'lr': a.lr_enc}, {'params': dec, 'lr': a.lr_dec}], weight_decay=1e-2)
    base = [a.lr_enc, a.lr_dec]
    ds = Crops(idx, meta, a.crop, a.steps * a.bs, a.seed, a.cache)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, num_workers=a.workers, persistent_workers=True, prefetch_factor=4)
    t0 = time.time(); log = open(os.path.join(out, 'log.txt'), 'a', encoding='utf-8')
    model.train()
    for step, (x, y) in enumerate(dl, 1):
        f = step / a.warmup if step < a.warmup else 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (step - a.warmup) / (a.steps - a.warmup)))
        for g, b in zip(opt.param_groups, base):
            g['lr'] = b * f
        x = x.to(dev, non_blocking=True).to(memory_format=torch.channels_last); y = y.to(dev, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            o = model(x).float()
        loss = F.binary_cross_entropy_with_logits(o[:, 0], y[:, 0]) + 0.5 * dice_loss(o[:, 0], y[:, 0]) \
            + 0.3 * F.binary_cross_entropy_with_logits(o[:, 1], y[:, 1])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); ema.update_parameters(model)
        if step % 200 == 0:
            msg = f'step {step} loss {loss.item():.4f} lr {f:.3f} {(time.time()-t0)/step:.3f}s/it mem {torch.cuda.max_memory_allocated()/2**30:.2f}G'
            print(msg, flush=True); log.write(msg + '\n'); log.flush()
        if step % 5000 == 0 or step == a.steps:
            torch.save(ema.module.state_dict(), os.path.join(out, 'ema.pt'))
    log.close()
    with open(os.path.join(out, 'done.json'), 'w', encoding='utf-8') as fh:
        json.dump({'done': True, 'sec': time.time() - t0}, fh)


def parse(argv=None):
    t = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    t.add_argument('--config', default='', help='read every argument from configs/<config>.json (e.g. cnx-f0)')
    t.add_argument('--name'); t.add_argument('--fold', type=int)
    t.add_argument('--arch', default='unet'); t.add_argument('--encoder', default='resnet34')
    t.add_argument('--steps', type=int, default=20000); t.add_argument('--bs', type=int, default=8)
    t.add_argument('--crop', type=int, default=512); t.add_argument('--warmup', type=int, default=500)
    t.add_argument('--lr_enc', type=float, default=2e-4); t.add_argument('--lr_dec', type=float, default=1e-3)
    t.add_argument('--workers', type=int, default=6); t.add_argument('--seed', type=int, default=0)
    t.add_argument('--cache', default=CACHE); t.add_argument('--runs', default=str(paths.WORK / 'runs'))
    a = t.parse_args(argv)
    if a.config:
        with open(paths.CONFIGS / f'{a.config}.json', encoding='utf-8') as fh:
            rec = json.load(fh)
        for k in ('name', 'fold', 'arch', 'encoder', 'steps', 'bs', 'crop', 'warmup', 'lr_enc', 'lr_dec', 'workers', 'seed'):
            setattr(a, k, rec[k])
    if a.name is None or a.fold is None:
        t.error('--name and --fold (or --config) are required')
    a.cmd = 'train'
    return a


if __name__ == '__main__':
    train(parse())
