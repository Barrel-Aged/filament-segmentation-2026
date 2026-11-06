"""Training cache for the dense U-Nets and the Mask R-CNN (the training half of the original ``prep.py``). CPU only.

usage (repository root):  python -m training.prep_train [--cache DIR] [--data DIR] [--split configs/split-v2.json]

Writes into ``--cache`` (default ``<FILSEG_WORK>/train_cache``):
  train_img.npy   (N, 2048, 2048) uint8  the grayscale training images, in file-name order of the split manifest
  train_soft.npy  (N, 2048, 2048) uint8  round(255 * mean over the image's annotator records of the record's filled mask)
  train_spine.npy (N, 2048, 2048) uint8  the same mean over 3 px spine polylines
  meta.json       {'train': [{file_name, fold, group, records, disk_mean, disk_std, disk_thr}]}
  centers.json    {image index: [(x, y) box centre of every instance of every record]}
The folds come from ``configs/split-v2.json`` (folds 0-3 = development, 4 = holdout; UTC-day groups). N = 707.
The three arrays need about 8.9 GB of disk.

Changes from the original run
-----------------------------
The arithmetic is that of ``src/cfil/prep.py`` (``disk_stats`` L18-22, ``record_masks`` L25-37, ``build_one`` L40-51,
the training half of ``main`` L54-73, L82-84, L88-97).
- Paths are arguments (default ``filseg.paths``) instead of ``<root>/data/MAGFiLO_1.0_Kaggle_2026``,
  ``<root>/data/annotations/...`` and ``<root>/claude/configs/split-v2.json`` built from ``sys.argv[1]`` (L55-57, L62).
  The competition's ``train/MAGFiLO_1.0_Annotations_kaggle2026_train.json`` is byte-identical to the
  ``data/annotations`` copy the run read.
- The test half (L74-81 and the ``test`` key of meta.json, L85) is ``filseg.data.write_test_cache``; this meta.json
  holds only the ``train`` key, which is all the trainers read.
- ``ProcessPoolExecutor(8)`` is now ``--workers`` (default 8). Files are opened with ``with`` blocks.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

from filseg import paths
from filseg.data import disk_stats

H = W = 2048


def record_masks(anns):
    """Union mask and spine raster for one annotator record."""
    full = np.zeros((H, W), np.uint8)
    spine = np.zeros((H, W), np.uint8)
    for a in anns:
        segs = json.loads(a['segmentation']) if isinstance(a['segmentation'], str) else a['segmentation']
        rles = mu.frPyObjects([s for s in segs if len(s) >= 6], H, W)
        full |= mu.decode(mu.merge(rles)).astype(np.uint8)
        sp = json.loads(a['spine']) if isinstance(a['spine'], str) else a['spine']
        if sp and len(sp) >= 4:
            pts = np.round(np.asarray(sp, np.float64).reshape(-1, 2)).astype(np.int32)
            cv2.polylines(spine, [pts], False, 1, thickness=3)
    return full, spine


def build_one(args):
    path, recs = args
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    assert img.shape == (H, W), path
    soft = np.zeros((H, W), np.float32)
    sp = np.zeros((H, W), np.float32)
    for anns in recs:
        f, s = record_masks(anns)
        soft += f
        sp += s
    n = max(len(recs), 1)
    return img, np.round(soft / n * 255).astype(np.uint8), np.round(sp / n * 255).astype(np.uint8), disk_stats(img)


def main(cache, data, split_path, workers=8):
    with open(data / 'train' / 'MAGFiLO_1.0_Annotations_kaggle2026_train.json', encoding='utf-8') as fh:
        ann = json.load(fh)
    with open(split_path, encoding='utf-8') as fh:
        split = json.load(fh)
    by_rec = {}
    for a in ann['annotations']:
        by_rec.setdefault(a['image_id'], []).append(a)
    obs = sorted(split['observations'], key=lambda o: o['file_name'])
    os.makedirs(cache, exist_ok=True)
    N = len(obs)
    mm = {k: np.lib.format.open_memmap(os.path.join(cache, f'train_{k}.npy'), 'w+', np.uint8, (N, H, W))
          for k in ('img', 'soft', 'spine')}
    jobs = [(os.path.join(data, 'train', 'train_images', o['file_name']), [by_rec[r] for r in o['annotator_image_ids']])
            for o in obs]
    stats = []
    with ProcessPoolExecutor(workers) as ex:
        for i, (img, soft, sp, st) in enumerate(ex.map(build_one, jobs, chunksize=4)):
            mm['img'][i], mm['soft'][i], mm['spine'][i] = img, soft, sp
            stats.append(st)
    for m in mm.values():
        m.flush()
    meta = {'train': [dict(file_name=o['file_name'], fold=o['fold'], group=o['group_id'], records=o['annotator_image_ids'],
                           disk_mean=s[0], disk_std=s[1], disk_thr=s[2]) for o, s in zip(obs, stats)]}
    with open(os.path.join(cache, 'meta.json'), 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, indent=1)
    centers = {}
    for i, o in enumerate(obs):
        cs = []
        for r in o['annotator_image_ids']:
            for a in by_rec[r]:
                x, y, w, h = json.loads(a['bbox']) if isinstance(a['bbox'], str) else a['bbox']
                cs.append((x + w / 2, y + h / 2))
        centers[str(i)] = cs
    with open(os.path.join(cache, 'centers.json'), 'w', encoding='utf-8') as fh:
        json.dump(centers, fh)
    print('train', N)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, default=paths.WORK / 'train_cache')
    p.add_argument('--data', type=Path, default=paths.DATA)
    p.add_argument('--split', type=Path, default=paths.CONFIGS / 'split-v2.json')
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    main(str(a.cache), a.data, a.split, a.workers)
