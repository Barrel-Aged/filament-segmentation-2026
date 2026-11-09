"""Detector candidate-root records (torch-free, shared by det.py inference and det_rebin.py)."""
import math
import numpy as np
from pycocotools import mask as mu

H = W = 2048


def _rle(m):
    r = mu.encode(np.asfortranarray(m.astype(np.uint8)))
    return {'size': [H, W], 'counts': r['counts'].decode()}


def build_record(masks, sc):
    """masks: iterable of (binary mask HxW, mean probability inside it) per detection in score order; sc: scores.
    Returns (rows, selected) in the candidate-root JSON layout read by ext.py."""
    lg = lambda p: math.log(max(p, 1e-6) / max(1 - p, 1e-6))
    rows, kept = [], []
    for j, (bm, mp) in enumerate(masks):
        area = int(bm.sum())
        if area < 20:
            continue
        ys, xs = np.nonzero(bm)
        rho = float(np.hypot(xs.mean() - 1024, ys.mean() - 1024) / 965)
        rows.append({'rle': _rle(bm), 'score': float(sc[j]), 'features': [0.0, lg(float(sc[j])), lg(mp)], 'area': area, 'rho': rho})
        kept.append((j, bm))
    # disjoint "selected" set: greedy by score above 0.5, dropping overlaps with pixels already taken
    taken = np.zeros((H, W), bool); sel = []
    for j, bm in kept:
        if sc[j] < 0.5:
            break
        mm = bm & ~taken
        if mm.sum() >= 0.7 * bm.sum() and mm.sum() >= 20:
            taken |= mm; sel.append(_rle(mm))
    return rows, sel
