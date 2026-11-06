"""Test evidence features (stage 5a): tree candidates of the cnx map and their overlap with external candidates.

For every test image this module writes, under a work folder:

- ``cands/testcnx/<stem>.npz``: the component-tree candidates of the cnx mask map. A candidate is a connected
  component (8-connectivity) of ``P > round(255 t)`` for t in ``LEVELS``; components nest across levels and form a
  forest. ``F[N, 28]`` holds the ``FEATURES`` of each candidate and ``parents[N]`` its parent (-1 for a root).
- ``ext/testcnx-<source>/<stem>.npz``: stacking features against one external candidate source (the Codex I6 bank, or
  the roots of one Mask R-CNN fold model): ``X[N, 6]`` (``XFEAT``), the external candidate table ``E[M, 8]``
  (``EFEAT``), the IoU matrix ``ov[N, M]`` (float16), and the external masks ``ext_rle[M]`` and selected set
  ``ext_sel`` as COCO RLE count strings.
- ``feat/<rich name>/<stem>.npz``: ``X[N, 11]`` (``RFEAT``), features computed from ``ov`` and ``E`` only.

The I6 bank's own selector is not used: its selected set is replaced by a disjoint greedy set (``greedy_sel``). The
detector roots keep their stored ``set_selector_pq`` set. Everything runs on the CPU.

Changes from the original run
-----------------------------
The arithmetic, its order, the dtypes and the thresholds are those of ``src/cfil/cands.py``,
``src/cfil/build_cands.py``, ``src/cfil/ext.py`` and ``src/cfil/rich.py`` (commit 133e77d), test side only.
- One pass per image instead of one script per output folder (``work/run_final_feats.sh`` L11-20): the tree
  candidates and their RLEs are computed once and shared by ``cands``, every ``ext`` source and every ``rich`` file.
  ``ext.py`` L76-78 recomputed the same ``candidates(...)`` call for each source, and ``rich.py`` L50 read ``F`` and the
  float16 ``ov`` back from disk; here ``rich_features`` gets the same float32 ``F`` and the same float16 ``ov``.
- Removed the training branches: annotations, split file and per-record hit IoUs (``cands.py`` L55-67, L123-137;
  ``build_cands.py`` L23-29, L49-55; ``ext.py`` L55-60, L97-108) and ``tree_select``/``materialize`` (L165-177,
  L195-222; not on the test path).
- Not written any more, because nothing on the test path reads them: ``reg`` (``region_of``, ``build_cands.py`` L56,
  ``cands.py`` L38-48) and the empty training placeholders ``hits``/``ngt`` in ``cands``, ``Eh`` in ``ext``.
- ``CFIL_CLOSE_R`` is gone (``cands.py`` L20, L28-35): the run used 0, so a level is the plain thresholded map.
- ``materialize_one``'s per-process label cache keyed by ``id(P)`` (``cands.py`` L180-192, ``ext.py`` L77) is replaced
  by a per-image dict of level label maps (``tree_rles``); the RLEs are the same.
- Paths come from arguments (default ``filseg.paths``) instead of ``PARENT``/``RAW`` derived from the file location
  (``build_cands.py`` L12-15); ``sys.path`` inserts and the ``CUDA_VISIBLE_DEVICES`` import side effect are gone.
- Outputs are always rewritten; ``build_cands.py`` L41/L65 and ``ext.py`` L66/L120 skipped existing files. An image that
  a source does not cover raises ``KeyError`` (``ext.py`` L120 skipped it, which made the row stage fail later).
- Worker processes: at most 4 (``NPROC`` defaulted to 14, ``build_cands.py`` L68).
"""
from __future__ import annotations

import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu
from scipy import ndimage as ndi

from filseg import paths

H = W = 2048
LEVELS = np.round(np.arange(0.10, 0.851, 0.05), 3)
MIN_AREA = 40
_RING = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
MAX_WORKERS = 4

FEATURES = ['t', 'lvl', 'log_area', 'bw', 'bh', 'aspect', 'fill', 'p_mean', 'p_max', 'p_std', 'p_edge', 'persist',
            'par_ratio', 'n_child', 'depth_up', 's_mean', 's_max', 'contrast', 'i_mean', 'r_disk',
            'major', 'minor', 'width', 'elong', 'img_n50', 'img_pmean', 'sib_n', 'rank_in_img']
XFEAT = ['x_best_iou', 'x_best_score', 'x_best_det', 'x_score_ov50', 'x_n_ov30', 'x_cov_sel']
EFEAT = ['e_score', 'e_det', 'e_mprob', 'e_log_area', 'e_rho', 'e_thr', 'e_best_mine_iou', 'e_rank']
RFEAT = ['r_ov_score_max', 'r_ov_score_sum', 'r_score_ov3050', 'r_score_ov1030', 'r_best_mprob', 'r_cov_mine',
         'r_cov_ext', 'r_inside_score', 'r_inside_n', 'r_best_rank', 'r_is_best_match']

CANDS_DIR = 'cands/testcnx'


def _sig(z):
    return 1 / (1 + math.exp(-z))


# ----------------------------------------------------------------------------------------------- output folder layout

def test_sources(i6_root, det_roots) -> dict:
    """The external sources of the final, in the original folder names.

    Returns ``{source name: (candidate root, greedy selection, ext folder, rich folder)}``, folders relative to the work
    folder. ``det_roots`` lists the four detector roots ``det/det-f{k}/test`` in fold order.
    """
    out = {'i6g': (str(i6_root), True, 'ext/testcnx-i6g', 'feat/testcnx-i6g-rich')}
    for k, root in enumerate(det_roots):
        out[f'det-f{k}'] = (str(root), False, f'ext/testcnx-det-f{k}', f'feat/testcnx-detrich-f{k}')
    return out


# ------------------------------------------------------------------------------------------------- tree candidates

def disk_stats(img):
    """Mean and std of the solar-disk pixels (above 30% of the 99th percentile) of a uint8 image."""
    thr = 0.3 * np.percentile(img, 99)
    v = img[img > thr].astype(np.float32)
    return float(v.mean()), float(v.std() + 1e-6)


def read_image(image_dir, fname):
    """The test JPEG as a uint8 [2048, 2048] array, decoded by OpenCV as in the original run."""
    img = cv2.imread(os.path.join(str(image_dir), fname), cv2.IMREAD_GRAYSCALE)
    if img is None or img.shape != (H, W):
        raise ValueError(f'cannot read a 2048 x 2048 image: {fname}')
    return img


def level_labels(P, t):
    """(n, label map) of the components of P > round(255 t) (label 0 = background)."""
    B = (P > round(t * 255)).astype(np.uint8)
    return cv2.connectedComponents(B, connectivity=8)


def candidates(P, S, img, disk_mean, disk_std):
    """Tree candidates of the uint8 mask map P, spine map S and image img.

    Returns ``(F[N, 28] float32, parents[N] int64, keys)``; ``keys[i] = (level index, component label, bbox slices)``.
    Candidates are ordered by level, then by component label, so a child always comes after its parent.
    """
    Pf = P.astype(np.float32) / 255
    I = (img.astype(np.float32) - disk_mean) / disk_std
    rows, parents, keys = [], [], []
    prev_lab, prev_ids = None, None
    for li, t in enumerate(LEVELS):
        n, lab = level_labels(P, t)
        st = np.zeros((n, 5), np.int64); st[:, cv2.CC_STAT_AREA] = np.bincount(lab.ravel(), minlength=n)
        ids = np.full(n, -1, np.int64)
        slices = ndi.find_objects(lab)
        for k in range(1, n):
            area = st[k, cv2.CC_STAT_AREA]
            if area < MIN_AREA:
                continue
            sl = slices[k - 1]
            m = lab[sl] == k
            pv = Pf[sl][m]
            par = -1
            if prev_lab is not None:
                pl = prev_lab[sl][m][0]
                par = prev_ids[pl] if pl > 0 else -1
            y0, x0 = sl[0].start, sl[1].start
            bh, bw = m.shape
            # boundary and ring
            mu8 = m.astype(np.uint8)
            er = cv2.erode(mu8, None)
            edge = (mu8 - er).astype(bool)
            pad = 8
            ys, xs = max(y0 - pad, 0), max(x0 - pad, 0)
            ye, xe = min(y0 + bh + pad, H), min(x0 + bw + pad, W)
            big = np.zeros((ye - ys, xe - xs), np.uint8)
            big[y0 - ys:y0 - ys + bh, x0 - xs:x0 - xs + bw] = mu8
            ring = cv2.dilate(big, _RING).astype(bool) & ~big.astype(bool)
            iv = I[sl][m]
            iring = I[ys:ye, xs:xe][ring]
            cy, cx = np.nonzero(m)
            cy = cy + y0; cx = cx + x0
            cov = np.cov(np.stack([cx, cy]).astype(np.float64)) if area > 2 else np.eye(2)
            ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
            major = 4 * np.sqrt(max(ev[0], 1e-6)); minor = 4 * np.sqrt(max(ev[1], 1e-6))
            r = np.hypot(cx.mean() - 1024, cy.mean() - 1024) / 965
            sv = S[sl][m].astype(np.float32) / 255
            rows.append([t, li, np.log(area), bw, bh, max(bw, bh) / max(min(bw, bh), 1), area / (bw * bh),
                         pv.mean(), pv.max(), pv.std(), Pf[sl][edge].mean() if edge.any() else pv.mean(),
                         pv.max() - t, 0.0, 0, 0, sv.mean(), sv.max(),
                         (iring.mean() - iv.mean()) if iring.size else 0.0, iv.mean(), r,
                         major, minor, area / max(major, 1), major / max(minor, 1), 0, 0, 0, 0])
            ids[k] = len(rows) - 1
            parents.append(par)
            keys.append((li, k, sl))
        prev_lab, prev_ids = lab, ids
    F = np.asarray(rows, np.float32).reshape(-1, len(FEATURES))
    parents = np.asarray(parents, np.int64)
    _tree_columns(F, parents, Pf)
    return F, parents, keys


def _tree_columns(F, parents, Pf):
    """Fill the columns that need the whole forest (par_ratio, n_child, depth_up, image-level and sibling columns)."""
    N = len(F)
    if not N:
        return
    area = np.exp(F[:, 2])
    nchild = np.bincount(parents[parents >= 0], minlength=N)
    F[:, 13] = nchild
    pr = np.where(parents >= 0, area / np.exp(F[np.maximum(parents, 0), 2]), 1.0)
    F[:, 12] = pr
    depth = np.zeros(N)
    for i in range(N):
        if parents[i] >= 0:
            depth[i] = depth[parents[i]] + 1
    F[:, 14] = depth
    F[:, 24] = np.sum((F[:, 0] == 0.5))
    F[:, 25] = Pf.mean()
    # siblings: number of components sharing the parent
    sib = np.bincount(np.where(parents >= 0, parents, N), minlength=N + 1)
    F[:, 26] = np.where(parents >= 0, sib[np.where(parents >= 0, parents, N)], 0)
    order = np.argsort(-F[:, 7])
    rk = np.empty(N); rk[order] = np.arange(N) / max(N - 1, 1)
    F[:, 27] = rk


def tree_rles(P, keys):
    """COCO RLE (Fortran order) of every candidate, from the level label maps of P."""
    labs = {}
    out = []
    for li, k, sl in keys:
        if li not in labs:
            labs[li] = level_labels(P, LEVELS[li])[1]
        m = np.zeros((H, W), np.uint8)
        m[sl] = labs[li][sl] == k
        out.append(mu.encode(np.asfortranarray(m)))
    return out


# ------------------------------------------------------------------------------------------- external candidates

def load_roots(root):
    """{image file name: record path} of the clean records in a candidate root's ``manifest.json``."""
    with open(os.path.join(root, 'manifest.json'), encoding='utf-8') as fh:
        records = json.load(fh)['records']
    return {rec['file_name']: os.path.join(root, rec['file']) for rec in records if rec.get('kind', 'clean') == 'clean'}


def greedy_sel(ext, score):
    """Disjoint greedy set: by descending score >= 0.5, keep the unclaimed part of a mask if it is >= 70% of the mask
    and >= 20 px. Returns RLE dicts with str counts."""
    taken = np.zeros((2048, 2048), bool); out = []
    for i in np.argsort(-score):
        if score[i] < 0.5:
            break
        m = mu.decode(ext[i]).astype(bool); free = m & ~taken
        if free.sum() >= 0.7 * m.sum() and free.sum() >= 20:
            taken |= free
            e = mu.encode(np.asfortranarray(free.astype(np.uint8)))
            out.append({'size': list(e['size']), 'counts': e['counts'].decode()})
    return out


def ext_features(mine, record_path, greedy):
    """Stacking features of the tree candidates (RLEs ``mine``) against one external record file.

    Returns the arrays of the ``ext`` file: X[N, 6], E[M, 8], ov[N, M] float16, ext_rle[M], ext_sel (object arrays).
    """
    with open(record_path, encoding='utf-8') as fh:
        r = json.load(fh)
    (thr, rows), = r['views'].items()
    ext = [{'size': c['rle']['size'], 'counts': c['rle']['counts'].encode()} for c in rows]
    sel = greedy_sel(ext, np.asarray([c['score'] for c in rows], np.float32)) if greedy else \
        r['predictions'].get('set_selector_pq', [])
    N, M = len(mine), len(ext)
    ov = np.asarray(mu.iou(mine, ext, [0] * M), np.float32).reshape(N, M) if N and M else np.zeros((N, M), np.float32)
    score = np.asarray([c['score'] for c in rows], np.float32)
    det = np.asarray([_sig(c['features'][1]) for c in rows], np.float32)
    selr = [{'size': s['size'], 'counts': s['counts'].encode()} for s in sel]
    X = np.zeros((N, len(XFEAT)), np.float32)
    if N and M:
        b = ov.argmax(1)
        X[:, 0] = ov.max(1); X[:, 1] = score[b]; X[:, 2] = det[b]
        X[:, 3] = np.where(ov > 0.5, score[None], 0).max(1)
        X[:, 4] = (ov > 0.3).sum(1)
    if N and selr:
        X[:, 5] = np.asarray(mu.iou(mine, selr, [0] * len(selr))).reshape(N, -1).max(1)
    E = np.zeros((M, len(EFEAT)), np.float32)
    order = np.argsort(-score); rk = np.empty(M); rk[order] = np.arange(M) / max(M - 1, 1)
    for i, c in enumerate(rows):
        E[i] = [c['score'], det[i], _sig(c['features'][2]), math.log(max(c['area'], 1)), c['rho'], float(thr),
                ov[:, i].max() if N else 0, rk[i]]
    return dict(X=X, E=E, ov=ov.astype(np.float16), ext_rle=np.asarray([c['rle']['counts'] for c in rows], object),
                ext_sel=np.asarray([s['counts'] for s in sel], object))


def rich_features(F, ov, E):
    """X[N, 11] (RFEAT) from the tree features F (FEATURES order), IoU ov[N, M] and external table E (EFEAT order)."""
    N = len(F)
    out = np.zeros((N, len(RFEAT)), np.float32)
    if not N or ov is None or ov.shape[1] == 0:
        return out
    ov = ov.astype(np.float32)
    sc = E[:, 0]; A = np.exp(F[:, 2])[:, None]; B = np.exp(E[:, 3])[None]
    inter = ov * (A + B) / (1 + ov)
    b = ov.argmax(1); r = np.arange(N)
    out[:, 0] = (ov * sc[None]).max(1)
    out[:, 1] = (ov * sc[None]).sum(1)
    out[:, 2] = np.where((ov > 0.3) & (ov <= 0.5), sc[None], 0).max(1)
    out[:, 3] = np.where((ov > 0.1) & (ov <= 0.3), sc[None], 0).max(1)
    out[:, 4] = E[b, 2]
    out[:, 5] = inter[r, b] / A[:, 0]
    out[:, 6] = inter[r, b] / B[0, b]
    inside = inter / B > 0.8
    out[:, 7] = np.where(inside, sc[None], 0).sum(1)
    out[:, 8] = inside.sum(1)
    out[:, 9] = E[b, 7]
    out[:, 10] = (np.abs(ov[r, b] - E[b, 6]) < 1e-3).astype(np.float32)
    return out


# ------------------------------------------------------------------------------------------------------- driver

_G = {}


def _init(maps_path, sources, image_dir, work_root):
    _G['maps'] = np.load(maps_path, mmap_mode='r')
    _G['records'] = {name: load_roots(src[0]) for name, src in sources.items()}
    _G['sources'] = sources
    _G['image_dir'] = image_dir
    _G['work'] = Path(work_root)


def _one(job):
    j, fname = job
    stem = fname[:-5]
    work = _G['work']
    m = np.asarray(_G['maps'][j])
    img = read_image(_G['image_dir'], fname)
    dm, ds = disk_stats(img)
    F, par, keys = candidates(m[0], m[1], img, dm, ds)
    np.savez_compressed(work / CANDS_DIR / f'{stem}.npz', F=F, parents=par)
    mine = tree_rles(m[0], keys)
    for name, (_, greedy, ext_dir, rich_dir) in _G['sources'].items():
        e = ext_features(mine, _G['records'][name][fname], greedy)
        np.savez_compressed(work / ext_dir / f'{stem}.npz', **e)
        np.savez_compressed(work / rich_dir / f'{stem}.npz', X=rich_features(F, e['ov'], e['E']))
    return fname, len(F)


def build_evidence(maps_path, sources, work_root=None, image_dir=None, nproc=MAX_WORKERS, limit=None):
    """Write the cands / ext / rich files of every image listed in ``<maps_path>.json``.

    maps_path: the cnx test maps ``(n, 2, 2048, 2048)`` uint8 (channel 0 mask, channel 1 spine);
    sources: ``test_sources(...)``; work_root: default ``filseg.paths.WORK``; image_dir: default
    ``filseg.paths.TEST_IMAGES``. Returns the number of tree candidates per image.
    """
    work_root = Path(work_root or paths.WORK)
    image_dir = str(image_dir or paths.TEST_IMAGES)
    with open(str(maps_path) + '.json', encoding='utf-8') as fh:
        files = json.load(fh)['files']
    jobs = list(enumerate(files))[:limit]
    for d in [CANDS_DIR] + [s[2] for s in sources.values()] + [s[3] for s in sources.values()]:
        (work_root / d).mkdir(parents=True, exist_ok=True)
    counts = {}
    with ProcessPoolExecutor(min(nproc, MAX_WORKERS), initializer=_init,
                             initargs=(str(maps_path), sources, image_dir, str(work_root))) as ex:
        for n, (fname, nc) in enumerate(ex.map(_one, jobs, chunksize=2)):
            counts[fname] = nc
            if n % 20 == 0:
                print('evidence', n, fname, 'candidates', nc, flush=True)
    return counts
