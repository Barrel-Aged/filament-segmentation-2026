"""Candidate rows of the combS pool (stage 5b): every candidate mask of a test image with its 74 + 18 features.

The pool of one image, in row order:

- fam 0, tree rows: the cnx component-tree candidates (``filseg.evidence.candidates``), first;
- fam 2, closing merges: components of ``closing(P > t, disk r)``, r in 6 / 12, that join >= 2 tree nodes of level t;
- fam 4, pairwise merges: two same-level tree nodes within 32 px joined by a straight bridge;
- fam 5 / fam 6: the tree nodes of the cnxL / R34 maps, with features computed on their own map.

New rows (fam != 0) smaller than 40 px or with exactly the pixels of an earlier new row are not added; new rows with
IoU >= 0.9 to a tree row or to an earlier kept new row are dropped.

Each image gives ``rows/<stem>.npz`` with
  F[4, N, 74]   one block per detector fold model k: 28 tree features | I6 X 6 | I6 rich 11 | det X 6 | det rich 11 |
                d_i6 | d_det | one-hot (mg, pm, cnxL, R34) | 6 family extras. Only columns 45-61 and 63 differ across k.
  xm[N, 18]     6 map-agreement columns against the R34 | cnxL | cnx maps (``xm_block``)
  fam[N] int8, parents[N] (-1 for new rows), conf[M, 2] (pairs that may not be selected together), nb (tree rows),
  lvl[N] (level index), area[N], bt[N] (new rows: best IoU to a tree row), pairs / piou (new-new pairs with
  IoU >= 0.5), ext_selfcheck (max |difference| between the tree rows' evidence recomputed from pixel sets and the
  stored evidence files), nraw (cnxL and R34 tree sizes before deduplication), set_flat / set_len (the row pixel
  sets: sorted flat indices y * 2048 + x, see ``pack_sets``).

Changes from the original run
-----------------------------
The arithmetic, its order, the dtypes and the thresholds are those of ``work/ideas8/bundle-hold/build_rows.py``
(``gen`` L41-235, ``xm18`` L238-245), ``work/ideas8/bundle-test/rowgen.py`` (``_gen``, ``one_test`` L40-106),
``work/ideas7/playbook/xcands.py`` (``THR``, ``K``, ``bbox``, ``feat_row`` L30-82), ``work/ideas7/playbook/xcands2.py``
(``csr_of``, ``iou_sparse``, ``rle_sets``, ``ext_block``, ``base_delta`` L36-87), ``work/ideas7/playbook/pool_oracle.py``
(``tree_sets`` L36-44), ``work/ideas7/exp/p1_poolxm/xm_rows.py`` (``level_stack``, ``xm_block`` L47-81) and
``work/ideas8/bundle-test/bt_common.py`` (``pack_sets``, ``unpack_sets``, ``rle_of_set``, ``rasterize`` L24-55).
- Label-free only: the ``gts`` argument and the hit-IoU branches are gone (``build_rows.py`` L148, L152, L174,
  L184-187, L231-234; ``xcands.py`` ``hits_of`` L85-94). The run called ``gen(..., gts=None)`` (``rowgen.py`` L50).
- Row files no longer hold ``ok``, ``xmR_maxdiff`` and ``xmL_maxdiff`` (``rowgen.py`` L85-99): they recorded equality
  checks against ``cands/testcnx`` and ``feat/testcnx-xm{R,L}`` (``src/cfil/xmap.py``), which are not inputs of the rows.
  The checks themselves are not done here; ``tools/verify_rows.py`` compares the rows with the original run instead.
  ``gen`` no longer returns the check-only ``F28``/``par`` (``build_rows.py`` L230).
- Paths come from arguments instead of the hard-coded ``ROOT`` (``bt_common.py`` L13, ``build_rows.py`` L21,
  ``xcands.py`` L18, ``xcands2.py`` L14, ``pool_oracle.py`` L10, ``xm_rows.py`` L14); the ``sys.path`` inserts, the
  ``CUDA_VISIBLE_DEVICES`` import side effects, the unused scikit-image imports (``xcands.py`` L22-23,
  ``xcands2.py`` L20-21) and the dev-side output constants are gone.
- Outputs are always rewritten (``rowgen.py`` L80-82 skipped existing row files). At most 4 worker processes.
"""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu
from scipy import ndimage as ndi
from scipy.sparse import csr_matrix

from filseg import paths
from filseg.evidence import FEATURES, LEVELS, MAX_WORKERS, MIN_AREA, candidates, disk_stats, read_image, rich_features

H = W = 2048
THR = [int(round(t * 255)) for t in LEVELS]
_RING = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
_K1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
NE = 6                                 # family extras
GAP = 32                               # pairwise merges: largest gap in px
ONEHOT = {2: 0, 4: 1, 5: 2, 6: 3}      # fam -> one-hot column (mg, pm, cnxL, R34)
PAD = 4                                # xm_block crop margin
MAPS = ('r34', 'cnxL', 'cnx')          # xm block order
NV = 4                                 # detector fold models = F variants
ROWS_DIR = 'rows'


def K(r):
    """Elliptic structuring element of radius r."""
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


# ------------------------------------------------------------------------------------------------ pixel-set helpers

def bbox(Q):
    """(y0, y1, x0, x1), end exclusive, of the flat pixel indices Q."""
    y, x = Q // W, Q % W
    return y.min(), y.max() + 1, x.min(), x.max() + 1


def csr_of(sets, n=H * W):
    """Sparse 0/1 matrix [len(sets), n] of pixel sets, and the set sizes."""
    lens = np.array([len(q) for q in sets], np.int64)
    ind = np.concatenate(sets) if len(sets) else np.zeros(0, np.int64)
    ptr = np.concatenate([[0], np.cumsum(lens)])
    return csr_matrix((np.ones(len(ind), np.float32), ind, ptr), shape=(len(sets), n)), lens


def iou_sparse(Ca, la, Cb, lb):
    """IoU matrix [a, b] (float32) of two ``csr_of`` set lists."""
    if Ca.shape[0] == 0 or Cb.shape[0] == 0:
        return np.zeros((Ca.shape[0], Cb.shape[0]), np.float32)
    inter = (Ca @ Cb.T).toarray()
    return (inter / np.maximum(la[:, None] + lb[None, :] - inter, 1)).astype(np.float32)


def rle_sets(counts):
    """Flat pixel indices of COCO RLE count strings, in Fortran (column-major) order as pycocotools decodes."""
    out = []
    for c in counts:
        m = mu.decode({'size': [H, W], 'counts': c.encode() if isinstance(c, str) else c})
        out.append(np.flatnonzero(m).astype(np.int64))
    return out


def tree_sets(P, keys):
    """Row-major flat pixel indices of the tree candidates ``keys`` of map P."""
    labs = {}
    out = []
    for (li, kk, sl) in keys:
        if li not in labs:
            labs[li] = cv2.connectedComponents((P > THR[li]).astype(np.uint8), connectivity=8)[1]
        ys, xs = np.nonzero(labs[li][sl] == kk)
        out.append(((ys + sl[0].start) * W + xs + sl[1].start).astype(np.int64))
    return out


def pack_sets(sets):
    """Pixel sets -> (int32 flat array, int64 lengths)."""
    lens = np.array([len(q) for q in sets], np.int64)
    flat = np.concatenate([np.asarray(q, np.int64) for q in sets]).astype(np.int32) if len(sets) else np.zeros(0, np.int32)
    assert flat.size == 0 or (flat.min() >= 0 and flat.max() < H * W)
    return flat, lens


def unpack_sets(flat, lens):
    """Inverse of ``pack_sets``: a list of int64 flat index arrays."""
    off = np.concatenate([[0], np.cumsum(lens)])
    return [flat[off[i]:off[i + 1]].astype(np.int64) for i in range(len(lens))]


def rle_of_set(Q):
    """COCO RLE dict (ASCII counts) of one instance given its row-major flat pixel indices."""
    m = np.zeros(H * W, np.uint8)
    m[Q] = 1
    m = m.reshape(H, W)
    e = mu.encode(np.asfortranarray(m.astype(np.uint8)))
    return {'size': list(e['size']), 'counts': e['counts'].decode('ascii')}


def rasterize(sets):
    """RLEs of one image's instances; asserts they are non-empty and pairwise disjoint."""
    occ = np.zeros(H * W, bool)
    out = []
    for Q in sets:
        Q = np.asarray(Q, np.int64)
        assert len(Q) > 0, 'empty instance'
        assert not occ[Q].any(), 'overlapping instances'
        occ[Q] = True
        out.append(rle_of_set(Q))
    return out


# ------------------------------------------------------------------------------------------------ row features

def feat_row(Q, t, li, Pf, I, S, extra_tree):
    """The 28 FEATURES of ``evidence.candidates`` for an arbitrary mask given by flat pixel indices Q."""
    y0, y1, x0, x1 = bbox(Q)
    bh, bw = y1 - y0, x1 - x0
    m = np.zeros((bh, bw), bool); m[Q // W - y0, Q % W - x0] = True
    area = len(Q)
    sl = (slice(y0, y1), slice(x0, x1))
    pv = Pf[sl][m]
    mu8 = m.astype(np.uint8)
    edge = (mu8 - cv2.erode(mu8, None)).astype(bool)
    pad = 8
    ys, xs = max(y0 - pad, 0), max(x0 - pad, 0)
    ye, xe = min(y1 + pad, H), min(x1 + pad, W)
    big = np.zeros((ye - ys, xe - xs), np.uint8)
    big[y0 - ys:y0 - ys + bh, x0 - xs:x0 - xs + bw] = mu8
    ring = cv2.dilate(big, _RING).astype(bool) & ~big.astype(bool)
    iv = I[sl][m]; iring = I[ys:ye, xs:xe][ring]
    cy, cx = Q // W, Q % W
    cov = np.cov(np.stack([cx, cy]).astype(np.float64)) if area > 2 else np.eye(2)
    ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
    major = 4 * np.sqrt(max(ev[0], 1e-6)); minor = 4 * np.sqrt(max(ev[1], 1e-6))
    r = np.hypot(cx.mean() - 1024, cy.mean() - 1024) / 965
    sv = S[sl][m].astype(np.float32) / 255
    par_ratio, n_child, depth_up, sib_n = extra_tree
    return [t, li, np.log(area), bw, bh, max(bw, bh) / max(min(bw, bh), 1), area / (bw * bh),
            pv.mean(), pv.max(), pv.std(), Pf[sl][edge].mean() if edge.any() else pv.mean(),
            pv.max() - t, par_ratio, n_child, depth_up, sv.mean(), sv.max(),
            (iring.mean() - iv.mean()) if iring.size else 0.0, iv.mean(), r,
            major, minor, area / max(major, 1), major / max(minor, 1), 0, 0, sib_n, 0]


def ext_block(Cn, ln, src, Fbase28):
    """Evidence X (6) + rich (11) + best-match delta of arbitrary pixel sets against a stored ``ext`` file, with the
    formulas of ``evidence.ext_features`` / ``evidence.rich_features`` and IoUs from exact pixel sets."""
    z = np.load(src, allow_pickle=True)
    E = z['E']; M = len(E); N = Cn.shape[0]
    Ce, le = csr_of(rle_sets(z['ext_rle'])) if M else (None, None)
    ov = iou_sparse(Cn, ln, Ce, le) if M else np.zeros((N, 0), np.float32)
    Xb = np.zeros((N, 6), np.float32)
    if N and M:
        score, det = E[:, 0], E[:, 1]
        b = ov.argmax(1)
        Xb[:, 0] = ov.max(1); Xb[:, 1] = score[b]; Xb[:, 2] = det[b]
        Xb[:, 3] = np.where(ov > 0.5, score[None], 0).max(1)
        Xb[:, 4] = (ov > 0.3).sum(1)
    if N and len(z['ext_sel']):
        Cs, ls = csr_of(rle_sets(z['ext_sel']))
        Xb[:, 5] = iou_sparse(Cn, ln, Cs, ls).max(1)
    R = rich_features(Fbase28, ov.astype(np.float16), E)
    d = (ov.max(1) - E[ov.argmax(1), 6]).astype(np.float32) if N and M else np.zeros(N, np.float32)
    return Xb, R, d


def base_delta(src):
    """Best-match IoU minus the tree's best IoU for that external candidate (<= 0 for tree rows), from an ext file."""
    z = np.load(src, allow_pickle=True)
    ov = z['ov'].astype(np.float32); E = z['E']
    if not ov.size:
        return np.zeros(len(ov), np.float32)
    return (ov.max(1) - E[ov.argmax(1), 6]).astype(np.float32)


def level_stack(Pm):
    """Per level: flat component labels of Pm > THR[li] and the component areas."""
    out = []
    for t8 in THR:
        lab = cv2.connectedComponents((Pm > t8).astype(np.uint8), connectivity=8)[1]
        out.append((lab.ravel(), np.bincount(lab.ravel())))
    return out


def xm_block(sets, lis, Pm, levs):
    """6 map-agreement columns of pixel sets (sorted flat indices) against map Pm (uint8): mean, max, fraction > 0.5,
    edge mean, IoU with the best-overlap component at the row's own level, best such IoU over all levels."""
    Qf = Pm.astype(np.float32) / 255
    Xb = np.zeros((len(sets), 6), np.float32)
    for i, (Q, li) in enumerate(zip(sets, lis)):
        if not len(Q):
            continue
        y, x = Q // W, Q % W
        y0, y1 = max(int(y.min()) - PAD, 0), min(int(y.max()) + 1 + PAD, H)
        x0, x1 = max(int(x.min()) - PAD, 0), min(int(x.max()) + 1 + PAD, W)
        c = np.zeros((y1 - y0, x1 - x0), bool); c[y - y0, x - x0] = True
        q = Qf[y0:y1, x0:x1]
        qc = q[c]
        edge = c & ~cv2.erode(c.astype(np.uint8), _K1).astype(bool)
        n = int(c.sum())
        ious = []
        for lj in range(len(LEVELS)):
            labf, oarea = levs[lj]
            cnt = np.bincount(labf[Q], minlength=1); cnt[0] = 0
            if cnt.max() == 0:
                ious.append(0.0); continue
            kk = int(cnt.argmax())
            ious.append(cnt[kk] / (n + oarea[kk] - cnt[kk]))
        Xb[i] = [qc.mean(), qc.max(), (qc > 0.5).mean(), q[edge].mean() if edge.any() else 0, ious[li], max(ious)]
    return Xb


def xm18(sets, lis, P_r34, P_cnxL, P_cnx):
    """X[N, 18] = R34 | cnxL | cnx ``xm_block`` columns (MAPS order)."""
    Xall = np.zeros((len(sets), 18), np.float32)
    src = {'r34': P_r34, 'cnxL': P_cnxL, 'cnx': P_cnx}
    for b, name in enumerate(MAPS):
        Xall[:, 6 * b:6 * b + 6] = xm_block(sets, lis, src[name], level_stack(src[name]))
    return Xall


# ------------------------------------------------------------------------------------------------ pool generation

def _closing_merges(P, labs, node, area_b, depth, add):
    """fam 2: closing(P > t, disk r) components that join >= 2 tree nodes of that level, r in (6, 12)."""
    for r in (6, 12):
        for li, t8 in enumerate(THR):
            B = (P > t8).astype(np.uint8)
            if not B.any():
                continue
            cl = cv2.morphologyEx(B, cv2.MORPH_CLOSE, K(r))
            _, lc = cv2.connectedComponents(cl, connectivity=8)
            lab = labs[li]; msk = lab > 0
            nl = int(lab.max()) + 1
            code = np.unique(lc[msk].astype(np.int64) * nl + lab[msk])
            pairs = np.stack([code // nl, code % nl], 1)
            pairs = pairs[np.array([(li, int(b)) in node for b in pairs[:, 1]], bool)]
            if not len(pairs):
                continue
            cid, cnt = np.unique(pairs[:, 0], return_counts=True)
            for c_ in cid[cnt >= 2]:
                mem = pairs[pairs[:, 0] == c_, 1]
                ys, xs = np.nonzero(lc == c_)
                Q = ys.astype(np.int64) * W + xs
                ma = sum(area_b[node[(li, int(b))]] for b in mem)
                add(2, Q, float(LEVELS[li]), li, (1.0, len(mem), np.mean([depth[node[(li, int(b))]] for b in mem]), len(mem)),
                    [np.log(len(Q) / max(ma, 1)), ma / len(Q), len(mem), r])


def _axis(Q):
    """Principal axis (unit vector, x then y) of a pixel set."""
    y, x = Q // W, Q % W
    if len(Q) < 3:
        return np.array([1.0, 0.0])
    return np.linalg.eigh(np.cov(np.stack([x, y]).astype(np.float64)))[1][:, -1]


def _pairwise_merges(F, keys, labs, node, pix, area_b, depth, Pf, add):
    """fam 4: two same-level tree nodes at most GAP px apart, joined by a straight bridge of the thinner one's width."""
    Kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * GAP + 1, 2 * GAP + 1))
    for i, (li, kk, sl) in enumerate(keys):
        y0, y1, x0, x1 = bbox(pix[i])
        ya, yb, xa, xb = max(y0 - GAP - 1, 0), min(y1 + GAP + 1, H), max(x0 - GAP - 1, 0), min(x1 + GAP + 1, W)
        cm = np.zeros((yb - ya, xb - xa), np.uint8); cm[pix[i] // W - ya, pix[i] % W - xa] = 1
        dil = cv2.dilate(cm, Kd).astype(bool)
        L = labs[li][ya:yb, xa:xb]
        for kb in np.unique(L[dil & ~cm.astype(bool)]):
            if kb == 0 or kb == kk:
                continue
            jn = node.get((li, int(kb)))
            if jn is None or jn <= i:
                continue
            Y0, Y1, X0, X1 = bbox(np.concatenate([pix[i], pix[jn]]))
            Y0, Y1, X0, X1 = max(Y0 - 2, 0), min(Y1 + 2, H), max(X0 - 2, 0), min(X1 + 2, W)
            A = np.zeros((Y1 - Y0, X1 - X0), bool); A[pix[i] // W - Y0, pix[i] % W - X0] = True
            B = np.zeros_like(A); B[pix[jn] // W - Y0, pix[jn] % W - X0] = True
            dA = ndi.distance_transform_edt(~A)
            by, bx = np.nonzero(B); t_ = np.argmin(dA[by, bx]); pb = (by[t_], bx[t_]); gap = dA[pb]
            if gap > GAP:
                continue
            dB = ndi.distance_transform_edt(~B)
            ay, ax = np.nonzero(A); t_ = np.argmin(dB[ay, ax]); pa = (ay[t_], ax[t_])
            th = int(max(1, round(min(F[i, 22], F[jn, 22]))))
            br = np.zeros(A.shape, np.uint8)
            cv2.line(br, (int(pa[1]), int(pa[0])), (int(pb[1]), int(pb[0])), 1, thickness=th)
            U = A | B | br.astype(bool)
            yy, xx = np.nonzero(U)
            Q = (yy + Y0).astype(np.int64) * W + xx + X0
            bonly = br.astype(bool) & ~A & ~B
            pbr = Pf[Y0:Y1, X0:X1][bonly].mean() if bonly.any() else Pf[Y0:Y1, X0:X1][U].min()
            ua, ub = _axis(pix[i]), _axis(pix[jn])
            g = np.array([pb[1] - pa[1], pb[0] - pa[0]], np.float64); g = g / (np.linalg.norm(g) + 1e-9)
            colin = 0.5 * (abs(g @ ua) + abs(g @ ub)) if gap > 0 else 1.0
            add(4, Q, float(LEVELS[li]), li, (1.0, 2, 0.5 * (depth[i] + depth[jn]), 2),
                [gap, pbr, abs(ua @ ub), colin, min(area_b[i], area_b[jn]) / max(area_b[i], area_b[jn]),
                 np.log((bonly.sum() + 1) / len(Q))])


def _drop_near_duplicates(new, pix, Nb):
    """Drop new rows with IoU >= 0.9 to a tree row or to an earlier kept new row. Returns (kept, IoU[kept, Nb])."""
    Cb, lb = csr_of(pix)
    Inb = np.zeros((0, Nb), np.float32)
    if new:
        Cn0, ln0 = csr_of([x[1] for x in new])
        Inb = iou_sparse(Cn0, ln0, Cb, lb) if Nb else np.zeros((len(new), 0), np.float32)
        keepn = (Inb.max(1) < 0.9) if Nb else np.ones(len(new), bool)
        Inn = (Cn0 @ Cn0.T).tocoo()
        inter = {}
        for a, b, v in zip(Inn.row, Inn.col, Inn.data):
            if a < b:
                inter.setdefault(b, []).append((a, v))
        for b in range(len(new)):
            if not keepn[b]:
                continue
            for a, v in inter.get(b, []):
                if keepn[a] and v / (ln0[a] + ln0[b] - v) >= 0.9:
                    keepn[b] = False; break
        new = [x for x, kp in zip(new, keepn) if kp]
        Inb = Inb[keepn]
    return new, Inb


def gen(P, S, others, img, i6, i6rich, dets, detrichs):
    """The pool of one image.

    P, S: cnx mask and spine maps (uint8); others: ``{'cnxL': (P, S), 'r34': (P, S)}``; img: the uint8 image;
    i6 / i6rich: the image's I6 ``ext`` and ``rich`` files; dets / detrichs: one ``ext`` / ``rich`` file per detector
    fold model. Returns a dict with F (leading variant axis), fam, parents, conf, nb, sets, lis, bt, pairs, piou,
    nraw and ext_selfcheck.
    """
    dm, ds = disk_stats(img)
    F, par, keys = candidates(P, S, img, dm, ds)
    Nb = len(F)
    bI = [np.load(i6, allow_pickle=True)['X'], np.load(i6rich)['X']]
    dI = base_delta(i6)
    Fb = []
    for dp, rp in zip(dets, detrichs):
        bD = [np.load(dp, allow_pickle=True)['X'], np.load(rp)['X']]
        dD = base_delta(dp)
        Fb.append(np.concatenate([F] + bI + bD + [dI[:, None], dD[:, None]], 1))
    Pf = P.astype(np.float32) / 255
    I = (img.astype(np.float32) - dm) / ds
    labs = [cv2.connectedComponents((P > t8).astype(np.uint8), connectivity=8)[1] for t8 in THR]
    node = {(li, kk): i for i, (li, kk, sl) in enumerate(keys)}
    pix = []
    for (li, kk, sl) in keys:
        ys, xs = np.nonzero(labs[li][sl] == kk)
        pix.append(((ys + sl[0].start) * W + xs + sl[1].start).astype(np.int64))
    area_b = np.array([len(q) for q in pix])
    depth = F[:, 14] if Nb else np.zeros(0)
    seen, new = set(), []   # (fam, Q, t, li, extra_tree, extras, row28 or None)

    def add(fam, Q, t, li, et, ex, row=None):
        if len(Q) < MIN_AREA:
            return
        Q = np.sort(Q)
        h = hashlib.blake2b(Q.tobytes(), digest_size=12).digest()
        if h in seen:
            return
        seen.add(h)
        new.append((fam, Q, t, li, et, list(ex) + [0.0] * (NE - len(ex)), row))

    _closing_merges(P, labs, node, area_b, depth, add)
    _pairwise_merges(F, keys, labs, node, pix, area_b, depth, Pf, add)
    del labs
    # pooled tree nodes of the other dense models (own-map features)
    nraw = {}
    for code, name in ((5, 'cnxL'), (6, 'r34')):
        P2, S2 = others[name]
        F2, _, keys2 = candidates(P2, S2, img, dm, ds)
        s2 = tree_sets(P2, keys2)
        nraw[name] = len(s2)
        for i in range(len(s2)):
            add(code, s2[i], float(F2[i, 0]), int(F2[i, 1]), None, [], row=F2[i])
    new, Inb = _drop_near_duplicates(new, pix, Nb)
    Nn = len(new)
    img_n50, img_pmean = (F[0, 24], F[0, 25]) if Nb else (0, Pf.mean())
    pm_base = np.sort(F[:, 7]) if Nb else np.zeros(1)
    Fn = np.zeros((Nn, len(FEATURES)), np.float32)
    Xn = np.zeros((Nn, 4 + NE), np.float32)
    for a, (fam, Q, t, li, et, ex, row) in enumerate(new):
        if row is None:
            row = feat_row(Q, t, li, Pf, I, S, et)
            row[24], row[25] = img_n50, img_pmean
            row[27] = 1 - np.searchsorted(pm_base, row[7]) / max(len(pm_base), 1)
        Fn[a] = row; Xn[a, ONEHOT[fam]] = 1; Xn[a, 4:] = ex
    conf = []
    for i in range(Nb):
        p = par[i]
        while p >= 0:
            conf.append((p, i)); p = par[p]
    bt = np.zeros(Nb + Nn, np.float32)
    pairs = np.zeros((0, 2), np.int32); piou = np.zeros(0, np.float32)
    # external evidence: one ext_block call per source over tree + new rows (row-independent formulas); the tree part is
    # a self-check against the stored evidence, the new part is used
    sets = pix + [x[1] for x in new]
    Ca, la = csr_of(sets)
    F28 = np.concatenate([F, Fn]) if Nn else F
    xi, ri, di = ext_block(Ca, la, i6, F28)
    chk_i6 = max(float(np.max(np.abs(xi[:Nb] - bI[0]))) if Nb else 0.0, float(np.max(np.abs(ri[:Nb] - bI[1]))) if Nb else 0.0,
                 float(np.max(np.abs(di[:Nb] - dI))) if Nb else 0.0)
    Fnew, chk_det = [], []
    for dp, rp in zip(dets, detrichs):
        xd, rd, dd = ext_block(Ca, la, dp, F28)
        bD = [np.load(dp, allow_pickle=True)['X'], np.load(rp)['X']]; dD = base_delta(dp)
        chk_det.append(max(float(np.max(np.abs(xd[:Nb] - bD[0]))) if Nb else 0.0,
                           float(np.max(np.abs(rd[:Nb] - bD[1]))) if Nb else 0.0, float(np.max(np.abs(dd[:Nb] - dD))) if Nb else 0.0))
        Fnew.append(np.concatenate([Fn, xi[Nb:], ri[Nb:], xd[Nb:], rd[Nb:], di[Nb:, None], dd[Nb:, None], Xn], 1) if Nn
                    else np.zeros((0, Fb[0].shape[1] + 4 + NE), np.float32))
    if Nn:
        Cn, ln = csr_of([x[1] for x in new])
        a_, b_ = np.nonzero(Inb > 0)
        conf += list(zip(b_.tolist(), (Nb + a_).tolist()))
        nn = (Cn @ Cn.T).tocoo(); u = nn.row < nn.col
        conf += list(zip((Nb + nn.row[u]).tolist(), (Nb + nn.col[u]).tolist()))
        if Nb:
            bt[Nb:] = Inb.max(1)
        ra, rb, v = nn.row[u], nn.col[u], nn.data[u]
        iou = v / (ln[ra] + ln[rb] - v)
        k5 = iou >= 0.5
        pairs = np.stack([Nb + ra[k5], Nb + rb[k5]], 1).astype(np.int32); piou = iou[k5].astype(np.float32)
    Fall = np.stack([np.concatenate([np.concatenate([fb, np.zeros((Nb, 4 + NE), np.float32)], 1), fn]).astype(np.float32)
                     for fb, fn in zip(Fb, Fnew)])
    fam = np.concatenate([np.zeros(Nb, np.int8), np.array([x[0] for x in new], np.int8)])
    return dict(F=Fall, fam=fam, parents=np.concatenate([par, np.full(Nn, -1)]), conf=np.asarray(conf, np.int32).reshape(-1, 2),
                nb=Nb, sets=sets, lis=[k_[0] for k_ in keys] + [int(x[3]) for x in new], bt=bt, pairs=pairs, piou=piou,
                nraw=nraw, ext_selfcheck=max([chk_i6] + chk_det))


# ------------------------------------------------------------------------------------------------------- driver

_G = {}


def _init(maps, evidence_root, image_dir, out_dir):
    _G['maps'] = {n: np.load(maps[n], mmap_mode='r') for n in ('cnx', 'cnxL', 'r34')}
    _G['ev'] = Path(evidence_root)
    _G['image_dir'] = image_dir
    _G['out'] = Path(out_dir)


def row_file(j, fname, maps, ev, image_dir, out_dir):
    """Build and write ``<out_dir>/<stem>.npz`` for image j of the map files. Returns (number of rows, fam counts)."""
    stem = fname[:-5]
    m = np.asarray(maps['cnx'][j]); mL = np.asarray(maps['cnxL'][j]); mR = np.asarray(maps['r34'][j])
    img = read_image(image_dir, fname)
    ev = Path(ev)
    g = gen(m[0], m[1], {'cnxL': (mL[0], mL[1]), 'r34': (mR[0], mR[1])}, img,
            ev / f'ext/testcnx-i6g/{stem}.npz', ev / f'feat/testcnx-i6g-rich/{stem}.npz',
            [ev / f'ext/testcnx-det-f{k}/{stem}.npz' for k in range(NV)],
            [ev / f'feat/testcnx-detrich-f{k}/{stem}.npz' for k in range(NV)])
    Xm = xm18(g['sets'], g['lis'], mR[0], mL[0], m[0])
    flat, lens = pack_sets(g['sets'])
    np.savez_compressed(Path(out_dir) / f'{stem}.npz', F=g['F'], xm=Xm, fam=g['fam'], parents=g['parents'], conf=g['conf'],
                        nb=g['nb'], lvl=np.asarray(g['lis'], np.int8), area=lens, bt=g['bt'], pairs=g['pairs'],
                        piou=g['piou'], ext_selfcheck=g['ext_selfcheck'],
                        nraw=np.array([g['nraw']['cnxL'], g['nraw']['r34']], np.int64), set_flat=flat, set_len=lens)
    return len(g['fam']), [int((g['fam'] == f).sum()) for f in (0, 2, 4, 5, 6)]


def _one(job):
    j, fname = job
    return (fname,) + row_file(j, fname, _G['maps'], _G['ev'], _G['image_dir'], _G['out'])


def build_rows(maps, evidence_root=None, out_dir=None, image_dir=None, nproc=MAX_WORKERS, limit=None):
    """Write the row file of every test image.

    maps: ``{'cnx': path, 'cnxL': path, 'r34': path}`` of the ``(n, 2, 2048, 2048)`` uint8 test maps (their ``.json``
    sidecars must list the same files); evidence_root: the work folder holding the stage-5a ``ext/`` and ``feat/``
    folders (default ``filseg.paths.WORK``); out_dir: default ``<work>/rows``. Returns totals.
    """
    files = {}
    for n in ('cnx', 'cnxL', 'r34'):
        with open(str(maps[n]) + '.json', encoding='utf-8') as fh:
            files[n] = json.load(fh)['files']
    if not files['cnx'] == files['cnxL'] == files['r34']:
        raise ValueError('the three map files list different images')
    evidence_root = Path(evidence_root or paths.WORK)
    out_dir = Path(out_dir or evidence_root / ROWS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = list(enumerate(files['cnx']))[:limit]
    rows, fams = 0, np.zeros(5, np.int64)
    with ProcessPoolExecutor(min(nproc, MAX_WORKERS), initializer=_init,
                             initargs=({k: str(v) for k, v in maps.items()}, str(evidence_root),
                                       str(image_dir or paths.TEST_IMAGES), str(out_dir))) as ex:
        for n, (fname, nrow, fam) in enumerate(ex.map(_one, jobs, chunksize=1)):
            rows += nrow; fams += np.array(fam)
            if n % 10 == 0:
                print('rows', n, fname, nrow, flush=True)
    return dict(images=len(jobs), rows=rows, fam_0_2_4_5_6=fams.tolist())
