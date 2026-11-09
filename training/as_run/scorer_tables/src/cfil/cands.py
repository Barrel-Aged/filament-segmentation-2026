"""Component-tree candidates from probability maps, with features and per-record targets (CPU only).

A candidate is a connected component of P > t for t in LEVELS. Components nest across levels,
forming a forest; selection picks an antichain (disjoint masks) by exact tree DP.
For each candidate and each annotator record we store the hit IoU (IoU if > 0.5 else 0);
since predictions are disjoint and GTs within a record are (almost) disjoint, this is
sufficient to compute the official pooled PQ of any antichain exactly.
"""
import json, os
import numpy as np, cv2
from scipy import ndimage as ndi
from pycocotools import mask as mu

H = W = 2048
LEVELS = np.round(np.arange(0.10, 0.851, 0.05), 3)
MIN_AREA = 40
_RING = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
# Bridged forests: with CFIL_CLOSE_R=r>0 a candidate is a connected component of closing(P>t, disk r) restricted to
# P>t, i.e. a group of nearby fragments (annotators often draw one polygon across gaps). Nested across levels.
CLOSE_R = int(os.environ.get('CFIL_CLOSE_R', 0))
REGION_R = 16  # selection regions: components of closing(P>LEVELS[0], disk REGION_R)


def disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def level_labels(P, t, close_r=None):
    """(n, label map) of the candidates at threshold t (label 0 = background)."""
    r = CLOSE_R if close_r is None else close_r
    B = (P > round(t * 255)).astype(np.uint8)
    if r <= 0:
        return cv2.connectedComponents(B, connectivity=8)
    n, lab = cv2.connectedComponents(cv2.morphologyEx(B, cv2.MORPH_CLOSE, disk(r)), connectivity=8)
    return n, lab * B


def region_of(P, keys):
    """Selection-region id per candidate (forests with any CLOSE_R <= REGION_R nest inside these regions)."""
    _, reg = level_labels(P, LEVELS[0], REGION_R)
    out = np.zeros(len(keys), np.int64)
    cache = {}
    for i, (li, k, sl) in enumerate(keys):
        if li not in cache:
            cache[li] = level_labels(P, LEVELS[li])[1]
        ys, xs = np.nonzero(cache[li][sl] == k)
        out[i] = reg[sl][ys[0], xs[0]]
    return out

FEATURES = ['t', 'lvl', 'log_area', 'bw', 'bh', 'aspect', 'fill', 'p_mean', 'p_max', 'p_std', 'p_edge', 'persist',
            'par_ratio', 'n_child', 'depth_up', 's_mean', 's_max', 'contrast', 'i_mean', 'r_disk',
            'major', 'minor', 'width', 'elong', 'img_n50', 'img_pmean', 'sib_n', 'rank_in_img']


def gt_label_maps(anns_by_record, records):
    """Per record: int16 label map (1..n) and GT areas."""
    out = []
    for r in records:
        lab = np.zeros((H, W), np.int16)
        areas = []
        for j, a in enumerate(anns_by_record[r], 1):
            segs = json.loads(a['segmentation']) if isinstance(a['segmentation'], str) else a['segmentation']
            m = mu.decode(mu.merge(mu.frPyObjects([s for s in segs if len(s) >= 6], H, W))).astype(bool)
            lab[m] = j
            areas.append(int(m.sum()))
        out.append((lab, np.asarray([0] + areas, np.int64)))
    return out


def candidates(P, S, img, disk_mean, disk_std, gts=None, aux_out=None):
    """P,S uint8 maps (mask prob, spine prob), img uint8. Returns (feat[N,F], parent[N], hit_iou[N,R], pix list).
    aux_out: optional list, filled with per-candidate per-record (raw IoU, precision, recall) vs the best-overlap GT."""
    Pf = P.astype(np.float32) / 255
    I = (img.astype(np.float32) - disk_mean) / disk_std
    rows, parents, hits, keys = [], [], [], []
    prev_lab, prev_ids = None, None
    yy, xx = None, None
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
            if gts is not None:
                h, ax = [], []
                for lab_g, areas in gts:
                    g = lab_g[sl][m]
                    cnt = np.bincount(g, minlength=len(areas))
                    cnt[0] = 0
                    j = int(cnt.argmax())
                    if cnt[j] == 0:
                        h.append(0.0); ax.append((0.0, 0.0, 0.0)); continue
                    iou = cnt[j] / (area + areas[j] - cnt[j])
                    h.append(float(iou) if iou > 0.5 else 0.0)
                    ax.append((float(iou), cnt[j] / area, cnt[j] / areas[j]))
                hits.append(h)
                if aux_out is not None:
                    aux_out.append(ax)
        prev_lab, prev_ids = lab, ids
    F = np.asarray(rows, np.float32).reshape(-1, len(FEATURES))
    parents = np.asarray(parents, np.int64)
    N = len(F)
    if N:
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
    H_ = np.asarray(hits, np.float32).reshape(N, len(gts)) if gts is not None else None
    return F, parents, H_, keys


def materialize(P, keys, chosen):
    """Label map for chosen candidate indices (must be an antichain)."""
    out = np.zeros((H, W), np.int32)
    by_level = {}
    for c in chosen:
        li, k, sl = keys[c]
        by_level.setdefault(li, []).append((c, k, sl))
    for li, items in by_level.items():
        t = LEVELS[li]
        _, lab = level_labels(P, t)
        for n, (c, k, sl) in enumerate(items):
            out[sl][lab[sl] == k] = c + 1
    return out


_CC = {}


def materialize_one(P, key):
    """COCO RLE (Fortran order) of one candidate; component labels cached per (map, level)."""
    li, k, sl = key
    ck = (id(P), li)
    if ck not in _CC:
        _CC.clear() if len(_CC) > len(LEVELS) else None
        _CC[ck] = level_labels(P, LEVELS[li])[1]
    m = np.zeros((H, W), np.uint8)
    m[sl] = _CC[ck][sl] == k
    return mu.encode(np.asfortranarray(m))


def tree_select(score, parents, tau):
    """Exact max of sum(score - tau) over antichains of the component forest."""
    N = len(score)
    best = np.maximum(score - tau, 0.0)  # leaf default, updated bottom-up
    child_sum = np.zeros(N)
    take = np.zeros(N, bool)
    order = np.arange(N)[::-1]  # children always have larger index than parents
    for i in order:
        own = score[i] - tau
        if own >= child_sum[i] and own > 0:
            best[i] = own; take[i] = True
        else:
            best[i] = child_sum[i]
        if parents[i] >= 0:
            child_sum[parents[i]] += best[i]
    chosen = []
    stack = [i for i in range(N) if parents[i] < 0]
    kids = [[] for _ in range(N)]
    for i in range(N):
        if parents[i] >= 0:
            kids[parents[i]].append(i)
    while stack:
        i = stack.pop()
        if take[i]:
            chosen.append(i)
        else:
            stack.extend(kids[i])
    return np.asarray(chosen, np.int64)
