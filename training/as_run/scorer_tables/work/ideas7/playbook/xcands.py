"""Playbook diagnostic 2: extended candidate pools (base threshold tree + three new generator families) with features,
per-record hits and the full conflict graph, so a scorer + exact selection can measure what the new shapes are worth.

Families (all deployable: they use only the dense maps):
  sp     spine-seeded split (Omnipose / skeleton-flow analog): inside every tree node that holds >= 2 spine markers
         (components of S > s, s in 0.15 / 0.25 / 0.40, >= 10 px inside the node), watershed basins flooding by descending P
  mg     closing merge (agglomeration with a fixed rule): components of closing(P > t, disk r), r in 6 / 12, that join
         >= 2 tree nodes of that level, bridge pixels included
  tb     constant-width tube (human polygon / frame-field analog): node skeleton dilated by radius r_hat - 1 and r_hat,
         r_hat = round((area / skeleton length - 1) / 2)
New shapes that duplicate a tree node (IoU >= 0.9) or an earlier new shape (identical pixels) are dropped.
Output work/ideas7/playbook/xc/<stem>.npz: F[N, 28 + 7] (base features + one-hot sp/mg/tb + 4 extras), fam[N] (0 base),
parents[N] (-1 for new), hits[N, R], ngt[R], conf[M, 2] (all overlapping pairs i < j), nb (number of base rows).
Dev folds 0-3 only, CPU only. usage: xcands.py [--nproc=5] [--limit=N]
"""
import json, os, sys, time, hashlib
os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = 'D:/FilamentSegmentationKaggle/claude'
sys.path.insert(0, ROOT + '/src')
from concurrent.futures import ProcessPoolExecutor
import numpy as np, cv2
from skimage.segmentation import watershed
from skimage.morphology import skeletonize
from cfil.cands import LEVELS, MIN_AREA, candidates, gt_label_maps, FEATURES
from cfil.build_cands import disk_stats, RAW

ANN = ROOT + '/../data/annotations/MAGFiLO_1.0_Annotations_kaggle2026_train.json'
OUT = ROOT + '/work/ideas7/playbook/xc'
H = W = 2048
THR = [int(round(t * 255)) for t in LEVELS]
_RING = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
K = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
NX = 7  # one-hot sp/mg/tb + 4 extras
_G = {}


def _init():
    obs = [o for o in json.load(open(ROOT + '/configs/split-v2.json'))['observations'] if o['fold'] in (0, 1, 2, 3)]
    keep = {r for o in obs for r in o['annotator_image_ids']}
    by = {}
    for a in json.load(open(ANN))['annotations']:
        if a['image_id'] in keep:
            by.setdefault(a['image_id'], []).append(a)
    _G['by'] = by
    _G['recs'] = {o['file_name']: o['annotator_image_ids'] for o in obs}
    _G['maps'] = {k: np.load(ROOT + f'/work/maps/cnx-f{k}/oof.npy', mmap_mode='r') for k in range(4)}


def bbox(Q):
    y, x = Q // W, Q % W
    return y.min(), y.max() + 1, x.min(), x.max() + 1


def feat_row(Q, t, li, Pf, I, S, extra_tree):
    """The 28 FEATURES of cands.candidates for an arbitrary mask given by flat pixel indices Q."""
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


def hits_of(Q, gts):
    h = []
    for lab_g, ga in gts:
        cnt = np.bincount(lab_g.ravel()[Q], minlength=len(ga)); cnt[0] = 0
        j = int(cnt.argmax())
        if cnt[j] == 0:
            h.append(0.0); continue
        iou = cnt[j] / (len(Q) + ga[j] - cnt[j])
        h.append(float(iou) if iou > 0.5 else 0.0)
    return h


def one(args):
    k, j, fname = args
    dst = os.path.join(OUT, fname.replace('.jpeg', '.npz'))
    if os.path.exists(dst):
        return fname, None, 0.0
    t0 = time.time()
    m = np.asarray(_G['maps'][k][j]); P, S = m[0], m[1]
    img = cv2.imread(os.path.join(RAW, 'train/train_images', fname), cv2.IMREAD_GRAYSCALE)
    dm, ds = disk_stats(img)
    gts = gt_label_maps(_G['by'], _G['recs'][fname])
    F, par, Hh, keys = candidates(P, S, img, dm, ds, gts)
    z = np.load(ROOT + '/work/cands/cnx/' + fname[:-5] + '.npz')
    same = bool(np.array_equal(z['hits'], Hh) and np.allclose(z['F'], F))
    Nb = len(F)
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
    seen = set()
    new = []  # (fam, Q, t, li, extra_tree, extras4)

    def overl_base(Q):
        """base node indices overlapping Q, with intersection counts"""
        out = {}
        for li in range(len(LEVELS)):
            v = labs[li].ravel()[Q]
            u, c = np.unique(v[v > 0], return_counts=True)
            for kk, cc in zip(u, c):
                i = node.get((li, int(kk)))
                if i is not None:
                    out[i] = cc
        return out

    def add(fam, Q, t, li, extra_tree, extras):
        if len(Q) < MIN_AREA:
            return
        Q = np.sort(Q)
        h = hashlib.blake2b(Q.tobytes(), digest_size=12).digest()
        if h in seen:
            return
        ob = overl_base(Q)
        for i, c in ob.items():
            if c / (len(Q) + area_b[i] - c) >= 0.9:
                return
        seen.add(h)
        new.append((fam, Q, t, li, extra_tree, extras, ob))

    # --- sp: spine-seeded split ---
    for fs, s in ((0, 0.15), (1, 0.25), (2, 0.40)):
        n_m, ml = cv2.connectedComponents((S > int(round(s * 255))).astype(np.uint8), connectivity=8)
        if n_m < 3:
            continue
        mlf = ml.ravel()
        for i, (li, kk, sl) in enumerate(keys):
            mv = mlf[pix[i]]
            u, c = np.unique(mv[mv > 0], return_counts=True)
            u = u[c >= 10]
            if len(u) < 2:
                continue
            y0, y1, x0, x1 = bbox(pix[i])
            cm = np.zeros((y1 - y0, x1 - x0), bool); cm[pix[i] // W - y0, pix[i] % W - x0] = True
            mk = np.zeros(cm.shape, np.int32)
            mlc = ml[y0:y1, x0:x1]
            for n_, mid in enumerate(u, 1):
                mk[(mlc == mid) & cm] = n_
            ws = watershed((255 - P[y0:y1, x0:x1]).astype(np.uint8), mk, mask=cm)
            for n_ in range(1, len(u) + 1):
                yy, xx = np.nonzero(ws == n_)
                Q = (yy + y0) * W + xx + x0
                add(1, Q, float(LEVELS[li]), li, (len(Q) / area_b[i], 0, depth[i], len(u)),
                    [np.log(area_b[i] / max(len(Q), 1)), len(Q) / area_b[i], len(u), s])
    # --- mg: closing merges ---
    for r in (6, 12):
        for li, t8 in enumerate(THR):
            B = (P > t8).astype(np.uint8)
            if not B.any():
                continue
            cl = cv2.morphologyEx(B, cv2.MORPH_CLOSE, K(r))
            n_c, lc = cv2.connectedComponents(cl, connectivity=8)
            # which closing components hold >= 2 tree nodes of this level
            lab = labs[li]
            msk = (lab > 0)
            nl = int(lab.max()) + 1
            code = np.unique(lc[msk].astype(np.int64) * nl + lab[msk])
            pairs = np.stack([code // nl, code % nl], 1)
            okn = np.array([(li, int(b)) in node for b in pairs[:, 1]], bool)
            pairs = pairs[okn]
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
    # --- tb: constant-width tubes ---
    for i, (li, kk, sl) in enumerate(keys):
        if area_b[i] < 60:
            continue
        y0, y1, x0, x1 = bbox(pix[i])
        pad = 10
        ya, yb, xa, xb = max(y0 - pad, 0), min(y1 + pad, H), max(x0 - pad, 0), min(x1 + pad, W)
        cm = np.zeros((yb - ya, xb - xa), np.uint8); cm[pix[i] // W - ya, pix[i] % W - xa] = 1
        sk = skeletonize(cm > 0)
        L = max(int(sk.sum()), 1)
        rh = int(round((area_b[i] / L - 1) / 2))
        for dr in (-1, 0):
            rr = rh + dr
            if rr < 1:
                continue
            tube = cv2.dilate(sk.astype(np.uint8), K(rr)) > 0
            yy, xx = np.nonzero(tube)
            Q = (yy + ya).astype(np.int64) * W + xx + xa
            inter = np.count_nonzero(tube & (cm > 0))
            add(3, Q, float(LEVELS[li]), li, (len(Q) / area_b[i], 0, depth[i], 0),
                [np.log(len(Q) / area_b[i]), inter / (len(Q) + area_b[i] - inter), rr, dr])
    # --- assemble ---
    Nn = len(new)
    img_n50, img_pmean = (F[0, 24], F[0, 25]) if Nb else (0, Pf.mean())
    Fn = np.zeros((Nn, len(FEATURES)), np.float32); Xn = np.zeros((Nn, NX), np.float32)
    Hn = np.zeros((Nn, len(gts)), np.float32)
    pm_base = np.sort(F[:, 7]) if Nb else np.zeros(1)
    for a, (fam, Q, t, li, et, ex, ob) in enumerate(new):
        row = feat_row(Q, t, li, Pf, I, S, et)
        row[24], row[25] = img_n50, img_pmean
        row[27] = 1 - np.searchsorted(pm_base, row[7]) / max(len(pm_base), 1)
        Fn[a] = row
        Xn[a, fam - 1] = 1; Xn[a, 3:] = ex
        Hn[a] = hits_of(Q, gts)
    Fall = np.concatenate([np.concatenate([F, np.zeros((Nb, NX), np.float32)], 1), np.concatenate([Fn, Xn], 1)])
    fam = np.concatenate([np.zeros(Nb, np.int8), np.array([x[0] for x in new], np.int8)])
    parents = np.concatenate([par, np.full(Nn, -1)])
    hits = np.concatenate([Hh, Hn]) if Nn else Hh
    # conflicts: base ancestor pairs, new-base, new-new
    conf = []
    for i in range(Nb):
        p = par[i]
        while p >= 0:
            conf.append((p, i)); p = par[p]
    for a, (fam_, Q, t, li, et, ex, ob) in enumerate(new):
        for i in ob:
            conf.append((i, Nb + a))
    bb = [bbox(x[1]) for x in new]
    for a in range(Nn):
        ya0, ya1, xa0, xa1 = bb[a]
        for b in range(a + 1, Nn):
            yb0, yb1, xb0, xb1 = bb[b]
            if ya1 <= yb0 or yb1 <= ya0 or xa1 <= xb0 or xb1 <= xa0:
                continue
            if len(np.intersect1d(new[a][1], new[b][1], assume_unique=True)):
                conf.append((Nb + a, Nb + b))
    conf = np.asarray(conf, np.int32).reshape(-1, 2)
    np.savez_compressed(dst, F=Fall, fam=fam, parents=parents, hits=hits, ngt=np.asarray([len(a) - 1 for _, a in gts]),
                        conf=conf, nb=Nb, same=same)
    return fname, (Nb, Nn, same, [int((fam == f).sum()) for f in (1, 2, 3)]), time.time() - t0


def main():
    opt = dict(a[2:].split('=', 1) for a in sys.argv[1:] if a.startswith('--') and '=' in a)
    os.makedirs(OUT, exist_ok=True)
    jobs = []
    for k in range(4):
        mp = ROOT + f'/work/maps/cnx-f{k}/oof.npy'
        jobs += [(k, j, f) for j, f in enumerate(json.load(open(mp + '.json'))['files'])]
    if 'limit' in opt:
        rng = np.random.default_rng(0)
        jobs = [jobs[i] for i in sorted(rng.choice(len(jobs), int(opt['limit']), replace=False))]
    t0 = time.time()
    with ProcessPoolExecutor(int(opt.get('nproc', 5)), initializer=_init) as ex:
        for n, (f, info, dt) in enumerate(ex.map(one, jobs, chunksize=1)):
            if n % 20 == 0:
                print(n, f, info, f'{dt:.1f}s', f'elapsed {time.time() - t0:.0f}s', flush=True)
    print('done', len(jobs), f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
