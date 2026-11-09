"""Playbook diagnostic 3: extended candidate pools with the full C2 evidence for every row.

Base rows: the stored C2 feature blocks (28 tree features + I6 X 6 + I6 rich 11 + det X 6 + det rich 11 = 62).
New rows (families sp / mg / tb as in xcands.py, pm as in xcands_pm.py, deduplicated across families): the same 62
columns, with the I6 and detector blocks recomputed from the stored external masks (work/ext/cnx-i6g, work/ext/cnx-det:
ext_rle, E, ext_sel) exactly as ext.py / rich.py compute them for tree candidates.
Plus d_i6, d_det (best-match IoU minus the tree's best IoU for that external candidate, all rows), one-hot
sp/mg/tb/pm (4) and 6 family extras (zeros on base rows) -> F[N, 74].
Output work/ideas7/playbook/xc2/<stem>.npz: F, fam (0 base, 1 sp, 2 mg, 3 tb, 4 pm), parents, hits, ngt, conf, nb, same.
Dev folds 0-3 only, CPU only. usage: xcands2.py [--nproc=5] [--folds=0,1,2,3]
"""
import json, os, sys, time, hashlib
os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = 'D:/FilamentSegmentationKaggle/claude'
sys.path.insert(0, ROOT + '/src'); sys.path.insert(0, ROOT + '/work/ideas7/playbook')
from concurrent.futures import ProcessPoolExecutor
import numpy as np, cv2
from scipy import ndimage as ndi
from pycocotools import mask as mu
from skimage.segmentation import watershed
from skimage.morphology import skeletonize
from cfil.cands import LEVELS, MIN_AREA, candidates, gt_label_maps, FEATURES
from cfil.build_cands import disk_stats, RAW
from cfil.rich import rich_features
import xcands as X

OUT = ROOT + '/work/ideas7/playbook/xc2'
H = W = 2048
THR = X.THR
K = X.K
NE = 6
GAP = 32
_sig = lambda z: 1 / (1 + np.exp(-z))


def csr_of(sets, n=H * W):
    from scipy.sparse import csr_matrix
    lens = np.array([len(q) for q in sets], np.int64)
    ind = np.concatenate(sets) if len(sets) else np.zeros(0, np.int64)
    ptr = np.concatenate([[0], np.cumsum(lens)])
    return csr_matrix((np.ones(len(ind), np.float32), ind, ptr), shape=(len(sets), n)), lens


def iou_sparse(Ca, la, Cb, lb):
    if Ca.shape[0] == 0 or Cb.shape[0] == 0:
        return np.zeros((Ca.shape[0], Cb.shape[0]), np.float32)
    inter = (Ca @ Cb.T).toarray()
    return (inter / np.maximum(la[:, None] + lb[None, :] - inter, 1)).astype(np.float32)


def rle_sets(counts):
    out = []
    for c in counts:
        m = mu.decode({'size': [H, W], 'counts': c.encode() if isinstance(c, str) else c})
        out.append(np.flatnonzero(m).astype(np.int64))
    return out


def ext_block(Cn, ln, src, Fbase28):
    """I6/det X (6) + rich (11) + best-match delta for new candidates, from the stored external source file
    (same formulas as ext.py / rich.py; IoUs from exact pixel sets)."""
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
    """best-match IoU minus the tree's best IoU for that external candidate (<= 0 for tree rows)"""
    z = np.load(src, allow_pickle=True)
    ov = z['ov'].astype(np.float32); E = z['E']
    if not ov.size:
        return np.zeros(len(ov), np.float32)
    return (ov.max(1) - E[ov.argmax(1), 6]).astype(np.float32)


def one(args):
    k, j, fname = args
    stem = fname[:-5]
    dst = os.path.join(OUT, stem + '.npz')
    if os.path.exists(dst):
        return fname, None, 0.0
    t0 = time.time()
    _G = X._G
    m = np.asarray(_G['maps'][k][j]); P, S = m[0], m[1]
    img = cv2.imread(os.path.join(RAW, 'train/train_images', fname), cv2.IMREAD_GRAYSCALE)
    dm, ds = disk_stats(img)
    gts = gt_label_maps(_G['by'], _G['recs'][fname])
    F, par, Hh, keys = candidates(P, S, img, dm, ds, gts)
    z = np.load(ROOT + '/work/cands/cnx/' + stem + '.npz')
    same = bool(np.array_equal(z['hits'], Hh) and np.allclose(z['F'], F))
    Nb = len(F)
    # stored C2 blocks for the base rows
    blocks = [np.load(ROOT + '/work/ext/cnx-i6g/' + stem + '.npz', allow_pickle=True)['X'],
              np.load(ROOT + '/work/feat/cnx-i6g-rich/' + stem + '.npz')['X'],
              np.load(ROOT + '/work/ext/cnx-det/' + stem + '.npz', allow_pickle=True)['X'],
              np.load(ROOT + '/work/feat/cnx-detrich/' + stem + '.npz')['X']]
    dI = base_delta(ROOT + '/work/ext/cnx-i6g/' + stem + '.npz'); dD = base_delta(ROOT + '/work/ext/cnx-det/' + stem + '.npz')
    Fb = np.concatenate([F] + blocks + [dI[:, None], dD[:, None]], 1)
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
    seen, new = set(), []

    def overl_base(Q):
        out = {}
        for li in range(len(LEVELS)):
            v = labs[li].ravel()[Q]
            u, c = np.unique(v[v > 0], return_counts=True)
            for kk, cc in zip(u, c):
                i = node.get((li, int(kk)))
                if i is not None:
                    out[i] = cc
        return out

    def add(fam, Q, t, li, et, ex):
        if len(Q) < MIN_AREA:
            return
        Q = np.sort(Q)
        h = hashlib.blake2b(Q.tobytes(), digest_size=12).digest()
        if h in seen:
            return
        seen.add(h)
        ex = list(ex) + [0.0] * (NE - len(ex))
        new.append((fam, Q, t, li, et, ex, None))

    # sp: spine-seeded split
    for s in (0.15, 0.25, 0.40):
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
            y0, y1, x0, x1 = X.bbox(pix[i])
            cm = np.zeros((y1 - y0, x1 - x0), bool); cm[pix[i] // W - y0, pix[i] % W - x0] = True
            mk = np.zeros(cm.shape, np.int32); mlc = ml[y0:y1, x0:x1]
            for n_, mid in enumerate(u, 1):
                mk[(mlc == mid) & cm] = n_
            ws = watershed((255 - P[y0:y1, x0:x1]).astype(np.uint8), mk, mask=cm)
            for n_ in range(1, len(u) + 1):
                yy, xx = np.nonzero(ws == n_)
                Q = (yy + y0).astype(np.int64) * W + xx + x0
                add(1, Q, float(LEVELS[li]), li, (len(Q) / area_b[i], 0, depth[i], len(u)),
                    [np.log(area_b[i] / max(len(Q), 1)), len(Q) / area_b[i], len(u), s])
    # mg: closing merges
    for r in (6, 12):
        for li, t8 in enumerate(THR):
            B = (P > t8).astype(np.uint8)
            if not B.any():
                continue
            cl = cv2.morphologyEx(B, cv2.MORPH_CLOSE, K(r))
            n_c, lc = cv2.connectedComponents(cl, connectivity=8)
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
    # tb: tubes
    for i, (li, kk, sl) in enumerate(keys):
        if area_b[i] < 60:
            continue
        y0, y1, x0, x1 = X.bbox(pix[i]); pad = 10
        ya, yb, xa, xb = max(y0 - pad, 0), min(y1 + pad, H), max(x0 - pad, 0), min(x1 + pad, W)
        cm = np.zeros((yb - ya, xb - xa), np.uint8); cm[pix[i] // W - ya, pix[i] % W - xa] = 1
        sk = skeletonize(cm > 0); Ls = max(int(sk.sum()), 1)
        rh = int(round((area_b[i] / Ls - 1) / 2))
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
    # pm: pairwise same-level merges with a straight bridge
    Kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * GAP + 1, 2 * GAP + 1))
    def axis(Q):
        y, x = Q // W, Q % W
        if len(Q) < 3:
            return np.array([1.0, 0.0])
        return np.linalg.eigh(np.cov(np.stack([x, y]).astype(np.float64)))[1][:, -1]
    for i, (li, kk, sl) in enumerate(keys):
        y0, y1, x0, x1 = X.bbox(pix[i])
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
            Y0, Y1, X0, X1 = X.bbox(np.concatenate([pix[i], pix[jn]]))
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
            ua, ub = axis(pix[i]), axis(pix[jn])
            g = np.array([pb[1] - pa[1], pb[0] - pa[0]], np.float64); g = g / (np.linalg.norm(g) + 1e-9)
            colin = 0.5 * (abs(g @ ua) + abs(g @ ub)) if gap > 0 else 1.0
            add(4, Q, float(LEVELS[li]), li, (1.0, 2, 0.5 * (depth[i] + depth[jn]), 2),
                [gap, pbr, abs(ua @ ub), colin, min(area_b[i], area_b[jn]) / max(area_b[i], area_b[jn]),
                 np.log((bonly.sum() + 1) / len(Q))])
    # assemble: drop near-duplicates of tree nodes (IoU >= 0.9), then features, ext evidence, conflicts
    Cb, lb = csr_of(pix)
    if new:
        Cn0, ln0 = csr_of([x[1] for x in new])
        Inb = iou_sparse(Cn0, ln0, Cb, lb)
        keepn = (Inb.max(1) < 0.9) if Nb else np.ones(len(new), bool)
        new = [x for x, kp in zip(new, keepn) if kp]
        Inb = Inb[keepn]
    Nn = len(new)
    img_n50, img_pmean = (F[0, 24], F[0, 25]) if Nb else (0, Pf.mean())
    pm_base = np.sort(F[:, 7]) if Nb else np.zeros(1)
    Fn = np.zeros((Nn, len(FEATURES)), np.float32); Hn = np.zeros((Nn, len(gts)), np.float32)
    Xn = np.zeros((Nn, 4 + NE), np.float32)
    for a, (fam, Q, t, li, et, ex, ob) in enumerate(new):
        row = X.feat_row(Q, t, li, Pf, I, S, et)
        row[24], row[25] = img_n50, img_pmean
        row[27] = 1 - np.searchsorted(pm_base, row[7]) / max(len(pm_base), 1)
        Fn[a] = row; Xn[a, fam - 1] = 1; Xn[a, 4:] = ex
        Hn[a] = X.hits_of(Q, gts)
    conf = []
    for i in range(Nb):
        p = par[i]
        while p >= 0:
            conf.append((p, i)); p = par[p]
    if Nn:
        Cn, ln = csr_of([x[1] for x in new])
        xi, ri, di = ext_block(Cn, ln, ROOT + '/work/ext/cnx-i6g/' + stem + '.npz', Fn)
        xd, rd, dd = ext_block(Cn, ln, ROOT + '/work/ext/cnx-det/' + stem + '.npz', Fn)
        Fnew = np.concatenate([Fn, xi, ri, xd, rd, di[:, None], dd[:, None], Xn], 1)
        a_, b_ = np.nonzero(Inb > 0)
        conf += list(zip(b_.tolist(), (Nb + a_).tolist()))
        nn = (Cn @ Cn.T).tocoo()
        u = nn.row < nn.col
        conf += list(zip((Nb + nn.row[u]).tolist(), (Nb + nn.col[u]).tolist()))
    else:
        Fnew = np.zeros((0, Fb.shape[1] + 4 + NE), np.float32)
    Fall = np.concatenate([np.concatenate([Fb, np.zeros((Nb, 4 + NE), np.float32)], 1), Fnew]).astype(np.float32)
    fam = np.concatenate([np.zeros(Nb, np.int8), np.array([x[0] for x in new], np.int8)])
    np.savez_compressed(dst, F=Fall, fam=fam, parents=np.concatenate([par, np.full(Nn, -1)]),
                        hits=np.concatenate([Hh, Hn]) if Nn else Hh, ngt=np.asarray([len(a) - 1 for _, a in gts]),
                        conf=np.asarray(conf, np.int32).reshape(-1, 2), nb=Nb, same=same)
    return fname, (Nb, Nn, same, [int((fam == f).sum()) for f in (1, 2, 3, 4)]), time.time() - t0


def main():
    opt = dict(a[2:].split('=', 1) for a in sys.argv[1:] if a.startswith('--') and '=' in a)
    folds = [int(f) for f in opt.get('folds', '0,1,2,3').split(',')]
    os.makedirs(OUT, exist_ok=True)
    jobs = []
    for k in folds:
        mp = ROOT + f'/work/maps/cnx-f{k}/oof.npy'
        jobs += [(k, j, f) for j, f in enumerate(json.load(open(mp + '.json'))['files'])]
    t0 = time.time()
    with ProcessPoolExecutor(int(opt.get('nproc', 5)), initializer=X._init) as ex:
        for n, (f, info, dt) in enumerate(ex.map(one, jobs, chunksize=1)):
            if n % 20 == 0:
                print(n, f, info, f'{dt:.1f}s', f'elapsed {time.time() - t0:.0f}s', flush=True)
    print('done', len(jobs), f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
