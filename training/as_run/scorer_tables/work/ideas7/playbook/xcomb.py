"""Playbook diagnostic 5: the two positive families in one pool, with the full C2 evidence on every row:
merges (mg closing groups r 6/12 + pm same-level pairs within 32 px, as xcands2.py) and pooled tree nodes of the cnxL
and R34 OOF maps (as pool2.py). Near-duplicates (IoU >= 0.9 to any earlier row) dropped; exact duplicates hashed.
F[N, 74] = 28 tree features (merges: on the cnx map; pooled nodes: their own map) + I6 X/rich + det X/rich + d_i6/d_det
+ one-hot (mg, pm, cnxL, R34) + 6 extras. fam: 0 cnx tree, 2 mg, 4 pm, 5 cnxL, 6 R34.
Output work/ideas7/playbook/xcomb/<stem>.npz. Dev folds 0-3 only, CPU only. usage: xcomb.py [--nproc=5]
"""
import json, os, sys, time, hashlib
os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = 'D:/FilamentSegmentationKaggle/claude'
sys.path.insert(0, ROOT + '/src'); sys.path.insert(0, ROOT + '/work/ideas7/playbook')
from concurrent.futures import ProcessPoolExecutor
import numpy as np, cv2
from scipy import ndimage as ndi
from cfil.cands import LEVELS, MIN_AREA, candidates, gt_label_maps, FEATURES
from cfil.build_cands import disk_stats, RAW
import xcands as X, xcands2 as X2, pool_oracle as PO

OUT = ROOT + '/work/ideas7/playbook/xcomb'
H = W = 2048
NE = 6
GAP = 32
ONEHOT = {2: 0, 4: 1, 5: 2, 6: 3}


def one(args):
    k, j, fname = args
    stem = fname[:-5]
    dst = os.path.join(OUT, stem + '.npz')
    if os.path.exists(dst):
        return fname, None
    t0 = time.time()
    m = np.asarray(X._G['maps'][k][j]); P, S = m[0], m[1]
    img = cv2.imread(os.path.join(RAW, 'train/train_images', fname), cv2.IMREAD_GRAYSCALE)
    dm, ds = disk_stats(img)
    gts = gt_label_maps(X._G['by'], X._G['recs'][fname])
    F, par, Hh, keys = candidates(P, S, img, dm, ds, gts)
    Nb = len(F)
    blocks = [np.load(ROOT + '/work/ext/cnx-i6g/' + stem + '.npz', allow_pickle=True)['X'],
              np.load(ROOT + '/work/feat/cnx-i6g-rich/' + stem + '.npz')['X'],
              np.load(ROOT + '/work/ext/cnx-det/' + stem + '.npz', allow_pickle=True)['X'],
              np.load(ROOT + '/work/feat/cnx-detrich/' + stem + '.npz')['X']]
    dI = X2.base_delta(ROOT + '/work/ext/cnx-i6g/' + stem + '.npz'); dD = X2.base_delta(ROOT + '/work/ext/cnx-det/' + stem + '.npz')
    Fb = np.concatenate([F] + blocks + [dI[:, None], dD[:, None]], 1)
    Pf = P.astype(np.float32) / 255
    I = (img.astype(np.float32) - dm) / ds
    labs = [cv2.connectedComponents((P > t8).astype(np.uint8), connectivity=8)[1] for t8 in X.THR]
    node = {(li, kk): i for i, (li, kk, sl) in enumerate(keys)}
    pix = []
    for (li, kk, sl) in keys:
        ys, xs = np.nonzero(labs[li][sl] == kk)
        pix.append(((ys + sl[0].start) * W + xs + sl[1].start).astype(np.int64))
    area_b = np.array([len(q) for q in pix])
    depth = F[:, 14] if Nb else np.zeros(0)
    seen, new = set(), []   # (fam, Q, row28 or None, t, li, et, ex)

    def add(fam, Q, t, li, et, ex, row=None, hits=None):
        if len(Q) < MIN_AREA:
            return
        Q = np.sort(Q)
        h = hashlib.blake2b(Q.tobytes(), digest_size=12).digest()
        if h in seen:
            return
        seen.add(h)
        new.append((fam, Q, t, li, et, list(ex) + [0.0] * (NE - len(ex)), row, hits))

    # mg: closing merges
    for r in (6, 12):
        for li, t8 in enumerate(X.THR):
            B = (P > t8).astype(np.uint8)
            if not B.any():
                continue
            cl = cv2.morphologyEx(B, cv2.MORPH_CLOSE, X.K(r))
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
    # pm: pairwise merges
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
    # pooled tree nodes of the other dense models (own-map features and hits)
    for code, name in ((5, 'cnxL'), (6, 'r34')):
        arr, jj = PO._G['other'][(name, fname)]
        m2 = np.asarray(arr[jj])
        F2, par2, H2, keys2 = candidates(m2[0], m2[1], img, dm, ds, gts)
        s2 = PO.tree_sets(m2[0], keys2)
        for i in range(len(s2)):
            add(code, s2[i], float(F2[i, 0]), int(F2[i, 1]), None, [], row=F2[i], hits=H2[i])
    # drop near-duplicates of tree nodes and of earlier new rows
    Cb, lb = X2.csr_of(pix)
    if new:
        Cn0, ln0 = X2.csr_of([x[1] for x in new])
        Inb = X2.iou_sparse(Cn0, ln0, Cb, lb) if Nb else np.zeros((len(new), 0), np.float32)
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
    Nn = len(new)
    img_n50, img_pmean = (F[0, 24], F[0, 25]) if Nb else (0, Pf.mean())
    pm_base = np.sort(F[:, 7]) if Nb else np.zeros(1)
    Fn = np.zeros((Nn, len(FEATURES)), np.float32); Hn = np.zeros((Nn, len(gts)), np.float32)
    Xn = np.zeros((Nn, 4 + NE), np.float32)
    for a, (fam, Q, t, li, et, ex, row, hits) in enumerate(new):
        if row is None:
            row = X.feat_row(Q, t, li, Pf, I, S, et)
            row[24], row[25] = img_n50, img_pmean
            row[27] = 1 - np.searchsorted(pm_base, row[7]) / max(len(pm_base), 1)
            Hn[a] = X.hits_of(Q, gts)
        else:
            Hn[a] = hits
        Fn[a] = row; Xn[a, ONEHOT[fam]] = 1; Xn[a, 4:] = ex
    conf = []
    for i in range(Nb):
        p = par[i]
        while p >= 0:
            conf.append((p, i)); p = par[p]
    if Nn:
        Cn, ln = X2.csr_of([x[1] for x in new])
        xi, ri, di = X2.ext_block(Cn, ln, ROOT + '/work/ext/cnx-i6g/' + stem + '.npz', Fn)
        xd, rd, dd = X2.ext_block(Cn, ln, ROOT + '/work/ext/cnx-det/' + stem + '.npz', Fn)
        Fnew = np.concatenate([Fn, xi, ri, xd, rd, di[:, None], dd[:, None], Xn], 1)
        a_, b_ = np.nonzero(Inb > 0)
        conf += list(zip(b_.tolist(), (Nb + a_).tolist()))
        nn = (Cn @ Cn.T).tocoo(); u = nn.row < nn.col
        conf += list(zip((Nb + nn.row[u]).tolist(), (Nb + nn.col[u]).tolist()))
    else:
        Fnew = np.zeros((0, Fb.shape[1] + 4 + NE), np.float32)
    Fall = np.concatenate([np.concatenate([Fb, np.zeros((Nb, 4 + NE), np.float32)], 1), Fnew]).astype(np.float32)
    fam = np.concatenate([np.zeros(Nb, np.int8), np.array([x[0] for x in new], np.int8)])
    np.savez_compressed(dst, F=Fall, fam=fam, parents=np.concatenate([par, np.full(Nn, -1)]),
                        hits=np.concatenate([Hh, Hn]) if Nn else Hh, ngt=np.asarray([len(a) - 1 for _, a in gts]),
                        conf=np.asarray(conf, np.int32).reshape(-1, 2), nb=Nb)
    return fname, (Nb, [int((fam == f).sum()) for f in (2, 4, 5, 6)], round(time.time() - t0, 1))


def main():
    opt = dict(a[2:].split('=', 1) for a in sys.argv[1:] if a.startswith('--') and '=' in a)
    os.makedirs(OUT, exist_ok=True)
    jobs = []
    for k in range(4):
        mp = ROOT + f'/work/maps/cnx-f{k}/oof.npy'
        jobs += [(k, j, f) for j, f in enumerate(json.load(open(mp + '.json'))['files'])]
    t0 = time.time()
    with ProcessPoolExecutor(int(opt.get('nproc', 5)), initializer=PO._init) as ex:
        for n, (f, info) in enumerate(ex.map(one, jobs, chunksize=1)):
            if n % 50 == 0:
                print(n, f, info, f'elapsed {time.time() - t0:.0f}s', flush=True)
    print('done', len(jobs), f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
