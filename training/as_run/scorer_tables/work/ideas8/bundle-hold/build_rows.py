"""Fold-4 holdout candidate rows for the comb pool, built exactly as work/ideas7/playbook/xcomb.py builds the dev rows
(same code path, copied; checked against the stored dev rows by `--devcheck`):
  cnx tree rows first (== work/cands/holdcnx), then closing-group merges (mg, fam 2, r 6/12), pairwise merges (pm, fam 4,
  gap <= 32 px), cnxL tree nodes (fam 5) and R34 tree nodes (fam 6) of the holdout maps; exact duplicates hashed,
  IoU >= 0.9 near-duplicates of tree rows / earlier new rows dropped, MIN_AREA 40.
Holdout inputs (4-model averages): work/maps/final/holdcnx.npy, holdcnxL.npy, holdr34.npy; evidence: work/ext/holdcnx-i6g,
work/feat/holdcnx-i6g-rich, and per detector variant k = 0..3 work/ext/holdcnx-det-f{k}, work/feat/holdcnx-detrich-f{k}.
Per image writes rows/<stem>.npz:
  F[4, N, 74]  variant axis first (k = detector fold model); per variant the dev xcomb layout: 28 tree features | I6 X 6 |
               I6 rich 11 | det X 6 | det rich 11 | d_i6 | d_det | one-hot (mg, pm, cnxL, R34) | 6 family extras.
               Only det X / det rich / d_det (columns 45-61 and 63) differ between variants.
  xm[N, 18]    src/cfil/xmap.py's 6 columns against the holdout R34 | cnxL | cnx maps (xm_rows.py layout)
  fam[N] (0 tree, 2 mg, 4 pm, 5 cnxL, 6 R34), parents[N] (-1 for new rows), conf[M, 2] (overlap / ancestor pairs),
  nb (number of tree rows), hits[N, R] (strict IoU > 0.5 hit IoU per fold-4 record), ngt[R], lvl[N], area[N],
  bt[N] (new rows: best IoU to a tree row), pairs / piou (new-new pairs with IoU >= 0.5), plus self-check scalars.
Fold-4 labels are read only to store per-record hits (and to assert the tree hits equal work/cands/holdcnx); nothing is
evaluated. CPU only. usage: build_rows.py [--nproc=5] [--limit=N] | build_rows.py --devcheck=N
"""
import json, os, sys, time, hashlib
os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = 'D:/FilamentSegmentationKaggle/claude'
sys.path.insert(0, ROOT + '/src'); sys.path.insert(0, ROOT + '/work/ideas7/playbook')
sys.path.insert(0, ROOT + '/work/ideas7/exp/p1_poolxm')
from concurrent.futures import ProcessPoolExecutor
import numpy as np, cv2
from scipy import ndimage as ndi
from cfil.cands import LEVELS, MIN_AREA, candidates, gt_label_maps, FEATURES
from cfil.build_cands import disk_stats, RAW, ANN, SPLIT
import xcands as X, xcands2 as X2, pool_oracle as PO, xm_rows as XM

HERE = ROOT + '/work/ideas8/bundle-hold'
OUT = HERE + '/rows'
H = W = 2048
NE = 6
GAP = 32
ONEHOT = {2: 0, 4: 1, 5: 2, 6: 3}
NV = 4
_G = {}


def gen(P, S, others, img, i6, i6rich, dets, detrichs, gts):
    """xcomb.py one() body for one image. others: {'cnxL': (P, S), 'r34': (P, S)} uint8 maps; dets / detrichs: one ext /
    rich path per detector variant. Returns a dict (F has a leading variant axis)."""
    dm, ds = disk_stats(img)
    F, par, Hh, keys = candidates(P, S, img, dm, ds, gts)
    Nb = len(F)
    bI = [np.load(i6, allow_pickle=True)['X'], np.load(i6rich)['X']]
    dI = X2.base_delta(i6)
    Fb = []
    for dp, rp in zip(dets, detrichs):
        bD = [np.load(dp, allow_pickle=True)['X'], np.load(rp)['X']]
        dD = X2.base_delta(dp)
        Fb.append(np.concatenate([F] + bI + bD + [dI[:, None], dD[:, None]], 1))
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
    seen, new = set(), []   # (fam, Q, t, li, et, ex, row28 or None, hits or None)

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
    del labs
    # pooled tree nodes of the other dense models (own-map features and hits)
    nraw = {}
    for code, name in ((5, 'cnxL'), (6, 'r34')):
        P2, S2 = others[name]
        F2, par2, H2, keys2 = candidates(P2, S2, img, dm, ds, gts)
        s2 = PO.tree_sets(P2, keys2)
        nraw[name] = len(s2)
        for i in range(len(s2)):
            add(code, s2[i], float(F2[i, 0]), int(F2[i, 1]), None, [], row=F2[i], hits=None if H2 is None else H2[i])
    # drop near-duplicates of tree nodes and of earlier new rows
    Cb, lb = X2.csr_of(pix)
    Inb = np.zeros((0, Nb), np.float32)
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
    R = len(gts) if gts is not None else 0
    img_n50, img_pmean = (F[0, 24], F[0, 25]) if Nb else (0, Pf.mean())
    pm_base = np.sort(F[:, 7]) if Nb else np.zeros(1)
    Fn = np.zeros((Nn, len(FEATURES)), np.float32); Hn = np.zeros((Nn, R), np.float32)
    Xn = np.zeros((Nn, 4 + NE), np.float32)
    for a, (fam, Q, t, li, et, ex, row, hits) in enumerate(new):
        if row is None:
            row = X.feat_row(Q, t, li, Pf, I, S, et)
            row[24], row[25] = img_n50, img_pmean
            row[27] = 1 - np.searchsorted(pm_base, row[7]) / max(len(pm_base), 1)
            if gts is not None:
                Hn[a] = X.hits_of(Q, gts)
        elif gts is not None:
            Hn[a] = hits
        Fn[a] = row; Xn[a, ONEHOT[fam]] = 1; Xn[a, 4:] = ex
    conf = []
    for i in range(Nb):
        p = par[i]
        while p >= 0:
            conf.append((p, i)); p = par[p]
    bt = np.zeros(Nb + Nn, np.float32)
    pairs = np.zeros((0, 2), np.int32); piou = np.zeros(0, np.float32)
    # external evidence: one ext_block call per source over tree + new rows (row-independent formulas); the tree part is
    # a self-check against the stored blocks, the new part is used
    sets = pix + [x[1] for x in new]
    Ca, la = X2.csr_of(sets)
    F28 = np.concatenate([F, Fn]) if Nn else F
    chk = {}
    xi, ri, di = X2.ext_block(Ca, la, i6, F28)
    chk['i6'] = max(float(np.max(np.abs(xi[:Nb] - bI[0]))) if Nb else 0.0, float(np.max(np.abs(ri[:Nb] - bI[1]))) if Nb else 0.0,
                    float(np.max(np.abs(di[:Nb] - dI))) if Nb else 0.0)
    Fnew, chk['det'] = [], []
    for dp, rp in zip(dets, detrichs):
        xd, rd, dd = X2.ext_block(Ca, la, dp, F28)
        bD = [np.load(dp, allow_pickle=True)['X'], np.load(rp)['X']]; dD = X2.base_delta(dp)
        chk['det'].append(max(float(np.max(np.abs(xd[:Nb] - bD[0]))) if Nb else 0.0,
                              float(np.max(np.abs(rd[:Nb] - bD[1]))) if Nb else 0.0, float(np.max(np.abs(dd[:Nb] - dD))) if Nb else 0.0))
        Fnew.append(np.concatenate([Fn, xi[Nb:], ri[Nb:], xd[Nb:], rd[Nb:], di[Nb:, None], dd[Nb:, None], Xn], 1) if Nn
                    else np.zeros((0, Fb[0].shape[1] + 4 + NE), np.float32))
    if Nn:
        Cn, ln = X2.csr_of([x[1] for x in new])
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
    out = dict(F=Fall, fam=fam, parents=np.concatenate([par, np.full(Nn, -1)]), conf=np.asarray(conf, np.int32).reshape(-1, 2),
               nb=Nb, sets=sets, lis=[k_[0] for k_ in keys] + [int(x[3]) for x in new], bt=bt, pairs=pairs, piou=piou,
               F28=F, par=par, chk=chk, nraw=nraw)
    if gts is not None:
        out['hits'] = np.concatenate([Hh, Hn]) if Nn else Hh
        out['ngt'] = np.asarray([len(a) - 1 for _, a in gts])
        out['Hh'] = Hh
    return out


def xm18(sets, lis, P_r34, P_cnxL, P_cnx):
    """X[N, 18] = R34 | cnxL | cnx blocks (xm_rows.py MAPS order)."""
    Xall = np.zeros((len(sets), 18), np.float32)
    src = {'r34': P_r34, 'cnxL': P_cnxL, 'cnx': P_cnx}
    for b, name in enumerate(XM.MAPS):
        Xall[:, 6 * b:6 * b + 6] = XM.xm_block(sets, lis, src[name], XM.level_stack(src[name]))
    return Xall


def _init():
    obs = [o for o in json.load(open(SPLIT))['observations'] if o['fold'] == 4]
    keep = {r for o in obs for r in o['annotator_image_ids']}
    by = {}
    for a in json.load(open(ANN))['annotations']:
        if a['image_id'] in keep:
            by.setdefault(a['image_id'], []).append(a)
    _G['by'] = by
    _G['recs'] = {o['file_name']: o['annotator_image_ids'] for o in obs}
    _G['maps'] = {n: np.load(ROOT + f'/work/maps/final/hold{n}.npy', mmap_mode='r') for n in ('cnx', 'cnxL', 'r34')}
    files = {n: json.load(open(ROOT + f'/work/maps/final/hold{n}.npy.json'))['files'] for n in ('cnx', 'cnxL', 'r34')}
    assert files['cnx'] == files['cnxL'] == files['r34']


def one(args):
    j, fname = args
    stem = fname[:-5]
    dst = os.path.join(OUT, stem + '.npz')
    if os.path.exists(dst):
        return fname, dict(skip=True)
    t0 = time.time()
    m = np.asarray(_G['maps']['cnx'][j]); mL = np.asarray(_G['maps']['cnxL'][j]); mR = np.asarray(_G['maps']['r34'][j])
    img = cv2.imread(os.path.join(RAW, 'train/train_images', fname), cv2.IMREAD_GRAYSCALE)
    gts = gt_label_maps(_G['by'], _G['recs'][fname])
    E = ROOT + '/work/ext/'; Fd = ROOT + '/work/feat/'
    g = gen(m[0], m[1], {'cnxL': (mL[0], mL[1]), 'r34': (mR[0], mR[1])}, img,
            E + f'holdcnx-i6g/{stem}.npz', Fd + f'holdcnx-i6g-rich/{stem}.npz',
            [E + f'holdcnx-det-f{k}/{stem}.npz' for k in range(NV)], [Fd + f'holdcnx-detrich-f{k}/{stem}.npz' for k in range(NV)], gts)
    nb = g['nb']
    zc = np.load(ROOT + f'/work/cands/holdcnx/{stem}.npz')
    info = dict(N=len(g['fam']), nb=nb, fam=[int((g['fam'] == f).sum()) for f in (0, 2, 4, 5, 6)], nraw=g['nraw'])
    info['cand_ok'] = bool(zc['F'].shape == g['F28'].shape and np.allclose(zc['F'], g['F28'], rtol=0, atol=1e-5)
                           and np.array_equal(zc['parents'], g['par']))
    info['hits_ok'] = bool(np.array_equal(zc['hits'], g['Hh']) and np.array_equal(zc['ngt'], g['ngt']))  # integrity only
    info['lvl_ok'] = bool(np.array_equal(np.asarray(g['lis'], np.float32), g['F'][0][:, 1])) if info['N'] else True
    info['ext_selfcheck'] = max([g['chk']['i6']] + g['chk']['det'])
    Xm = xm18(g['sets'], g['lis'], mR[0], mL[0], m[0])
    a = np.load(Fd + f'holdcnx-xmR/{stem}.npz')['X']; c = np.load(Fd + f'holdcnx-xmL/{stem}.npz')['X']
    info['xmR_maxdiff'] = float(np.max(np.abs(a - Xm[:nb, :6]))) if nb else 0.0
    info['xmL_maxdiff'] = float(np.max(np.abs(c - Xm[:nb, 6:12]))) if nb else 0.0
    info['xm_ok'] = bool(a.shape[0] == nb and c.shape[0] == nb and info['xmR_maxdiff'] <= 1e-4 and info['xmL_maxdiff'] <= 1e-4)
    ok = info['cand_ok'] and info['hits_ok'] and info['lvl_ok'] and info['xm_ok']
    info['ok'] = ok
    info['sec'] = round(time.time() - t0, 1)
    np.savez_compressed(dst, F=g['F'], xm=Xm, fam=g['fam'], parents=g['parents'], hits=g['hits'], ngt=g['ngt'], conf=g['conf'],
                        nb=nb, lvl=np.asarray(g['lis'], np.int8), area=np.array([len(s) for s in g['sets']], np.int64),
                        bt=g['bt'], pairs=g['pairs'], piou=g['piou'], ok=ok, ext_selfcheck=info['ext_selfcheck'],
                        xmR_maxdiff=info['xmR_maxdiff'], xmL_maxdiff=info['xmL_maxdiff'],
                        nraw=np.array([g['nraw']['cnxL'], g['nraw']['r34']], np.int64))
    return fname, info


def devcheck(n):
    """Run gen()/xm18() on n dev images with the dev inputs (fold OOF maps, work/ext/cnx-i6g, cnx-det, dev labels) and
    compare with the stored dev rows (work/ideas7/playbook/xcomb) and xm columns (work/ideas7/exp/p1_poolxm/xm)."""
    obs = [o for o in json.load(open(SPLIT))['observations'] if o['fold'] in (0, 1, 2, 3)]
    keep = {r for o in obs for r in o['annotator_image_ids']}
    by = {}
    for a in json.load(open(ANN))['annotations']:
        if a['image_id'] in keep:
            by.setdefault(a['image_id'], []).append(a)
    recs = {o['file_name']: o['annotator_image_ids'] for o in obs}
    jobs = []
    for k in range(4):
        jobs += [(k, f) for f in json.load(open(ROOT + f'/work/maps/cnx-f{k}/oof.npy.json'))['files']]
    jobs = jobs[::max(len(jobs) // n, 1)][:n]
    for k, fname in jobs:
        stem = fname[:-5]
        t0 = time.time()
        mp = {}
        for name in ('cnx', 'cnxL', 'r34'):
            p = ROOT + f'/work/maps/{name}-f{k}/oof.npy'
            mp[name] = np.asarray(np.load(p, mmap_mode='r')[json.load(open(p + '.json'))['files'].index(fname)])
        img = cv2.imread(os.path.join(RAW, 'train/train_images', fname), cv2.IMREAD_GRAYSCALE)
        gts = gt_label_maps(by, recs[fname])
        g = gen(mp['cnx'][0], mp['cnx'][1], {'cnxL': (mp['cnxL'][0], mp['cnxL'][1]), 'r34': (mp['r34'][0], mp['r34'][1])}, img,
                ROOT + f'/work/ext/cnx-i6g/{stem}.npz', ROOT + f'/work/feat/cnx-i6g-rich/{stem}.npz',
                [ROOT + f'/work/ext/cnx-det/{stem}.npz'], [ROOT + f'/work/feat/cnx-detrich/{stem}.npz'], gts)
        z = np.load(ROOT + f'/work/ideas7/playbook/xcomb/{stem}.npz')
        x = np.load(ROOT + f'/work/ideas7/exp/p1_poolxm/xm/{stem}.npz')
        Xm = xm18(g['sets'], g['lis'], mp['r34'][0], mp['cnxL'][0], mp['cnx'][0])
        F0 = g['F'][0]
        same_shape = F0.shape == z['F'].shape
        print(fname, 'fold', k, 'N', len(F0), 'nb', g['nb'], '| F shape eq', same_shape,
              'max|dF|', float(np.max(np.abs(F0 - z['F']))) if same_shape and len(F0) else None,
              '| fam/parents/conf/nb/hits/ngt eq', np.array_equal(g['fam'], z['fam']), np.array_equal(g['parents'], z['parents']),
              np.array_equal(g['conf'], z['conf']), g['nb'] == int(z['nb']), np.array_equal(g['hits'], z['hits']),
              np.array_equal(g['ngt'], z['ngt']),
              '| max|dxm|', float(np.max(np.abs(Xm - x['X']))) if Xm.shape == x['X'].shape and len(Xm) else None,
              '| bt/pairs/piou eq', np.array_equal(g['bt'], x['bt']), np.array_equal(g['pairs'], x['pairs']),
              np.allclose(g['piou'], x['piou']), '| ext self-check', round(max([g['chk']['i6']] + g['chk']['det']), 6),
              f'| {time.time() - t0:.1f}s', flush=True)


def main():
    opt = dict(a[2:].split('=', 1) for a in sys.argv[1:] if a.startswith('--') and '=' in a)
    if 'devcheck' in opt:
        devcheck(int(opt['devcheck'])); return
    os.makedirs(OUT, exist_ok=True)
    files = json.load(open(ROOT + '/work/maps/final/holdcnx.npy.json'))['files']
    jobs = list(enumerate(files))
    if 'limit' in opt:
        jobs = jobs[:int(opt['limit'])]
    t0 = time.time()
    bad, rows = [], 0
    with ProcessPoolExecutor(int(opt.get('nproc', 5)), initializer=_init) as ex:
        for n, (f, info) in enumerate(ex.map(one, jobs, chunksize=1)):
            if info.get('skip'):
                continue
            rows += info['N']
            if not info['ok']:
                bad.append((f, info)); print('MISMATCH', f, info, flush=True)
            if n % 10 == 0:
                print(n, f, info, f'elapsed {time.time() - t0:.0f}s', flush=True)
    print('done', len(jobs), 'rows', rows, 'mismatches', len(bad), f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
