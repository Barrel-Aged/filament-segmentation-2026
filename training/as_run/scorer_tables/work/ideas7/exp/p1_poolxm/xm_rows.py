"""P1 pool-xm, step 1: regenerate every row of work/ideas7/playbook/xcomb/<stem>.npz with its pixel set (same code path as
work/ideas7/playbook/xcomb.py: cnx tree, closing merges mg, pairwise merges pm, pooled cnxL / R34 tree nodes, IoU >= 0.9
near-duplicate filter), assert the regenerated feature rows equal the stored F (atol 1e-5) and fam / parents / nb / conf,
then compute the 6 src/cfil/xmap.py columns of every row against the R34, cnxL and cnx OOF maps of the image's fold
-> X[N, 18] (R34 block, cnxL block, cnx block). The level of a row is F[:, 1] (merges: the level of the merged nodes).
Tree rows: the R34 / cnxL blocks are checked against work/feat/cnx-xmR / cnx-xmL (atol 1e-4).
Also stores, for the deployment-loss emulation, each new row's best IoU to the cnx tree rows (bt) and the new-new pairs with
IoU >= 0.5 (pairs, piou; row indices into the full row list).
No labels are read (candidates(..., gts=None); hits are not regenerated). Dev folds 0-3 only, CPU only.
Output work/ideas7/exp/p1_poolxm/xm/<stem>.npz. usage: xm_rows.py [--nproc=5] [--limit=N]
"""
import json, os, sys, time, hashlib
os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = 'D:/FilamentSegmentationKaggle/claude'
sys.path.insert(0, ROOT + '/src'); sys.path.insert(0, ROOT + '/work/ideas7/playbook')
from concurrent.futures import ProcessPoolExecutor
import numpy as np, cv2
from scipy import ndimage as ndi
from cfil.cands import LEVELS, MIN_AREA, candidates, FEATURES
from cfil.build_cands import disk_stats, RAW
import xcands as X, xcands2 as X2, pool_oracle as PO

HERE = ROOT + '/work/ideas7/exp/p1_poolxm'
OUT = HERE + '/xm'
XC = ROOT + '/work/ideas7/playbook/xcomb'
H = W = 2048
NE = 6
GAP = 32
ONEHOT = {2: 0, 4: 1, 5: 2, 6: 3}
PAD = 4
_K1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
MAPS = ('r34', 'cnxL', 'cnx')   # xm block order
_G = {}


def _init():
    _G['maps'] = {}
    _G['idx'] = {}
    for name in ('cnx', 'cnxL', 'r34'):
        for k in range(4):
            mp = ROOT + f'/work/maps/{name}-f{k}/oof.npy'
            arr = np.load(mp, mmap_mode='r')
            for j, f in enumerate(json.load(open(mp + '.json'))['files']):
                _G['idx'][(name, f)] = (arr, j)


def level_stack(Pm):
    """per level: flat component labels of Pm > THR[li] and component areas (as xmap.py: level_labels, no closing)"""
    out = []
    for t8 in X.THR:
        lab = cv2.connectedComponents((Pm > t8).astype(np.uint8), connectivity=8)[1]
        out.append((lab.ravel(), np.bincount(lab.ravel())))
    return out


def xm_block(sets, lis, Pm, levs):
    """xmap.py's 6 columns for arbitrary pixel sets (sorted flat indices) against map Pm (uint8)."""
    Qf = Pm.astype(np.float32) / 255
    Qflat = Qf.ravel()
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


def regen(fname):
    """xcomb.py one() without labels; returns Fall, fam, parents, conf, nb, row pixel sets, levels, bt, new-new pairs."""
    stem = fname[:-5]
    arr, j = _G['idx'][('cnx', fname)]
    m = np.asarray(arr[j]); P, S = m[0], m[1]
    img = cv2.imread(os.path.join(RAW, 'train/train_images', fname), cv2.IMREAD_GRAYSCALE)
    dm, ds = disk_stats(img)
    F, par, _, keys = candidates(P, S, img, dm, ds, None)
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
    seen, new = set(), []

    def add(fam, Q, t, li, et, ex, row=None):
        if len(Q) < MIN_AREA:
            return
        Q = np.sort(Q)
        h = hashlib.blake2b(Q.tobytes(), digest_size=12).digest()
        if h in seen:
            return
        seen.add(h)
        new.append((fam, Q, t, li, et, list(ex) + [0.0] * (NE - len(ex)), row))

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
    # pooled tree nodes of the other dense models (own-map features)
    other_P = {}
    for code, name in ((5, 'cnxL'), (6, 'r34')):
        arr2, jj = _G['idx'][(name, fname)]
        m2 = np.asarray(arr2[jj]); other_P[name] = m2[0]
        F2, par2, _, keys2 = candidates(m2[0], m2[1], img, dm, ds, None)
        s2 = PO.tree_sets(m2[0], keys2)
        for i in range(len(s2)):
            add(code, s2[i], float(F2[i, 0]), int(F2[i, 1]), None, [], row=F2[i])
    # near-duplicate filter (as xcomb.py)
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
    Fn = np.zeros((Nn, len(FEATURES)), np.float32)
    Xn = np.zeros((Nn, 4 + NE), np.float32)
    for a, (fam, Q, t, li, et, ex, row) in enumerate(new):
        if row is None:
            row = X.feat_row(Q, t, li, Pf, I, S, et)
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
    if Nn:
        Cn, ln = X2.csr_of([x[1] for x in new])
        xi, ri, di = X2.ext_block(Cn, ln, ROOT + '/work/ext/cnx-i6g/' + stem + '.npz', Fn)
        xd, rd, dd = X2.ext_block(Cn, ln, ROOT + '/work/ext/cnx-det/' + stem + '.npz', Fn)
        Fnew = np.concatenate([Fn, xi, ri, xd, rd, di[:, None], dd[:, None], Xn], 1)
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
    else:
        Fnew = np.zeros((0, Fb.shape[1] + 4 + NE), np.float32)
    Fall = np.concatenate([np.concatenate([Fb, np.zeros((Nb, 4 + NE), np.float32)], 1), Fnew]).astype(np.float32)
    fam = np.concatenate([np.zeros(Nb, np.int8), np.array([x[0] for x in new], np.int8)])
    sets = pix + [x[1] for x in new]
    lis = [k_[0] for k_ in keys] + [int(x[3]) for x in new]
    return dict(F=Fall, fam=fam, parents=np.concatenate([par, np.full(Nn, -1)]), conf=np.asarray(conf, np.int32).reshape(-1, 2),
                nb=Nb, sets=sets, lis=lis, bt=bt, pairs=pairs, piou=piou, P=P, other_P=other_P)


def one(args):
    k, fname = args
    stem = fname[:-5]
    dst = os.path.join(OUT, stem + '.npz')
    if os.path.exists(dst):
        z = np.load(dst)
        return fname, dict(skip=True, ok=bool(z['ok']))
    t0 = time.time()
    g = regen(fname)
    z = np.load(os.path.join(XC, stem + '.npz'))
    Fs = z['F']
    info = dict(N=len(g['F']), nb=g['nb'])
    shape_ok = Fs.shape == g['F'].shape
    info['F_maxdiff'] = float(np.max(np.abs(Fs - g['F']))) if shape_ok and len(Fs) else (0.0 if shape_ok else np.inf)
    info['struct_ok'] = bool(shape_ok and np.array_equal(z['fam'], g['fam']) and np.array_equal(z['parents'], g['parents'])
                             and int(z['nb']) == g['nb'] and np.array_equal(z['conf'], g['conf']))
    info['F_ok'] = bool(shape_ok and np.allclose(Fs, g['F'], rtol=0, atol=1e-5))
    # level index of each row must equal F[:, 1]
    info['lvl_ok'] = bool(np.array_equal(np.asarray(g['lis'], np.float32), g['F'][:, 1])) if len(g['F']) else True
    Xall = np.zeros((len(g['F']), 18), np.float32)
    srcP = {'r34': g['other_P']['r34'], 'cnxL': g['other_P']['cnxL'], 'cnx': g['P']}
    for b, name in enumerate(MAPS):
        levs = level_stack(srcP[name])
        Xall[:, 6 * b:6 * b + 6] = xm_block(g['sets'], g['lis'], srcP[name], levs)
    nb = g['nb']
    a = np.load(ROOT + f'/work/feat/cnx-xmR/{stem}.npz')['X']; c = np.load(ROOT + f'/work/feat/cnx-xmL/{stem}.npz')['X']
    info['xmR_maxdiff'] = float(np.max(np.abs(a - Xall[:nb, :6]))) if nb else 0.0
    info['xmL_maxdiff'] = float(np.max(np.abs(c - Xall[:nb, 6:12]))) if nb else 0.0
    info['xm_ok'] = bool(a.shape[0] == nb and c.shape[0] == nb and info['xmR_maxdiff'] <= 1e-4 and info['xmL_maxdiff'] <= 1e-4)
    ok = info['F_ok'] and info['struct_ok'] and info['lvl_ok'] and info['xm_ok']
    info['ok'] = ok
    info['sec'] = round(time.time() - t0, 1)
    np.savez_compressed(dst, X=Xall, nb=nb, fam=g['fam'], lvl=np.asarray(g['lis'], np.int8), bt=g['bt'], pairs=g['pairs'],
                        piou=g['piou'], area=np.array([len(s) for s in g['sets']], np.int64), ok=ok,
                        F_maxdiff=info['F_maxdiff'], xmR_maxdiff=info['xmR_maxdiff'], xmL_maxdiff=info['xmL_maxdiff'])
    return fname, info


def main():
    opt = dict(a[2:].split('=', 1) for a in sys.argv[1:] if a.startswith('--') and '=' in a)
    os.makedirs(OUT, exist_ok=True)
    jobs = []
    for k in [int(x) for x in opt.get('folds', '0,1,2,3').split(',')]:
        mp = ROOT + f'/work/maps/cnx-f{k}/oof.npy'
        jobs += [(k, f) for f in json.load(open(mp + '.json'))['files']]
    if 'limit' in opt:
        jobs = jobs[::max(len(jobs) // int(opt['limit']), 1)][:int(opt['limit'])]
    t0 = time.time()
    bad, rows, mx = [], 0, dict(F=0.0, R=0.0, L=0.0)
    with ProcessPoolExecutor(int(opt.get('nproc', 5)), initializer=_init) as ex:
        for n, (f, info) in enumerate(ex.map(one, jobs, chunksize=1)):
            if info.get('skip'):
                if not info['ok']:
                    bad.append((f, 'stored not ok'))
                continue
            rows += info['N']
            mx['F'] = max(mx['F'], info['F_maxdiff']); mx['R'] = max(mx['R'], info['xmR_maxdiff']); mx['L'] = max(mx['L'], info['xmL_maxdiff'])
            if not info['ok']:
                bad.append((f, info)); print('MISMATCH', f, info, flush=True)
            if n % 25 == 0:
                print(n, f, info, f'elapsed {time.time() - t0:.0f}s', flush=True)
    print('done', len(jobs), 'rows', rows, 'max|dF|', mx['F'], 'max|dxmR|', mx['R'], 'max|dxmL|', mx['L'],
          'mismatches', len(bad), f'{time.time() - t0:.0f}s', flush=True)
    for b in bad[:20]:
        print('BAD', b)


if __name__ == '__main__':
    main()
