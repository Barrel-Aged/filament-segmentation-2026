"""ideas11/synth: own evaluation library (CPU only, dev folds 0-3 only; nothing outside work/ideas11/synth is written).

Rows: run_arms.load_imgs('sym') + xcv.sub(im, 'combS') (read-only import of the harness loader; same rows as every lens).
Selection (own code): exact maximum of sum(s - tau) over conflict-free subsets of the rows with s > tau, solved per connected
component of the conflict graph among those rows: components of <= 20 rows by exhaustive branch-and-bound on bitsets (no
solver), larger ones by scipy.optimize.milp (HiGHS) with mip_rel_gap = 0.
Counting (own code): per annotator record r, TP_r = selected rows with a stored hit IoU > 0 (the stored hits hold the IoU
of a strict IoU > 0.5 match, else 0), FP_r = n_sel - TP_r, FN_r = ngt_r - TP_r, S_r = sum of the stored hit IoUs; pooled.
Score: s = q - 0.2 |t - 0.475| - 0.03 [fam != 0], t = F[:, 0] (the frozen B3f shrinkage).
"""
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = 'D:/FilamentSegmentationKaggle/claude'
HERE = ROOT + '/work/ideas11/synth'
for p in ('/src', '/work/ideas7/playbook', '/work/ideas7/exp/p1_poolxm'):
    if ROOT + p not in sys.path:
        sys.path.insert(0, ROOT + p)
_argv = sys.argv
sys.argv = [_argv[0], '0,1,2,3,4', 'base', 'unused']   # run_arms parses argv at import, then imports xcv (5-thread cap)
import run_arms  # noqa: E402
import xcv  # noqa: E402
sys.argv = _argv
import numpy as np  # noqa: E402
from scipy.optimize import milp, LinearConstraint, Bounds  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402

FAMNAME = {0: 'tree', 2: 'mg', 4: 'pm', 5: 'cnxL', 6: 'R34'}
T0, LAM, DELTA = 0.475, 0.2, 0.03
R_TASK = 0.755


def load():
    imgs = run_arms.load_imgs('sym')
    s = [xcv.sub(im, 'combS') for im in imgs]
    assert len(s) == 566 and sorted({im['fold'] for im in s}) == [0, 1, 2, 3]
    return s


def b3f_score(s, q):
    t = s['F'][:, 0].astype(np.float64)
    return np.asarray(q, np.float64) - LAM * np.abs(t - T0) - DELTA * (s['fam'] != 0).astype(np.float64)


def _mwis_small(w, nb):
    """exact max-weight independent set on <= 20 nodes; w > 0; nb[i] = bitmask of i's neighbours. Returns bitmask."""
    n = len(w)
    best = [0.0, 0]
    suffix = np.concatenate([np.cumsum(w[::-1])[::-1], [0.0]])

    def rec(cand, cur, curm):
        if cand == 0:
            if cur > best[0]:
                best[0], best[1] = cur, curm
            return
        # bound: current + all remaining candidate weights
        ub = cur
        m = cand
        while m:
            b = m & -m
            ub += w[b.bit_length() - 1]
            m ^= b
        if ub <= best[0]:
            return
        b = cand & -cand
        v = b.bit_length() - 1
        rec(cand & ~b & ~nb[v], cur + w[v], curm | b)   # take v
        rec(cand & ~b, cur, curm)                         # skip v

    rec((1 << n) - 1, 0.0, 0)
    return best[1]


def select(s, sc, tau):
    w = np.asarray(sc, np.float64) - tau
    pos = np.flatnonzero(w > 0)
    if not len(pos):
        return pos
    loc = -np.ones(len(w), np.int64); loc[pos] = np.arange(len(pos))
    c = s['conf']
    if len(c):
        c = c[(loc[c[:, 0]] >= 0) & (loc[c[:, 1]] >= 0)]
    if not len(c):
        return pos
    a, b = loc[c[:, 0]], loc[c[:, 1]]
    G = coo_matrix((np.ones(len(a)), (a, b)), shape=(len(pos), len(pos)))
    ncc, lab = connected_components(G, directed=False)
    out = []
    order = np.argsort(lab, kind='stable')
    starts = np.r_[0, np.flatnonzero(np.diff(lab[order])) + 1, len(order)]
    for k in range(ncc):
        nodes = order[starts[k]:starts[k + 1]]
        if len(nodes) == 1:
            out.append(pos[nodes[0]]); continue
        ln = -np.ones(len(pos), np.int64); ln[nodes] = np.arange(len(nodes))
        e = (ln[a] >= 0) & (ln[b] >= 0)
        ea, eb = ln[a[e]], ln[b[e]]
        ww = w[pos[nodes]]
        if len(nodes) <= 20:
            nb = [0] * len(nodes)
            for x, y in zip(ea.tolist(), eb.tolist()):
                if x != y:
                    nb[x] |= 1 << y; nb[y] |= 1 << x
            m = _mwis_small(ww, nb)
            out.extend(pos[nodes[[i for i in range(len(nodes)) if (m >> i) & 1]]])
        else:
            keep = ea != eb
            ea, eb = ea[keep], eb[keep]
            A = coo_matrix((np.ones(2 * len(ea)), (np.repeat(np.arange(len(ea)), 2), np.stack([ea, eb], 1).ravel())),
                           shape=(len(ea), len(nodes)))
            r = milp(c=-ww, constraints=LinearConstraint(A, -np.inf, 1), integrality=np.ones(len(nodes)),
                     bounds=Bounds(0, 1), options=dict(mip_rel_gap=0.0))
            out.extend(pos[nodes[r.x > 0.5]])
    return np.sort(np.asarray(out, np.int64))


def conflict_free(s, sel):
    if len(sel) < 2 or not len(s['conf']):
        return True
    m = np.zeros(len(s['F']), bool); m[sel] = True
    c = s['conf']
    return not np.any(m[c[:, 0]] & m[c[:, 1]] & (c[:, 0] != c[:, 1]))


def counts(s, sel):
    h = np.asarray(s['hits'][sel], np.float64)
    tp = (h > 0).sum(0)
    return np.array([h.sum(), tp.sum(), (len(sel) - tp).sum(), (np.asarray(s['ngt']) - tp).sum()], np.float64)


def pq(c):
    c = np.asarray(c, np.float64)
    return c[..., 0] / (c[..., 1] + 0.5 * (c[..., 2] + c[..., 3]))


def thin(c, r=R_TASK):
    c = np.asarray(c, np.float64)
    S, TP, FP, FN = c[..., 0], c[..., 1], c[..., 2], c[..., 3]
    return r * S / (0.5 * (1 + r) * TP + 0.5 * FP + 0.5 * r * FN)


def boot_weights(m, n=2000, seed=0):
    """the resampling of P1's compare.boot (numpy default_rng(seed), integers(m, size=m) per draw, bincount weights)"""
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.integers(m, size=m), minlength=m) for _ in range(n)]).astype(np.float64)


_W = {}


def boot_pairs(pairs, fn=pq):
    """paired image bootstrap of the replicate-mean delta mean_r [fn(W Ra_r) - fn(W Rb_r)]; one pair = compare.boot."""
    m = len(pairs[0][0])
    if m not in _W:
        _W[m] = boot_weights(m)
    W = _W[m]
    d = np.mean([fn(W @ Ra) - fn(W @ Rb) for Ra, Rb in pairs], 0)
    return float(d.std()), float(np.mean(d > 0))


# ---------------------------------------------------------------- multiprocess grid evaluation
_IM = {}


def _init():
    os.environ['OMP_NUM_THREADS'] = '1'
    for i, s in enumerate(load()):
        _IM[i] = s


def _work(args):
    idx, qd, taus = args
    out = {}
    for j, i in enumerate(idx):
        s = _IM[i]
        res = {}
        for name, qs in qd.items():
            sc = b3f_score(s, qs[j])
            C = np.zeros((len(taus), 4)); FS = np.zeros((len(taus), 7), np.int64); ok = True
            for a, tau in enumerate(taus):
                sel = select(s, sc, tau)
                ok &= conflict_free(s, sel)
                C[a] = counts(s, sel)
                FS[a] = np.bincount(s['fam'][sel].astype(np.int64), minlength=7)
            res[name] = (C, FS, ok)
        out[i] = res
    return out


def grid(jobs, taus, n_img, nproc=4, sizes=None):
    """jobs: {name: [per-image q arrays]} -> {name: (C[ntau, nimg, 4], FS[ntau, nimg, 7])}, all selections conflict-free."""
    from concurrent.futures import ProcessPoolExecutor
    names = list(jobs)
    order = np.argsort(-np.asarray(sizes)) if sizes is not None else np.arange(n_img)
    chunks = [order[i::nproc * 6] for i in range(nproc * 6)]
    res = {}
    with ProcessPoolExecutor(nproc, initializer=_init) as ex:
        futs = [ex.submit(_work, (c.tolist(), {n: [jobs[n][i] for i in c] for n in names}, list(taus))) for c in chunks]
        for f in futs:
            res.update(f.result())
    out = {}
    bad = 0
    for n in names:
        C = np.zeros((len(taus), n_img, 4)); FS = np.zeros((len(taus), n_img, 7), np.int64)
        for i in range(n_img):
            C[:, i], FS[:, i], ok = res[i][n]
            bad += (not ok)
        out[n] = (C, FS)
    assert bad == 0, f'{bad} selections with a conflicting pair'
    return out


def load_q(path, names):
    z = np.load(path, allow_pickle=True)
    assert list(z['names']) == list(names), path
    return [np.asarray(x, np.float64) for x in z['q']]


def mix(ws):
    n = len(ws[0][1])
    return [sum(w * x[i] for w, x in ws) for i in range(n)]
