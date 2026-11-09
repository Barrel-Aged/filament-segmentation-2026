"""P1 pool-xm, step 2: the xm_check.py protocol (work/ideas7/verify-playbook/xm_check.py: xcv.py bag of 5 LightGBM seeds,
dev folds 0-3, day-group cross-fitting, tree DP on base, HiGHS ILP elsewhere) on the work/ideas7/playbook/xcomb rows with
REAL second-opinion columns on every row (work/ideas7/exp/p1_poolxm/xm/<stem>.npz, X[N, 18] = R34 | cnxL | cnx blocks of
src/cfil/xmap.py's 6 columns) instead of NaN on new rows.
Arms (identical rows and seeds):
  base   = xm12: tree rows (fam 0), F[74] + R34 + cnxL blocks (reproduces xm_check base: 0.4636 s0-4 / 0.4648 s5-9)
  pool   = xm18: fam 0, 5, 6        comb  = xm18: fam 0, 2, 4, 5, 6
  poolL  = xm18: fam 0, 5           poolR = xm18: fam 0, 6
  base18 = xm18 on tree rows (diagnostic only)
Writes q/<arm>_<tag>.npz: names, folds, taus, R[ntau, nimg, 4] (S, TP, FP, FN per image), C[ntau, nimg, 4] (tree-rows
control: the arm's scorer selecting from tree rows only, tree DP), q (object array of per-row scores of the arm's rows),
famsel (selected rows per family at tau 0.30), seeds. CPU only; LightGBM 5 threads.
usage: run_arms.py <seeds e.g. 0,1,2,3,4> <arms e.g. base,pool,comb> <tag e.g. s04>"""
import glob, os, sys, time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = 'D:/FilamentSegmentationKaggle/claude'
seeds, arms, tag = sys.argv[1], sys.argv[2].split(','), sys.argv[3]
sys.argv = [sys.argv[0], f'--seeds={seeds}']
sys.path.insert(0, ROOT + '/src'); sys.path.insert(0, ROOT + '/work/ideas7/playbook')
import numpy as np
import xcv  # functions only (sub, select, counts, pq, fit_predict); also caps LightGBM at 5 threads
from cfil.decide import FOLD, thin
HERE = ROOT + '/work/ideas7/exp/p1_poolxm'
QD = HERE + '/q'
xcv.FAMS.update({'comb': (0, 2, 4, 5, 6), 'pool': (0, 5, 6), 'poolL': (0, 5), 'poolR': (0, 6), 'base18': (0,),
                  'poolS': (0, 5, 6), 'combS': (0, 2, 4, 5, 6)})
NCOL = {'base': 12, 'poolS': 'sym', 'combS': 'sym'}  # xm columns per arm (default 18)
# 'sym' (exploratory, added after the main arms ran): 12 columns = the opinions of the two maps the row does NOT come from,
# slot A then slot B: cnx-source rows (tree, merges) R34 | cnxL (identical to base(xm12) on tree rows), cnxL rows R34 | cnx,
# R34 rows cnxL | cnx. No self-agreement block.


def xm_cols(X, fam, ncol):
    if ncol != 'sym':
        return X[:, :ncol]
    A = np.where((fam == 6)[:, None], X[:, 6:12], X[:, 0:6])
    B = np.where(np.isin(fam, (5, 6))[:, None], X[:, 12:18], X[:, 6:12])
    return np.concatenate([A, B], 1)


def load_imgs(ncol):
    imgs = []
    for p in sorted(glob.glob(ROOT + '/work/ideas7/playbook/xcomb/*.npz')):
        name = os.path.basename(p)[:-4]
        if FOLD[name] not in (0, 1, 2, 3):
            continue
        z = np.load(p); nb = int(z['nb'])
        x = np.load(HERE + f'/xm/{name}.npz')
        assert bool(x['ok']) and int(x['nb']) == nb and len(x['X']) == len(z['F']) and np.array_equal(x['fam'], z['fam'])
        imgs.append(dict(name=name, fold=FOLD[name], F=np.concatenate([z['F'], xm_cols(x['X'], z['fam'], ncol)], 1), fam=z['fam'],
                         parents=z['parents'], hits=z['hits'], ngt=z['ngt'], conf=z['conf'], nb=nb))
    return imgs


def control(im, qq, t):
    bb = np.flatnonzero(im['fam'] == 0)
    sb = dict(im, F=im['F'][bb], hits=im['hits'][bb], parents=im['parents'][bb], fam=im['fam'][bb], base_only=True)
    return xcv.counts(sb, xcv.select(sb, np.asarray(qq)[bb], t))


def main():
    os.makedirs(QD, exist_ok=True)
    pq, TAUS = xcv.pq, xcv.TAUS
    cache = {}
    t0 = time.time()
    for arm in arms:
        nc = NCOL.get(arm, 18)
        if nc not in cache:
            cache[nc] = load_imgs(nc)
        imgs = cache[nc]
        simgs = [xcv.sub(im, 'base' if arm == 'base18' else arm) for im in imgs]
        print(f'{arm}: images {len(simgs)} rows {sum(len(im["F"]) for im in simgs)} cols {simgs[0]["F"].shape[1]} seeds {xcv.SEEDS}',
              flush=True)
        q = xcv.fit_predict(simgs)
        R = np.stack([np.array([xcv.counts(im, xcv.select(im, qq, t)) for im, qq in zip(simgs, q)]) for t in TAUS])
        C = np.stack([np.array([control(im, qq, t) for im, qq in zip(simgs, q)]) for t in TAUS])
        i30 = int(np.flatnonzero(np.isclose(TAUS, 0.30))[0])
        sel30 = [xcv.select(im, qq, 0.30) for im, qq in zip(simgs, q)]
        famsel = np.bincount(np.concatenate([im['fam'][s] for im, s in zip(simgs, sel30)]).astype(np.int64), minlength=7)
        np.savez(QD + f'/{arm}_{tag}.npz', names=np.array([im['name'] for im in simgs]), folds=np.array([im['fold'] for im in simgs]),
                 taus=TAUS, R=R, C=C, q=np.array(q, dtype=object), famsel=famsel, seeds=np.array(xcv.SEEDS), fams=np.array(xcv.FAMS[arm]))
        print(f'{arm:6s} {tag} | scorer ' + ' '.join(f'{t}:{pq(R[i].sum(0)):.4f}' for i, t in enumerate(TAUS)) +
              f' | thin@0.30 {thin(*R[i30].sum(0)):.4f} | sel@0.30 by fam {famsel.tolist()} | TP/FP/FN@0.30 {R[i30].sum(0)[1:].astype(int).tolist()}'
              f' | control ' + ' '.join(f'{t}:{pq(C[i].sum(0)):.4f}' for i, t in enumerate(TAUS)) + f' | {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
