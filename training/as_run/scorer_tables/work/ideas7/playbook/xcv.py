"""Playbook diagnostic 2b: what are the new candidate families worth after scoring and exact selection?

Reads work/ideas7/playbook/xc/*.npz (xcands.py). Arms = candidate pools: base tree only, base + sp (spine-seeded
split), base + mg (closing merges), base + tb (tubes), base + all. Same 35 columns in every arm (28 tree features +
family one-hot + 4 family extras; zeros on tree rows). Scorer = bagged LightGBM (seeds 0-2, production params, 5
threads) on the record-mean hit IoU, cross-fitted over dev folds 0-3 (day groups). Selection = exact max of
sum(q - tau) over conflict-free sets: tree DP on the base arm, ILP (HiGHS) per conflict component elsewhere.
Also reports the oracle (true record-mean hits as scores) per arm. CPU only.
usage: xcv.py [--seeds=0,1,2] [--arms=base,sp,mg,tb,all]
"""
import glob, os, sys, time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = 'D:/FilamentSegmentationKaggle/claude'
sys.path.insert(0, ROOT + '/src')
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import cfil.stack as ST
from cfil.cands import tree_select
from cfil.decide import FOLD, thin

_g = ST.gbm_params
ST.gbm_params = lambda seed, threads=5: _g(seed, 5)
opt = dict(a[2:].split('=', 1) for a in sys.argv[1:] if a.startswith('--') and '=' in a)
SEEDS = [int(s) for s in opt.get('seeds', '0,1,2').split(',')]
ARMS = opt.get('arms', 'base,sp,mg,tb,all').split(',')
FAMS = {'base': (0,), 'sp': (0, 1), 'mg': (0, 2), 'tb': (0, 3), 'all': (0, 1, 2, 3), 'spmg': (0, 1, 2), 'pm': (0, 4),
        'all5': (0, 1, 2, 3, 4), 'nontube': (0, 1, 2, 4), 'mgpm': (0, 2, 4), 'pool': (0, 5, 6)}
TAUS = np.round(np.arange(0.24, 0.37, 0.02), 2)

def load_imgs(d=ROOT + '/work/ideas7/playbook/' + opt.get('dir', 'xc')):
    imgs = []
    for p in sorted(glob.glob(d + '/*.npz')):
        z = np.load(p)
        name = os.path.basename(p)[:-4]
        imgs.append(dict(name=name, fold=FOLD[name], F=z['F'], fam=z['fam'], parents=z['parents'], hits=z['hits'],
                         ngt=z['ngt'], conf=z['conf'], nb=int(z['nb'])))
    return [im for im in imgs if im['fold'] in (0, 1, 2, 3)]


def sub(im, arm):
    keep = np.isin(im['fam'], FAMS[arm])
    idx = np.flatnonzero(keep)
    remap = -np.ones(len(keep), np.int64); remap[idx] = np.arange(len(idx))
    c = im['conf']
    c = c[keep[c[:, 0]] & keep[c[:, 1]]]
    return dict(name=im['name'], fold=im['fold'], F=im['F'][idx], hits=im['hits'][idx], ngt=im['ngt'],
                parents=np.where(im['parents'][idx] >= 0, remap[np.maximum(im['parents'][idx], 0)], -1),
                conf=remap[c], nb=im['nb'], fam=im['fam'][idx], base_only=(arm == 'base'))


def select(im, q, tau):
    if im['base_only']:
        return tree_select(q, im['parents'], tau)
    w = q - tau
    pos = np.flatnonzero(w > 0)
    if not len(pos):
        return pos
    loc = -np.ones(len(q), np.int64); loc[pos] = np.arange(len(pos))
    c = im['conf']
    c = c[(loc[c[:, 0]] >= 0) & (loc[c[:, 1]] >= 0)]
    if not len(c):
        return pos
    a, b = loc[c[:, 0]], loc[c[:, 1]]
    G = coo_matrix((np.ones(len(a)), (a, b)), shape=(len(pos), len(pos)))
    ncc, lab = connected_components(G, directed=False)
    out = []
    for cc in range(ncc):
        nodes = np.flatnonzero(lab == cc)
        if len(nodes) == 1:
            out.append(pos[nodes[0]]); continue
        ln = -np.ones(len(pos), np.int64); ln[nodes] = np.arange(len(nodes))
        e = (ln[a] >= 0)
        ea, eb = ln[a[e]], ln[b[e]]
        A = coo_matrix((np.ones(2 * len(ea)), (np.repeat(np.arange(len(ea)), 2), np.stack([ea, eb], 1).ravel())),
                       shape=(len(ea), len(nodes)))
        r = milp(c=-w[pos[nodes]], constraints=LinearConstraint(A, -np.inf, 1), integrality=np.ones(len(nodes)),
                 bounds=Bounds(0, 1))
        out.extend(pos[nodes[r.x > 0.5]])
    return np.asarray(out, np.int64)


def counts(im, s):
    h = im['hits'][s]; tp = (h > 0).sum(0)
    return np.array([h.sum(), tp.sum(), (len(s) - tp).sum(), (im['ngt'] - tp).sum()], np.float64)


pq = lambda c: c[0] / (c[1] + 0.5 * (c[2] + c[3]))


def fit_predict(simgs):
    import lightgbm as lgb
    preds = {}
    for k in range(4):
        tr = [im for im in simgs if im['fold'] != k]
        X = np.concatenate([im['F'] for im in tr]); y = np.concatenate([im['hits'].mean(1) for im in tr])
        w = np.concatenate([np.full(len(im['F']), im['hits'].shape[1]) for im in tr])
        ms = [lgb.train(ST.gbm_params(s), lgb.Dataset(X, y, weight=w), 600) for s in SEEDS]
        for im in simgs:
            if im['fold'] == k:
                preds[im['name']] = np.mean([m.predict(im['F']) for m in ms], 0) if len(im['F']) else np.zeros(0)
    return [preds[im['name']] for im in simgs]


def main():
  imgs = load_imgs()
  print('images', len(imgs), 'rows', sum(len(im['F']) for im in imgs),
        'by family', [int(sum((im['fam'] == f).sum() for im in imgs)) for f in range(5)], flush=True)
  res = {}
  t0 = time.time()
  for arm in ARMS:
      simgs = [sub(im, arm) for im in imgs]
      # oracle: true record-mean hits as scores
      orc = {t: np.array([counts(im, select(im, im['hits'].mean(1), t)) for im in simgs]) for t in (0.26, 0.28, 0.30, 0.32, 0.34)}
      ob = max(orc, key=lambda t: pq(orc[t].sum(0)))
      q = fit_predict(simgs)
      R = {t: np.array([counts(im, select(im, qq, t)) for im, qq in zip(simgs, q)]) for t in TAUS}
      res[arm] = R
      tag = opt.get('tag', '')
      np.savez(ROOT + f'/work/ideas7/playbook/xcv_{arm}{tag}.npz', names=np.array([im['name'] for im in simgs]),
               folds=np.array([im['fold'] for im in simgs]), taus=TAUS, R=np.stack([R[t] for t in TAUS]),
               q=np.array(q, dtype=object), seeds=np.array(SEEDS))
      bp = ROOT + f'/work/ideas7/playbook/xcv_base{tag}.npz'
      if 'base' not in res and os.path.exists(bp):
          zb = np.load(bp, allow_pickle=True)
          res['base'] = {t: zb['R'][i] for i, t in enumerate(zb['taus'])}
      sel30 = [select(im, qq, 0.30) for im, qq in zip(simgs, q)]
      famsel = np.bincount(np.concatenate([im['fam'][s] for im, s in zip(simgs, sel30)]).astype(np.int64), minlength=5)
      # control: the arm-trained scorer selecting from tree rows only (separates new-candidate value from the
      # effect of the extra training rows)
      ctl = {}
      for t in (0.28, 0.30, 0.32):
          cc = []
          for im, qq in zip(simgs, q):
              b = np.flatnonzero(im['fam'] == 0)
              sb = dict(im, F=im['F'][b], hits=im['hits'][b], parents=im['parents'][b], fam=im['fam'][b], base_only=True)
              cc.append(counts(sb, select(sb, np.asarray(qq)[b], t)))
          ctl[t] = pq(np.sum(cc, 0))
      print(f'{arm:5s} rows {sum(len(im["F"]) for im in simgs):7d} | oracle best tau {ob} PQ {pq(orc[ob].sum(0)):.4f} | scorer ' +
            ' '.join(f'{t}:{pq(R[t].sum(0)):.4f}' for t in TAUS) + f' | thin@0.30 {thin(*R[0.30].sum(0)):.4f}'
            f' | selected@0.30 by family {famsel.tolist()} | TP/FP/FN@0.30 {R[0.30].sum(0)[1:].astype(int).tolist()}'
            f' | control (tree rows only) ' + ' '.join(f'{t}:{v:.4f}' for t, v in ctl.items()) + f' | {time.time() - t0:.0f}s', flush=True)
      if arm != 'base' and 'base' in res:
          B = res['base']
          rng = np.random.default_rng(0); d = []
          for _ in range(2000):
              i = rng.integers(len(imgs), size=len(imgs))
              d.append(pq(R[0.30][i].sum(0)) - pq(B[0.30][i].sum(0)))
          d = np.asarray(d)
          tb_ = max(TAUS, key=lambda t: pq(B[t].sum(0))); ta = max(TAUS, key=lambda t: pq(R[t].sum(0)))
          fold = ' '.join(f'{k}:{pq(R[0.30][[j for j, im in enumerate(imgs) if im["fold"] == k]].sum(0)) - pq(B[0.30][[j for j, im in enumerate(imgs) if im["fold"] == k]].sum(0)):+.4f}' for k in range(4))
          print(f'      vs base: at 0.30 {pq(R[0.30].sum(0)) - pq(B[0.30].sum(0)):+.4f} (boot SE {d.std():.4f}, P>0 {np.mean(d > 0):.2f}); '
                f'best-tau {pq(R[ta].sum(0)) - pq(B[tb_].sum(0)):+.4f}; folds {fold}', flush=True)


if __name__ == '__main__':
    main()
