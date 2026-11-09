"""Decision layer: fixed-threshold baselines, grouped-CV LightGBM expected-IoU scorer, tree DP + Dinkelbach tau. CPU only."""
import glob, json, os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cfil.cands import FEATURES, LEVELS, tree_select

SPLIT = os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'split-v2.json')
_OBS = json.load(open(SPLIT))['observations']
FOLD = {o['file_name'].replace('.jpeg', ''): o['fold'] for o in _OBS}
GROUP = {o['file_name'].replace('.jpeg', ''): o['group_id'] for o in _OBS}


def load(dirs, ext_dirs=None, pseudo=False, cnn_dirs=None, feat_dirs=None):
    """pseudo=True: re-split the loaded images into 4 group-preserving pseudo folds (for single-fold iteration).
    feat_dirs: per candidate dir, a list of directories whose <stem>.npz 'X' blocks are appended to F."""
    imgs = []
    widths = {}
    for d, e, c in zip(dirs, ext_dirs or [None] * len(dirs), cnn_dirs or [None] * len(dirs)):
        for p in sorted(glob.glob(os.path.join(d, '*.npz'))):
            z = np.load(p)
            name = os.path.basename(p)[:-4]
            im = dict(name=name, fold=FOLD.get(name, -1), F=z['F'], parents=z['parents'], hits=z['hits'], ngt=z['ngt'])
            if e:
                q = os.path.join(e, name + '.npz')
                if not os.path.exists(q):
                    continue
                y = np.load(q, allow_pickle=True)
                im.update(X=y['X'], E=y['E'], Eh=y['Eh'], ov=y['ov'].astype(np.float32), ext_rle=y['ext_rle'])
                im['F'] = np.concatenate([im['F'], im['X']], 1)
            for fd in (feat_dirs[dirs.index(d)] if feat_dirs else []):
                q = os.path.join(fd, name + '.npz')
                if fd not in widths:
                    widths[fd] = np.load(glob.glob(os.path.join(fd, '*.npz'))[0], allow_pickle=True)['X'].shape[1]
                if os.path.exists(q):
                    x = np.load(q, allow_pickle=True)['X']
                else:  # zero-filled; callers that need full coverage must check im['missing']
                    x = np.zeros((len(im['F']), widths[fd]), np.float32); im.setdefault('missing', []).append(fd)
                im['F'] = np.concatenate([im['F'], x], 1)
            if c:
                q = os.path.join(c, name + '.npz')  # images without candidates have no crop scores
                cs = np.load(q)['c'] if os.path.exists(q) else np.zeros(len(im['F']), np.float32)
                im['F'] = np.concatenate([im['F'], cs[:, None]], 1)
            imgs.append(im)
    if pseudo:
        for im in imgs:
            im['fold'] = int(GROUP[im['name']], 16) % 4
    return imgs


def pq_of(imgs, selections):
    S = TP = FP = FN = 0.0
    for im, sel in zip(imgs, selections):
        h = im['hits'][sel]
        for r in range(h.shape[1]):
            tp = (h[:, r] > 0).sum()
            S += h[:, r].sum(); TP += tp; FP += len(sel) - tp; FN += im['ngt'][r] - tp
    d = TP + 0.5 * (FP + FN)
    return dict(pq=S / d if d else 0, thin=thin(S, TP, FP, FN), tp=int(TP), fp=int(FP), fn=int(FN), sq=S / TP if TP else 0)


R_THIN = 0.77  # public-proxy: test annotators keep each filament w.p. r (calibrated on Codex I5/H7 public scores)


def thin(S, TP, FP, FN, r=R_THIN):
    """Expected PQ when each GT instance is kept independently with probability r."""
    d = r * TP + 0.5 * (FP + (1 - r) * TP + r * FN)
    return r * S / d if d else 0


def fixed(imgs, t, min_area=64):
    li = int(np.argmin(np.abs(LEVELS - t)))
    return [np.nonzero((im['F'][:, 1] == li) & (np.exp(im['F'][:, 2]) >= min_area))[0] for im in imgs]


def fit_predict(imgs, folds, params=None, feats=None):
    import lightgbm as lgb
    params = params or dict(objective='regression', learning_rate=0.05, num_leaves=31, min_data_in_leaf=100,
                            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=16)
    cols = list(range(imgs[0]['F'].shape[1])) if feats is None else [FEATURES.index(f) for f in feats]
    preds = [None] * len(imgs)
    for k in folds:
        tr = [im for im in imgs if im['fold'] != k and im['fold'] in folds]
        X = np.concatenate([im['F'][:, cols] for im in tr]); y = np.concatenate([im['hits'].mean(1) for im in tr])
        w = np.concatenate([np.full(len(im['F']), im['hits'].shape[1]) for im in tr])
        m = lgb.train(params, lgb.Dataset(X, y, weight=w), num_boost_round=600)
        for i, im in enumerate(imgs):
            if im['fold'] == k:
                preds[i] = m.predict(im['F'][:, cols])
    return preds


def dinkelbach(imgs, preds, tau=0.2, iters=8, idx=None, r_thin=None):
    """Optimal global tau for dev PQ (r_thin=None) or for the thinned public proxy (keep iff q > PQ'/(2r))."""
    idx = range(len(imgs)) if idx is None else idx
    sub = [imgs[i] for i in idx]
    for _ in range(iters):
        sel = [tree_select(preds[i], imgs[i]['parents'], tau) for i in idx]
        r = pq_of(sub, sel)
        new = r['pq'] / 2 if r_thin is None else r['thin'] / (2 * r_thin)
        if abs(new - tau) < 1e-4:
            break
        tau = new
    return tau, r


def sweep(imgs, preds, key='thin', grid=np.arange(0.16, 0.40, 0.01)):
    """Grid-search the global tau directly (scorer outputs are not perfectly calibrated, so the Dinkelbach
    fixed point is not the empirical optimum). Returns (best_tau, {tau: metrics})."""
    res = {}
    for t in grid:
        res[round(float(t), 3)] = pq_of(imgs, [tree_select(preds[i], im['parents'], t) for i, im in enumerate(imgs)])
    best = max(res, key=lambda t: res[t][key])
    return best, res


def report(imgs, preds, tau, folds):
    out = {}
    for k in folds:
        idx = [i for i, im in enumerate(imgs) if im['fold'] == k]
        sel = [tree_select(preds[i], imgs[i]['parents'], tau) for i in idx]
        out[k] = pq_of([imgs[i] for i in idx], sel)
    sel = [tree_select(preds[i], imgs[i]['parents'], tau) for i in range(len(imgs))]
    out['all'] = pq_of(imgs, sel)
    return out


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    kw = dict(a[2:].split('=', 1) for a in sys.argv[1:] if a.startswith('--') and '=' in a)
    dirs = args
    imgs = load(dirs, kw['ext'].split(',') if 'ext' in kw else None, pseudo='--pseudo' in sys.argv,
                cnn_dirs=kw['cnn'].split(',') if 'cnn' in kw else None)
    folds = sorted({im['fold'] for im in imgs})
    print('images', len(imgs), 'folds', folds, 'candidates', sum(len(im['F']) for im in imgs))
    for t in (0.3, 0.4, 0.5, 0.6):
        print('fixed', t, {k: round(v, 4) if isinstance(v, float) else v for k, v in pq_of(imgs, fixed(imgs, t)).items()})
    if len(folds) > 1:
        preds = fit_predict(imgs, folds)
        tt, _ = dinkelbach(imgs, preds, r_thin=R_THIN)
        print('gbm thin-dinkelbach tau', round(tt, 4), {k: round(float(v), 4) for k, v in report(imgs, preds, tt, folds)['all'].items()})
        for key in ('pq', 'thin'):
            bt, res = sweep(imgs, preds, key)
            print(f'sweep best-{key} tau', bt, {k: round(float(v), 4) for k, v in res[bt].items()})
        print('sweep curve', {t: (round(float(v['pq']), 4), round(float(v['thin']), 4)) for t, v in res.items()})
        tau, r = dinkelbach(imgs, preds)
        print('gbm tau', round(tau, 4), {k: {a: round(b, 4) if isinstance(b, float) else b for a, b in v.items()} for k, v in report(imgs, preds, tau, folds).items()})
