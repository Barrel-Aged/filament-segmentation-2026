"""Bagged LightGBM expected-IoU scorer with an optional level-choice re-stack (stage 2 sees stage-1 scores of each
candidate's tree relatives). One code path for CV, holdout and test. CPU only."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np

TAU_CTX = 0.29  # tau used only inside the subtree-value context feature


def gbm_params(seed, threads=12):
    return dict(objective='regression', learning_rate=0.05, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.8,
                bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=threads, seed=seed)


def target(im, t):
    """t=None: record-mean hit IoU; t=0/1/2: record-mean raw IoU / precision / recall vs the best-overlap GT."""
    return im['hits'].mean(1) if t is None else im['aux'][:, :, t].mean(1)


def fit_gbm(imgs, key, seed, rounds=600, t=None):
    import lightgbm as lgb
    X = np.concatenate([key(im) for im in imgs]); y = np.concatenate([target(im, t) for im in imgs])
    w = np.concatenate([np.full(len(im['F']), im['hits'].shape[1]) for im in imgs])
    return lgb.train(gbm_params(seed), lgb.Dataset(X, y, weight=w), rounds)


def context(q, par):
    """Tree-context features from stage-1 scores q (parent, grandparent, children, subtree max/DP value, ancestors)."""
    N = len(q)
    if N == 0:
        return np.zeros((0, 11), np.float32)
    pq = np.where(par >= 0, q[np.maximum(par, 0)], -1)
    gp = np.where(par >= 0, par[np.maximum(par, 0)], -1)
    gq = np.where(gp >= 0, q[np.maximum(gp, 0)], -1)
    cmax = np.full(N, -1.0); csum = np.zeros(N); nch = np.zeros(N)
    dmax = q.copy(); best = np.zeros(N); chs = np.zeros(N)
    for i in range(N - 1, -1, -1):  # children always have larger indices than parents
        own = q[i] - TAU_CTX
        best[i] = own if (own >= chs[i] and own > 0) else max(chs[i], 0)
        p = par[i]
        if p >= 0:
            cmax[p] = max(cmax[p], q[i]); csum[p] += q[i]; nch[p] += 1
            dmax[p] = max(dmax[p], dmax[i]); chs[p] += best[i]
    amax = np.full(N, -1.0)
    for i in range(N):
        if par[i] >= 0:
            amax[i] = max(amax[par[i]], q[par[i]])
    return np.stack([q, pq, gq, cmax, csum, nch, dmax, amax, q - pq, q - cmax, chs], 1).astype(np.float32)


def aux_context(a, par):
    """Stage-1 auxiliary predictions a[N, 3] (raw IoU, precision, recall) with parent and best-child values."""
    N = len(a)
    if N == 0:
        return np.zeros((0, 12), np.float32)
    pa = np.where(par[:, None] >= 0, a[np.maximum(par, 0)], -1)
    cb = np.full((N, 3), -1.0)
    for i in range(N - 1, -1, -1):
        p = par[i]
        if p >= 0 and a[i, 0] > cb[p, 0]:
            cb[p] = a[i]
    return np.concatenate([a, pa, cb, a - pa], 1).astype(np.float32)


class Stack:
    """restack: stage 2 sees tree context of stage-1 scores. aux: stage 1 also predicts raw IoU/precision/recall
    (needs im['aux']) and stage 2 sees them with their tree context."""
    def __init__(self, seeds=(0, 1, 2, 3, 4), restack=True, aux=False):
        self.seeds, self.restack, self.aux = tuple(seeds), restack or aux, aux
        self.targets = [None] + ([0, 1, 2] if aux else [])

    def _stage1(self, models, F):
        """models: {target: [gbm per seed]} -> (q[N], a[N, 3] or None)"""
        if not len(F):
            return np.zeros(0), (np.zeros((0, 3)) if self.aux else None)
        q = np.mean([m.predict(F) for m in models[None]], 0)
        a = np.stack([np.mean([m.predict(F) for m in models[t]], 0) for t in (0, 1, 2)], 1) if self.aux else None
        return q, a

    def _key2(self, F, par, q, a):
        blocks = [F, context(q, par)] + ([aux_context(a, par)] if self.aux else [])
        return np.concatenate(blocks, 1)

    def fit(self, imgs):
        """imgs: list of dicts with F, hits, parents, fold (cross-fitting folds for stage 2) [, aux]."""
        fit1 = lambda sub: {t: [fit_gbm(sub, lambda im: im['F'], s, t=t) for s in self.seeds] for t in self.targets}
        self.m1 = fit1(imgs)
        if not self.restack:
            return self
        oof = {}
        for j in sorted({im['fold'] for im in imgs}):  # out-of-fold stage-1 outputs for the stage-2 training rows
            ms = fit1([im for im in imgs if im['fold'] != j])
            for im in imgs:
                if im['fold'] == j:
                    oof[im['name']] = self._stage1(ms, im['F'])
        key = lambda im: self._key2(im['F'], im['parents'], *oof[im['name']])
        self.m2 = [fit_gbm(imgs, key, s) for s in self.seeds]
        return self

    def predict(self, F, parents):
        q, a = self._stage1(self.m1, F)
        if not self.restack or not len(F):
            return q
        X = self._key2(F, parents, q, a)
        return np.mean([m.predict(X) for m in self.m2], 0)


def cross_val(imgs, folds=(0, 1, 2, 3), **kw):
    """Outer CV of the whole Stack: predictions for every image from a Stack fit on the other folds."""
    preds = {}
    for k in folds:
        st = Stack(**kw).fit([im for im in imgs if im['fold'] != k and im['fold'] in folds])
        for im in imgs:
            if im['fold'] == k:
                preds[im['name']] = st.predict(im['F'], im['parents'])
    return [preds[im['name']] for im in imgs]
