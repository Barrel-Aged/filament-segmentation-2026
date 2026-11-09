"""B3f scorer refit on all 707 labelled images (dev folds 0-3 = 566 images + fold 4 = 141 images). CPU only.

Training set (frozen B3f recipe, nothing re-tuned):
  dev   : work/ideas7/playbook/xcomb rows ++ run_arms.xm_cols(xm18, fam, 'sym') (run_arms.load_imgs('sym')), xcv.sub(im, 'combS')
          (families 0 tree, 2 mg, 4 pm, 5 cnxL, 6 R34), y = hits.mean(1), weight = number of records
  fold 4: work/ideas8/bundle-hold/rows via holdrows.load(k, 'sym', with_labels=True), xcv.sub(im, 'combS'), every image
          once per detector variant k = 0..3, y = hits.mean(1), weight = number of records / 4 (each image counts once)
Scorer: LightGBM regression, production parameters (cfil.stack.gbm_params(seed)) with num_threads = 4, 600 rounds,
seeds 10-19, trained exactly as work/ideas8/hold-read/hold_read.py train() (one lgb.Dataset per model).

  train707.py cols      step 1: column-order / provenance assertions, row counts, weights, per-column mean / sd
  train707.py fit       step 2: the 10 models -> models/combS707_s<seed>.txt (+ training-set hashes)
  train707.py repro566  check: the same code on the 566 dev images only (seed 10, 4 threads) vs the saved 5-thread
                        work/ideas8/hold-read/models/combS_A_s10.txt (predictions on the test rows, all 4 variants)
"""
import hashlib, json, os, sys, time
sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = 'D:/FilamentSegmentationKaggle/claude'
HERE = ROOT + '/work/ideas8/b3f707'
for p in (ROOT + '/src', ROOT + '/work/ideas7/playbook', ROOT + '/work/ideas7/exp/p1_poolxm', ROOT + '/work/ideas8/bundle-hold',
          ROOT + '/work/ideas8/bundle-test'):
    if p not in sys.path:
        sys.path.insert(0, p)
_argv = sys.argv
sys.argv = [_argv[0], '0,1,2,3,4', 'combS', 'unused']   # run_arms parses argv at import, then imports xcv
import run_arms  # noqa: E402  (adds combS to xcv.FAMS)
import xcv  # noqa: E402
sys.argv = _argv
import holdrows  # noqa: E402
import numpy as np  # noqa: E402
import cfil.stack as ST  # noqa: E402
from cfil.cands import FEATURES  # noqa: E402
from cfil.ext import XFEAT  # noqa: E402
from cfil.rich import RFEAT  # noqa: E402
from cfil.xmap import XMFEAT  # noqa: E402

SEEDS = list(range(10, 20))
NTHREADS = 4
ROUNDS = 600
VARCOLS = set(range(45, 62)) | {63}


def params(seed):
    """production LightGBM parameters (cfil.stack.gbm_params, unpatched original) with num_threads = 4"""
    p = xcv._g(seed, NTHREADS)
    ref = ST.gbm_params(seed)   # xcv-patched production params (5 threads)
    assert {k: v for k, v in p.items() if k != 'num_threads'} == {k: v for k, v in ref.items() if k != 'num_threads'}
    assert p['num_threads'] == NTHREADS
    return p


def colnames():
    """86 column names of the combS layout, built from the source modules (xcomb.py / build_rows.py order)"""
    f74 = (list(FEATURES) + ['i6_' + c for c in XFEAT] + ['i6_' + c for c in RFEAT] + ['det_' + c for c in XFEAT] +
           ['det_' + c for c in RFEAT] + ['d_i6', 'd_det', 'is_mg', 'is_pm', 'is_cnxL', 'is_R34'] + [f'fam_extra{i}' for i in range(6)])
    sym = ['symA_' + c for c in XMFEAT] + ['symB_' + c for c in XMFEAT]
    return f74, sym


def load_dev():
    imgs = run_arms.load_imgs('sym')
    simgs = [xcv.sub(im, 'combS') for im in imgs]
    assert len(simgs) == 566 and sorted({im['fold'] for im in simgs}) == [0, 1, 2, 3]
    return imgs, simgs


def load_hold():
    hv = []
    for k in range(4):
        hv.append([xcv.sub(im, 'combS') for im in holdrows.load(k, 'sym', with_labels=True)])
    names = [im['name'] for im in hv[0]]
    assert len(names) == 141 and all([im['name'] for im in v] == names for v in hv)
    for v in hv[1:]:
        for a, b in zip(hv[0], v):
            assert a['F'].shape == b['F'].shape and np.array_equal(a['fam'], b['fam']) and np.array_equal(a['hits'], b['hits'])
            assert np.array_equal(a['parents'], b['parents']) and np.array_equal(a['conf'], b['conf'])
    return hv


def assemble(simgs, hv):
    Xd = np.concatenate([im['F'] for im in simgs]); yd = np.concatenate([im['hits'].mean(1) for im in simgs])
    wd = np.concatenate([np.full(len(im['F']), im['hits'].shape[1]) for im in simgs]).astype(np.float64)
    Xh = np.concatenate([im['F'] for v in hv for im in v]); yh = np.concatenate([im['hits'].mean(1) for v in hv for im in v])
    wh = np.concatenate([np.full(len(im['F']), im['hits'].shape[1] / 4.0) for v in hv for im in v])
    return (Xd, yd, wd), (Xh, yh, wh)


def sha_arr(*arrs):
    h = hashlib.sha256()
    for a in arrs:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def cols():
    t0 = time.time()
    f74, sym = colnames()
    names86 = f74 + sym
    # (a) names: source-module names == the fold-4 loader's FCOLS (dev xcomb order documented / reproduced by build_rows)
    assert len(f74) == 74 and f74 == holdrows.FCOLS, 'F[74] names differ from holdrows.FCOLS'
    assert holdrows.XMCOLS == [f'{m}_{c}' for m in ('r34', 'cnxL', 'cnx') for c in XMFEAT]
    print(f'(a) names: 74 F names from cfil.cands.FEATURES / cfil.ext.XFEAT / cfil.rich.RFEAT (xcomb.py concatenation order) '
          f'== holdrows.FCOLS: True | xm18 names == holdrows.XMCOLS (R34 | cnxL | cnx, xmap.XMFEAT): True', flush=True)
    # (b) provenance: build_rows.py (the fold-4 / test row builder) reproduced 16 dev xcomb + xm files bit-exactly
    L = [l for l in open(ROOT + '/work/ideas8/bundle-hold/devcheck16.log') if 'max|dF|' in l]
    ok16 = [('max|dF| 0.0 ' in l) and ('max|dxm| 0.0 ' in l) and ('fam/parents/conf/nb/hits/ngt eq True True True True True True' in l)
            for l in L]
    print(f'(b) provenance: bundle-hold/devcheck16.log: {len(L)} dev images regenerated by build_rows.gen/xm18 (the fold-4 '
          f'and test row code), F max|d| 0.0 and xm max|d| 0.0 and fam/parents/conf/nb/hits/ngt equal on {sum(ok16)}/{len(L)}', flush=True)
    assert len(L) == 16 and all(ok16)
    # (c) xm 'sym' functions identical (dev run_arms.xm_cols vs fold-4/test holdrows.xm_cols) on random input
    rng = np.random.default_rng(0)
    Xr = rng.normal(size=(5000, 18)).astype(np.float32); fr = rng.choice(np.array([0, 2, 4, 5, 6], np.int8), 5000)
    assert np.array_equal(run_arms.xm_cols(Xr, fr, 'sym'), holdrows.xm_cols(Xr, fr, 'sym'))
    print('(c) run_arms.xm_cols(., ., "sym") == holdrows.xm_cols(., ., "sym") on 5000 random rows: True', flush=True)
    imgs, simgs = load_dev()
    hv = load_hold()
    # (d) structural checks of the assembled 86 columns in both sources
    split = json.load(open(ROOT + '/configs/split-v2.json'))
    obs = {o['file_name'][:-5]: o for o in split['observations']}
    dn = [im['name'] for im in simgs]; hn = [im['name'] for im in hv[0]]
    assert not (set(dn) & set(hn))
    assert all(obs[n]['fold'] in (0, 1, 2, 3) for n in dn) and all(obs[n]['fold'] == 4 for n in hn)
    gd = {obs[n]['group_id'] for n in dn}; gh = {obs[n]['group_id'] for n in hn}
    rec_ok_d = all(im['hits'].shape[1] == len(obs[im['name']]['annotator_image_ids']) == len(im['ngt']) for im in simgs)
    rec_ok_h = all(im['hits'].shape[1] == len(obs[im['name']]['annotator_image_ids']) == len(im['ngt']) for im in hv[0])
    print(f'(d) split-v2: dev {len(dn)} images folds {sorted({obs[n]["fold"] for n in dn})}, fold-4 {len(hn)} images; shared names '
          f'{len(set(dn) & set(hn))}, shared day groups {len(gd & gh)} | records per image == len(annotator_image_ids) == len(ngt): '
          f'dev {rec_ok_d}, fold 4 {rec_ok_h}', flush=True)
    assert not (gd & gh) and rec_ok_d and rec_ok_h
    src = {'dev': simgs, 'fold4': [im for v in hv for im in v]}
    for tag, L_ in src.items():
        F = np.concatenate([im['F'] for im in L_]); fam = np.concatenate([im['fam'] for im in L_])
        assert F.shape[1] == 86
        oh = all(np.array_equal(F[:, 64 + j] == 1, fam == f) for j, f in enumerate((2, 4, 5, 6)))
        oh01 = np.isin(F[:, 64:68], (0, 1)).all()
        ex0 = not np.any(F[fam == 0, 64:74])
        tree = fam == 0
        # tree rows: sym slot A = R34 block, slot B = cnxL block (base(xm12) order)
        print(f'(d) {tag}: rows {len(F)} | one-hot cols 64-67 == fam (mg, pm, cnxL, R34) {oh} (values in {{0,1}} {oh01}) | '
              f'cols 64-73 zero on tree rows {ex0} | t (col 0) range [{F[:, 0].min():.3f}, {F[:, 0].max():.3f}] | '
              f'families ' + ' / '.join(f'{f}:{int((fam == f).sum())}' for f in (0, 2, 4, 5, 6)), flush=True)
        assert oh and oh01 and ex0
    # raw-xm consistency: the assembled sym columns equal xm_cols of the stored raw xm (dev xm/<stem>.npz, fold-4 rows xm)
    md = 0.0
    for im in imgs[:: 7]:
        x = np.load(ROOT + f'/work/ideas7/exp/p1_poolxm/xm/{im["name"]}.npz')
        md = max(md, float(np.abs(im['F'][:, 74:] - holdrows.xm_cols(x['X'], x['fam'], 'sym')).max()) if len(im['F']) else 0.0)
        z = np.load(ROOT + f'/work/ideas7/playbook/xcomb/{im["name"]}.npz')
        assert np.array_equal(im['F'][:, :74], z['F'])
    mh = 0.0
    for im in holdrows.load(0, 'sym', with_labels=False)[:: 7]:
        z = np.load(ROOT + f'/work/ideas8/bundle-hold/rows/{im["name"]}.npz')
        mh = max(mh, float(np.abs(im['F'][:, 74:] - run_arms.xm_cols(z['xm'], z['fam'], 'sym')).max()))
        assert np.array_equal(im['F'][:, :74], z['F'][0])
    print(f'(d) assembled cols 0-73 == stored F (dev xcomb / fold-4 F[0]) and cols 74-85 == xm_cols(stored xm18, fam, sym) '
          f'(every 7th image): max|d| dev {md} fold4 {mh}', flush=True)
    assert md == 0 and mh == 0
    # variant columns
    dif = set()
    for a, b in zip(hv[0], hv[1]):
        dif |= set(np.flatnonzero(np.any(a['F'] != b['F'], 0)).tolist())
    for v in hv[2:]:
        for a, b in zip(hv[0], v):
            dif |= set(np.flatnonzero(np.any(a['F'] != b['F'], 0)).tolist())
    print(f'(d) fold-4 columns that differ between variants: {sorted(dif)} (subset of 45-61, 63: {dif <= VARCOLS})', flush=True)
    assert dif <= VARCOLS
    (Xd, yd, wd), (Xh, yh, wh) = assemble(simgs, hv)
    nimg_d, nimg_h = len(simgs), len(hv[0])
    print(f'rows: dev {len(Xd)} ({nimg_d} images, {len(Xd) / nimg_d:.1f}/image) | fold 4 {len(Xh)} = 4 variants x {len(Xh) // 4} '
          f'({nimg_h} images, {len(Xh) / 4 / nimg_h:.1f}/image) | total {len(Xd) + len(Xh)}', flush=True)
    print(f'weight: dev total {wd.sum():.0f} (records/image mean {np.mean([im["hits"].shape[1] for im in simgs]):.3f}) | fold 4 total '
          f'{wh.sum():.0f} (records/image mean {np.mean([im["hits"].shape[1] for im in hv[0]]):.3f}; per variant {wh.sum() / 4:.0f}) | '
          f'fold-4 share of weight {wh.sum() / (wd.sum() + wh.sum()):.4f}', flush=True)
    print(f'target y = record-mean hit IoU: dev mean {yd.mean():.4f} (weighted {np.average(yd, weights=wd):.4f}, share > 0 '
          f'{np.mean(yd > 0):.4f}) | fold 4 mean {yh.mean():.4f} (weighted {np.average(yh, weights=wh):.4f}, share > 0 {np.mean(yh > 0):.4f})',
          flush=True)
    md_, sd_ = Xd.astype(np.float64).mean(0), Xd.astype(np.float64).std(0)
    mh_, sh_ = Xh.astype(np.float64).mean(0), Xh.astype(np.float64).std(0)
    flags = []
    print('per-column mean / sd (unweighted rows; fold 4 = all 4 variants) | d/sd_dev = (mean_f4 - mean_dev) / sd_dev', flush=True)
    for j, nm in enumerate(names86):
        r_dev = (mh_[j] - md_[j]) / sd_[j] if sd_[j] > 0 else (0.0 if mh_[j] == md_[j] else np.inf)
        r_f4 = (mh_[j] - md_[j]) / sh_[j] if sh_[j] > 0 else (0.0 if mh_[j] == md_[j] else np.inf)
        fl = abs(r_dev) > 1 or abs(r_f4) > 1
        if fl:
            flags.append(nm)
        print(f'  {j:2d} {nm:24s} dev {md_[j]:10.4f} / {sd_[j]:9.4f} | f4 {mh_[j]:10.4f} / {sh_[j]:9.4f} | d/sd_dev {r_dev:+7.3f} '
              f'd/sd_f4 {r_f4:+7.3f}' + ('  FLAG' if fl else ''), flush=True)
    print(f'columns with |mean_f4 - mean_dev| > 1 sd (dev sd or fold-4 sd): {len(flags)} {flags}', flush=True)
    json.dump(dict(names=names86, dev_mean=md_.tolist(), dev_sd=sd_.tolist(), f4_mean=mh_.tolist(), f4_sd=sh_.tolist(), flags=flags,
                   rows=dict(dev=int(len(Xd)), fold4=int(len(Xh))), weight=dict(dev=float(wd.sum()), fold4=float(wh.sum()))),
              open(HERE + '/colstats.json', 'w'), indent=1)
    print(f'cols done ({time.time() - t0:.0f}s)', flush=True)


def fit():
    import lightgbm as lgb
    t0 = time.time()
    imgs, simgs = load_dev()
    hv = load_hold()
    (Xd, yd, wd), (Xh, yh, wh) = assemble(simgs, hv)
    assert Xd.shape[1] == Xh.shape[1] == 86 and Xd.dtype == Xh.dtype == np.float32
    X = np.concatenate([Xd, Xh]); y = np.concatenate([yd, yh]); w = np.concatenate([wd, wh])
    meta = dict(rows_dev=int(len(Xd)), rows_fold4=int(len(Xh)), weight_dev=float(wd.sum()), weight_fold4=float(wh.sum()),
                images_dev=len(simgs), images_fold4=len(hv[0]), sha256_Xyw=sha_arr(X, y, w), sha256_dev=sha_arr(Xd, yd, wd),
                sha256_fold4=sha_arr(Xh, yh, wh), seeds=SEEDS, rounds=ROUNDS, params={s: params(s) for s in SEEDS},
                lightgbm=lgb.__version__, numpy=np.__version__, models={})
    print(f'fit: X {X.shape} {X.dtype} | y mean {y.mean():.4f} | w sum {w.sum():.0f} | sha256(X,y,w) {meta["sha256_Xyw"][:16]}... '
          f'| params {params(SEEDS[0])}', flush=True)
    for s in SEEDS:
        t1 = time.time()
        m = lgb.train(params(s), lgb.Dataset(X, y, weight=w), ROUNDS)
        p = HERE + f'/models/combS707_s{s}.txt'
        m.save_model(p)
        h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
        meta['models'][str(s)] = dict(path=p, sha256=h, num_trees=m.num_trees(), num_feature=m.num_feature())
        print(f'  seed {s}: {m.num_trees()} trees, {m.num_feature()} features, train-fit mean pred {m.predict(X[:20000], num_threads=NTHREADS).mean():.4f} '
              f'| sha256 {h[:16]}... | {time.time() - t1:.0f}s', flush=True)
    json.dump(meta, open(HERE + '/models/train_meta.json', 'w'), indent=1)
    print(f'fit done ({time.time() - t0:.0f}s)', flush=True)


def repro566():
    """same training code on the 566 dev images only (seed 10, 4 threads) vs the saved 5-thread hold-read model"""
    import lightgbm as lgb
    import score_select as SS
    t0 = time.time()
    imgs, simgs = load_dev()
    Xd = np.concatenate([im['F'] for im in simgs]); yd = np.concatenate([im['hits'].mean(1) for im in simgs])
    wd = np.concatenate([np.full(len(im['F']), im['hits'].shape[1]) for im in simgs])
    m = lgb.train(params(10), lgb.Dataset(Xd, yd, weight=wd), ROUNDS)
    ref = lgb.Booster(model_file=ROOT + '/work/ideas8/hold-read/models/combS_A_s10.txt')
    md = 0.0
    for k in range(4):
        T = np.concatenate([SS.nl_sub(im, 'combS')['F'] for im in SS.load_test(k, 'sym')]).astype(np.float64)
        a = m.predict(T, num_threads=NTHREADS); b = ref.predict(T, num_threads=NTHREADS)
        md = max(md, float(np.max(np.abs(a - b))))
        print(f'repro566 variant {k}: test rows {len(T)} | max|q_mine(4 threads) - q_saved(5 threads)| {np.max(np.abs(a - b)):.3g} '
              f'mean|d| {np.mean(np.abs(a - b)):.3g} corr {np.corrcoef(a, b)[0, 1]:.6f}', flush=True)
    a = m.predict(Xd.astype(np.float64), num_threads=NTHREADS); b = ref.predict(Xd.astype(np.float64), num_threads=NTHREADS)
    print(f'repro566 dev rows {len(Xd)}: max|d| {np.max(np.abs(a - b)):.3g} | ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    dict(cols=cols, fit=fit, repro566=repro566)[sys.argv[1]]()
