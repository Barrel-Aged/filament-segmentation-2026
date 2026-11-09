"""Export the two stage-6 scorer training tables into the release (training/scorer_data/).

Loads the rows with the ORIGINAL loaders (read-only imports):
  lgbm_707: work/ideas8/b3f707/train707.py load_dev() + load_hold() + assemble()  (dev 566 images ++ fold 4 x 4 variants)
  et_566  : work/ideas11/synth/slib.py load()  (the loader work/package/et/save_et.py used)
Checks the assembled (X, y, w) hashes against work/ideas8/b3f707/models/train_meta.json, writes compressed npz files and
re-reads them to check they round-trip. CPU only.
"""
import hashlib, json, os, sys, time
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
sys.dont_write_bytecode = True
ROOT = 'D:/FilamentSegmentationKaggle/claude'
REL = ROOT + '/release/filament-segmentation-2026'
OUT = REL + '/training/scorer_data'
sys.path.insert(0, ROOT + '/work/ideas8/b3f707')
import train707 as T  # noqa: E402  (sets CUDA_VISIBLE_DEVICES='' at import; reset below, nothing here uses a GPU)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
sys.path.insert(0, ROOT + '/work/ideas11/synth')
import slib  # noqa: E402
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np  # noqa: E402


def table(simgs_by_part):
    """per-row arrays + per-image names/folds from lists of (images, variant, weight divisor)"""
    names, folds, index = [], [], {}
    X, y, w, img, var, fam = [], [], [], [], [], []
    for subs, k, div in simgs_by_part:
        for im in subs:
            if im['name'] not in index:
                index[im['name']] = len(names); names.append(im['name']); folds.append(im['fold'])
            n = len(im['F'])
            X.append(im['F']); y.append(im['hits'].mean(1)); w.append(np.full(n, im['hits'].shape[1] / div))
            img.append(np.full(n, index[im['name']], np.int32)); var.append(np.full(n, k, np.int8)); fam.append(im['fam'].astype(np.int8))
    return dict(X=np.concatenate(X), y=np.concatenate(y), w=np.concatenate(w).astype(np.float64), image=np.concatenate(img),
                variant=np.concatenate(var), fam=np.concatenate(fam), names=np.array(names), folds=np.array(folds, np.int8))


def main():
    t0 = time.time()
    meta = json.load(open(ROOT + '/work/ideas8/b3f707/models/train_meta.json'))
    f74, sym = T.colnames(); cols = np.array(f74 + sym)
    imgs, simgs = T.load_dev()
    hv = T.load_hold()
    (Xd, yd, wd), (Xh, yh, wh) = T.assemble(simgs, hv)
    X = np.concatenate([Xd, Xh]); y = np.concatenate([yd, yh]); w = np.concatenate([wd, wh])
    L = table([(simgs, -1, 1.0)] + [(hv[k], k, 4.0) for k in range(4)])
    for a, b in ((L['X'], X), (L['y'], y), (L['w'], w)):
        assert a.dtype == b.dtype and np.array_equal(a, b)
    hL = T.sha_arr(L['X'], L['y'], L['w'])
    print(f'lgbm_707: X {L["X"].shape} {L["X"].dtype} | images {len(L["names"])} | sha256(X,y,w) {hL} | == train_meta '
          f'sha256_Xyw {hL == meta["sha256_Xyw"]}', flush=True)
    assert hL == meta['sha256_Xyw']
    se = slib.load()
    E = table([(se, -1, 1.0)])
    hE = T.sha_arr(E['X'], E['y'], E['w'])
    same_dev = all(np.array_equal(E[k], L[k][:len(Xd)]) for k in ('X', 'y', 'w', 'fam')) and list(E['names']) == list(L['names'][:566])
    print(f'et_566: X {E["X"].shape} {E["X"].dtype} | images {len(E["names"])} | sha256(X,y,w) {hE} | == train_meta '
          f'sha256_dev {hE == meta["sha256_dev"]} | == the dev part of lgbm_707 {same_dev}', flush=True)
    assert hE == meta['sha256_dev'] and same_dev
    os.makedirs(OUT, exist_ok=True)
    for tag, D, h in (('lgbm_707', L, hL), ('et_566', E, hE)):
        p = f'{OUT}/{tag}.npz'
        np.savez_compressed(p, columns=cols, sha256_Xyw=np.array(h), **D)
        z = np.load(p)
        rt = all(np.array_equal(z[k], D[k]) and z[k].dtype == D[k].dtype for k in D) and list(z['columns']) == list(cols)
        print(f'{tag}.npz: {os.path.getsize(p):,} B | keys {sorted(z.files)} | round-trip equal {rt} | '
              f'sha256(file) {hashlib.sha256(open(p, "rb").read()).hexdigest()}', flush=True)
        assert rt
    print(f'done ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
