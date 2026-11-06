# Scorer training tables

The two tables the stage-6 scorer models in `weights/models/` were fit on. `filseg.train.fit_scorers()` reads them
by default; `filseg.train.load_scorer_table("lgbm" | "et")` returns one as a dict.

| file | used for | images | rows | size |
|---|---|---|---|---|
| `lgbm_707.npz` | the 10 LightGBM models `combS707_s10..19.txt` | 707 (dev folds 0-3 and fold 4) | 292,535 | 53.3 MB |
| `et_566.npz` | the 10 ExtraTrees models `et_seed10..19.joblib` | 566 (dev folds 0-3) | 148,711 | 27.0 MB |

These are derived from the competition's training annotations and from model outputs on the training images. They
are data-derived tables, not the raw data: they hold no images and no annotation masks, only per-candidate columns
and a per-candidate target.

## Contents (both files)

- `X` (rows, 86) float32: the 86 scorer columns, in the order of `columns`.
- `y` float64: the target, the mean over the image's annotator records of the candidate's hit IoU (its IoU with
  the ground-truth object it overlaps most when that IoU is > 0.5, else 0).
- `w` float64: the sample weight, the number of annotator records of the image (fold-4 rows: that number / 4).
- `image` int32: per row, an index into `names`; `names`, `folds` int8: per image.
- `variant` int8: per row, the detector variant 0-3 of a fold-4 row; -1 on dev rows.
- `fam` int8: per row, the candidate family (0 tree, 2 merge, 4 cut, 5 cnxL, 6 R34).
- `columns`: the 86 column names; `sha256_Xyw`: the hash of (X, y, w) recorded with the original training run
  (`load_scorer_table` checks it).

`et_566.npz` is the dev block of `lgbm_707.npz` (its first 148,711 rows).

## How they were built

Only the candidates of the "combS" pool are rows (families 0, 2, 4, 5, 6). The columns come from model outputs on
the training images:

- Dev images (folds 0-3): the tree columns and the cross-map columns use out-of-fold U-Net maps (each image's maps
  from the fold model that did not train on its fold); the detector columns come from the one Mask R-CNN fold model
  that did not train on the image; the I6 columns come from the two Codex-track I6 "miner" models (each image from
  the miner that did not train on it).
- Fold-4 images: the U-Net maps are 4-fold-model averages, as on test. None of the four detector fold models saw
  fold 4, so each image enters four times, once per detector variant, at weight records / 4. Only the detector
  columns (45-61 and 63) differ between the four copies.

The original run assembled the dev rows with `work/ideas7/exp/p1_poolxm/run_arms.load_imgs('sym')` and
`xcv.sub(.., 'combS')`, and the fold-4 rows with `work/ideas8/bundle-hold/holdrows.load(k, 'sym')` for k = 0..3
(`work/ideas8/b3f707/train707.py` `load_dev`, `load_hold`, `assemble`); the ExtraTrees rows with
`work/ideas11/synth/slib.load()` (`work/package/et/save_et.py`). These files were exported with those loaders; the
(X, y, w) hashes equal the ones recorded in `b3f707/models/train_meta.json` (`sha256_Xyw` for 707 images,
`sha256_dev` for 566).

Building such tables from scratch needs the out-of-fold maps and detector candidates of the fold models on the
training images, the candidate and column code and the label matching. That code is not part of this release.

## Refit check

Refitting with `fit_scorers()` on these tables (LightGBM with 4 threads, the recorded parameters and seeds 10-19;
ExtraTrees with `n_jobs=4`) gave LightGBM model files byte-identical to the shipped ones. On the 180 test images,
the refit models' q_L, q_ET and blended q were bit-equal to the shipped models' (47,101 rows, 4 detector variants),
and the selection was the same on 180/180 images. The refit ExtraTrees `.joblib` files are not byte-identical to
the shipped ones, but their predictions are.
