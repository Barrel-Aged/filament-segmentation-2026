# Scorer training tables

These two tables are what the step-6 scorer models in `weights/models/` were trained on. `filseg.train.fit_scorers()`
reads them by default, and `filseg.train.load_scorer_table("lgbm")` or `load_scorer_table("et")` returns one as a
dictionary.

| File | Used to train | Images | Rows | Size |
|---|---|---|---|---|
| `lgbm_707.npz` | the 10 LightGBM models (`combS707_s10..19.txt`) | 707 (development folds 0-3 plus fold 4) | 292,535 | 53.3 MB |
| `et_566.npz` | the 10 ExtraTrees models (`et_seed10..19.joblib`) | 566 (development folds 0-3) | 148,711 | 27.0 MB |

Each row is one candidate mask on one training image. The tables come from the competition's training annotations
and from my models' outputs on the training images, but they hold no images and no annotation masks, only 86
numbers per candidate and its target.

## What's in each file

- `X` (rows × 86, float32): the 86 evidence numbers for each candidate, in the order given by `columns`.
- `y` (float64): the target. For each annotator who drew the image, the code takes the IoU between the candidate and
  the annotator's filament it overlaps most, counting it only when that IoU is above 0.5 (otherwise 0). The target is
  the average over the image's annotators.
- `w` (float64): the row's weight, which is the number of annotators who drew the image. Fold-4 rows get a quarter of
  that, for the reason given below.
- `image` (int32): which image each row belongs to, as an index into `names`. `names` and `folds` are per image.
- `variant` (int8): which of the four detector copies a fold-4 row came from (0-3), or -1 for development rows.
- `fam` (int8): the kind of candidate: 0 a blob of the main map, 2 a merge, 4 a cut, 5 a blob of the cnxL map, 6 a
  blob of the R34 map.
- `columns`: the names of the 86 numbers. `sha256_Xyw`: a hash of (X, y, w) recorded when the original models were
  trained. `load_scorer_table` checks it.

`et_566.npz` is the first 148,711 rows of `lgbm_707.npz`, the development part.

## How the tables were built

The evidence numbers come from model outputs on the training images, and each image always gets outputs from models
that didn't train on it.

- **Development images (folds 0-3).** Each image's U-Net maps and Mask R-CNN candidates come from the fold copy that
  didn't see that image's fold. The I6 numbers come from two I6 models trained on separate halves of the data, each
  image from the half that didn't include it.
- **Fold-4 images.** No U-Net or Mask R-CNN trained on fold 4, so these images use the 4-copy averages, just like
  the test images. Their I6 numbers come from the same two half-data I6 models, which did train on other observation
  groups in fold 4, so fold 4 is clean per image but not untouched. Each fold-4 image appears four times, once per Mask R-CNN copy, at a quarter of the weight. Only the
  detector columns (45-61 and 63) differ between the four copies.

Claude exported these files with the same loading code the original training used, and their hashes match the ones
recorded at training time. Building tables like these from scratch needs the out-of-fold maps and detector outputs
on the training images, plus the candidate, evidence and matching code for the training side. That code is in
`training/as_run/scorer_tables/`, as it originally ran.

## Retraining check

Retraining with `fit_scorers()` on these tables gave LightGBM model files that are byte-identical to the shipped
ones. On the 180 test images, the retrained models' scores matched the shipped models' exactly, and the selection
was the same on all 180 images. The retrained ExtraTrees files aren't byte-identical to the shipped ones, but their
predictions are.
