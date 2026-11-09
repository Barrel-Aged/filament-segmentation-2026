"""Optional retraining: a small interface to ``training/`` for the demo notebook. Nothing here runs on import.

The pipeline always reads the shipped weights in ``weights/``; this module does not replace them. A retrained file
is written to ``<runs>/<name>/`` (default ``<FILSEG_WORK>/runs``) and stays there.

What can be retrained, and what it costs
- The 16 networks with an entry point here: 12 U-Nets (``cnx-f0..3``, ``cnxL-f0..3``, ``r34-f0..3``,
  ``training/train_dense.py``) and 4 Mask R-CNN fold models (``det-f0..3``, ``training/train_det.py``). Each is one
  CUDA run with the arguments recorded in ``configs/<name>.json``. Recorded times on one RTX 4070 SUPER:
  cnx 3,943-5,146 s, cnxL 8,891-9,221 s, R34 4,316-4,613 s and det 5,857-8,122 s per fold model, so about
  26-30 GPU hours for all 16. Both trainers need CUDA (bf16 autocast) and download their pretrained starting
  weights (ImageNet encoders through timm / segmentation-models-pytorch, COCO Mask R-CNN through torchvision).
  Training is not bit-reproducible, so a retrained network differs from the shipped one.
- The I6 detector and refiner (stage 4) were trained in the Codex track. There is no entry point here: the original
  training scripts are in ``training/as_run/i6/`` exactly as they ran, not re-tested in that location (see
  ``training/as_run/README.md``).
- The candidate scorer (stage 6): 10 LightGBM and 10 ExtraTrees models (seeds 10-19). ``fit_lgbm`` and ``fit_et`` hold
  their exact recipes; on a desktop CPU each LightGBM model takes about 10 s (292,535 rows) and all 10 ExtraTrees
  about 90 s (148,711 rows). They are fit on labelled candidate rows: the 86 scorer columns of every candidate on the
  training images, computed from out-of-fold U-Net maps and detector candidates, with the target y = the mean over
  the image's annotator records of the candidate's IoU with the ground-truth object it shares most pixels with
  (0 when that IoU is <= 0.5), and the weight w = the number of records. The two tables the shipped models were fit
  on are in ``training/scorer_data/`` (``lgbm_707.npz``, ``et_566.npz``; see its README); ``fit_scorers()`` reads
  them by default. The original scripts that built those rows from the images and annotations (the out-of-fold maps,
  detector and I6 banks, evidence, targets and table assembly) are in ``training/as_run/scorer_tables/`` exactly as
  they ran, not re-tested in that location (see ``training/as_run/README.md``).

Usage (repository root on ``sys.path``):
    from filseg import train
    train.prepare_cache()            # training cache, about 8.9 GB (once)
    train.train_network('r34-f0')    # one fold model, CUDA
    train.fit_scorers()              # the 10 + 10 scorer models from training/scorer_data/, CPU
"""
from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path

import numpy as np

from filseg import paths

DENSE = [f"{s}-f{k}" for s in ("cnx", "cnxL", "r34") for k in range(4)]
DET = [f"det-f{k}" for k in range(4)]
NETWORKS = DENSE + DET
SEEDS = list(range(10, 20))

# recorded seconds per fold model on one RTX 4070 SUPER (training/train_dense.py, training/train_det.py)
GPU_SECONDS = {"cnx": (3943, 5146), "cnxL": (8891, 9221), "r34": (4316, 4613), "det": (5857, 8122)}

# stage 6 scorer recipes (cfil.stack.gbm_params with 4 threads, 600 rounds; work/package/et/save_et.py)
LGBM_PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.8,
                   bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=4)
LGBM_ROUNDS = 600
ET_PARAMS = dict(n_estimators=60, min_samples_leaf=100, max_features=0.5, bootstrap=False)
SCORER_DATA = paths.REPO / "training" / "scorer_data"
SCORER_TABLES = {"lgbm": "lgbm_707.npz", "et": "et_566.npz"}


def _training(module: str):
    """``training.<module>``; the ``training`` folder sits next to ``filseg`` in the repository."""
    if str(paths.REPO) not in sys.path:
        sys.path.insert(0, str(paths.REPO))
    return importlib.import_module(f"training.{module}")


def default_cache() -> Path:
    return paths.WORK / "train_cache"


def default_runs() -> Path:
    return paths.WORK / "runs"


def prepare_cache(cache=None, data=None, split=None, workers: int = 4) -> Path:
    """Write the training cache (``training/prep_train.py``): images, soft and spine targets, folds, crop centres.
    Needs ``<data>/train`` (images and annotations). CPU only."""
    cache = Path(cache or default_cache())
    _training("prep_train").main(str(cache), Path(data or paths.DATA), Path(split or paths.CONFIGS / "split-v2.json"),
                                 workers)
    return cache


def estimated_hours(names=None) -> float:
    """Upper end of the recorded GPU time of the given networks (default all 16), in hours."""
    names = NETWORKS if names is None else names
    return sum(GPU_SECONDS[n.rsplit("-f", 1)[0]][1] for n in names) / 3600


def train_network(name: str, cache=None, runs=None) -> Path:
    """Train one fold model with its recorded arguments (``configs/<name>.json``). CUDA only.

    Returns the weight file written: ``<runs>/<name>/ema.pt`` (U-Nets) or ``<runs>/<name>/model.pt`` (Mask R-CNN).
    """
    if name not in NETWORKS:
        raise ValueError(f"{name}: not one of {NETWORKS}")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("training needs a CUDA GPU (the trainers use bf16 autocast on CUDA)")
    cache, runs = Path(cache or default_cache()), Path(runs or default_runs())
    if not (cache / "meta.json").exists():
        raise FileNotFoundError(f"{cache}/meta.json: run prepare_cache() first")
    mod = _training("train_det" if name in DET else "train_dense")
    a = mod.parse(["--config", name, "--cache", str(cache), "--runs", str(runs)])
    mod.train(a)
    return runs / name / ("model.pt" if name in DET else "ema.pt")


def load_scorer_table(which: str, folder=None) -> dict:
    """One shipped scorer training table: ``which`` = ``"lgbm"`` (all 707 labelled images) or ``"et"`` (the 566
    development images). Keys: X (rows, 86) float32, y, w (float64), image (row -> index into names), names, folds
    (per image), variant (detector variant 0-3 of a fold-4 row, -1 for development rows), fam, columns, sha256_Xyw."""
    with np.load(Path(folder or SCORER_DATA) / SCORER_TABLES[which]) as z:
        t = {k: z[k] for k in z.files}
    t["sha256_Xyw"] = str(t["sha256_Xyw"])
    if _sha_arr(t["X"], t["y"], t["w"]) != t["sha256_Xyw"]:
        raise ValueError(f"{SCORER_TABLES[which]}: (X, y, w) do not match the stored sha256")
    return t


def _sha_arr(*arrs) -> str:
    """sha256 over dtype, shape and bytes of each array (the hash recorded with the original training run)."""
    h = hashlib.sha256()
    for a in arrs:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def fit_lgbm(X, y, w, seed: int):
    """One stage-6 LightGBM model (600 rounds) on labelled rows X (n, 86) float32, target y, weight w."""
    import lightgbm as lgb
    return lgb.train(dict(LGBM_PARAMS, seed=seed), lgb.Dataset(X, y, weight=w), LGBM_ROUNDS)


def fit_et(X, y, w, seed: int, n_jobs: int = 4):
    """One stage-6 ExtraTrees model on labelled rows (the shipped ones were fit on the 566 development images)."""
    from sklearn.ensemble import ExtraTreesRegressor
    return ExtraTreesRegressor(**ET_PARAMS, n_jobs=n_jobs, random_state=seed).fit(X, y, sample_weight=w)


def fit_scorers(X_lgbm=None, y_lgbm=None, w_lgbm=None, X_et=None, y_et=None, w_et=None, out_dir=None,
                seeds=SEEDS) -> Path:
    """Fit and save the 10 + 10 scorer models as ``combS707_s<seed>.txt`` and ``et_seed<seed>.joblib`` in out_dir
    (default ``<runs>/scorer``). The shipped LightGBM models used all 707 labelled images, the ExtraTrees the 566
    development images. Arrays left as None are read from the shipped tables (``load_scorer_table``)."""
    import joblib
    if X_lgbm is None:
        t = load_scorer_table("lgbm"); X_lgbm, y_lgbm, w_lgbm = t["X"], t["y"], t["w"]
    if X_et is None:
        t = load_scorer_table("et"); X_et, y_et, w_et = t["X"], t["y"], t["w"]
    out = Path(out_dir or default_runs() / "scorer")
    out.mkdir(parents=True, exist_ok=True)
    for s in seeds:
        fit_lgbm(X_lgbm, y_lgbm, w_lgbm, s).save_model(str(out / f"combS707_s{s}.txt"))
        joblib.dump(fit_et(X_et, y_et, w_et, s), out / f"et_seed{s}.joblib", compress=3)
        print(f"scorer seed {s} saved", flush=True)
    return out
