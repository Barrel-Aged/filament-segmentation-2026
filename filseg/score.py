"""Stage 6: the row scorer q = 0.7 q_L + 0.3 q_ET.

Every candidate row (one row file per test image, written by stage 5) has a 74-column block per detector variant
k = 0..3 (``F[k]``; the variants differ only in which of the four Mask R-CNN fold models gave the detector columns)
and 18 cross-map columns (``xm``). The scorer sees 86 columns: ``F[k]`` plus 12 of the cross-map columns, namely the
two dense maps the row did not come from (``xm_cols``). Only the rows of the "combS" pool are scored: tree rows
(family 0) and the merge, bridged-pair, cnxL and R34 families (2, 4, 5, 6).

q_L  : 10 LightGBM models (seeds 10-19, trained on 707 images). Per variant, the mean of the 10 models; then the mean
       of the 4 variants.
q_ET : 10 ExtraTrees models (seeds 10-19, trained on the 566 development images), averaged in the same order.
q    : 0.7 q_L + 0.3 q_ET.

Changes from the original run
- Inputs and paths. The original read ``<bundle-test>/rows/<stem>.npz`` for the sorted stems of the cnx map sidecar
  ``work/maps/final/testcnx.npy.json`` (score_select.py:87-98). Here the rows folder is an argument and the stems are
  the sorted ``*.npz`` names in it. All absolute paths, ``sys.path`` inserts and the ``CUDA_VISIBLE_DEVICES`` and
  ``OMP_NUM_THREADS`` assignments at import are gone (score707.py:10-15, score_select.py:19-21, csv_score.py:16-25).
- Import side effects. score_select.py:23-26 swapped ``sys.argv`` to import xcv, which parses argv and patches
  ``cfil.stack.gbm_params`` at import (xcv.py:23-27) and imports ``cfil.decide``, which reads the split file at
  import (xcv.py:21). None of that is imported here; ``nl_sub``, ``xm_cols`` and ``predict`` are copied.
- Each row file is read once and the four variants are built from it; the original read every file once per
  variant (score707.py:41, csv_score.py:57). The arrays are the same.
- Only the combS layout is kept. ``nl_sub`` (score_select.py:49-56) no longer carries ``parents``, ``nb`` and
  ``base_only``: they feed only the tree DP of the "base" layout (xcv.py:54-55), which this pool never uses.
  ``xm_cols`` keeps only the ``'sym'`` branch (holdrows.py:37-45).
- LightGBM models come from ``weights/models/combS707_s1*.txt`` through ``filseg.weights`` (sha256 checked against
  ``weights/manifest.json``) and are built with ``lgb.Booster(model_str=...)``. The original built them with
  ``model_file=`` after checking the sha256 against ``b3f707/models/train_meta.json`` (score707.py:33-39).
- q_L is computed here. The blend step of the original run read it back from ``test_q707_combS.npz`` (written by
  score707.py:84-85) after checking that file's sha256 (csv_score.py:35, 55, 61-63).
- ExtraTrees models are loaded from ``weights/models/et_seed1*.joblib`` (saved by work/package/et/save_et.py with the
  same recipe, data and seeds) instead of being fit and discarded (csv_score.py:66-85); the training-row loader is
  gone. Prediction uses ``n_jobs=1`` instead of 5 threads (csv_score.py:18, 76, 79). With several threads sklearn sums
  the tree predictions in completion order, so the result can change from run to run in the last bits; with one job
  the trees are summed in their stored order and the result is the same on every run (measured: two n_jobs=4 runs
  differ by up to 3.3e-16, two n_jobs=1 runs are bit-equal; n_jobs=1 takes about 9 s for all 40 predictions instead
  of 3 s). Against the original q_ET the largest difference is 3.3e-16 and the blend's is 1.1e-16; the selection is
  unchanged (tools/verify_score_select.py).
- Removed: the comparisons with older outputs and their prints (score707.py:50-69, 77-82; csv_score.py:89-113), the
  selection (now filseg.select) and the writes of the original outputs (score707.py:84-91, csv_score.py:114-125).
  ``save`` writes the same keys as the original ``test_q_s1.npz`` (csv_score.py:115-118) to a path you choose.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np

from filseg import paths, weights

FAMILIES = (0, 2, 4, 5, 6)           # combS pool: tree, merge, bridged pair, cnxL, R34
FAMNAME = {0: "tree", 2: "mg", 4: "pm", 5: "cnxL", 6: "R34"}
N_VARIANTS = 4
N_COLUMNS = 86
SEEDS = list(range(10, 20))
LGBM_THREADS = 4
W_L, W_ET = 0.7, 0.3


def objarr(xs: list) -> np.ndarray:
    """A 1-d object array holding the given arrays (score_select.objarr)."""
    a = np.empty(len(xs), object)
    for i, x in enumerate(xs):
        a[i] = x
    return a


def xm_cols(X: np.ndarray, fam: np.ndarray) -> np.ndarray:
    """The 12 cross-map columns of the two maps a row does not come from (holdrows.xm_cols, ncol 'sym').

    ``X`` holds 6 columns per map in the order R34 | cnxL | cnx. R34 rows (family 6) take cnxL instead of R34 for
    the first block; cnxL and R34 rows (families 5, 6) take cnx instead of cnxL for the second.
    """
    A = np.where((fam == 6)[:, None], X[:, 6:12], X[:, 0:6])
    B = np.where(np.isin(fam, (5, 6))[:, None], X[:, 12:18], X[:, 6:12])
    return np.concatenate([A, B], 1)


def read_rows(rows_dir: Path) -> list[dict]:
    """The row files of every image in ``rows_dir``, sorted by name."""
    out = []
    for p in sorted(Path(rows_dir).glob("*.npz")):
        z = np.load(p)
        if "ok" in z and not bool(z["ok"]):  # rows written by filseg.rows have no "ok" key
            raise ValueError(f"{p.name}: row file is not marked ok")
        out.append(dict(name=p.stem, F=z["F"], xm=z["xm"], fam=z["fam"], conf=z["conf"]))
    if not out:
        raise FileNotFoundError(f"no row files in {rows_dir}")
    return out


def variant_image(z: dict, k: int) -> dict:
    """All rows of one image with the 86 scorer columns of detector variant ``k`` (score_select.load_test)."""
    return dict(name=z["name"], F=np.concatenate([z["F"][k], xm_cols(z["xm"], z["fam"])], 1), fam=z["fam"],
                conf=z["conf"])


def nl_sub(im: dict) -> dict:
    """The combS rows of one image and the conflict pairs among them (score_select.nl_sub).

    ``idx`` maps a sub-row back to its row in the row file; ``conf`` is re-indexed to sub-rows.
    """
    keep = np.isin(im["fam"], FAMILIES); idx = np.flatnonzero(keep)
    remap = -np.ones(len(keep), np.int64); remap[idx] = np.arange(len(idx))
    c = im["conf"]; c = c[keep[c[:, 0]] & keep[c[:, 1]]]
    return dict(name=im["name"], F=im["F"][idx], idx=idx, fam=im["fam"][idx], conf=remap[c])


def load_views(rows_dir: Path) -> list[list[dict]]:
    """``tv[k][i]``: the combS sub of image i under detector variant k, for k = 0..3."""
    rows = read_rows(rows_dir)
    tv = [[nl_sub(variant_image(z, k)) for z in rows] for k in range(N_VARIANTS)]
    assert all(v[i]["F"].shape == tv[0][i]["F"].shape for v in tv for i in range(len(rows)))
    assert tv[0][0]["F"].shape[1] == N_COLUMNS
    return tv


def offsets(subs: list[dict]) -> np.ndarray:
    """Start of each image's rows in the concatenated row axis, plus the total at the end."""
    return np.concatenate([[0], np.cumsum([len(im["F"]) for im in subs])])


def load_lgbm() -> list[lgb.Booster]:
    """The 10 LightGBM models, seeds 10-19."""
    ms = [lgb.Booster(model_str=weights.read_bytes(f"models/combS707_s{s}.txt").decode("utf-8")) for s in SEEDS]
    assert all(m.num_feature() == N_COLUMNS and m.num_trees() == 600 for m in ms)
    return ms


def load_et(n_jobs: int = 1) -> list:
    """The 10 ExtraTrees models, seeds 10-19, set to predict with ``n_jobs`` workers."""
    ms = [joblib.load(weights.open_file(f"models/et_seed{s}.joblib")) for s in SEEDS]
    for m in ms:
        m.n_jobs = n_jobs
    return ms


def predict_lgbm(ms: list[lgb.Booster], tv: list[list[dict]]) -> np.ndarray:
    """P[k, j, row]: model j on variant k (score_select.predict; float64 input, 4 threads)."""
    out = []
    for subs in tv:
        X = np.concatenate([im["F"] for im in subs]).astype(np.float64)
        out.append(np.stack([m.predict(X, num_threads=LGBM_THREADS) for m in ms]))
    return np.stack(out)


def predict_et(ms: list, tv: list[list[dict]]) -> np.ndarray:
    """P[k, j, row]: model j on variant k (csv_score.py:75-85; float32 input as stored)."""
    Xv = [np.concatenate([im["F"] for im in v]) for v in tv]
    P = np.zeros((len(tv), len(ms), len(Xv[0])))
    for j, m in enumerate(ms):
        for k in range(len(tv)):
            P[k, j] = m.predict(Xv[k])
    return P


def score(tv: list[list[dict]], lgbm: list[lgb.Booster], et: list) -> dict:
    """q_L, q_ET and the blend for all rows, concatenated over images in ``tv[0]`` order."""
    qv_l = predict_lgbm(lgbm, tv).mean(1)          # per variant: mean of the 10 models
    q_l = qv_l.mean(0)                             # mean of the 4 variants
    qv_et = predict_et(et, tv).mean(1)
    q_et = qv_et.mean(0)
    q = W_L * q_l + W_ET * q_et
    return dict(off=offsets(tv[0]), q=q, q_l=q_l, qv_l=qv_l, q_et=q_et, qv_et=qv_et)


def per_image(a: np.ndarray, off: np.ndarray) -> list[np.ndarray]:
    """Split the last (row) axis of ``a`` into one array per image."""
    return [a[..., off[i]:off[i + 1]] for i in range(len(off) - 1)]


def save(path: Path, tv: list[list[dict]], sc: dict) -> str:
    """Write the scores with the keys of the original ``test_q_s1.npz``; return the file's sha256.

    names; per image: q (blend), q707 (q_L), qet (q_ET), qetvar (q_ET per variant, 4 x rows), rows (row indices in the
    row file); weights (0.7, 0.3); et_seeds. Per-image arrays are stored as object arrays (load with allow_pickle).
    """
    off = sc["off"]
    np.savez(path, names=np.array([im["name"] for im in tv[0]]), q=objarr(per_image(sc["q"], off)),
             q707=objarr(per_image(sc["q_l"], off)), qet=objarr(per_image(sc["q_et"], off)),
             qetvar=objarr(per_image(sc["qv_et"], off)), rows=objarr([im["idx"] for im in tv[0]]),
             weights=np.array([W_L, W_ET]), et_seeds=np.array(SEEDS))
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="Score the candidate rows: q = 0.7 LightGBM + 0.3 ExtraTrees.")
    ap.add_argument("--rows", type=Path, default=paths.WORK / "rows", help="folder of row files (stage 5)")
    ap.add_argument("--out", type=Path, default=None, help="output .npz (default <FILSEG_WORK>/scores.npz)")
    a = ap.parse_args()
    tv = load_views(a.rows)
    sc = score(tv, load_lgbm(), load_et())
    out = a.out or paths.work("scores.npz")
    print(f"scored {len(sc['q'])} rows of {len(tv[0])} images | q mean {sc['q'].mean():.4f} | {out} sha256 "
          f"{save(out, tv, sc)}")


if __name__ == "__main__":
    main()
