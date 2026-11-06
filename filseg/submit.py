"""Stages 8-9 together: from the selection to the submission CSV.

For every test image: take the selected rows, reshape the family 0/5/6 masks with the 0.23 px shift on their own map
(filseg.shift), let rows claim pixels in descending row score so the instances stay disjoint, then write all
instances as column-major COCO RLE (filseg.write). ``shift=None`` writes the stored masks unshifted.

This is the glue that tools/verify_write_shift.py checks piece by piece against the original run.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from filseg import paths
from filseg import shift as S
from filseg import write as WR


def load_selection(sel_path: Path) -> dict:
    with open(sel_path, "rb") as fh:
        return pickle.load(fh)["sel"]["S1"]


def load_scores(scores_path: Path) -> dict:
    """{stem: (row indices of the scored pool, q)} from the output of filseg.score."""
    z = np.load(scores_path, allow_pickle=True)
    return {str(n): (np.asarray(r, np.int64), np.asarray(v, np.float64)) for n, r, v in zip(z["names"], z["rows"], z["q"])}


def image_instances(rows_file: Path, sel_rows: np.ndarray, idx: np.ndarray, q: np.ndarray, own: dict | None,
                    shift=S.SHIFT, counters: dict | None = None) -> list:
    """The final masks (sorted pixel indices) of one image, in selection order."""
    with np.load(rows_file) as z:
        sets = WR.unpack_sets(z["set_flat"], z["set_len"])
        fam, lvl, t32 = z["fam"], z["lvl"].astype(np.int64), z["F"][0][:, 0]
    if shift is None:
        return S.image_masks(sets, fam, lvl, sel_rows, shift=None)
    if not len(sel_rows):
        return []
    s_row = S.row_scores(q, idx, t32, fam)
    finals = S.image_masks(sets, fam, lvl, sel_rows, s_row, own, counters=counters)
    return [m for m in finals if m is not None]


def make_submission(rows_dir: Path, scores_path: Path, sel_path: Path, map_dir: Path, out_csv: Path,
                    shift=S.SHIFT) -> dict:
    """Write the submission CSV. ``map_dir`` holds test{cnx,cnxL,r34}.npy and their .json (the file order)."""
    files = json.loads((Path(map_dir) / "testcnx.npy.json").read_text())["files"]
    sel, q = load_selection(sel_path), load_scores(scores_path)
    maps = S.open_maps(map_dir, files) if shift is not None else None
    counters = S.new_counters()
    masks = {}
    for j, f in enumerate(files):
        stem = f[:-5]
        idx, qq = q[stem]
        own = S.own_maps(maps, j) if maps is not None else None
        masks[f] = image_instances(Path(rows_dir) / f"{stem}.npz", np.asarray(sel["sel_rows"][stem], np.int64),
                                   idx, qq, own, shift, counters)
    res, _ = WR.write_csv(out_csv, masks, files, overwrite=True)
    res["shift_counters"] = counters
    res["sha256"] = WR.sha256_file(out_csv)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Write the submission CSV from the selection.")
    ap.add_argument("--rows", type=Path, default=paths.WORK / "rows")
    ap.add_argument("--scores", type=Path, default=paths.WORK / "scores.npz")
    ap.add_argument("--selection", type=Path, default=paths.WORK / "selection.pkl")
    ap.add_argument("--maps", type=Path, default=paths.WORK / "maps")
    ap.add_argument("--out", type=Path, default=paths.WORK / "submission.csv")
    ap.add_argument("--no-shift", action="store_true", help="write the selected masks without the 0.23 px shift")
    a = ap.parse_args()
    res = make_submission(a.rows, a.scores, a.selection, a.maps, a.out, None if a.no_shift else S.SHIFT)
    print(f"{a.out}: {res['instance_rows']} instances, sha256 {res['sha256']}")


if __name__ == "__main__":
    main()
