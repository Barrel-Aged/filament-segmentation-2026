"""Check filseg.write / filseg.shift / filseg.validate / filseg.metrics against the original run's inputs and outputs.

usage (from the repository root):
    python -m tools.verify_write_shift --orig-root <the original project tree> [--out DIR] [--log FILE] [--pq-check]

Inputs, all read-only, at their original places under --orig-root:
  work/ideas11/synth/csv/sel_s1.pkl (+ .sha256)   the final's selection (tau 0.24)
  work/ideas11/synth/csv/test_q_s1.npz            its q, for the row scores s that set the claiming order
  work/ideas8/bundle-test/rows/<stem>.npz         the candidate rows (pixel sets, family, level)
  work/maps/final/test{cnx,cnxL,r34}.npy (+ .json) the test maps the shift reads (channel 0)
It writes (a) the unshifted CSV and requires sha256 3357e64d... (= work/ideas11/synth/csv/s1-t024.csv) and (b) the shifted
CSV and requires sha256 18bf87d6... (= work/ideas15/hyg/csv/s1-S-t024.csv, the submitted final), both also compared byte
for byte with the original files; then runs filseg.validate on both, the shift's own checks, and the shifted-vs-unshifted
comparison. --pq-check also scores the original run's two dev fold-0 prediction files (work/ideas15/hyg/csv/
dev_f0_L110_{BASE,S}.pred.json, development labels only) with filseg.metrics and requires the PQ recorded in e2e_dev.json.
Outputs go to --out (default: $FILSEG_WORK/verify_write_shift); the log to --log (default: $FILSEG_WORK/verify_write_shift.log).
CPU only. Exit code 0 only if every check passes.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse
import filecmp
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from filseg import paths
from filseg import shift as S
from filseg import validate as V
from filseg import write as WR
from filseg.metrics import score_predictions

EXPECT_UNSHIFTED = "3357e64d78958bc5578a48ab749cad1c9c9d87929abb373e197bc9fc7caabd21"
EXPECT_SHIFTED = "18bf87d67f7c663c4a0a82d8f17fe77cfb378ae81c50dbe8c71a99b48ea74cd4"
CFG = ("combS", 0.475, 0.2, 0.03, 0.24)
POOL_FAMILIES = (0, 2, 4, 5, 6)
ORIG = dict(
    inventory="work/maps/final/testcnx.npy.json", maps="work/maps/final", rows="work/ideas8/bundle-test/rows",
    sel="work/ideas11/synth/csv/sel_s1.pkl", sel_sha="work/ideas11/synth/csv/sel_s1.sha256",
    q="work/ideas11/synth/csv/test_q_s1.npz", unshifted="work/ideas11/synth/csv/s1-t024.csv",
    shifted="work/ideas15/hyg/csv/s1-S-t024.csv", e2e="work/ideas15/hyg/csv/e2e_dev.json",
    pred="work/ideas15/hyg/csv/dev_f0_L110_{arm}.pred.json", split="configs/split-v2.json",
    annotations="../data/annotations/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
CHECKS: list = []


class Log:
    """print to stdout and append to the log file."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = path.open("w", encoding="utf-8")

    def __call__(self, msg: str) -> None:
        print(msg, flush=True)
        self.fh.write(msg + "\n"); self.fh.flush()


def check(log, name, ok, detail=""):
    CHECKS.append((name, bool(ok), detail))
    log(f"[{'PASS' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}")


def load_inputs(root: Path, log):
    """Test inventory, the S1 selection (sha-checked) and q per image."""
    files = json.loads((root / ORIG["inventory"]).read_text())["files"]
    check(log, "test inventory: 180 unique files", len(files) == 180 and len(set(files)) == 180)
    h = WR.sha256_file(root / ORIG["sel"])
    check(log, "sel_s1.pkl sha256 == its .sha256 file", (root / ORIG["sel_sha"]).read_text().split()[0] == h, h)
    with open(root / ORIG["sel"], "rb") as fh:
        sel = pickle.load(fh)["sel"]["S1"]
    check(log, f"selection config {CFG}", tuple(sel["cfg"]) == CFG)
    Z = np.load(root / ORIG["q"], allow_pickle=True)
    q = {str(n): (np.asarray(r, np.int64), np.asarray(v, np.float64)) for n, r, v in zip(Z["names"], Z["rows"], Z["q"])}
    log(f"test_q_s1.npz sha256 {WR.sha256_file(root / ORIG['q'])}")
    return files, sel, q


def build_masks(root: Path, files, sel, q, log):
    """Per image: unshifted masks, shifted masks (S + claiming), families per instance; shift counters."""
    maps = S.open_maps(root / ORIG["maps"], files)
    cnt, direction = S.new_counters(), None
    unshifted, shifted, fams = {}, {}, []
    bad_pool = 0
    for j, f in enumerate(files):
        stem = f[:-5]
        with np.load(root / ORIG["rows"] / f"{stem}.npz") as z:
            if not bool(z["ok"]):
                raise ValueError(f"{stem}: rows ok flag is False")
            sets = WR.unpack_sets(z["set_flat"], z["set_len"])
            fam, lvl, t32 = z["fam"], z["lvl"].astype(np.int64), z["F"][0][:, 0]
            lvl_ok = np.array_equal(z["F"][0][:, 1].astype(np.int64), lvl) and np.array_equal(t32, S.LEV32[lvl])
        sel_rows = np.asarray(sel["sel_rows"][stem], np.int64)
        idx, qq = q[stem]
        sel_sub = np.asarray(sel["sel_sub"][stem], np.int64)
        bad_pool += not (lvl_ok and np.array_equal(idx, np.flatnonzero(np.isin(fam, POOL_FAMILIES)))
                         and np.array_equal(idx[sel_sub], sel_rows))
        unshifted[f] = S.image_masks(sets, fam, lvl, sel_rows, shift=None)
        if not len(sel_rows):
            shifted[f] = []
            continue
        own = S.own_maps(maps, j)
        s_row = S.row_scores(qq, idx, t32, fam)
        finals = S.image_masks(sets, fam, lvl, sel_rows, s_row, own, counters=cnt)
        for r, m in zip(sel_rows, finals):
            if m is not None and fam[r] in S.SHIFT_FAMILIES:
                direction = S.direction_check(sets[r], m, own[int(fam[r])], direction)
        shifted[f] = [m for m in finals if m is not None]
        fams += [int(fam[r]) for r, m in zip(sel_rows, finals) if m is not None]
    check(log, "rows: level / t columns consistent, q rows == combS pool, sel_rows == idx[sel_sub]", bad_pool == 0,
          f"{180 - bad_pool}/180 images")
    return unshifted, shifted, fams, cnt, direction


def shift_checks(cnt, d, log):
    check(log, "stored fam 0/5/6 masks strictly above THR[lvl] on their own map", cnt["above_thr"] == cnt["reshaped"],
          f"{cnt['above_thr']}/{cnt['reshaped']}")
    check(log, "zero-shift identity (reshape at (0, 0) == stored mask)", cnt["zero_identity"] == cnt["reshaped"],
          f"{cnt['zero_identity']}/{cnt['reshaped']}")
    check(log, "no row emptied by reshape or claiming", cnt["kept"] == cnt["selected"] and cnt["emptied_by_claim"] == 0
          and cnt["emptied_by_reshape"] == 0, ", ".join(f"{k} {v}" for k, v in cnt.items()))
    n = d["changed_px"] - d["edge"]
    check(log, "every changed pixel consistent with content moving toward row 0",
          d["up_ok"] == n and d["inst_all_up"] == d["instances"] and d["edge"] == 0,
          f"{d['up_ok']}/{n} pixels, {d['inst_all_up']}/{d['instances']} instances (toward row 2047 {d['down_ok']}/{n}, "
          f"columns {d['colp_ok']}/{n} and {d['colm_ok']}/{n})")


def write_and_compare(tag, out: Path, masks, files, expect_sha, original: Path, meta, log):
    t0 = time.time()
    res, _ = WR.write_csv(out, masks, files, meta=meta, overwrite=True)
    sha = WR.sha256_file(out)
    log(f"{tag}: wrote {out} ({res['instance_rows']} rows, {res['images_with_instances']} images with instances, "
        f"empty {res['empty_images']}) in {time.time() - t0:.0f}s")
    check(log, f"{tag}: sha256 == {expect_sha[:8]}...", sha == expect_sha, sha)
    check(log, f"{tag}: byte-identical to the original {original.name}", filecmp.cmp(out, original, shallow=False))
    checks = V.validate_csv(out, files, expected=masks)
    for n, ok, d in checks:
        check(log, f"{tag}: {n}", ok, d)


def pq_check(root: Path, log):
    """filseg.metrics on the original run's dev fold-0 prediction files == the PQ recorded in e2e_dev.json."""
    rec = json.loads((root / ORIG["e2e"]).read_text())
    for arm in ("BASE", "S"):
        t0 = time.time()
        r = score_predictions(root / ORIG["annotations"], root / ORIG["split"], root / ORIG["pred"].format(arm=arm), 0)
        want = rec["arms"][arm]["pq_csv_official"]
        check(log, f"PQ replica, dev fold 0 {arm}: PQ == e2e_dev.json pq_csv_official", r["pq"] == want,
              f"{r['pq']!r} vs {want!r} (TP {r['tp']} FP {r['fp']} FN {r['fn']}, {len(r['records'])} records, "
              f"{time.time() - t0:.0f}s)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--orig-root", required=True, type=Path, help="root of the original project tree (read-only)")
    ap.add_argument("--out", type=Path, default=paths.WORK / "verify_write_shift")
    ap.add_argument("--log", type=Path, default=paths.WORK / "verify_write_shift.log")
    ap.add_argument("--pq-check", action="store_true", help="also check filseg.metrics on the dev fold-0 predictions")
    a = ap.parse_args(argv)
    root = a.orig_root.resolve()
    log = Log(a.log)
    t0 = time.time()
    import scipy, cv2  # noqa: E401  (versions for the log)
    from importlib import metadata
    log(f"verify_write_shift: python {sys.version.split()[0]}, numpy {np.__version__}, scipy {scipy.__version__}, "
        f"opencv {cv2.__version__}, pycocotools {metadata.version('pycocotools')}; CUDA_VISIBLE_DEVICES="
        f"{os.environ['CUDA_VISIBLE_DEVICES']!r}")
    files, sel, q = load_inputs(root, log)
    unshifted, shifted, fams, cnt, direction = build_masks(root, files, sel, q, log)
    log(f"masks built ({time.time() - t0:.0f}s)")
    shift_checks(cnt, direction, log)
    meta = dict(selection_sha256=WR.sha256_file(root / ORIG["sel"]), cfg=list(CFG))
    write_and_compare("unshifted", a.out / "s1-t024.csv", unshifted, files, EXPECT_UNSHIFTED, root / ORIG["unshifted"],
                      dict(meta, shift=None), log)
    write_and_compare("shifted", a.out / "s1-S-t024.csv", shifted, files, EXPECT_SHIFTED, root / ORIG["shifted"],
                      dict(meta, shift_dy_dx=list(S.SHIFT), shifted_families=list(S.SHIFT_FAMILIES)), log)
    for n, ok, d in V.compare_with_unshifted(a.out / "s1-S-t024.csv", a.out / "s1-t024.csv", fams):
        check(log, f"shifted vs unshifted: {n}", ok, d)
    if a.pq_check:
        pq_check(root, log)
    npass = sum(ok for _, ok, _ in CHECKS)
    log(f"verify_write_shift: {npass}/{len(CHECKS)} checks PASS -> {'PASS' if npass == len(CHECKS) else 'FAIL'} "
        f"({time.time() - t0:.0f}s)")
    return 0 if npass == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
