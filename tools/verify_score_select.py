"""Check filseg.score and filseg.select against the original run, on the original stored rows.

usage: python tools/verify_score_select.py --orig <original Claude track root> [--log FILE]

Reads (never writes) under --orig:
  work/ideas8/bundle-test/rows/*.npz           the 180 row files the original run scored
  work/maps/final/testcnx.npy.json             the stem list the original run used to find them
  work/ideas8/b3f707/test_q707_combS.npz       original q_L                         -> expect bit-equal
  work/ideas11/synth/csv/test_q_s1.npz         original q_ET, its per-variant means and the blend -> report max |d|
  work/ideas11/synth/csv/sel_s1.pkl            original selection                   -> expect 180/180 images equal
Also measures whether ExtraTrees prediction repeats bit for bit with n_jobs=1 and with n_jobs=4, and whether every
HiGHS solve is proven optimal (status, mip_gap), re-solving with mip_rel_gap=0 to compare. Writes the log and a copy
of the scores and selection under FILSEG_WORK/verify_score_select/. CPU only.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filseg import paths, score, select  # noqa: E402


class Log:
    """Print to stdout and append to a file."""

    def __init__(self, path: Path):
        self.f = open(path, "w", encoding="utf-8", newline="\n")

    def __call__(self, msg: str) -> None:
        print(msg, flush=True)
        self.f.write(msg + "\n"); self.f.flush()


def flat(objs) -> np.ndarray:
    return np.concatenate([np.asarray(x, np.float64) for x in objs], -1)


def maxd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def check_stems(orig: Path, tv: list[list[dict]], log: Log) -> None:
    files = json.loads((orig / "work/maps/final/testcnx.npy.json").read_text())["files"]
    ok = sorted(f[:-5] for f in files) == [im["name"] for im in tv[0]]
    log(f"stems: {len(tv[0])} row files; sorted names == original sorted sidecar stems: {ok}")
    assert ok


def check_lgbm(orig: Path, tv: list[list[dict]], sc: dict, log: Log) -> None:
    z = np.load(orig / "work/ideas8/b3f707/test_q707_combS.npz", allow_pickle=True)
    rows_eq = list(z["names"]) == [im["name"] for im in tv[0]] and all(
        np.array_equal(np.asarray(r), im["idx"]) for r, im in zip(z["rows"], tv[0]))
    q_eq = np.array_equal(sc["q_l"], flat(z["q"]))
    v_eq = np.array_equal(sc["qv_l"], flat(z["qvar"]))
    log(f"q_L vs test_q707_combS.npz: names and row indices equal {rows_eq} | q bit-equal {q_eq} "
        f"(max |d| {maxd(sc['q_l'], flat(z['q'])):.3e}) | per-variant q bit-equal {v_eq} | rows {len(sc['q_l'])}")
    assert rows_eq and q_eq and v_eq


def check_et_blend(orig: Path, tv: list[list[dict]], sc: dict, log: Log) -> dict:
    z = np.load(orig / "work/ideas11/synth/csv/test_q_s1.npz", allow_pickle=True)
    assert list(z["names"]) == [im["name"] for im in tv[0]]
    assert all(np.array_equal(np.asarray(r), im["idx"]) for r, im in zip(z["rows"], tv[0]))
    ref = dict(q=flat(z["q"]), q707=flat(z["q707"]), qet=flat(z["qet"]), qetvar=flat(z["qetvar"]))
    log(f"vs test_q_s1.npz: q707 bit-equal to q_L {np.array_equal(ref['q707'], sc['q_l'])} | "
        f"q_ET max |d| {maxd(sc['q_et'], ref['qet']):.3e} (bit-equal rows {int(np.sum(sc['q_et'] == ref['qet']))}"
        f"/{len(ref['qet'])}) | per-variant q_ET max |d| {maxd(sc['qv_et'], ref['qetvar']):.3e} | "
        f"blend q max |d| {maxd(sc['q'], ref['q']):.3e} (bit-equal rows {int(np.sum(sc['q'] == ref['q']))})")
    log(f"weights {z['weights'].tolist()} == {[score.W_L, score.W_ET]}, et_seeds {z['et_seeds'].tolist()} == "
        f"{score.SEEDS}: {z['weights'].tolist() == [score.W_L, score.W_ET] and z['et_seeds'].tolist() == score.SEEDS}")
    return ref


def check_et_determinism(et: list, tv: list[list[dict]], sc: dict, log: Log) -> None:
    for n_jobs in (1, 4):
        for m in et:
            m.n_jobs = n_jobs
        t0 = time.time()
        runs = [score.predict_et(et, tv).mean(1).mean(0) for _ in range(2)]
        same_run = np.array_equal(runs[0], runs[1])
        log(f"ET n_jobs={n_jobs}: two repeat predictions bit-equal {same_run} (max |d| {maxd(*runs):.3e}) | vs the "
            f"n_jobs=1 q_ET of the scoring run: bit-equal {np.array_equal(runs[0], sc['q_et'])}, max |d| "
            f"{max(maxd(r, sc['q_et']) for r in runs):.3e} | {(time.time() - t0) / 2:.1f}s per prediction")
    for m in et:
        m.n_jobs = 1


def compare_sel(res: dict, ref: dict) -> tuple[int, int, int]:
    """Images whose sel_sub matches in order, as a set, and whose sel_rows match as a set."""
    names = list(ref["sel_sub"])
    same_order = sum(np.array_equal(res["sel_sub"][n], np.asarray(ref["sel_sub"][n])) for n in names)
    same_set = sum(np.array_equal(np.sort(res["sel_sub"][n]), np.sort(ref["sel_sub"][n])) for n in names)
    same_rows = sum(np.array_equal(np.sort(res["sel_rows"][n]), np.sort(ref["sel_rows"][n])) for n in names)
    return same_order, same_set, same_rows


def milp_summary(results: list) -> str:
    st = np.bincount([r.status for r in results], minlength=1).tolist()
    gaps = np.array([r.mip_gap for r in results], np.float64)
    bound = np.array([abs(r.fun - r.mip_dual_bound) for r in results], np.float64)
    return (f"{len(results)} milp solves | status counts {st} (0 = optimal) | success {all(r.success for r in results)} "
            f"| mip_gap max {gaps.max():.3e}, solves with gap > 0: {int(np.sum(gaps > 0))} | "
            f"max |objective - dual bound| {bound.max():.3e}")


def check_selection(orig: Path, tv: list[list[dict]], sc: dict, ref_q: dict, log: Log) -> dict:
    ref = pickle.load(open(orig / "work/ideas11/synth/csv/sel_s1.pkl", "rb"))["sel"]["S1"]
    assert tuple(ref["cfg"]) == select.CFG
    subs, off = tv[0], sc["off"]
    out = {}
    for tag, q, opts in (("release q, default HiGHS options", sc["q"], None),
                         ("release q, mip_rel_gap=0", sc["q"], {"mip_rel_gap": 0}),
                         ("original stored q, default HiGHS options", ref_q["q"], None)):
        results = []
        t0 = time.time()
        res = select.select_all(subs, score.per_image(q, off), opts, results)
        o, s, r = compare_sel(res, ref)
        fs = np.bincount(np.concatenate([im["fam"][res["sel_sub"][im["name"]]] for im in subs]).astype(np.int64),
                         minlength=7)
        log(f"selection [{tag}] vs sel_s1.pkl: images equal in order {o}/180, as sets {s}/180, sel_rows {r}/180 | "
            f"selected {int(fs.sum())}: " + " / ".join(f"{score.FAMNAME[f]} {int(fs[f])}" for f in score.FAMILIES)
            + f" | {time.time() - t0:.1f}s")
        log(f"   {milp_summary(results)}")
        out[tag] = res
    a, b = out["release q, default HiGHS options"], out["release q, mip_rel_gap=0"]
    log(f"default options vs mip_rel_gap=0: same selection on "
        f"{sum(np.array_equal(a['sel_sub'][n], b['sel_sub'][n]) for n in a['sel_sub'])}/180 images")
    return a


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--orig", type=Path, required=True, help="root of the original Claude track (read only)")
    ap.add_argument("--log", type=Path, default=None, help="default <FILSEG_WORK>/verify_score_select.log")
    a = ap.parse_args()
    log = Log(a.log or paths.work("verify_score_select.log"))
    import joblib, lightgbm, scipy, sklearn  # noqa: E401
    log(f"numpy {np.__version__} | scipy {scipy.__version__} | sklearn {sklearn.__version__} | lightgbm "
        f"{lightgbm.__version__} | joblib {joblib.__version__} | OMP_NUM_THREADS {os.environ['OMP_NUM_THREADS']}")
    t0 = time.time()
    tv = score.load_views(a.orig / "work/ideas8/bundle-test/rows")
    lg, et = score.load_lgbm(), score.load_et(n_jobs=1)
    t1 = time.time()
    sc = score.score(tv, lg, et)
    log(f"loaded rows and 20 models in {t1 - t0:.1f}s; scored {len(sc['q'])} rows in {time.time() - t1:.1f}s")
    check_stems(a.orig, tv, log)
    check_lgbm(a.orig, tv, sc, log)
    ref_q = check_et_blend(a.orig, tv, sc, log)
    check_et_determinism(et, tv, sc, log)
    res = check_selection(a.orig, tv, sc, ref_q, log)
    out = paths.work("verify_score_select", "scores.npz")
    h = score.save(out, tv, sc)
    select.save(out.with_name("selection.pkl"), res, h)
    back = pickle.load(open(out.with_name("selection.pkl"), "rb"))["sel"]["S1"]
    log(f"wrote {out} (sha256 {h}) and selection.pkl; reloaded selection equals the in-memory one: "
        f"{all(np.array_equal(back['sel_rows'][n], res['sel_rows'][n]) for n in res['sel_rows'])}")
    log(f"total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
