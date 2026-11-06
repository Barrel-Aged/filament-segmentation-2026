"""Check stages 6-9 end to end: rebuilt rows -> scores -> selection -> shifted CSV == the submitted s1-S-t024.csv.

usage (repo root): python -m tools.verify_cpu_chain --rows <rows folder> --maps <folder with test{cnx,cnxL,r34}.npy>
"""
import argparse
import hashlib
import time
from pathlib import Path

from filseg import paths, score, select, submit

SUBMITTED = "18bf87d67f7c663c4a0a82d8f17fe77cfb378ae81c50dbe8c71a99b48ea74cd4"   # s1-S-t024.csv, Kaggle ref 57928175


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--maps", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=paths.WORK / "cpu_chain")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tv = score.load_views(a.rows)
    sc = score.score(tv, score.load_lgbm(), score.load_et())
    sha_scores = score.save(a.out / "scores.npz", tv, sc)
    print(f"scores: {len(sc['q'])} rows | {time.time() - t0:.0f}s", flush=True)
    res = select.select_all(tv[0], score.per_image(sc["q"], score.offsets(tv[0])))
    select.save(a.out / "selection.pkl", res, sha_scores)
    print(f"selection: {sum(len(v) for v in res['sel_sub'].values())} rows | {time.time() - t0:.0f}s", flush=True)
    r = submit.make_submission(a.rows, a.out / "scores.npz", a.out / "selection.pkl", a.maps, a.out / "submission.csv")
    print(f"submission: {r['instance_rows']} instances | sha256 {r['sha256']} | {time.time() - t0:.0f}s")
    print("MATCHES the submitted s1-S-t024.csv" if r["sha256"] == SUBMITTED else "DIFFERS from the submitted file")


if __name__ == "__main__":
    main()
