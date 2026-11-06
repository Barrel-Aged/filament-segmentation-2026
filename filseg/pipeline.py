"""The pipeline driver: one function per stage, and ``run`` for the whole chain. Used by ``demo.ipynb``.

    from filseg import pipeline
    pipeline.run(mode='replay', device='cpu', data='<competition folder>', work='work', out_csv='work/submission.csv')

Modes
- ``mode='full'``: stages 1-10 from the test images. Stages 2-4 (U-Nets, Mask R-CNN, I6) run on ``device``.
- ``mode='replay'``: unpack ``replay/`` (the stage 2-4 outputs of the run that produced the submission, see
  ``filseg.replay``) and run stages 5-10 on them, on the CPU. The CSV is byte-identical to the submitted
  ``s1-S-t024.csv`` (``SUBMITTED_SHA256``).

Every stage function takes its inputs as arguments and returns what the next stage needs, so a notebook cell is one
call. Each prints its wall time when it finishes; ``TIMES`` collects them. The computation is that of the stage
modules; this file only connects them. Stages 8 and 9 are ``filseg.submit.make_submission`` split in two (the shift
and claiming loop, then the writer), so that the final masks can be inspected before they are written.

Windows: stages 5a and 5b start worker processes (``ProcessPoolExecutor``, at most 4). Their worker functions live at
module level in ``filseg.evidence`` and ``filseg.rows``, so they work under the spawn start method from a script (put
the call under ``if __name__ == '__main__':``) and from a normal Jupyter kernel.

Work folder layout (``work``): ``cache/`` (stage 1), ``maps/`` (2), ``det/`` (3), ``i6-candidates/test`` (4),
``replay/`` (replay mode), ``cands/``, ``ext/``, ``feat/`` (5a), ``rows/`` (5b), ``scores.npz`` (6),
``selection.pkl`` (7). Use a separate work folder for each image set (the scorer reads every row file in ``rows/``).
"""
from __future__ import annotations

import contextlib
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from filseg import paths

SUBMITTED_SHA256 = "18bf87d67f7c663c4a0a82d8f17fe77cfb378ae81c50dbe8c71a99b48ea74cd4"   # s1-S-t024.csv
SUBMITTED_CSV = paths.REPO / "submission" / "s1-S-t024.csv"
MAP_SOURCES = ("cnx", "cnxL", "r34")
NPROC = 4
H = W = 2048

TIMES: dict = {}


# ---- helpers ------------------------------------------------------------------------------------------------------

@contextlib.contextmanager
def _stage(key: str, label: str):
    print(f"[{key}] {label} ...", flush=True)
    t0 = time.time()
    yield
    TIMES[key] = time.time() - t0
    print(f"[{key}] {label}: {TIMES[key]:.0f} s", flush=True)


def cpu_only() -> None:
    """Hide the GPUs from torch in this process and its workers. Call before torch is imported.

    On Windows an empty ``CUDA_VISIBLE_DEVICES`` does not hide the GPU from torch; ``-1`` does.
    """
    if "torch" in sys.modules:
        print("note: torch is already imported; CUDA_VISIBLE_DEVICES set now does not hide the GPU", flush=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def _free_gpu(device: str) -> None:
    gc.collect()
    if device == "cuda":
        import torch
        torch.cuda.empty_cache()


def test_images(data=None) -> Path:
    """``<data>/test/test_images`` (default ``filseg.paths.TEST_IMAGES``)."""
    return Path(data) / "test" / "test_images" if data else paths.TEST_IMAGES


def map_paths(map_dir) -> dict:
    """{'cnx': .../testcnx.npy, 'cnxL': ..., 'r34': ...}."""
    return {n: Path(map_dir) / f"test{n}.npy" for n in MAP_SOURCES}


def map_files(maps: dict) -> list:
    """The image file names in map order (the sidecar of the cnx map)."""
    return json.loads(Path(str(maps["cnx"]) + ".json").read_text(encoding="utf-8"))["files"]


# ---- stages 1-4 (full mode) ---------------------------------------------------------------------------------------

def stage1_images(image_dir, work, names=None):
    """Stage 1: read the images and their disk statistics (``filseg.data``).

    ``names=None`` reads every image of ``image_dir`` and writes ``<work>/cache`` (the original run's layout); a list
    of file names reads only those, in sorted order, into memory (used for the dev-fold evaluation).
    Returns (records, images uint8 (n, 2048, 2048)).
    """
    from filseg import data
    with _stage("1", "images"):
        if names is None:
            data.write_test_cache(Path(work) / "cache", image_dir)
            recs, images = data.open_test_cache(Path(work) / "cache")
        else:
            names = sorted(names)
            images = np.empty((len(names), H, W), np.uint8)
            recs = []
            for i, n in enumerate(names):
                images[i] = data.read_image(image_dir, n)
                s = data.disk_stats(images[i])
                recs.append(dict(file_name=n, disk_mean=s[0], disk_std=s[1], disk_thr=s[2]))
        print(f"    {len(recs)} images", flush=True)
    return recs, images


def stage2_dense(recs, images, work, device="cuda") -> dict:
    """Stage 2: the three U-Net probability maps, each the mean of 4 fold models (``filseg.dense``).
    Returns the map paths ``{'cnx', 'cnxL', 'r34'}`` under ``<work>/maps``."""
    from filseg import dense
    with _stage("2", "dense maps (cnx, cnxL, r34)"):
        for src in MAP_SOURCES:
            t0 = time.time()
            dense.infer_source(src, images, recs, Path(work) / "maps", device=device)
            TIMES[f"2-{src}"] = time.time() - t0
            _free_gpu(device)
    return map_paths(Path(work) / "maps")


def stage3_det(recs, images, work, device="cuda") -> list:
    """Stage 3: Mask R-CNN candidates of the 4 fold models (``filseg.det``). Returns the 4 candidate roots."""
    from filseg import det
    with _stage("3", "Mask R-CNN, 4 folds"):
        det.infer_all(images, recs, Path(work) / "det", device=device)
        _free_gpu(device)
    return [Path(work) / "det" / f"det-f{k}" / "test" for k in det.FOLDS]


def stage4_i6(image_dir, work, device="cuda", names=None) -> Path:
    """Stage 4: the Codex I6 candidate bank (``filseg.i6``). Returns the bank folder."""
    from filseg import i6
    out = Path(work) / i6.BANK_DIR
    with _stage("4", "I6 detector + refiner"):
        i6.run(image_dir, out, device=device, names=names)
        _free_gpu(device)
    return out


def stage_replay(work):
    """Replay mode, instead of stages 1-4: unpack ``replay/`` into ``<work>/replay`` (every file sha256-checked).
    Returns (map paths, the 4 detector roots, the I6 bank folder)."""
    from filseg import replay
    with _stage("replay", "unpack replay/ (stage 2-4 outputs of the submitted run)"):
        r = replay.unpack(work)
    return map_paths(r["maps"]), list(r["det_roots"]), r["i6"]


# ---- stages 5-10 (CPU) --------------------------------------------------------------------------------------------

def stage5a_evidence(maps, det_roots, i6_root, work, image_dir, nproc=NPROC) -> dict:
    """Stage 5a: tree candidates of the cnx map and their overlap with the I6 and detector candidates
    (``filseg.evidence``). Returns the number of tree candidates per image."""
    from filseg import evidence
    with _stage("5a", f"evidence ({min(nproc, NPROC)} processes)"):
        counts = evidence.build_evidence(maps["cnx"], evidence.test_sources(i6_root, det_roots), work, image_dir,
                                         min(nproc, NPROC))
        print(f"    {sum(counts.values())} tree candidates on {len(counts)} images", flush=True)
    return counts


def stage5b_rows(maps, work, image_dir, nproc=NPROC) -> Path:
    """Stage 5b: the candidate pool (tree nodes, merges, cnxL and R34 nodes) and its 86 scorer columns
    (``filseg.rows``). Returns the rows folder."""
    from filseg import rows
    out = Path(work) / rows.ROWS_DIR
    with _stage("5b", f"candidate rows ({min(nproc, NPROC)} processes)"):
        tot = rows.build_rows(maps, work, out, image_dir, min(nproc, NPROC))
        print(f"    {tot}", flush=True)
    return out


def _check_rows_dir(rows_dir, files) -> None:
    have = sorted(p.stem for p in Path(rows_dir).glob("*.npz"))
    want = sorted(f[:-5] for f in files)
    if have != want:
        raise ValueError(f"{rows_dir} holds {len(have)} row files but the maps list {len(want)} images; "
                         "use a separate work folder for each image set")


def stage6_score(rows_dir, work, files=None) -> dict:
    """Stage 6: q = 0.7 x mean of 10 LightGBM + 0.3 x mean of 10 ExtraTrees (``filseg.score``).
    Returns {'tv', 'sc', 'path', 'sha256'}; ``files`` (optional) checks that ``rows_dir`` holds exactly those images."""
    from filseg import score
    if files is not None:
        _check_rows_dir(rows_dir, files)
    with _stage("6", "scoring"):
        tv = score.load_views(rows_dir)
        sc = score.score(tv, score.load_lgbm(), score.load_et())
        path = Path(work) / "scores.npz"
        sha = score.save(path, tv, sc)
        print(f"    {len(sc['q'])} rows, q mean {sc['q'].mean():.4f}", flush=True)
    return dict(tv=tv, sc=sc, path=path, sha256=sha)


def stage7_select(scores: dict, work) -> dict:
    """Stage 7: per image, the conflict-free set maximising sum(s - 0.24) (``filseg.select``).
    Returns {'res', 'path'}."""
    from filseg import score, select
    with _stage("7", "selection"):
        tv, sc = scores["tv"], scores["sc"]
        res = select.select_all(tv[0], score.per_image(sc["q"], score.offsets(tv[0])))
        path = Path(work) / "selection.pkl"
        select.save(path, res, scores["sha256"])
        print(f"    {sum(len(v) for v in res['sel_sub'].values())} rows selected", flush=True)
    return dict(res=res, path=path)


def stage8_shift(rows_dir, scores: dict, selection: dict, maps) -> dict:
    """Stage 8: the 0.23 px shift of the family 0/5/6 masks and pixel claiming in descending s (``filseg.shift``).

    The loop of ``filseg.submit.make_submission``. Returns {'files', 'masks': {file name: [sorted flat pixel indices]},
    'counters'}."""
    from filseg import shift as S
    from filseg import submit
    with _stage("8", "shift and claiming"):
        map_dir = Path(maps["cnx"]).parent
        files = json.loads((map_dir / "testcnx.npy.json").read_text())["files"]
        sel, q = submit.load_selection(selection["path"]), submit.load_scores(scores["path"])
        mp = S.open_maps(map_dir, files)
        counters = S.new_counters()
        masks = {}
        for j, f in enumerate(files):
            stem = f[:-5]
            idx, qq = q[stem]
            masks[f] = submit.image_instances(Path(rows_dir) / f"{stem}.npz", np.asarray(sel["sel_rows"][stem], np.int64),
                                              idx, qq, S.own_maps(mp, j), S.SHIFT, counters)
        del mp
        print(f"    {sum(len(v) for v in masks.values())} instances | {counters}", flush=True)
    return dict(files=files, masks=masks, counters=counters)


def stage9_write(final: dict, out_csv) -> dict:
    """Stage 9: column-major COCO RLE CSV, read back and checked by the writer (``filseg.write``).
    Returns the writer's result plus 'sha256' and 'shift_counters'."""
    from filseg import write as WR
    with _stage("9", "CSV"):
        res, _ = WR.write_csv(out_csv, final["masks"], final["files"], overwrite=True)
        res["shift_counters"] = final["counters"]
        res["sha256"] = WR.sha256_file(out_csv)
        print(f"    {out_csv}: {res['instance_rows']} instances, sha256 {res['sha256']}", flush=True)
    return res


def stage10_validate(out_csv, final: dict) -> bool:
    """Stage 10: the submission checks of ``filseg.validate`` (header, ids, disjointness, RLE round trip, orientation
    with an independent decoder, decoded masks == intended masks). Prints one line per check; returns all passed."""
    from filseg import validate
    with _stage("10", "validation"):
        checks = validate.validate_csv(out_csv, final["files"], expected=final["masks"])
        print(validate.format_checks(checks), flush=True)
        ok = validate.all_pass(checks)
        print("    all checks passed" if ok else "    SOME CHECKS FAILED", flush=True)
    return ok


# ---- comparison and evaluation ------------------------------------------------------------------------------------

def compare(reference, candidate) -> None:
    """sha256 of both files and the instance-by-instance comparison of ``tools/compare_csv.py``."""
    from filseg.write import sha256_file
    if str(paths.REPO) not in sys.path:
        sys.path.insert(0, str(paths.REPO))
    from tools import compare_csv
    a, b = sha256_file(reference), sha256_file(candidate)
    print(f"reference {a}\ncandidate {b}\n" + ("byte-identical" if a == b else "files differ"), flush=True)
    compare_csv.main(str(reference), str(candidate))


def dev_fold_images(fold: int, n: int | None = None, split=None) -> list:
    """The first ``n`` file names (sorted) of development fold ``fold`` (0-3) of ``configs/split-v2.json``."""
    split = json.loads(Path(split or paths.CONFIGS / "split-v2.json").read_text(encoding="utf-8"))
    if fold == split["reserved_holdout_fold"] or fold not in split["development_folds"]:
        raise ValueError(f"fold {fold} is not a development fold {split['development_folds']}")
    return sorted(o["file_name"] for o in split["observations"] if o["fold"] == fold)[:n]


def csv_predictions(csv_path, files) -> dict:
    """{file name: [RLE dicts]} of a submission CSV, ``[]`` for an image without rows."""
    from filseg import validate
    by_stem = {Path(f).stem: f for f in files}
    out = {f: [] for f in files}
    for fid, counts in validate.read_raw(csv_path)[1]:
        out[by_stem[fid.rsplit("_", 1)[0]]].append({"size": [H, W], "counts": counts})
    return out


def evaluate(csv_path, files, annotations) -> dict:
    """Pooled PQ (``filseg.metrics``, the organisers' rule) of a CSV over the annotator records of ``files``."""
    from filseg import metrics
    ann = json.loads(Path(annotations).read_text(encoding="utf-8-sig"))
    r = metrics.score_records(ann, csv_predictions(csv_path, files), set(files))
    print(f"PQ {r['pq']:.4f} (SQ {r['sq']:.4f}, RQ {r['rq']:.4f}) | TP {r['tp']} FP {r['fp']} FN {r['fn']} | "
          f"{len(r['records'])} annotator records on {len(files)} images", flush=True)
    return r


# ---- the whole chain ----------------------------------------------------------------------------------------------

def run(mode="full", device="cuda", work=None, data=None, out_csv=None, image_dir=None, names=None,
        nproc=NPROC) -> dict:
    """Run the pipeline and write the submission CSV.

    mode: 'full' (stages 1-10 from the images) or 'replay' (unpack ``replay/``, then stages 5-10);
    device: 'cuda' or 'cpu' for stages 2-4 (stages 5-10 always run on the CPU); 'cpu' hides the GPUs first;
    work: work folder (default ``filseg.paths.WORK``); data: the competition folder (default ``filseg.paths.DATA``);
    out_csv: default ``<work>/submission.csv``; image_dir: default ``<data>/test/test_images``;
    names: full mode only, a subset of the image file names (for example dev-fold training images);
    nproc: worker processes of stage 5 (at most 4).
    Returns {'csv', 'sha256', 'instances', 'checks_passed', 'matches_submitted', 'times'}.
    """
    if mode not in ("full", "replay"):
        raise ValueError("mode must be 'full' or 'replay'")
    if device == "cpu":
        cpu_only()
    work = Path(work or paths.WORK)
    work.mkdir(parents=True, exist_ok=True)
    image_dir = Path(image_dir) if image_dir else test_images(data)
    out_csv = Path(out_csv or work / "submission.csv")
    TIMES.clear()
    t0 = time.time()
    if mode == "full":
        recs, images = stage1_images(image_dir, work, names)
        maps = stage2_dense(recs, images, work, device)
        det_roots = stage3_det(recs, images, work, device)
        i6_root = stage4_i6(image_dir, work, device, [r["file_name"] for r in recs])
        del images
    else:
        if names is not None:
            raise ValueError("replay mode covers the 180 test images only")
        maps, det_roots, i6_root = stage_replay(work)
    stage5a_evidence(maps, det_roots, i6_root, work, image_dir, nproc)
    rows_dir = stage5b_rows(maps, work, image_dir, nproc)
    scores = stage6_score(rows_dir, work, map_files(maps))
    selection = stage7_select(scores, work)
    final = stage8_shift(rows_dir, scores, selection, maps)
    res = stage9_write(final, out_csv)
    ok = stage10_validate(out_csv, final)
    TIMES["total"] = time.time() - t0
    same = res["sha256"] == SUBMITTED_SHA256
    print("\nstage times (s): " + " | ".join(f"{k} {v:.0f}" for k, v in TIMES.items()), flush=True)
    if mode == "replay":
        print("MATCHES the submitted s1-S-t024.csv" if same else "DIFFERS from the submitted s1-S-t024.csv", flush=True)
    return dict(csv=out_csv, sha256=res["sha256"], instances=res["instance_rows"], checks_passed=ok,
                matches_submitted=same, times=dict(TIMES))
