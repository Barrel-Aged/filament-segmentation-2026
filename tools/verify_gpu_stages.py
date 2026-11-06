"""Rerun stages 1-3 (test images, dense maps, detector roots) and compare them with the original run's files.

usage (from the repository root; one stage per call, each appends to ``<work>/verify_gpu_stages.log``):
  python -m tools.verify_gpu_stages images  --original <claude tree> [--work DIR] [--data DIR]
  python -m tools.verify_gpu_stages dense   --original <claude tree> [--sources cnx,cnxL,r34] [--device cuda]
  python -m tools.verify_gpu_stages det     --original <claude tree> [--folds 0,1,2,3] [--device cuda]
  python -m tools.verify_gpu_stages compare-det --original <claude tree> [--folds 0,1,2,3]   (no inference)
  python -m tools.verify_gpu_stages chain   --original <claude tree> --i6-root <I6 test bank> [--nproc 4]
  python -m tools.verify_gpu_stages compare-csv --original <claude tree> --csv <submission.csv>

--original is the original Claude tree (the folder holding ``work/cache``, ``work/maps/final`` and ``work/det``); its
files are only read. ``--work`` (default ``filseg.paths.WORK``) receives ``cache/``, ``maps/``, ``det/`` and ``chain/``.
``--data`` overrides ``FILSEG_DATA`` (the competition folder with ``test/test_images``).

Checks:
- images: the rebuilt ``test_img.npy`` equals the original ``work/cache/test_img.npy`` array (per image, exact) and the
  test disk statistics equal the ``test`` records of the original ``work/cache/meta.json``.
- dense: each source's map vs ``work/maps/final/test<source>.npy``: the file order, the fraction of exactly equal
  values, and the histogram of |difference| in uint8 levels, per channel (0 mask, 1 spine).
- det: each fold model's root vs ``work/det/det-f<k>/test``: per image record, byte equality of the json, then the
  rows (RLE, score, features, area, rho) and the selected set.
- chain (CPU; run it with the environment that runs stages 5-10): stages 5a/5b (``filseg.evidence``, ``filseg.rows``)
  on the rebuilt maps and detector roots, then ``tools.verify_cpu_chain`` (scores, selection, shifted CSV), then the
  CSV comparison below.
- compare-csv: a CSV against the submitted ``work/ideas15/hyg/csv/s1-S-t024.csv``: sha256, and per image the instance
  count, identical instances and one-to-one IoU > 0.5 matches.
Every stage is timed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SUBMITTED_CSV = 'work/ideas15/hyg/csv/s1-S-t024.csv'
SUBMITTED_SHA = '18bf87d67f7c663c4a0a82d8f17fe77cfb378ae81c50dbe8c71a99b48ea74cd4'
ORIG_TEST_IMG_SHA = 'd71edb68c67fd23930ab95ee173b90a03ec4848041a91007eb3fa05383a4636d'   # PATH_MAP.md section 1

_LOG = None


def log(msg=''):
    print(msg, flush=True)
    if _LOG is not None:
        with open(_LOG, 'a', encoding='utf-8') as fh:
            fh.write(msg + '\n')


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------------- stage 1

def stage_images(a):
    from filseg import data
    t0 = time.time()
    path, recs = data.write_test_cache(a.work / 'cache', a.image_dir)
    log(f'images: wrote {len(recs)} images to {path} in {time.time() - t0:.1f}s')
    orig = np.load(a.original / 'work/cache/test_img.npy', mmap_mode='r')
    new = np.load(path, mmap_mode='r')
    ometa = json.loads((a.original / 'work/cache/meta.json').read_text(encoding='utf-8'))['test']
    same = sum(bool(np.array_equal(orig[i], new[i])) for i in range(len(new))) if orig.shape == new.shape else 0
    names_ok = [r['file_name'] for r in recs] == [r['file_name'] for r in ometa]
    stats_ok = sum(all(r[k] == o[k] for k in ('disk_mean', 'disk_std', 'disk_thr')) for r, o in zip(recs, ometa))
    log(f'  shape {new.shape} vs original {orig.shape}; images equal {same}/{len(new)}; file names in order: {names_ok}; '
        f'disk stats equal {stats_ok}/{len(recs)}')
    t1 = time.time()
    sha = sha256_file(path)
    log(f'  rebuilt test_img.npy sha256 {sha} ({"equal to" if sha == ORIG_TEST_IMG_SHA else "differs from"} the '
        f'original file {ORIG_TEST_IMG_SHA[:16]}); hashing {time.time() - t1:.0f}s')
    log(f'images: {"EXACT" if same == len(new) and names_ok and stats_ok == len(recs) else "NOT EXACT"} '
        f'({time.time() - t0:.0f}s)')


# ------------------------------------------------------------------------------------------------------- stage 2

def compare_maps(new_path, orig_path):
    new = np.load(new_path, mmap_mode='r')
    orig = np.load(orig_path, mmap_mode='r')
    nf = json.loads(Path(str(new_path) + '.json').read_text(encoding='utf-8'))['files']
    of = json.loads(Path(str(orig_path) + '.json').read_text(encoding='utf-8'))['files']
    out = {'files_equal': nf == of, 'shape': list(new.shape), 'orig_shape': list(orig.shape)}
    if new.shape != orig.shape:
        return out
    hist = np.zeros((2, 256), np.int64)
    img_diff = [0, 0]
    for i in range(new.shape[0]):
        d = np.abs(np.asarray(new[i]).astype(np.int16) - np.asarray(orig[i]).astype(np.int16))
        for c in range(2):
            h = np.bincount(d[c].ravel(), minlength=256)
            hist[c] += h
            img_diff[c] += int(h[1:].sum() > 0)
    out['hist'] = hist
    out['images_with_diff'] = img_diff
    return out


def _fmt_hist(h):
    tot = h.sum()
    nz = np.flatnonzero(h)
    parts = [f'{k}:{h[k]} ({h[k] / tot:.3e})' for k in nz[:12]]
    mean = float((np.arange(256) * h).sum() / tot)
    return (f'exact {h[0] / tot:.6f} | max {int(nz.max())} | mean |d| {mean:.5f} | '
            f'>=2 levels {h[2:].sum() / tot:.3e} | hist ' + ', '.join(parts))


def stage_dense(a):
    from filseg import data, dense
    import torch
    recs, images = data.open_test_cache(a.work / 'cache')
    if a.limit:
        recs = recs[:a.limit]
    log(f'dense: device {a.device} ({torch.cuda.get_device_name(0) if a.device.startswith("cuda") else platform.processor()}), '
        f'torch {torch.__version__}, {len(recs)} images')
    for src in a.sources.split(','):
        t0 = time.time()
        side = dense.infer_source(src, images, recs, a.work / 'maps', device=a.device, log=log)
        log(f'dense {src}: {len(recs)} images in {time.time() - t0:.0f}s (inference loop {side["sec"]:.0f}s)')
        t1 = time.time()
        newp = a.work / 'maps' / f'{dense.SOURCES[src]}.npy'
        origp = a.original / 'work/maps/final' / f'{dense.SOURCES[src]}.npy'
        if a.limit:
            log('  (--limit: compared with the first images of the original)')
            new = np.load(newp, mmap_mode='r'); orig = np.load(origp, mmap_mode='r')
            for c in range(2):
                d = np.abs(new[:, c].astype(np.int16) - orig[:len(new), c].astype(np.int16))
                log(f'  channel {c}: ' + _fmt_hist(np.bincount(d.ravel(), minlength=256)))
            continue
        r = compare_maps(newp, origp)
        log(f'  vs original {origp.name}: file order equal {r["files_equal"]}, shape {r["shape"]} vs {r["orig_shape"]}')
        for c, name in enumerate(('mask', 'spine')):
            log(f'  channel {c} ({name}): images with any difference {r["images_with_diff"][c]}/{r["shape"][0]} | '
                + _fmt_hist(r['hist'][c]))
        log(f'  comparison {time.time() - t1:.0f}s')


# ------------------------------------------------------------------------------------------------------- stage 3

def compare_root(new_dir, orig_dir):
    """Per record comparison of two detector roots."""
    from pycocotools import mask as mu
    nm = json.loads((new_dir / 'manifest.json').read_text(encoding='utf-8'))['records']
    om = json.loads((orig_dir / 'manifest.json').read_text(encoding='utf-8'))['records']
    res = {'manifest_records_equal': nm == om, 'files': len(om), 'bytes_equal': 0, 'rows_equal_count': 0,
           'rows_identical': 0, 'sel_identical': 0, 'rows_new': 0, 'rows_orig': 0, 'rle_equal_rows': 0,
           'score_maxdiff': 0.0, 'feat_maxdiff': 0.0, 'rho_maxdiff': 0.0, 'area_diff_rows': 0,
           'iou_matched': 0, 'iou_min_matched': 1.0, 'examples': []}
    for rec in om:
        a_b = (new_dir / rec['file']).read_bytes(); b_b = (orig_dir / rec['file']).read_bytes()
        if a_b == b_b:
            res['bytes_equal'] += 1
        A = json.loads(a_b); B = json.loads(b_b)
        ra, rb = A['views']['0.5'], B['views']['0.5']
        res['rows_new'] += len(ra); res['rows_orig'] += len(rb)
        sa, sb = A['predictions']['set_selector_pq'], B['predictions']['set_selector_pq']
        res['sel_identical'] += int(sa == sb)
        res['rows_identical'] += int(ra == rb)
        if len(ra) == len(rb):
            res['rows_equal_count'] += 1
            for x, y in zip(ra, rb):
                res['rle_equal_rows'] += int(x['rle'] == y['rle'])
                res['score_maxdiff'] = max(res['score_maxdiff'], abs(x['score'] - y['score']))
                res['feat_maxdiff'] = max(res['feat_maxdiff'], max(abs(u - v) for u, v in zip(x['features'], y['features'])))
                res['rho_maxdiff'] = max(res['rho_maxdiff'], abs(x['rho'] - y['rho']))
                res['area_diff_rows'] += int(x['area'] != y['area'])
        elif len(res['examples']) < 5:
            res['examples'].append(f'{rec["file"]}: {len(ra)} rows vs {len(rb)}')
        if ra and rb:  # one-to-one IoU > 0.5 matches between the row masks
            ea = [{'size': r['rle']['size'], 'counts': r['rle']['counts'].encode()} for r in ra]
            eb = [{'size': r['rle']['size'], 'counts': r['rle']['counts'].encode()} for r in rb]
            iou = np.asarray(mu.iou(ea, eb, [0] * len(eb))).reshape(len(ea), len(eb))
            used = set()
            for i in np.argsort(-iou.max(1)):
                for j in np.argsort(-iou[i]):
                    if iou[i, j] <= 0.5:
                        break
                    if j not in used:
                        used.add(j); res['iou_matched'] += 1
                        res['iou_min_matched'] = min(res['iou_min_matched'], float(iou[i, j]))
                        res.setdefault('pair_iou', []).append(float(iou[i, j]))
                        res.setdefault('pair_dscore', []).append(abs(ra[i]['score'] - rb[j]['score']))
                        break
    return res


def _pair_summary(r):
    """Distribution of IoU and |score difference| over the one-to-one matched rows."""
    iou = np.asarray(r.get('pair_iou', [1.0])); ds = np.asarray(r.get('pair_dscore', [0.0]))
    q = lambda v, p: float(np.quantile(v, p))  # noqa: E731
    return (f'matched pairs: IoU == 1 {np.mean(iou >= 1 - 1e-12):.3f}, IoU median {q(iou, .5):.4f} p1 {q(iou, .01):.4f}; '
            f'|d score| median {q(ds, .5):.2e} p99 {q(ds, .99):.2e} max {ds.max():.2e}')


def stage_det(a):
    from filseg import data, det
    import torch
    recs, images = data.open_test_cache(a.work / 'cache')
    if a.limit:
        recs = recs[:a.limit]
    log(f'det: device {a.device} ({torch.cuda.get_device_name(0) if a.device.startswith("cuda") else platform.processor()}), '
        f'torch {torch.__version__}, {len(recs)} images')
    for k in [int(x) for x in a.folds.split(',')]:
        name = f'det-f{k}'
        out = a.work / 'det' / name / 'test'
        t0 = time.time()
        man = det.infer(name, images, recs, out, device=a.device, log=log)
        log(f'det {name}: {len(recs)} images in {time.time() - t0:.0f}s (loop {man["sec"]:.0f}s)')
        if a.limit:
            continue
        t1 = time.time()
        r = compare_root(out, a.original / 'work/det' / name / 'test')
        log(f'  vs original: manifest records equal {r["manifest_records_equal"]}; record json bytes equal '
            f'{r["bytes_equal"]}/{r["files"]}; rows identical {r["rows_identical"]}/{r["files"]}; selected sets '
            f'identical {r["sel_identical"]}/{r["files"]}')
        log(f'  rows {r["rows_new"]} vs {r["rows_orig"]}; images with equal row count {r["rows_equal_count"]}/{r["files"]}; '
            f'in those: RLE equal {r["rle_equal_rows"]}, area differs {r["area_diff_rows"]}, max |d score| '
            f'{r["score_maxdiff"]:.3e}, max |d feature| {r["feat_maxdiff"]:.3e}, max |d rho| {r["rho_maxdiff"]:.3e}')
        log(f'  one-to-one IoU>0.5 row matches {r["iou_matched"]} (min IoU {r["iou_min_matched"]:.4f}); '
            f'examples {r["examples"]} | comparison {time.time() - t1:.0f}s')
        log('  ' + _pair_summary(r) + ' (rows are in score order, so the positional maxima above mix rows that swapped places)')


def stage_compare_det(a):
    """Compare already written roots (no inference)."""
    for k in [int(x) for x in a.folds.split(',')]:
        name = f'det-f{k}'
        r = compare_root(a.work / 'det' / name / 'test', a.original / 'work/det' / name / 'test')
        log(f'det {name} vs original: json bytes equal {r["bytes_equal"]}/{r["files"]}; rows {r["rows_new"]} vs '
            f'{r["rows_orig"]}; equal row count {r["rows_equal_count"]}/{r["files"]}; selected sets identical '
            f'{r["sel_identical"]}/{r["files"]}; IoU>0.5 one-to-one matches {r["iou_matched"]} '
            f'(unmatched {r["rows_new"] - r["iou_matched"]} new, {r["rows_orig"] - r["iou_matched"]} original)')
        log('  ' + _pair_summary(r))


# ------------------------------------------------------------------------------------------------------- stage 4

def compare_csv(a, csv_path):
    from filseg import validate
    sub = a.original / SUBMITTED_CSV
    files = json.loads((a.work / 'maps' / 'testcnx.npy.json').read_text(encoding='utf-8'))['files'] \
        if (a.work / 'maps' / 'testcnx.npy.json').exists() else \
        json.loads((a.original / 'work/maps/final/testcnx.npy.json').read_text(encoding='utf-8'))['files']
    sha = sha256_file(csv_path)
    log(f'csv: {csv_path} sha256 {sha} ({"EQUAL to" if sha == SUBMITTED_SHA else "differs from"} the submitted '
        f'{SUBMITTED_SHA[:16]})')
    A = validate.decode_csv(csv_path, files); B = validate.decode_csv(sub, files)
    na = sum(len(v) for v in A.values()); nb = sum(len(v) for v in B.values())
    same_count = identical_imgs = ident = matched = 0
    ious = []
    worst = []
    for f in files:
        ia = [q for _, q in A[f]]; ib = [q for _, q in B[f]]
        same_count += int(len(ia) == len(ib))
        sb = {q.tobytes(): j for j, q in enumerate(ib)}
        n_ident = sum(q.tobytes() in sb for q in ia)
        ident += n_ident
        identical_imgs += int(n_ident == len(ia) == len(ib))
        # instances are disjoint inside each file, so an IoU > 0.5 match is unique on both sides
        m = 0
        for q in ia:
            for r in ib:
                inter = np.intersect1d(q, r, assume_unique=True).size
                if inter and inter / (q.size + r.size - inter) > 0.5:
                    m += 1
                    ious.append(inter / (q.size + r.size - inter))
                    break
        matched += m
        if not (n_ident == len(ia) == len(ib)):
            worst.append((f, len(ia), len(ib), n_ident, m))
    log(f'  instances {na} vs submitted {nb}; images with equal count {same_count}/{len(files)}; images identical '
        f'{identical_imgs}/{len(files)}; identical instances {ident}; IoU>0.5 matches {matched} '
        f'(unmatched: {na - matched} new, {nb - matched} submitted)')
    iv = np.asarray(ious); ch = iv[iv < 1]
    if ch.size:
        log(f'  matched instances that changed: {ch.size}, IoU with the submitted instance mean {ch.mean():.4f}, '
            f'median {np.median(ch):.4f}, min {ch.min():.4f}')
    for f, x, y, n_ident, m in worst[:20]:
        log(f'    {f}: {x} vs {y} instances, identical {n_ident}, IoU>0.5 matched {m}')


def stage_chain(a):
    from filseg import evidence, rows
    t0 = time.time()
    chain = a.work / 'chain'
    det_roots = [a.work / 'det' / f'det-f{k}' / 'test' for k in range(4)]
    maps = {s: a.work / 'maps' / f'test{s}.npy' for s in ('cnx', 'cnxL', 'r34')}
    counts = evidence.build_evidence(maps['cnx'], evidence.test_sources(a.i6_root, det_roots), work_root=chain,
                                     image_dir=a.image_dir, nproc=a.nproc)
    log(f'chain 5a evidence: {len(counts)} images, {sum(counts.values())} tree candidates, {time.time() - t0:.0f}s')
    t1 = time.time()
    tot = rows.build_rows(maps, evidence_root=chain, out_dir=chain / 'rows', image_dir=a.image_dir, nproc=a.nproc)
    log(f'chain 5b rows: {tot}, {time.time() - t1:.0f}s')
    t2 = time.time()
    cmd = [sys.executable, '-m', 'tools.verify_cpu_chain', '--rows', str(chain / 'rows'), '--maps', str(a.work / 'maps'),
           '--out', str(chain / 'cpu_chain')]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1]))
    for line in (p.stdout + p.stderr).splitlines():
        log('  verify_cpu_chain | ' + line)
    log(f'chain 6-10: exit {p.returncode}, {time.time() - t2:.0f}s')
    compare_csv(a, chain / 'cpu_chain' / 'submission.csv')
    log(f'chain total {time.time() - t0:.0f}s')


# ------------------------------------------------------------------------------------------------------- driver

def main():
    global _LOG
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['images', 'dense', 'det', 'compare-det', 'chain', 'compare-csv'])
    ap.add_argument('--original', type=Path, required=True)
    ap.add_argument('--work', type=Path, default=None)
    ap.add_argument('--data', type=Path, default=None)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--sources', default='cnx,cnxL,r34')
    ap.add_argument('--folds', default='0,1,2,3')
    ap.add_argument('--limit', type=int, default=0, help='first N images only (smoke test; no full comparison)')
    ap.add_argument('--i6-root', type=Path, default=None)
    ap.add_argument('--nproc', type=int, default=4)
    ap.add_argument('--csv', type=Path, default=None)
    a = ap.parse_args()
    if a.data:
        os.environ['FILSEG_DATA'] = str(a.data)
    if a.work:
        os.environ['FILSEG_WORK'] = str(a.work)
    from filseg import paths
    a.work = Path(a.work or paths.WORK)
    a.work.mkdir(parents=True, exist_ok=True)
    a.image_dir = (a.data / 'test' / 'test_images') if a.data else paths.TEST_IMAGES
    _LOG = a.work / 'verify_gpu_stages.log'
    log(f'=== verify_gpu_stages {a.stage} {time.strftime("%Y-%m-%d %H:%M:%S %z")} | python {platform.python_version()} '
        f'| numpy {np.__version__} | CUDA_VISIBLE_DEVICES {os.environ.get("CUDA_VISIBLE_DEVICES")!r} | args '
        f'{ {k: str(v) for k, v in vars(a).items()} }')
    t0 = time.time()
    if a.stage == 'images':
        stage_images(a)
    elif a.stage == 'dense':
        stage_dense(a)
    elif a.stage == 'det':
        stage_det(a)
    elif a.stage == 'compare-det':
        stage_compare_det(a)
    elif a.stage == 'chain':
        stage_chain(a)
    else:
        compare_csv(a, a.csv)
    log(f'=== {a.stage} done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
