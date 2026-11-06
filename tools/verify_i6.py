"""Run the release I6 inference (stage 4) on the test images and compare its bank with the stored Codex bank.

usage (from the repository root):
  python -m tools.verify_i6 --stored <Codex outputs/improvement-i6/candidates/test> --original <claude tree>
                            [--image-dir DIR] [--work DIR] [--device cuda|cpu] [--limit N] [--nproc 4] [--skip-infer]

--stored is the bank of the original run (180 ``NNNN.json`` + ``manifest.json`` + ``repeat.json``); --original is the
original Claude tree (the folder holding ``work/ext``, ``work/feat`` and ``work/ideas8/bundle-test/rows``). Both are
only read. Outputs go to --work (default ``filseg.paths.WORK``): the bank in ``i6-candidates/test``, the stage-5 I6
evidence of both banks in ``i6-evidence/{stored,release}``, ``verify_i6_images.tsv`` and ``verify_i6.log``.

Checks, in order:
1. the stored bank: every record file against its sha256 in the stored ``manifest.json``, the manifest and directory
   digests recorded in PATH_MAP; the original row files' directory digest;
2. inference with ``filseg.i6.run``, timed;
3. bank vs bank, per image: exact record equality apart from ``predictions`` (the release writes ``{}``), and the
   sha256 of the release record re-saved with the stored ``predictions`` against the stored record sha256. If not
   exact: candidate counts, one-to-one matching of the candidate masks (maximum total IoU), matched IoU and score /
   feature differences, in ``verify_i6_images.tsv``;
4. stage-5 I6 columns of the 47,101 rows (columns 28-44 and 62 of every ``F`` variant: I6 X 6, I6 rich 11, d_i6),
   rebuilt with ``filseg.evidence`` / ``filseg.rows`` from the original row pixel sets for both banks. The stored bank
   must reproduce the original rows and ext/rich files exactly (this checks the harness); for the release bank the
   number of rows whose I6 columns change is reported.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from filseg import evidence, paths, rows

STORED_MANIFEST = '9cba127a222aed8129160bb499681630c3aaa1e37454ff7243f8b1f693967414'
STORED_DIR = '3223c8fb2358eb3dd3d8682e4ecdfde1c696a06f2c4a9d4550cfb04e5cf95e75'   # 182 JSON incl. manifest, repeat
ROWS_DIR = 'fcb0ca6dd01e7a4a3cb7144081399bdeb12da277872f0d953825ade07d566539'
ORIGINAL_ROWS = 'work/ideas8/bundle-test/rows'
I6COLS = list(range(28, 45)) + [62]
COLNAMES = list(evidence.XFEAT) + list(evidence.RFEAT) + ['d_i6']

_LOG = []


def log(*parts):
    line = ' '.join(str(p) for p in parts)
    print(line, flush=True)
    _LOG.append(line)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dir_digest(d, pattern):
    names = sorted(p.name for p in Path(d).glob(pattern))
    return hashlib.sha256('\n'.join(f'{n} {sha256(Path(d) / n)}' for n in names).encode()).hexdigest(), len(names)


def read_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def json_bytes(value):
    """The bytes ``filseg.i6.atomic_json`` writes."""
    return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode('utf-8')


# ------------------------------------------------------------------------------------------------- bank vs bank

def compare_record(r0, r1):
    """Per-image comparison of a stored record r0 and a release record r1."""
    from pycocotools import mask as mu
    from scipy.optimize import linear_sum_assignment
    (t0, c0), = r0['views'].items()
    (t1, c1), = r1['views'].items()
    out = dict(file_name=r0['file_name'], exact=({k: v for k, v in r0.items() if k != 'predictions'} ==
                                                {k: v for k, v in r1.items() if k != 'predictions'}),
               input_equal=r0['input_sha256'] == r1['input_sha256'], source_equal=r0['source_sha256'] == r1['source_sha256'],
               raw_equal=r0['raw_predictions_sha256'] == r1['raw_predictions_sha256'], thr_equal=t0 == t1,
               n_stored=len(c0), n_release=len(c1), identical=0, matched50=0, iou_mean=np.nan, iou_min=np.nan,
               score_maxdiff=np.nan, feat_maxdiff=np.nan, unmatched_stored=len(c0), unmatched_release=len(c1))
    out['identical'] = sum(a == b for a, b in zip(c0, c1))
    if c0 and c1:
        e0 = [{'size': c['rle']['size'], 'counts': c['rle']['counts'].encode()} for c in c0]
        e1 = [{'size': c['rle']['size'], 'counts': c['rle']['counts'].encode()} for c in c1]
        iou = np.asarray(mu.iou(e0, e1, [0] * len(e1)), np.float64).reshape(len(c0), len(c1))
        ra, rb = linear_sum_assignment(-iou)
        good = iou[ra, rb] > 0.5
        ra, rb = ra[good], rb[good]
        out['matched50'] = int(good.sum())
        out['unmatched_stored'] = len(c0) - out['matched50']
        out['unmatched_release'] = len(c1) - out['matched50']
        if len(ra):
            out['iou_mean'] = float(iou[ra, rb].mean()); out['iou_min'] = float(iou[ra, rb].min())
            out['score_maxdiff'] = float(max(abs(c0[i]['score'] - c1[j]['score']) for i, j in zip(ra, rb)))
            out['feat_maxdiff'] = float(max(np.abs(np.subtract(c0[i]['features'], c1[j]['features'])).max()
                                            for i, j in zip(ra, rb)))
    return out


# --------------------------------------------------------------------------------------- stage-5 I6 columns

def _ext_files(mine, record_path, F28, nb, Ca, la, out_root, stem):
    """Write the I6 ext / rich files of the tree rows for one record; return (ext arrays, rich X, I6 cols [N, 18])."""
    e = evidence.ext_features(mine, record_path, True)
    rich = evidence.rich_features(F28[:nb], e['ov'], e['E'])
    ext_path = out_root / 'ext/testcnx-i6g' / f'{stem}.npz'
    ext_path.parent.mkdir(parents=True, exist_ok=True)
    (out_root / 'feat/testcnx-i6g-rich').mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ext_path, **e)
    np.savez_compressed(out_root / 'feat/testcnx-i6g-rich' / f'{stem}.npz', X=rich)
    dI = rows.base_delta(ext_path)
    xi, ri, di = rows.ext_block(Ca, la, ext_path, F28)          # same call as rows.gen
    cols = np.concatenate([np.concatenate([e['X'], rich, dI[:, None]], 1),
                           np.concatenate([xi[nb:], ri[nb:], di[nb:, None]], 1)]).astype(np.float32)
    return e, rich, cols


def _equal(a, b):
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.dtype == object:
        return a.tolist() == b.tolist()
    return bool(np.array_equal(a, b))


def evidence_job(job):
    stem, row_file, banks, orig_ext, orig_rich, work = job
    z = np.load(row_file, allow_pickle=True)
    F = z['F']; nb = int(z['nb'])
    sets = rows.unpack_sets(z['set_flat'], z['set_len'])
    mine = []
    for Q in sets[:nb]:
        r = rows.rle_of_set(Q)
        mine.append({'size': r['size'], 'counts': r['counts'].encode()})
    F28 = F[0][:, :28]
    Ca, la = rows.csr_of(sets)
    orig = F[:, :, I6COLS]                                      # [4, N, 18]
    res = dict(stem=stem, n=len(sets), nb=nb, variants_same=bool(all(np.array_equal(orig[0], orig[k]) for k in range(4))))
    zo, ro = np.load(orig_ext, allow_pickle=True), np.load(orig_rich)
    for tag, rec in banks.items():
        e, rich, cols = _ext_files(mine, rec, F28, nb, Ca, la, Path(work) / 'i6-evidence' / tag, stem)
        d = np.abs(cols.astype(np.float64) - orig[0].astype(np.float64))
        res[tag] = dict(
            ext_equal=all(_equal(e[k], zo[k]) for k in ('X', 'E', 'ov', 'ext_rle', 'ext_sel')),
            rich_equal=_equal(rich, ro['X']),
            rows_changed=int((d > 0).any(1).sum()), tree_changed=int((d[:nb] > 0).any(1).sum()),
            new_changed=int((d[nb:] > 0).any(1).sum()), rows_changed_1e3=int((d > 1e-3).any(1).sum()),
            col_changed=(d > 0).sum(0).tolist(), col_maxdiff=d.max(0).tolist() if len(d) else [0.0] * len(I6COLS),
            n_ext=len(e['E']), n_sel=len(e['ext_sel']))
    return res


# ------------------------------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--stored', required=True)
    ap.add_argument('--original', required=True)
    ap.add_argument('--image-dir', default=str(paths.TEST_IMAGES))
    ap.add_argument('--work', default=str(paths.WORK))
    ap.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    ap.add_argument('--limit', type=int)
    ap.add_argument('--nproc', type=int, default=4)
    ap.add_argument('--skip-infer', action='store_true')
    a = ap.parse_args()
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    stored = Path(a.stored)
    bank = work / 'i6-candidates' / 'test'
    log('verify_i6', time.strftime('%Y-%m-%d %H:%M:%S %z'), '| python', platform.python_version(), '| numpy',
        np.__version__, '| device', a.device, '| CUDA_VISIBLE_DEVICES', repr(os.environ.get('CUDA_VISIBLE_DEVICES')))
    log('inputs: stored', stored, '| original', a.original, '| images', a.image_dir, '| work', work)

    log('1. stored bank and original rows')
    man = read_json(stored / 'manifest.json')
    got = sha256(stored / 'manifest.json')
    log(f'  stored manifest.json sha256 {got[:16]} recorded {"equal" if got == STORED_MANIFEST else "DIFFERENT"}')
    got, n = dir_digest(stored, '*.json')
    log(f'  stored directory digest ({n} JSON) {got[:16]} recorded {"equal" if got == STORED_DIR else "DIFFERENT"}')
    eq = sum(sha256(stored / r['file']) == r['sha256'] for r in man['records'])
    log(f'  stored record files vs manifest sha256: {eq}/{len(man["records"])} equal; repeat.json all_image_exact_repeat '
        f'{read_json(stored / "repeat.json")["all_image_exact_repeat"]}')
    rows_dir = Path(a.original) / ORIGINAL_ROWS
    got, n = dir_digest(rows_dir, '*.npz')
    log(f'  original rows directory digest ({n} files) {got[:16]} recorded {"equal" if got == ROWS_DIR else "DIFFERENT"}')

    names = man['names'][:a.limit]
    if not a.skip_infer:
        log('2. inference')
        from filseg import i6
        t0 = time.time()
        res = i6.run(a.image_dir, bank, a.device, names=names, log=lambda s: log('  ' + s))
        log(f'  {len(names)} images, wall {time.time() - t0:.1f} s (loop {res["seconds"]:.1f} s, model load '
            f'{res["model_load_seconds"]:.1f} s, peak {res["peak_mib"]} MiB) on {res["runtime"].get("gpu", "cpu")}; '
            f'stored run {man["seconds"]:.1f} s on {man["runtime"]["gpu"]["name"]}')
    new_man = read_json(bank / 'manifest.json')
    log(f'  release bank manifest: {len(new_man["names"])} images, loop {new_man["seconds"]:.1f} s, process '
        f'{new_man["process_seconds"]:.1f} s, peak {new_man["peak_mib"]} MiB, {new_man["runtime"]}')
    if new_man['names'] != sorted(new_man['names']) or new_man['names'] != names:
        log('  WARNING: release names differ from the stored names')

    log('3. release bank vs stored bank')
    stored_files = {r['file_name']: r for r in man['records']}
    new_files = {r['file_name']: r for r in new_man['records']}
    table = []
    resave_equal = 0
    for name in names:
        r0 = read_json(stored / stored_files[name]['file'])
        r1 = read_json(bank / new_files[name]['file'])
        resave_equal += hashlib.sha256(json_bytes({**r1, 'predictions': r0['predictions']})).hexdigest() == \
            stored_files[name]['sha256']
        table.append(compare_record(r0, r1))
    cols = list(table[0])
    with open(work / 'verify_i6_images.tsv', 'w', encoding='utf-8') as fh:
        fh.write('\t'.join(cols) + '\n')
        for t in table:
            fh.write('\t'.join(f'{t[c]:.6g}' if isinstance(t[c], float) else str(t[c]) for c in cols) + '\n')
    n = len(table)
    s = lambda k: sum(t[k] for t in table)
    log(f'  records exactly equal (all keys but predictions): {s("exact")}/{n}; re-saved with stored predictions, '
        f'sha256 equal to stored: {resave_equal}/{n}')
    log(f'  input_sha256 equal {s("input_equal")}/{n} | source_sha256 equal {s("source_equal")}/{n} | '
        f'raw_predictions_sha256 equal {s("raw_equal")}/{n} | view key equal {s("thr_equal")}/{n}')
    log(f'  candidates: stored {s("n_stored")}, release {s("n_release")}; images with equal counts '
        f'{sum(t["n_stored"] == t["n_release"] for t in table)}/{n}; candidates identical at the same position '
        f'{s("identical")}')
    log(f'  one-to-one matching (IoU > 0.5): {s("matched50")} matched, {s("unmatched_stored")} stored and '
        f'{s("unmatched_release")} release candidates unmatched')
    im = np.array([t['iou_mean'] for t in table], float); imin = np.array([t['iou_min'] for t in table], float)
    sd = np.array([t['score_maxdiff'] for t in table], float); fd = np.array([t['feat_maxdiff'] for t in table], float)
    log(f'  matched IoU: per-image mean {np.nanmean(im):.6f} (lowest image {np.nanmin(im):.6f}), lowest pair '
        f'{np.nanmin(imin):.6f}; max |score diff| {np.nanmax(sd):.3g} (median over images {np.nanmedian(sd):.3g}); '
        f'max |feature diff| {np.nanmax(fd):.3g}')
    worst = sorted(table, key=lambda t: -(t['unmatched_stored'] + t['unmatched_release']))[:5]
    for t in worst:
        log(f'    {t["file_name"]}: stored {t["n_stored"]} release {t["n_release"]} matched {t["matched50"]} '
            f'iou_mean {t["iou_mean"]:.4f} score_maxdiff {t["score_maxdiff"]:.3g}')

    log('4. stage-5 I6 columns of the rows (cols 28-44, 62), both banks')
    t0 = time.time()
    orig = Path(a.original) / 'work'
    jobs = []
    for name in names:
        stem = name[:-5]
        jobs.append((stem, str(rows_dir / f'{stem}.npz'),
                     {'stored': str(stored / stored_files[name]['file']), 'release': str(bank / new_files[name]['file'])},
                     str(orig / 'ext/testcnx-i6g' / f'{stem}.npz'), str(orig / 'feat/testcnx-i6g-rich' / f'{stem}.npz'),
                     str(work)))
    with ProcessPoolExecutor(min(a.nproc, evidence.MAX_WORKERS)) as ex:
        out = list(ex.map(evidence_job, jobs, chunksize=2))
    N = sum(o['n'] for o in out); NB = sum(o['nb'] for o in out)
    log(f'  {len(out)} images, {N} rows ({NB} tree, {N - NB} new); I6 columns equal across the 4 F variants in the '
        f'original: {sum(o["variants_same"] for o in out)}/{len(out)}; {time.time() - t0:.0f} s')
    for tag in ('stored', 'release'):
        g = lambda k: sum(o[tag][k] for o in out)
        cc = np.sum([o[tag]['col_changed'] for o in out], 0); cm = np.max([o[tag]['col_maxdiff'] for o in out], 0)
        log(f'  {tag} bank: ext files equal to the original {g("ext_equal")}/{len(out)}, rich files equal '
            f'{g("rich_equal")}/{len(out)}; rows whose I6 columns differ from the original rows: {g("rows_changed")}/{N} '
            f'({g("tree_changed")} tree, {g("new_changed")} new; {g("rows_changed_1e3")} by more than 1e-3); images with '
            f'a changed row {sum(o[tag]["rows_changed"] > 0 for o in out)}/{len(out)}; greedy-set sizes {g("n_sel")}')
        if g('rows_changed'):
            log('    per column (rows changed, max |diff|): ' + ', '.join(
                f'{c} {int(k)} {m:.3g}' for c, k, m in zip(COLNAMES, cc, cm)))
    (work / 'verify_i6.log').write_text('\n'.join(_LOG) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
