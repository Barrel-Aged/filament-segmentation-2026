"""Rebuild the test evidence features and the 180 row files, and compare them with the original run.

usage (from the repository root):
  python -m tools.verify_rows --original <claude tree> --i6-root <I6 test bank> --image-dir <test_images>
                              [--maps-dir DIR] [--det-dir DIR] [--work DIR] [--nproc 4] [--skip-build]

--original is the original Claude tree (the folder holding ``work/cands``, ``work/ext``, ``work/feat``, ``work/maps``,
``work/det`` and ``work/ideas8/bundle-test``); its files are only read. ``--maps-dir`` (default
``<original>/work/maps/final``) holds ``test{cnx,cnxL,r34}.npy`` + ``.json``; ``--det-dir`` (default
``<original>/work/det``) holds ``det-f{k}/test``. Outputs go to ``--work`` (default ``filseg.paths.WORK``), the log to
``<work>/verify_rows.log``.

Checks, in order:
1. the original folders still have the directory digests recorded for the final (and the rows their per-file
   sha256 in ``bundle-test/manifest.json``), so the comparison is against the files that made the submission;
2. rebuild (stage 5a ``filseg.evidence``, then stage 5b ``filseg.rows``), timed;
3. per folder and file: sha256 of the whole file; for every array of the original file, dtype, shape and exact
   content; the decompressed ``.npy`` member bytes; keys the release no longer writes are listed, not compared.
   Then each file is re-saved from the rebuilt arrays plus the original's not-written members, in the original key
   order, and its sha256 compared with the original file's (numpy 2.2 stamps zip entries 1980-01-01, so equal content
   gives equal bytes).
Directory digest recipe: ``sha256('\\n'.join(f'{name} {sha256(file)}' for name in sorted(names)))``.
"""
from __future__ import annotations

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import argparse  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402
import zipfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from filseg import evidence, paths, rows  # noqa: E402

# Directory digests of the original run (work/package/release/PATH_MAP.md, "Full digests computed 11-05"; the rows
# digest is also stored in work/ideas8/b3f707/manifest.json row_sets).
RECORDED = {
    'cands/testcnx': '993a17f104fc7452ffea4be891da0488510f9aedf2de25b88c06affd1993cdbb',
    'ext/testcnx-i6g': 'f5b91761d6d8c2b70852fdc9c6d92ffb7dfab2b3c61bad44709aa87ee2307210',
    'feat/testcnx-i6g-rich': '6603e4288ead2f10d8e788bdb130dba283987bfc4258b34d3a83b2dca2714a43',
    'ext/testcnx-det-f0': '8ac26282aa32245a4c35966aafd29096d7494f1ee51d25475fb1ac76d5156e93',
    'ext/testcnx-det-f1': '3e0a3deed7227f8f983b468f7b81b516bab9cca6dc596dd0366dce173ff1ea70',
    'ext/testcnx-det-f2': '0925c2abc5cbcdff327e1e84e4445f17068dc2b79ae38045fd462110ba5762da',
    'ext/testcnx-det-f3': '547b0ba1afe7979fcd2b13a351cc559f166de15ba21d073e3f544764adc3cf6b',
    'feat/testcnx-detrich-f0': 'f27622628e901de44a5f0f655b6f8364f122e141d287bc6d401e9d65cece7a4e',
    'feat/testcnx-detrich-f1': '5fd1f3dc491d6111c78e85c9bf3820602d732304bc4d5073c7e3e2ee1efb983a',
    'feat/testcnx-detrich-f2': '0f90e75670d3daf37d736c59b3217eecd1716c85698e1f17572cf85a97fef3b1',
    'feat/testcnx-detrich-f3': 'c96be5eb28c4f847645c4d6d561545b2df36f35a46bbc01aa17e9c6888ed764f',
    'rows': 'fcb0ca6dd01e7a4a3cb7144081399bdeb12da277872f0d953825ade07d566539',
}
ORIGINAL_ROWS = 'work/ideas8/bundle-test/rows'
# Keys the original files hold that the release does not write (see the module docstrings of evidence.py / rows.py).
DROPPED = {'cands': {'reg', 'hits', 'ngt'}, 'ext': {'Eh'}, 'feat': set(), 'rows': {'ok', 'xmR_maxdiff', 'xmL_maxdiff'}}

_LOG = []


def log(*parts):
    line = ' '.join(str(p) for p in parts)
    print(line, flush=True)
    _LOG.append(line)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dir_digest(d):
    """(directory digest, {name: sha256}) over the .npz files of d."""
    names = sorted(p.name for p in Path(d).glob('*.npz'))
    per = {n: sha256(Path(d) / n) for n in names}
    return hashlib.sha256('\n'.join(f'{n} {per[n]}' for n in names).encode()).hexdigest(), per


def original_dir(original, rel):
    return Path(original) / (ORIGINAL_ROWS if rel == 'rows' else f'work/{rel}')


def check_recorded(original):
    """Step 1: the original folders against the recorded digests."""
    ok = True
    for rel, want in RECORDED.items():
        got, per = dir_digest(original_dir(original, rel))
        same = got == want
        ok &= same
        log(f'  original {rel:26s} {len(per):3d} files  digest {got[:16]}  recorded {"equal" if same else "DIFFERENT: " + want}')
    man = json.loads((Path(original) / 'work/ideas8/bundle-test/manifest.json').read_text(encoding='utf-8'))['test_rows']
    _, per = dir_digest(original_dir(original, 'rows'))
    eq = sum(per.get(n) == h for n, h in man.items())
    ok &= eq == len(man) == len(per)
    log(f'  original rows vs bundle-test/manifest.json test_rows: {eq}/{len(man)} per-file sha256 equal')
    return ok


def _equal(a, b):
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.dtype == object:
        return a.tolist() == b.tolist()
    return bool(np.array_equal(a, b, equal_nan=a.dtype.kind == 'f'))


def _diff_note(a, b):
    if a.dtype != b.dtype or a.shape != b.shape:
        return f'dtype/shape {a.dtype}{a.shape} vs {b.dtype}{b.shape}'
    if a.dtype == object:
        return f'{sum(x != y for x, y in zip(a.ravel(), b.ravel()))} differing items'
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return f'{int((d > 0).sum())} differing values, max|d| {float(d.max()):.3g}'


def compare_dir(rel, orig_dir, new_dir, dropped):
    """Step 3 for one folder. Returns a summary dict."""
    names = sorted(p.name for p in Path(orig_dir).glob('*.npz'))
    s = dict(files=len(names), missing=0, file_sha_equal=0, arrays=0, arrays_equal=0, members_equal=0, extra_keys=set(),
             diffs=[], dropped_seen=set())
    for n in names:
        po, pn = Path(orig_dir) / n, Path(new_dir) / n
        if not pn.exists():
            s['missing'] += 1; s['diffs'].append(f'{n}: missing'); continue
        s['file_sha_equal'] += sha256(po) == sha256(pn)
        zo, zn = np.load(po, allow_pickle=True), np.load(pn, allow_pickle=True)
        fo, fn = set(zo.files), set(zn.files)
        s['extra_keys'] |= fn - fo
        with zipfile.ZipFile(po) as ao, zipfile.ZipFile(pn) as an:
            for k in sorted(fo):
                if k in dropped:
                    s['dropped_seen'].add(k); continue
                s['arrays'] += 1
                if k not in fn:
                    s['diffs'].append(f'{n}:{k}: not written'); continue
                a, b = zo[k], zn[k]
                if _equal(a, b):
                    s['arrays_equal'] += 1
                else:
                    s['diffs'].append(f'{n}:{k}: {_diff_note(a, b)}')
                s['members_equal'] += ao.read(k + '.npy') == an.read(k + '.npy')
    return s


def report(rel, s):
    log(f'  {rel:26s} files {s["files"]:3d} | file sha256 equal {s["file_sha_equal"]:3d} | arrays equal '
        f'{s["arrays_equal"]}/{s["arrays"]} | .npy member bytes equal {s["members_equal"]}/{s["arrays"]} | not written '
        f'{sorted(s["dropped_seen"]) or "-"} | extra keys {sorted(s["extra_keys"]) or "-"}')
    for d in s['diffs'][:20]:
        log('    DIFF', d)
    return s['missing'] == 0 and s['arrays_equal'] == s['arrays'] and not s['extra_keys']


def reassembled_sha(orig_file, new_file, dropped):
    """sha256 of an npz written from the rebuilt arrays plus the original's not-written members, in the original key
    order: equal to the original file's sha256 iff the release changes nothing but the left-out keys."""
    zo, zn = np.load(orig_file, allow_pickle=True), np.load(new_file, allow_pickle=True)
    buf = io.BytesIO()
    np.savez_compressed(buf, **{k: (zo[k] if k in dropped else zn[k]) for k in zo.files})
    return hashlib.sha256(buf.getvalue()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--original', required=True)
    ap.add_argument('--i6-root', required=True)
    ap.add_argument('--image-dir', default=str(paths.TEST_IMAGES))
    ap.add_argument('--maps-dir')
    ap.add_argument('--det-dir')
    ap.add_argument('--work', default=str(paths.WORK))
    ap.add_argument('--nproc', type=int, default=4)
    ap.add_argument('--skip-build', action='store_true')
    a = ap.parse_args()
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    maps_dir = Path(a.maps_dir or Path(a.original) / 'work/maps/final')
    det_dir = Path(a.det_dir or Path(a.original) / 'work/det')
    maps = {n: maps_dir / f'test{n}.npy' for n in ('cnx', 'cnxL', 'r34')}
    sources = evidence.test_sources(a.i6_root, [det_dir / f'det-f{k}/test' for k in range(4)])
    log('verify_rows', time.strftime('%Y-%m-%d %H:%M:%S %z'), '| python', platform.python_version(), '| numpy',
        np.__version__, '| CUDA_VISIBLE_DEVICES', repr(os.environ['CUDA_VISIBLE_DEVICES']))
    log('inputs: maps', maps_dir, '| det', det_dir, '| i6', a.i6_root, '| images', a.image_dir, '| work', work)
    for n, p in maps.items():
        log(f'  map {n}: {p.name} sha256 {sha256(str(p) + ".json")[:16]} (sidecar)')

    log('1. original folders vs recorded digests')
    rec_ok = check_recorded(a.original)

    if not a.skip_build:
        log('2. rebuild')
        t0 = time.time()
        counts = evidence.build_evidence(maps['cnx'], sources, work, a.image_dir, a.nproc)
        t1 = time.time()
        log(f'  evidence: {len(counts)} images, {sum(counts.values())} tree candidates, {t1 - t0:.0f} s')
        tot = rows.build_rows(maps, work, work / rows.ROWS_DIR, a.image_dir, a.nproc)
        log(f'  rows: {tot}, {time.time() - t1:.0f} s')

    log('3. rebuilt vs original')
    all_ok = True
    for rel in RECORDED:
        new_dir = work / rel
        s = compare_dir(rel, original_dir(a.original, rel), new_dir, DROPPED[rel.split('/')[0]])
        all_ok &= report(rel, s)
        got, _ = dir_digest(new_dir)
        log(f'  {"":26s} rebuilt directory digest {got}')
        od = original_dir(a.original, rel)
        n_eq = sum(reassembled_sha(od / n, new_dir / n, DROPPED[rel.split('/')[0]]) == sha256(od / n)
                   for n in sorted(p.name for p in od.glob('*.npz')))
        log(f'  {"":26s} rebuilt arrays + original not-written members re-saved: sha256 equal to the original {n_eq}/180')
        all_ok &= n_eq == 180
    z = [np.load(p) for p in sorted(original_dir(a.original, 'rows').glob('*.npz'))]
    log(f'  original rows, keys not written: ok all True {all(bool(x["ok"]) for x in z)}, max xmR_maxdiff '
        f'{max(float(x["xmR_maxdiff"]) for x in z)}, max xmL_maxdiff {max(float(x["xmL_maxdiff"]) for x in z)}')
    log('RESULT', 'recorded digests ok' if rec_ok else 'recorded digests DIFFER', '|',
        'every compared array equal' if all_ok else 'DIFFERENCES FOUND')
    (work / 'verify_rows.log').write_text('\n'.join(_LOG) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
