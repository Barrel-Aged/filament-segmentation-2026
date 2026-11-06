"""Checks of a written submission CSV, independent of the writer's own read-back. CPU only.

``validate_csv`` returns a list of ``(name, ok, detail)`` checks:
- header bytes ``filament_id,segmentation_rle`` + LF, and no CR anywhere;
- ids unique, ``<stem>_<1..n>`` sequential per image, rows ordered by sorted image name, every stem a test image;
- the test inventory has no duplicate, and (with ``expected``) every image with instances has rows, an image without
  instances has none, per-image counts are equal, and each decoded instance equals its intended pixel set;
- every instance non-empty, instances of an image pairwise disjoint (maximum pixel multiplicity 1);
- RLE round trip: pycocotools decode + re-encode gives back the same string;
- orientation: an independent decoder of the COCO counts string (no pycocotools), read column-major, gives the same
  pixels as pycocotools and as the intended sets. A transposed file is valid RLE and round-trips, so only a comparison
  with something outside the codec can catch it.

``compare_with_unshifted`` checks the shifted CSV against the unshifted one: same header bytes and id column, and only
instances of the shifted families changed.

Changes from the original run
- Collected from the validation block of ``work/ideas11/synth/csv/csv_write.py`` L111-145 and
  ``work/ideas15/hyg/csv/patch_test.py`` L296-352, L374-377, L404, the reader ``decode_csv`` and ``disjoint_check``
  of ``work/ideas8/bundle-test/bt_common.py`` L72-103, ``read_raw`` (patch_test.py L105-111, a copy of
  ``work/ideas8/bundle-test/write_test.py``'s), and the independent decoder ``coco_string_runs`` / ``colmajor_pixels``
  (patch_test.py L114-139). The tests themselves are unchanged.
- The header is compared with the literal bytes ``b"filament_id,segmentation_rle\\n"`` instead of the header of the
  accepted C2-plus CSV ``work/sub/c2plus-xmRL-t030.csv`` (csv_write.py L113-116, patch_test.py L274-275, L299-300);
  that file's header line is exactly these bytes.
- ``decode_csv`` raises ValueError where bt_common.py L80-90 used ``assert``; ``validate_csv`` turns a reader failure
  into a failed check instead of a crash. Each check is reported separately, where the original printed one line and
  asserted the conjunction.
- Not copied (they compared with legacy outputs): instances vs ``b3f707-t024.csv`` / ``b3f-t024.csv`` / C2-plus by IoU,
  RLE strings shared with b3f707 (csv_write.py L146-194); centroid and IoU statistics of the shift (patch_test.py
  L356-398, descriptive only). The pixel-direction check is in ``filseg.shift.direction_check``.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from pycocotools import mask as mu

H = W = 2048
HEADER_BYTES = b"filament_id,segmentation_rle\n"


# ---- readers ------------------------------------------------------------------------------------------------------

def read_raw(path):
    """(header, [(filament_id, counts)]) with the csv module, nothing decoded."""
    with open(path, encoding="utf-8", newline="") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        rows = [tuple(r) for r in rd]
    return header, rows


def decode_csv(path, files: list) -> dict:
    """{file name: [(filament_id, sorted flat row-major indices)]} via pycocotools; raises on a malformed file."""
    stems = {Path(f).stem: f for f in files}
    out = {f: [] for f in files}
    seen = set()
    with open(path, encoding="utf-8", newline="") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        if header != ["filament_id", "segmentation_rle"]:
            raise ValueError(f"header {header}")
        for row in rd:
            if len(row) != 2:
                raise ValueError(f"row with {len(row)} fields: {row[:1]}")
            fid, counts = row
            if fid in seen:
                raise ValueError("duplicate id " + fid)
            seen.add(fid)
            stem, num = fid.rsplit("_", 1)
            if stem not in stems:
                raise ValueError("unknown stem " + stem)
            if int(num) != len(out[stems[stem]]) + 1:
                raise ValueError("non-sequential id " + fid)
            m = mu.decode({"size": [H, W], "counts": counts.encode("ascii")})
            if m.shape != (H, W):
                raise ValueError(f"{fid}: decoded shape {m.shape}")
            out[stems[stem]].append((fid, np.flatnonzero(m.ravel(order="C")).astype(np.int64)))
    return out


def disjoint_check(dec: dict) -> int:
    """Maximum pixel multiplicity over the images after decoding (1 means disjoint)."""
    worst = 0
    for f, items in dec.items():
        if not items:
            continue
        cnt = np.bincount(np.concatenate([q for _, q in items]), minlength=H * W)
        worst = max(worst, int(cnt.max()))
    return worst


# ---- independent decoder ------------------------------------------------------------------------------------------

def coco_string_runs(s: str) -> list:
    """independent decoder of a COCO compressed counts string (maskApi.c rleFrString)"""
    cnts, p = [], 0
    while p < len(s):
        x, k, more = 0, 0, True
        while more:
            c = ord(s[p]) - 48
            x |= (c & 0x1f) << (5 * k)
            more = bool(c & 0x20)
            p += 1; k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(cnts) > 2:
            x += cnts[-2]
        cnts.append(x)
    return cnts


def colmajor_pixels(s: str, h: int = H, w: int = W):
    """Sorted row-major indices of the pixels of a counts string read column-major; None if the runs are invalid."""
    cnts = np.asarray(coco_string_runs(s), np.int64)
    if cnts.sum() != h * w or (cnts < 0).any():
        return None
    ends = np.cumsum(cnts); starts = ends - cnts
    on = [np.arange(a, b, dtype=np.int64) for a, b in zip(starts[1::2], ends[1::2])]
    p = np.concatenate(on) if on else np.zeros(0, np.int64)
    return np.sort((p % h) * w + p // h)


# ---- checks -------------------------------------------------------------------------------------------------------

def _check(out: list, name: str, ok, detail: str = "") -> bool:
    out.append((name, bool(ok), detail))
    return bool(ok)


def validate_csv(path, files: list, expected: dict | None = None) -> list:
    """All submission checks of one CSV. files: the test inventory (file names with extension);
    expected: optional {file name: [intended pixel sets in instance order]}."""
    checks = []
    raw = Path(path).read_bytes()
    _check(checks, "header bytes == 'filament_id,segmentation_rle' + LF", raw.split(b"\n", 1)[0] + b"\n" == HEADER_BYTES)
    _check(checks, "no CR in the file", b"\r" not in raw)
    _check(checks, "test inventory without duplicates", len(set(files)) == len(files), f"{len(files)} images")
    header, rows = read_raw(path)
    ids = [r[0] for r in rows]
    _check(checks, "every row has two fields", all(len(r) == 2 for r in rows), f"{len(rows)} rows")
    _check(checks, "ids unique", len(set(ids)) == len(ids))
    stem_order = [i.rsplit("_", 1)[0] for i in ids]
    _check(checks, "rows ordered by sorted image name", stem_order == sorted(stem_order))
    try:
        dec = decode_csv(path, files)
    except ValueError as e:
        _check(checks, "reader: known stems, sequential ids, decodable RLE", False, str(e))
        return checks
    _check(checks, "reader: known stems, sequential ids <stem>_<1..n>, decodable RLE", True)
    n_dec = sum(len(v) for v in dec.values())
    _check(checks, "every instance non-empty", all(len(q) > 0 for v in dec.values() for _, q in v))
    worst = disjoint_check(dec)
    _check(checks, "instances pairwise disjoint (max pixel multiplicity 1)", worst <= 1, f"max multiplicity {worst}")
    rt = sum(mu.encode(np.asfortranarray(mu.decode({"size": [H, W], "counts": c.encode("ascii")})))["counts"].decode("ascii") == c
             for _, c in rows)
    _check(checks, "RLE round trip (decode + re-encode == string)", rt == len(rows), f"{rt}/{len(rows)}")
    by_stem = {}
    for fid, c in rows:
        by_stem.setdefault(fid.rsplit("_", 1)[0], []).append(c)
    cm = sum(np.array_equal(colmajor_pixels(c), q) for f in files for c, (_, q) in zip(by_stem.get(Path(f).stem, []), dec[f]))
    _check(checks, "independent column-major decoder == pycocotools decode", cm == n_dec, f"{cm}/{n_dec}")
    if expected is not None:
        _expected_checks(checks, files, dec, by_stem, expected)
    return checks


def _expected_checks(checks: list, files: list, dec: dict, by_stem: dict, expected: dict) -> None:
    """The checks that compare the file with the intended masks."""
    n_exp = sum(len(expected.get(f, [])) for f in files)
    n_dec = sum(len(v) for v in dec.values())
    stems_in_csv = set(by_stem)
    empty = [f for f in files if not len(expected.get(f, []))]
    empty_ok = all(Path(f).stem not in stems_in_csv for f in empty) and \
        all(Path(f).stem in stems_in_csv for f in files if len(expected.get(f, [])))
    _check(checks, "images with instances have rows, empty images have none", empty_ok,
           f"{len(files) - len(empty)} with instances, empty {sorted(empty)}")
    cnt_ok = all(len(dec[f]) == len(expected.get(f, [])) for f in files)
    _check(checks, "per-image instance counts == intended", cnt_ok and n_dec == n_exp, f"decoded {n_dec}, intended {n_exp}")
    eq = sum(np.array_equal(q, np.asarray(e, np.int64)) for f in files for (_, q), e in zip(dec[f], expected.get(f, [])))
    _check(checks, "decoded instance == intended pixel set (orientation vs the pixel sets)", eq == n_exp, f"{eq}/{n_exp}")
    cm = sum(np.array_equal(colmajor_pixels(c), np.asarray(e, np.int64))
             for f in files for c, e in zip(by_stem.get(Path(f).stem, []), expected.get(f, [])))
    _check(checks, "independent column-major decoder == intended pixel set", cm == n_exp, f"{cm}/{n_exp}")


def compare_with_unshifted(path, unshifted_path, families: list, shifted_families=(0, 5, 6)) -> list:
    """Shifted CSV vs unshifted CSV: same header bytes and id column; changed RLE strings only for rows whose family
    (``families``, one per instance in file order) is in ``shifted_families``."""
    checks = []
    raw, oraw = Path(path).read_bytes(), Path(unshifted_path).read_bytes()
    h, rows = read_raw(path)
    oh, orows = read_raw(unshifted_path)
    _check(checks, "same header bytes as the unshifted CSV", raw.split(b"\n", 1)[0] == oraw.split(b"\n", 1)[0] and h == oh)
    _check(checks, "same id column as the unshifted CSV", [r[0] for r in rows] == [r[0] for r in orows],
           f"{len(rows)} vs {len(orows)} rows")
    if len(families) != len(rows) or len(rows) != len(orows):
        _check(checks, "one family per instance", False, f"{len(families)} families, {len(rows)} rows")
        return checks
    changed = [a[1] != b[1] for a, b in zip(rows, orows)]
    bad = sum(c and int(f) not in shifted_families for c, f in zip(changed, families))
    _check(checks, "changed instances only from the shifted families", bad == 0,
           f"changed {sum(changed)}, of which outside {tuple(shifted_families)}: {bad}")
    return checks


def all_pass(checks: list) -> bool:
    return all(ok for _, ok, _ in checks)


def format_checks(checks: list) -> str:
    """One line per check: [PASS] / [FAIL] name: detail."""
    return "\n".join(f"[{'PASS' if ok else 'FAIL'}] {n}{': ' + d if d else ''}" for n, ok, d in checks)
