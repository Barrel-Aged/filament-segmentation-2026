"""Write the submission CSV: selected rows -> masks -> column-major COCO RLE strings. CPU only.

A row's mask is stored as its sorted row-major pixel indices ``y * 2048 + x`` (the ``set_flat`` / ``set_len`` arrays of a
row file). The writer never re-rasterises a polygon: it writes these pixel sets (or their shifted versions, see
``filseg.shift``) as they are.

Encoding: each instance becomes a uint8 2048 x 2048 mask, ``np.asfortranarray`` + ``pycocotools.mask.encode``; the CSV
cell is the ASCII ``counts`` string. COCO RLE runs down the columns, so a pixel at row 0, column 1 has index 2048.

CSV layout (Kaggle's sample): header ``filament_id,segmentation_rle``, LF line ends, images sorted by file name, one row
per instance with id ``<image stem>_<1..n>`` in selection order, and no row for an image without instances. Every test
image still needs an entry (an empty list), so a missing image is an error and not a silent gap.

Checks made while writing (each raises): every instance non-empty and disjoint from the earlier ones of its image;
canonical RLE (decode + re-encode gives the same string); the written file is read back and must give exactly the RLEs
that were meant (``verify_csv``).

Changes from the original run
- Copied from ``work/ideas8/bundle-test/bt_common.py`` (``unpack_sets`` L31-33, ``rle_of_set`` L36-42, ``rasterize``
  L45-55, ``write_csv`` L58-69) and ``src/filament/s1_submission.py`` (``export_csv`` L39-61, ``verify_csv`` L64-98).
  The encoding, row order, id format, header and line ends are unchanged.
- Removed: the hard-coded ROOT and the ``sys.path`` inserts (bt_common.py L13-17), the in-function
  ``from filament.s1_submission import export_csv`` (L61), and ``os.environ['CUDA_VISIBLE_DEVICES'] = ''`` (L12; nothing
  here touches a GPU). ``SubmissionRefitIndex`` (s1_submission.py L14-36) is not copied: it is a training-data index the
  writer never used, and importing it pulled in ``filament.dataset`` -> scikit-image and Pillow.
- ``encode_mask``, ``decode_mask``, ``validate_instances`` now come from ``filseg.metrics`` (the same functions as
  ``src/filament/metrics.py`` L14-44); ``sha256`` (``src/filament/common.py`` L6-11) is copied here as ``sha256_file``.
- ``export_csv`` opens the file with mode ``'x'`` (s1_submission.py L43-44, L57), so the original caller deleted an old
  output first (``work/ideas11/synth/csv/csv_write.py`` L103-105). ``write_csv`` now takes ``overwrite`` (default False):
  with True it removes the old CSV and sidecar itself; with False an existing file is still an error.
- ``rasterize`` raises ValueError where bt_common.py L51-52 used ``assert`` (so the check survives ``python -O``).
- ``selected_sets`` and ``load_row_sets`` are new two-line helpers for what csv_write.py L97-106 did inline (rows of each
  image, picked in ``sel_rows`` order).
- The sidecar ``<csv>.json`` keeps the original layout (``instances``, ``verify``, ``meta``; bt_common.py L68); its
  ``meta`` content is whatever the caller passes, so it is not byte-identical to the original sidecar.
- The comparisons csv_write.py made against legacy files (``b3f707-t024.csv``, ``b3f-t024.csv``, the C2-plus CSV and its
  header bytes, ``sel707.pkl``; L28-34, L95-96, L113-115, L146-194) are not part of the writer. The validation block of
  L111-145 is in ``filseg.validate``, with the C2-plus header replaced by the literal header bytes.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as mu

from .metrics import decode_mask, encode_mask, validate_instances

H = W = 2048
HEADER = ["filament_id", "segmentation_rle"]


def sha256_file(path) -> str:
    """sha256 hex digest of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---- pixel sets ---------------------------------------------------------------------------------------------------

def unpack_sets(flat: np.ndarray, lens: np.ndarray) -> list:
    """Split the concatenated pixel indices of a row file into one int64 array per row."""
    off = np.concatenate([[0], np.cumsum(lens)])
    return [flat[off[i]:off[i + 1]].astype(np.int64) for i in range(len(lens))]


def load_row_sets(path) -> list:
    """Pixel sets of every row of one image's row file (``set_flat`` / ``set_len``)."""
    with np.load(path) as z:
        return unpack_sets(z["set_flat"], z["set_len"])


def selected_sets(sets: list, sel_rows) -> list:
    """The pixel sets of the selected rows, in selection order."""
    return [sets[i] for i in np.asarray(sel_rows, np.int64)]


# ---- encoding -----------------------------------------------------------------------------------------------------

def rle_of_set(Q: np.ndarray) -> dict:
    """COCO RLE dict of one instance given its flat row-major pixel indices."""
    m = np.zeros(H * W, np.uint8)
    m[Q] = 1
    m = m.reshape(H, W)
    e = mu.encode(np.asfortranarray(m.astype(np.uint8)))
    return {"size": list(e["size"]), "counts": e["counts"].decode("ascii")}


def rasterize(sets: list) -> list:
    """RLEs of one image's instances; raises if an instance is empty or overlaps an earlier one."""
    occ = np.zeros(H * W, bool)
    out = []
    for Q in sets:
        Q = np.asarray(Q, np.int64)
        if len(Q) == 0:
            raise ValueError("empty instance")
        if occ[Q].any():
            raise ValueError("overlapping instances")
        occ[Q] = True
        out.append(rle_of_set(Q))
    return out


# ---- CSV ----------------------------------------------------------------------------------------------------------

def export_csv(predictions: dict, path, shapes: dict) -> dict:
    """Write {file name: [RLE dicts]} as the submission CSV (the file must not exist) and verify it."""
    if set(predictions) != set(shapes):
        raise ValueError("Every test image needs an inference record, including empty images")
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    rows = []
    for name in sorted(shapes):
        stem = Path(name).stem
        if "_" in stem:
            raise ValueError("Unexpected observation identifier")
        validate_instances(predictions[name], shapes[name])
        for number, rle in enumerate(predictions[name], 1):
            mask = decode_mask(rle)
            if encode_mask(mask) != rle:
                raise ValueError("Noncanonical or malformed COCO RLE")
            rows.append((f"{stem}_{number}", rle["counts"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(HEADER)
        writer.writerows(rows)
    return verify_csv(path, predictions, shapes)


def verify_csv(path, predictions: dict, shapes: dict) -> dict:
    """Read the CSV back: columns, unique sequential ids, canonical RLE, exactly ``predictions``, disjoint instances."""
    if set(predictions) != set(shapes):
        raise ValueError("Prediction coverage differs from test inventory")
    stems = {Path(name).stem: name for name in shapes}
    if len(stems) != len(shapes):
        raise ValueError("Duplicate observation stem")
    reconstructed = {name: [] for name in shapes}
    seen = set()
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != HEADER:
            raise ValueError("Wrong submission columns")
        for row in reader:
            key = row["filament_id"]
            if key in seen or None in row:
                raise ValueError("Duplicate ID or extra CSV columns")
            seen.add(key)
            stem, number = key.rsplit("_", 1)
            if stem not in stems or int(number) != len(reconstructed[stems[stem]]) + 1:
                raise ValueError("Unknown observation or nonsequential instance ID")
            name = stems[stem]
            rle = {"size": list(shapes[name]), "counts": row["segmentation_rle"]}
            mask = decode_mask(rle)
            if encode_mask(mask) != rle:
                raise ValueError("CSV RLE does not round-trip")
            reconstructed[name].append(rle)
    if reconstructed != predictions:
        raise ValueError("CSV differs from frozen predictions")
    for name, rles in reconstructed.items():
        validate_instances(rles, shapes[name])
    return {"passed": True, "csv_sha256": sha256_file(path), "images_inferred": len(shapes),
            "images_with_instances": sum(bool(v) for v in reconstructed.values()),
            "empty_images": sorted(n for n, v in reconstructed.items() if not v),
            "instance_rows": len(seen), "rle_orientation": "COCO column-major; exact mask round-trip",
            "disjoint_instances": True, "invented_empty_image_predictions": False}


def write_csv(path, sets_by_file: dict, files: list, meta: dict | None = None, overwrite: bool = False):
    """Write the CSV and its ``.json`` sidecar.

    sets_by_file: {file name (with extension): [pixel index arrays in selection order]}; files: the full test inventory
    (an image missing from ``sets_by_file`` gets no row). Returns (verify result, {file name: [RLE dicts]}).
    """
    path = Path(path)
    sidecar = Path(str(path) + ".json")
    if overwrite:
        for p in (path, sidecar):
            if p.exists():
                p.unlink()
    predictions, shapes, n = {}, {}, 0
    for f in files:
        predictions[f] = rasterize(sets_by_file.get(f, []))
        shapes[f] = (H, W)
        n += len(predictions[f])
    res = export_csv(predictions, path, shapes)
    with sidecar.open("w") as fh:
        json.dump({"instances": n, "verify": res, "meta": meta or {}}, fh, indent=1, default=float)
    return res, predictions
