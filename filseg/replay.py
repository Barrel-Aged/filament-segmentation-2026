"""Replay mode: unpack the original run's GPU outputs instead of recomputing them.

The full pipeline recomputes stages 2-4 (U-Net maps, detector, Codex I6) on whatever GPU it runs on. GPUs differ
slightly in floating-point arithmetic, so the rebuilt submission is a close copy of the submitted one, not the same
bytes (see README, "Reproducibility"). Replay mode instead unpacks the stage 2-4 outputs of the run that produced the
submitted ``s1-S-t024.csv`` (``replay/``, packed by ``tools/pack_replay.py``) and runs stages 5-10 on them, which
rebuilds the submitted file byte for byte on any machine, CPU only.

These files are model outputs on the competition's test images, so they are data-derived (see README, "Licences").
The notebook never uses them unless replay mode is switched on.
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
import zlib
from pathlib import Path

import numpy as np

from filseg import paths

REPLAY = paths.REPO / "replay"


def _archive(name: str, entry: dict) -> zipfile.ZipFile:
    chunks = []
    for part in entry["parts"]:
        b = (REPLAY / part["path"]).read_bytes()
        if hashlib.sha256(b).hexdigest() != part["sha256"]:
            raise ValueError(f"{part['path']}: sha256 mismatch")
        chunks.append(b)
    blob = b"".join(chunks)
    if hashlib.sha256(blob).hexdigest() != entry["sha256"]:
        raise ValueError(f"{name}.zip: sha256 mismatch")
    return zipfile.ZipFile(io.BytesIO(blob))


def _write_map(zf: zipfile.ZipFile, n: str, info: dict, out: Path) -> None:
    """Rebuild test<n>.npy image by image and check its sha256 against the original file's."""
    shape, dtype = tuple(info["shape"]), np.dtype(info["dtype"])
    out.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    with open(out, "wb") as fh:
        hdr = io.BytesIO()
        np.lib.format.write_array_header_1_0(hdr, {"descr": np.lib.format.dtype_to_descr(dtype),
                                                   "fortran_order": False, "shape": shape})
        fh.write(hdr.getvalue()); h.update(hdr.getvalue())
        for i in range(shape[0]):
            b = zlib.decompress(zf.read(f"maps/test{n}/{i:04d}.zlib"))
            fh.write(b); h.update(b)
    if h.hexdigest() != info["sha256"]:
        raise ValueError(f"{out}: rebuilt map differs from the original (sha256)")


def unpack(work: Path | None = None) -> dict:
    """Unpack maps/, det/ and i6/ into the work folder; every file is checked against the original run's sha256.

    Returns the folders to feed stage 5: ``{'maps': ..., 'det_roots': [...], 'i6': ...}``.
    """
    work = Path(work or paths.WORK)
    man = json.loads((REPLAY / "manifest.json").read_text(encoding="utf-8"))["archives"]
    for name, entry in man.items():
        zf = _archive(name, entry)
        for rel, info in entry["files"].items():
            out = work / "replay" / rel
            if rel.endswith(".npy"):
                _write_map(zf, rel.split("test")[-1][:-4], info, out)
                continue
            b = zf.read(rel)
            if hashlib.sha256(b).hexdigest() != info["sha256"]:
                raise ValueError(f"{rel}: sha256 mismatch")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b)
    root = work / "replay"
    return {"maps": root / "maps", "det_roots": [root / f"det/det-f{k}/test" for k in range(4)], "i6": root / "i6/test"}
