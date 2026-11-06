"""Pack the original run's GPU outputs (stages 2-4) for the notebook's replay mode.

usage: python tools/pack_replay.py <maps dir> <det dir> <i6 bank dir>
  <maps dir>: holds test{cnx,cnxL,r34}.npy and .npy.json (uint8, (180, 2, 2048, 2048))
  <det dir>:  holds det-f{0..3}/test/
  <i6 bank dir>: the Codex I6 test bank (180 JSON records + manifest.json, repeat.json)
Writes replay/<name>.zip archives split into parts under 95 MB, and replay/manifest.json with the sha256 of every
original file inside each archive, so filseg.replay can check that what it unpacks is exactly what the run produced.
Each map is stored per image as zlib-compressed raw bytes; the maps are mostly zeros and shrink about 100-fold.
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import zipfile
import zlib
from pathlib import Path

import numpy as np

PART = 95_000_000
REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "replay"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def pack_maps(maps_dir: Path, zf: zipfile.ZipFile, files: dict) -> None:
    for n in ("cnx", "cnxL", "r34"):
        npy = maps_dir / f"test{n}.npy"
        side = (maps_dir / f"test{n}.npy.json").read_bytes()
        m = np.load(npy, mmap_mode="r")
        h = hashlib.sha256()
        with open(npy, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
        files[f"maps/test{n}.npy"] = dict(sha256=h.hexdigest(), shape=list(m.shape), dtype=str(m.dtype))
        files[f"maps/test{n}.npy.json"] = dict(sha256=sha(side))
        zf.writestr(f"maps/test{n}.npy.json", side)
        for i in range(m.shape[0]):
            zf.writestr(f"maps/test{n}/{i:04d}.zlib", zlib.compress(np.ascontiguousarray(m[i]).tobytes(), 6),
                        compress_type=zipfile.ZIP_STORED)
        print(f"maps {n}: {m.shape[0]} images", flush=True)


def pack_tree(root: Path, prefix: str, zf: zipfile.ZipFile, files: dict) -> None:
    for p in sorted(root.rglob("*")):
        if p.is_file():
            b = p.read_bytes()
            rel = f"{prefix}/{p.relative_to(root).as_posix()}"
            files[rel] = dict(sha256=sha(b))
            zf.writestr(rel, b)
    print(f"{prefix}: {sum(1 for k in files if k.startswith(prefix + '/'))} files", flush=True)


def split(blob: bytes, name: str) -> list:
    parts = []
    for i in range(0, len(blob), PART):
        chunk = blob[i:i + PART]
        rel = f"{name}.part{i // PART:03d}"
        (OUT / rel).write_bytes(chunk)
        parts.append(dict(path=rel, size=len(chunk), sha256=sha(chunk)))
    return parts


def main(maps_dir: str, det_dir: str, i6_dir: str) -> None:
    OUT.mkdir(exist_ok=True)
    for old in OUT.glob("*.part*"):
        old.unlink()
    manifest = {"note": "GPU outputs (stages 2-4) of the run that produced the submitted s1-S-t024.csv", "archives": {}}
    jobs = {"maps": lambda zf, f: pack_maps(Path(maps_dir), zf, f),
            "det": lambda zf, f: [pack_tree(Path(det_dir) / f"det-f{k}" / "test", f"det/det-f{k}/test", zf, f)
                                  for k in range(4)],
            "i6": lambda zf, f: pack_tree(Path(i6_dir), "i6/test", zf, f)}
    for name, job in jobs.items():
        buf, files = io.BytesIO(), {}
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            job(zf, files)
        blob = buf.getvalue()
        manifest["archives"][name] = dict(size=len(blob), sha256=sha(blob), parts=split(blob, f"{name}.zip"),
                                          files=files)
        print(f"{name}.zip: {len(blob):,} B", flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8",
                                       newline="\n")


if __name__ == "__main__":
    main(*sys.argv[1:4])
