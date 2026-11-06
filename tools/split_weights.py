"""Split weight files into parts under GitHub's 100 MB limit and write weights/manifest.json.

usage: python tools/split_weights.py <source_root> <relative/path> [<relative/path> ...]
Each source file <source_root>/<relative/path> becomes weights/<relative/path>.partNNN (95,000,000 bytes each, the last
one smaller). The manifest records the size and sha256 of the whole file and of every part. Re-running replaces the
entries it touches and keeps the others.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PART = 95_000_000
REPO = Path(__file__).resolve().parents[1]
WEIGHTS = REPO / "weights"


def main(argv: list[str]) -> None:
    src_root = Path(argv[0])
    man_path = WEIGHTS / "manifest.json"
    manifest = json.loads(man_path.read_text(encoding="utf-8")) if man_path.exists() else {"part_bytes": PART, "files": {}}
    for rel in argv[1:]:
        data = (src_root / rel).read_bytes()
        out_dir = (WEIGHTS / rel).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.glob(Path(rel).name + ".part*"):
            old.unlink()
        parts = []
        for i in range(0, len(data), PART):
            chunk = data[i:i + PART]
            part_rel = f"{rel}.part{i // PART:03d}"
            (WEIGHTS / part_rel).write_bytes(chunk)
            parts.append({"path": part_rel, "size": len(chunk), "sha256": hashlib.sha256(chunk).hexdigest()})
        manifest["files"][rel] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest(), "parts": parts}
        print(f"{rel}: {len(data):,} B -> {len(parts)} part(s)")
    man_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main(sys.argv[1:])
