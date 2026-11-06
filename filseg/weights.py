"""Load model weights that are stored in this repository as parts under GitHub's 100 MB per-file limit.

Each weight file is stored as ``weights/<name>.partNNN`` pieces plus an entry in ``weights/manifest.json`` that gives
the sha256 of every part and of the whole file. The parts are byte slices of the original fp32 file, so joining them
gives back exactly the file that produced the submission; nothing is converted or re-saved.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WEIGHTS = REPO / "weights"


def _manifest() -> dict:
    return json.loads((WEIGHTS / "manifest.json").read_text(encoding="utf-8"))


def read_bytes(name: str) -> bytes:
    """Join the parts of weight file ``name`` (for example ``"cnx-f0/ema.pt"``) and check its sha256."""
    entry = _manifest()["files"][name]
    chunks = []
    for part in entry["parts"]:
        data = (WEIGHTS / part["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != part["sha256"]:
            raise ValueError(f"{part['path']}: sha256 mismatch")
        chunks.append(data)
    blob = b"".join(chunks)
    if hashlib.sha256(blob).hexdigest() != entry["sha256"] or len(blob) != entry["size"]:
        raise ValueError(f"{name}: joined file does not match the manifest")
    return blob


def open_file(name: str) -> io.BytesIO:
    """A file-like object for ``torch.load``, ``joblib.load`` and similar readers."""
    return io.BytesIO(read_bytes(name))


def list_files() -> list[str]:
    return sorted(_manifest()["files"])
