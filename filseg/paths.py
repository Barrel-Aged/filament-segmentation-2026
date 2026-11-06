"""Where the pipeline reads the competition data and writes its intermediate files.

Set these with environment variables, or pass paths to the functions that take them:
- FILSEG_DATA: the unzipped Kaggle competition folder, the one that holds ``test/test_images`` and
  ``train/MAGFiLO_1.0_Annotations_kaggle2026_train.json``. Default: ``<repo>/data/MAGFiLO_1.0_Kaggle_2026``.
- FILSEG_WORK: a writable folder for intermediate files (maps, candidates, rows, scores). Default: ``<repo>/work``.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("FILSEG_DATA", REPO / "data" / "MAGFiLO_1.0_Kaggle_2026"))
WORK = Path(os.environ.get("FILSEG_WORK", REPO / "work"))
CONFIGS = REPO / "configs"

TEST_IMAGES = DATA / "test" / "test_images"
TRAIN_ANNOTATIONS = DATA / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"


def work(*parts: str) -> Path:
    """A path inside the work folder; its parent folders are created."""
    p = WORK.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
