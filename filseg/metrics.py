"""The organisers' panoptic quality (PQ), replicated on COCO RLE masks with pycocotools. CPU only.

Reference: the host's self-evaluation notebook, https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook

How the score is counted
- One record is one annotator's drawing of one image. Records are scored separately and then pooled; an image drawn by
  three annotators gives three records that share the same predictions.
- In each record, every (ground-truth object, prediction) pair with IoU strictly greater than 0.5 is a true positive.
  There is no Hungarian matching: the pairs are counted as they come. A prediction that matches nothing is a false
  positive, a ground-truth object that matches nothing is a false negative.
- Pooled over all records: PQ = (sum of the matched IoUs) / (TP + 0.5 FP + 0.5 FN).
- Ground-truth polygons are rasterised one by one with pycocotools (``frPyObjects`` + ``merge``); two polygons of the same
  record that overlap both keep the shared pixels.

The module also holds the mask codec the writer and the validators use (``encode_mask`` / ``decode_mask``: COCO RLE,
column-major) and ``validate_instances`` (right size, non-empty, pairwise disjoint).

Changes from the original run
- Copied from ``src/filament/metrics.py`` (102 lines); every formula, threshold and pycocotools call is unchanged.
- ``read_json`` came from ``src/filament/common.py`` L14-15 (``from .common import read_json``, metrics.py L11); it is now
  a private helper here with the same ``utf-8-sig`` decoding.
- ``score_predictions`` (metrics.py L75-102) is split in two: ``score_records`` holds the counting loop of L82-102 and
  takes the annotations, the predictions and the wanted image names directly; ``score_predictions`` keeps the original
  signature, the reserved-holdout refusal (L77-78) and the coverage check (L79-81), and calls it. The arithmetic and the
  record order (the order of ``annotations["images"]``) are those of the original.
- Docstrings and type hints added. Nothing removed.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from pycocotools import mask as coco_mask


def _read_json(path):
    """JSON file as Python objects (a UTF-8 byte-order mark is accepted)."""
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def encode_mask(mask: np.ndarray) -> dict:
    """COCO RLE of a binary 2-D mask: ``{"size": [h, w], "counts": <ascii string>}``, column-major as COCO defines it."""
    if mask.ndim != 2 or not np.isin(mask, [0, 1]).all():
        raise ValueError("Expected a binary 2-D mask")
    result = coco_mask.encode(np.asfortranarray(mask, dtype=np.uint8))
    return {"size": list(result["size"]), "counts": result["counts"].decode("ascii")}


def native_rle(rle: dict) -> dict:
    """The RLE dict pycocotools expects (counts as bytes)."""
    return {"size": list(rle["size"]), "counts": rle["counts"].encode("ascii") if isinstance(rle["counts"], str) else rle["counts"]}


def decode_mask(rle: dict) -> np.ndarray:
    """uint8 mask (h x w) of a COCO RLE dict."""
    return coco_mask.decode(native_rle(rle))


def polygon_rle(annotation: dict, height: int, width: int):
    """COCO RLE of one ground-truth annotation: its polygons rasterised by pycocotools and merged."""
    rles = coco_mask.frPyObjects(annotation["segmentation"], height, width)
    return coco_mask.merge(rles)


def validate_instances(rles: list, shape: tuple) -> None:
    """Raise ValueError unless every instance has the image's size, is non-empty and is disjoint from the others."""
    occupied = np.zeros(shape, dtype=bool)
    for rle in rles:
        if tuple(rle["size"]) != tuple(shape):
            raise ValueError("Prediction dimensions do not match source image")
        mask = decode_mask(rle).astype(bool)
        if not mask.any():
            raise ValueError("Empty predicted instance")
        if (occupied & mask).any():
            raise ValueError("Overlapping predicted instances")
        occupied |= mask


@dataclass
class PQCounts:
    """Pooled pair counts; ``add_matrix`` adds one record, ``result`` gives PQ, SQ and RQ."""
    tp: int = 0
    fp: int = 0
    fn: int = 0
    matched_iou_sum: float = 0.0

    def add_matrix(self, iou: np.ndarray) -> None:
        """Add one record from its IoU matrix (ground truth x prediction)."""
        hits = iou > 0.5
        self.tp += int(hits.sum())
        self.fp += int((~hits.any(axis=0)).sum())
        self.fn += int((~hits.any(axis=1)).sum())
        self.matched_iou_sum += float(iou[hits].sum())

    def result(self) -> dict:
        denominator = self.tp + 0.5 * (self.fp + self.fn)
        return {**asdict(self), "pq": self.matched_iou_sum / denominator if denominator else 0.0,
                "sq": self.matched_iou_sum / self.tp if self.tp else 0.0,
                "rq": self.tp / denominator if denominator else 0.0}


def overlap_matrix(gt: list, predictions: list) -> np.ndarray:
    """IoU matrix, ground truth x prediction (float64)."""
    if not gt or not predictions:
        return np.zeros((len(gt), len(predictions)), dtype=np.float64)
    # COCO returns prediction x GT; organizer uses GT x prediction.
    return coco_mask.iou([native_rle(r) for r in predictions], [native_rle(r) for r in gt], [0] * len(gt)).T


def score_records(annotations: dict, predictions: dict, wanted: set) -> dict:
    """Pooled PQ over every record (annotator-image) whose image is in ``wanted``.

    annotations: the competition's COCO-style annotation dict; predictions: {file name: [RLE dicts]} covering ``wanted``.
    Returns the pooled counts and PQ / SQ / RQ plus one entry per record under ``"records"``.
    """
    by_record = {}
    for ann in annotations["annotations"]:
        by_record.setdefault(ann["image_id"], []).append(ann)
    counts = PQCounts()
    details = []
    validated = set()
    for record in annotations["images"]:
        name = record["file_name"]
        if name not in wanted:
            continue
        pred = predictions[name]
        if name not in validated:
            validate_instances(pred, (record["height"], record["width"]))
            validated.add(name)
        gt = [polygon_rle(ann, record["height"], record["width"]) for ann in by_record.get(record["id"], [])]
        matrix = overlap_matrix(gt, pred)
        counts.add_matrix(matrix)
        individual = PQCounts()
        individual.add_matrix(matrix)
        details.append({"file_name": name, "annotator_image_id": record["id"], **individual.result()})
    return {**counts.result(), "records": details}


def score_predictions(annotation_path, split_path, prediction_path, fold: int) -> dict:
    """PQ of a prediction JSON ({file name: [RLE dicts]}) on one development fold of a split file.

    The split's reserved holdout fold is refused, and the predictions must cover exactly the fold's images (``[]`` for
    an image without detections).
    """
    annotations, split, predictions = map(_read_json, (annotation_path, split_path, prediction_path))
    if fold == split["reserved_holdout_fold"]:
        raise ValueError("Reserved holdout is locked during development. Release it through a documented final-selection change.")
    wanted = {row["file_name"] for row in split["observations"] if row["fold"] == fold}
    if not wanted or set(predictions) != wanted:
        raise ValueError("Predictions must contain exactly every observation in the selected fold (use [] for no detections)")
    return {"metric": "pooled_annotator_pq_strict_gt_0.5", "fold": fold, **score_records(annotations, predictions, wanted)}
