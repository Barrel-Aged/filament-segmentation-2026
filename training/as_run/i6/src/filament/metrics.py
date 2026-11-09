"""CPU COCO-RLE implementation of the organizer's pooled pair-counting PQ.

Reference: https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook
Keep annotator records separate. This deliberately does not use Hungarian matching.
"""
from dataclasses import asdict, dataclass

import numpy as np
from pycocotools import mask as coco_mask

from .common import read_json


def encode_mask(mask):
    if mask.ndim != 2 or not np.isin(mask, [0, 1]).all():
        raise ValueError("Expected a binary 2-D mask")
    result = coco_mask.encode(np.asfortranarray(mask, dtype=np.uint8))
    return {"size": list(result["size"]), "counts": result["counts"].decode("ascii")}


def native_rle(rle):
    return {"size": list(rle["size"]), "counts": rle["counts"].encode("ascii") if isinstance(rle["counts"], str) else rle["counts"]}


def decode_mask(rle):
    return coco_mask.decode(native_rle(rle))


def polygon_rle(annotation, height, width):
    rles = coco_mask.frPyObjects(annotation["segmentation"], height, width)
    return coco_mask.merge(rles)


def validate_instances(rles, shape):
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
    tp: int = 0
    fp: int = 0
    fn: int = 0
    matched_iou_sum: float = 0.0

    def add_matrix(self, iou):
        hits = iou > 0.5
        self.tp += int(hits.sum())
        self.fp += int((~hits.any(axis=0)).sum())
        self.fn += int((~hits.any(axis=1)).sum())
        self.matched_iou_sum += float(iou[hits].sum())

    def result(self):
        denominator = self.tp + 0.5 * (self.fp + self.fn)
        return {**asdict(self), "pq": self.matched_iou_sum / denominator if denominator else 0.0,
                "sq": self.matched_iou_sum / self.tp if self.tp else 0.0,
                "rq": self.tp / denominator if denominator else 0.0}


def overlap_matrix(gt, predictions):
    if not gt or not predictions:
        return np.zeros((len(gt), len(predictions)), dtype=np.float64)
    # COCO returns prediction x GT; organizer uses GT x prediction.
    return coco_mask.iou([native_rle(r) for r in predictions], [native_rle(r) for r in gt], [0] * len(gt)).T


def score_predictions(annotation_path, split_path, prediction_path, fold):
    annotations, split, predictions = map(read_json, (annotation_path, split_path, prediction_path))
    if fold == split["reserved_holdout_fold"]:
        raise ValueError("Reserved holdout is locked during development. Release it through a documented final-selection change.")
    wanted = {row["file_name"] for row in split["observations"] if row["fold"] == fold}
    if not wanted or set(predictions) != wanted:
        raise ValueError("Predictions must contain exactly every observation in the selected fold (use [] for no detections)")
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
    return {"metric": "pooled_annotator_pq_strict_gt_0.5", "fold": fold, **counts.result(), "records": details}
