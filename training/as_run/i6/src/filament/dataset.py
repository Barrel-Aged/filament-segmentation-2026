"""CPU-only, map-style training patches and separate annotator targets.

No torch dependency: the future remote DataLoader can consume __len__/__getitem__.
Sample randomness depends on (seed, epoch, index), not worker scheduling.
"""
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
import hashlib
import json

import cv2
import numpy as np
from PIL import Image

from .common import read_json, sha256, write_json
from .metrics import decode_mask, polygon_rle
from .views import cache_views
from .view_inputs import CachedViewReader


class ObservationIndex:
    def __init__(self, raw_root="data/raw", dataset_path="manifests/dataset-v1.json",
                 split_path="manifests/split-v2.json",
                 annotations_path="data/annotations/MAGFiLO_1.0_Annotations_kaggle2026_train.json",
                 validation_fold=0):
        self.raw_root = Path(raw_root)
        self.dataset_path, self.split_path, self.annotations_path = map(Path, (dataset_path, split_path, annotations_path))
        dataset, split, annotations = map(read_json, (dataset_path, split_path, annotations_path))
        if sha256(dataset_path) != split["dataset_manifest_sha256"]:
            raise ValueError("Dataset manifest differs from frozen split provenance")
        if sha256(annotations_path) != dataset["annotation_sha256"]:
            raise ValueError("Annotation file differs from dataset provenance")
        if validation_fold not in split["development_folds"] or validation_fold == split["reserved_holdout_fold"]:
            raise ValueError("Validation fold must be a development fold; holdout remains locked")
        self.images = {row["file_name"]: row for row in dataset["images"] if row["split"] == "train"}
        self.folds = {row["file_name"]: row["fold"] for row in split["observations"]}
        if len(self.folds) != len(split["observations"]) or set(self.images) != set(self.folds):
            raise ValueError("Split must contain every physical training observation exactly once")
        groups = defaultdict(set)
        for row in split["observations"]:
            groups[row["group_id"]].add(row["fold"])
            if sorted(row["annotator_image_ids"]) != sorted(self.images[row["file_name"]]["annotator_image_ids"]):
                raise ValueError("Split annotator membership differs from dataset")
        if any(len(folds) != 1 for folds in groups.values()):
            raise ValueError("Connected observation group crosses folds")
        self.development_names = sorted(n for n, f in self.folds.items() if f in split["development_folds"])
        self.train_names = sorted(n for n in self.development_names if self.folds[n] != validation_fold)
        self.validation_names = sorted(n for n in self.development_names if self.folds[n] == validation_fold)
        self.holdout_names = sorted(n for n, f in self.folds.items() if f == split["reserved_holdout_fold"])
        if not self.train_names or not self.validation_names or set(self.development_names) & set(self.holdout_names):
            raise ValueError("Invalid development/holdout partition")
        self.records = {record["id"]: record for record in annotations["images"]}
        if len(self.records) != len(annotations["images"]):
            raise ValueError("Duplicate annotation record ID")
        self.annotations = defaultdict(list)
        for ann in annotations["annotations"]:
            if ann["image_id"] not in self.records:
                raise ValueError("Orphan annotation")
            self.annotations[ann["image_id"]].append(ann)
        for name, image in self.images.items():
            for record_id in image["annotator_image_ids"]:
                record = self.records[record_id]
                if record["file_name"] != name or (record["height"], record["width"]) != (image["height"], image["width"]):
                    raise ValueError("Annotation/image geometry mismatch")
        self._verified = set()

    def load_image(self, name):
        if name not in self.images:
            raise ValueError("Unknown training observation")
        row = self.images[name]
        path = self.raw_root / row["relative_path"]
        if name not in self._verified:
            if sha256(path) != row["sha256"]:
                raise ValueError(f"Source image changed: {name}")
            self._verified.add(name)
        with Image.open(path) as image:
            image.load()
            if image.mode != "L" or image.size != (row["width"], row["height"]):
                raise ValueError(f"Unexpected image geometry: {name}")
            return np.asarray(image).copy()

    def development_record(self, record_id):
        record = self.records[record_id]
        if record["file_name"] not in self.development_names:
            raise ValueError("Holdout target access is locked during development")
        return record


class AnnotationCache:
    """One binary union per annotator, never a union across annotators."""
    def __init__(self, index, root="data/targets"):
        self.index = index
        self.provenance = {
            "schema_version": 1, "annotations_sha256": sha256(index.annotations_path),
            "dataset_sha256": sha256(index.dataset_path), "split_sha256": sha256(index.split_path),
            "implementation_sha256": sha256(__file__),
            "rasterizer_sha256": sha256(Path(__file__).with_name("metrics.py")),
            "pycocotools": version("pycocotools"), "numpy": version("numpy"),
            "target": "union_of_instances_within_one_annotator_record",
        }
        key = hashlib.sha256(json.dumps(self.provenance, sort_keys=True).encode()).hexdigest()[:20]
        self.root = Path(root) / key

    def load(self, record_id):
        record = self.index.development_record(record_id)
        key = hashlib.sha256(record_id.encode()).hexdigest()[:24]
        folder = self.root / key
        path, metadata_path = folder / "mask.npz", folder / "provenance.json"
        if metadata_path.exists():
            metadata = read_json(metadata_path)
            if metadata["record_id"] != record_id or metadata["provenance"] != self.provenance:
                raise ValueError("Annotation cache provenance mismatch")
            if sha256(path) != metadata["mask_sha256"]:
                raise ValueError("Corrupt annotation cache")
            with np.load(path, allow_pickle=False) as arrays:
                mask = arrays["mask"]
            if mask.dtype != np.uint8 or mask.shape != (record["height"], record["width"]) or not np.isin(mask, [0, 1]).all():
                raise ValueError("Invalid cached binary target")
            return mask, metadata
        mask = np.zeros((record["height"], record["width"]), dtype=np.uint8)
        instances = []
        for ann in sorted(self.index.annotations[record_id], key=lambda a: a["id"]):
            polygons = ann["segmentation"]
            if ann.get("iscrowd", 0) or not isinstance(polygons, list) or not polygons:
                raise ValueError(f"Unsupported annotation format: {ann['id']}")
            for polygon in polygons:
                coordinates = np.asarray(polygon)
                if len(coordinates) < 6 or len(coordinates) % 2 or not np.isfinite(coordinates).all():
                    raise ValueError(f"Malformed polygon: {ann['id']}")
            instance = decode_mask(polygon_rle(ann, record["height"], record["width"]))
            if not instance.any():
                raise ValueError(f"Empty rasterized annotation: {ann['id']}")
            mask |= instance
            instances.append({"id": ann["id"], "raster_area": int(instance.sum())})
        folder.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".partial")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, mask=mask)
        temporary.replace(path)
        metadata = {"provenance": self.provenance, "record_id": record_id, "file_name": record["file_name"],
                    "mask_sha256": sha256(path), "foreground_pixels": int(mask.sum()), "instances": instances}
        write_json(metadata_path, metadata)
        return mask, metadata


def estimate_disk(image):
    """Estimate a sampling region only. Never use this as a prediction veto."""
    height, width = image.shape
    small = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    threshold = max(5.0, float(np.percentile(small, 95)) * 0.1)
    binary = (small > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Cannot locate solar disk")
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 0.2 * small.size:
        raise ValueError("Solar-disk estimate too small; inspect image")
    (x, y), radius = cv2.minEnclosingCircle(contour)
    return {"center_x": x * width / 256, "center_y": y * height / 256,
            "radius": radius * min(height, width) / 256}


def aligned_geometry(array, rotations, flip):
    value = np.rot90(array, rotations, axes=(-2, -1))
    if flip:
        value = np.flip(value, axis=-1)
    return np.ascontiguousarray(value)


def crop_origin(y, x, shape, patch_size, rng):
    height, width = shape
    if not 0 <= y < height or not 0 <= x < width or not 1 <= patch_size <= min(shape):
        raise ValueError("Invalid patch geometry")
    top = int(rng.integers(max(0, y - patch_size + 1), min(y, height - patch_size) + 1))
    left = int(rng.integers(max(0, x - patch_size + 1), min(x, width - patch_size) + 1))
    return top, left


class TrainingPatches:
    def __init__(self, index, channels=("raw",), patch_size=512, samples_per_epoch=1024,
                 seed=20261014, foreground_probability=0.5, augment=True,
                 target_root="data/targets", view_root="data/derived", view_config="configs/views-v1.json"):
        if not channels or channels[0] != "raw" or len(set(channels)) != len(channels):
            raise ValueError("Unique ordered channels must start with raw")
        if not 0 <= foreground_probability <= 1 or patch_size < 1 or samples_per_epoch < 1 or seed < 0:
            raise ValueError("Invalid sampling configuration")
        self.index, self.channels, self.patch_size = index, tuple(channels), patch_size
        self.samples_per_epoch, self.seed, self.epoch = samples_per_epoch, seed, 0
        self.foreground_probability, self.augment = foreground_probability, augment
        self.targets = AnnotationCache(index, target_root)
        self.view_root, self.view_config = Path(view_root), Path(view_config)
        self.view_reader = CachedViewReader(channels, self.view_config, self.view_root) if len(channels) > 1 else None
        if set(channels) - set(read_json(view_config)["views"]):
            raise ValueError("Unknown channel in view configuration")

    def __len__(self):
        return self.samples_per_epoch

    def set_epoch(self, epoch):
        if epoch < 0:
            raise ValueError("Epoch must be nonnegative")
        self.epoch = epoch

    def __getitem__(self, item):
        if not 0 <= item < len(self):
            raise IndexError(item)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, item]))
        name = self.index.train_names[int(rng.integers(len(self.index.train_names)))]
        image_row = self.index.images[name]
        record_ids = sorted(image_row["annotator_image_ids"])
        record_id = record_ids[int(rng.integers(len(record_ids)))]
        image = self.index.load_image(name)
        target, target_metadata = self.targets.load(record_id)
        annotations = self.index.annotations[record_id]
        foreground = bool(rng.random() < self.foreground_probability and annotations)
        selected_id = None
        if foreground:
            annotation = sorted(annotations, key=lambda ann: ann["id"])[int(rng.integers(len(annotations)))]
            selected_id = annotation["id"]
            selected = decode_mask(polygon_rle(annotation, *image.shape))
            pixels = np.flatnonzero(selected)
            if not len(pixels):
                raise ValueError("Selected foreground annotation is empty")
            y, x = np.unravel_index(int(pixels[int(rng.integers(len(pixels)))]), image.shape)
        else:
            disk = estimate_disk(image)
            angle, radius = rng.uniform(0, 2 * np.pi), np.sqrt(rng.random()) * disk["radius"] * 0.98
            y = int(np.clip(round(disk["center_y"] + radius * np.sin(angle)), 0, image.shape[0] - 1))
            x = int(np.clip(round(disk["center_x"] + radius * np.cos(angle)), 0, image.shape[1] - 1))
        top, left = crop_origin(int(y), int(x), image.shape, self.patch_size, rng)
        region = np.s_[top:top + self.patch_size, left:left + self.patch_size]
        channel_arrays = {"raw": image.astype(np.float32) / 255}
        if len(self.channels) > 1:
            channel_arrays = self.view_reader.load(self.index.raw_root / image_row["relative_path"])
        inputs = np.stack([channel_arrays[channel][region] for channel in self.channels])
        target_patch = target[region][None].astype(np.float32)
        rotations = int(rng.integers(4)) if self.augment else 0
        flip = bool(rng.integers(2)) if self.augment else False
        inputs, target_patch = (aligned_geometry(value, rotations, flip) for value in (inputs, target_patch))
        return {"image": inputs, "target": target_patch, "metadata": {
            "file_name": name, "annotator_image_id": record_id, "fold": self.index.folds[name],
            "channels": list(self.channels), "seed": self.seed, "epoch": self.epoch, "sample_index": item,
            "top": top, "left": left, "patch_size": self.patch_size, "rotations_90": rotations,
            "horizontal_flip": flip, "sampling": "foreground" if foreground else "random_disk",
            "selected_instance_id": selected_id, "source_sha256": image_row["sha256"],
            "target_sha256": target_metadata["mask_sha256"],
        }}
