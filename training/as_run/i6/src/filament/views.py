"""Deterministic CPU views. Values are float32 in [0,1]; raw is always retained."""
from importlib.metadata import version
from pathlib import Path
import hashlib

import cv2
import numpy as np
from PIL import Image
from skimage.filters import sato

from .common import read_json, sha256, write_json


def make_views(image, config):
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("Expected 8-bit grayscale image")
    raw = image.astype(np.float32) / 255
    result = {}
    for name, params in config["views"].items():
        if name == "raw":
            value = raw
        elif name == "clahe":
            grid = params["tile_grid"]
            value = cv2.createCLAHE(clipLimit=params["clip_limit"], tileGridSize=(grid, grid)).apply(image) / 255
        elif name == "background_ratio":
            background = cv2.GaussianBlur(raw, (0, 0), params["sigma"])
            value = np.clip(raw / np.maximum(background, params["epsilon"]), 0, params["ratio_max"]) / params["ratio_max"]
        elif name == "gaussian":
            value = cv2.GaussianBlur(raw, (0, 0), params["sigma"])
        elif name == "dark_ridges":
            value = sato(raw, sigmas=params["sigmas"], black_ridges=True).astype(np.float32)
            value = value / max(float(value.max()), 1e-8)
        elif name == "blackhat":
            size = params["kernel_size"]
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            value = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel) / 255
        elif name == "adaptive_dark":
            value = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                          params["block_size"], params["offset"]) / 255
        else:
            raise ValueError(f"Unknown view: {name}")
        value = np.asarray(value, dtype=np.float32)
        if value.shape != raw.shape or not np.isfinite(value).all() or value.min() < 0 or value.max() > 1:
            raise ValueError(f"Invalid output from {name}")
        result[name] = value
    if "raw" not in result:
        raise ValueError("A view bank must retain raw input")
    return result


def cache_views(image_path, config_path, output, view_names=None):
    config = read_json(config_path)
    if view_names is not None:
        if not view_names or view_names[0] != "raw" or len(set(view_names)) != len(view_names):
            raise ValueError("Selected views must be unique and start with raw")
        config = {**config, "views": {name: config["views"][name] for name in view_names}}
    packages = {name: version(name) for name in ("numpy", "pillow", "opencv-python-headless", "scikit-image")}
    provenance = {"source_file": Path(image_path).name, "source_sha256": sha256(image_path),
                  "configuration": config, "configuration_sha256": sha256(config_path),
                  "implementation_sha256": sha256(__file__), "packages": packages,
                  "normalization": "image-local fixed operations; no population fitting",
                  "geometry": "full source resolution, no resizing or cropping"}
    import json
    key = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()[:20]
    destination = Path(output) / Path(image_path).stem / key
    metadata_path = destination / "provenance.json"
    if metadata_path.exists():
        existing = read_json(metadata_path)
        if existing["provenance"] != provenance:
            raise ValueError("Cache provenance mismatch")
        for name, digest in existing["output_sha256"].items():
            if sha256(destination / f"{name}.npy") != digest:
                raise ValueError(f"Corrupt cached view: {name}")
        return destination
    with Image.open(image_path) as image:
        views = make_views(np.asarray(image.convert("L")), config)
    destination.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, value in views.items():
        path = destination / f"{name}.npy"
        np.save(path, value, allow_pickle=False)
        hashes[name] = sha256(path)
    write_json(metadata_path, {"provenance": provenance, "output_sha256": hashes})
    return destination
