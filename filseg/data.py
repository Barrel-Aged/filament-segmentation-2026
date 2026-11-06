"""Test images (stage 1): the 180 competition JPEGs as uint8 arrays, plus the disk statistics the dense models need.

The images are read in ``sorted(os.listdir(test_images))`` order with OpenCV as 2048 x 2048 grayscale uint8 arrays. For
each image ``disk_stats`` gives the mean and standard deviation of the solar-disk pixels (those above 30 % of the 99th
percentile) and that threshold; ``filseg.dense`` normalises the image with the mean and std.

``write_test_cache`` writes the same two files the original run used: ``test_img.npy`` (n, 2048, 2048) uint8 and a
``test_meta.json`` that holds the ``test`` records of the original ``cache/meta.json`` (``file_name``, ``disk_mean``,
``disk_std``, ``disk_thr``).

Changes from the original run
-----------------------------
The arithmetic is that of ``src/cfil/prep.py`` (``disk_stats`` L18-22, the test half of ``main`` L74-81, L85).
- Test half only. The training arrays, soft and spine targets, folds and crop centres (prep.py L25-73, L83-84,
  L88-96) are training inputs; they are in ``training/prep_train.py``.
- The image folder is an argument (default ``filseg.paths.TEST_IMAGES``) instead of
  ``<root>/data/MAGFiLO_1.0_Kaggle_2026/test/test_images`` built from ``sys.argv[1]`` (prep.py L55, L100-101).
- The test records are written to ``test_meta.json`` instead of the ``test`` key of ``meta.json`` (prep.py L82-87),
  and ``load_test`` can return the arrays in memory without writing a cache.
- A decoded image that is missing or not 2048 x 2048 raises ``ValueError``; prep.py L78-79 stored whatever OpenCV
  returned (the training half asserted the shape, L43).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np

from filseg import paths

H = W = 2048


def disk_stats(img: np.ndarray) -> tuple[float, float, float]:
    """(mean, std + 1e-6, threshold) of the pixels above 0.3 x the 99th percentile of a uint8 image (prep.py L18-22)."""
    thr = 0.3 * np.percentile(img, 99)
    m = img > thr
    v = img[m].astype(np.float32)
    return float(v.mean()), float(v.std() + 1e-6), float(thr)


def test_files(image_dir=None) -> list[str]:
    """The test image file names in the order the original run used (``sorted(os.listdir(...))``, prep.py L74)."""
    return sorted(os.listdir(str(image_dir or paths.TEST_IMAGES)))


def read_image(image_dir, fname) -> np.ndarray:
    """One test JPEG as a uint8 (2048, 2048) array, decoded by ``cv2.imread(..., IMREAD_GRAYSCALE)`` (prep.py L78)."""
    img = cv2.imread(os.path.join(str(image_dir), fname), cv2.IMREAD_GRAYSCALE)
    if img is None or img.shape != (H, W):
        raise ValueError(f'cannot read a 2048 x 2048 image: {fname}')
    return img


def load_test(image_dir=None, limit: int | None = None):
    """(records, images): the test records (``file_name``, ``disk_mean``, ``disk_std``, ``disk_thr``) and a uint8
    (n, 2048, 2048) array, both in ``test_files`` order. ``limit`` keeps the first n images."""
    image_dir = image_dir or paths.TEST_IMAGES
    names = test_files(image_dir)[:limit]
    images = np.empty((len(names), H, W), np.uint8)
    recs = []
    for i, n in enumerate(names):
        img = read_image(image_dir, n)
        images[i] = img
        s = disk_stats(img)
        recs.append(dict(file_name=n, disk_mean=s[0], disk_std=s[1], disk_thr=s[2]))
    return recs, images


def write_test_cache(out_dir=None, image_dir=None, limit: int | None = None):
    """Write ``<out_dir>/test_img.npy`` and ``<out_dir>/test_meta.json`` (default out_dir ``<work>/cache``).

    Returns (path of test_img.npy, records). Existing files are overwritten.
    """
    out_dir = Path(out_dir or paths.WORK / 'cache')
    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = image_dir or paths.TEST_IMAGES
    names = test_files(image_dir)[:limit]
    tm = np.lib.format.open_memmap(str(out_dir / 'test_img.npy'), 'w+', np.uint8, (len(names), H, W))
    recs = []
    for i, n in enumerate(names):
        img = read_image(image_dir, n)
        tm[i] = img
        s = disk_stats(img)
        recs.append(dict(file_name=n, disk_mean=s[0], disk_std=s[1], disk_thr=s[2]))
    tm.flush()
    del tm
    with open(out_dir / 'test_meta.json', 'w', encoding='utf-8') as fh:
        json.dump({'test': recs}, fh, indent=1)
    return out_dir / 'test_img.npy', recs


def open_test_cache(cache_dir=None):
    """(records, images memmap) written by ``write_test_cache``."""
    cache_dir = Path(cache_dir or paths.WORK / 'cache')
    with open(cache_dir / 'test_meta.json', encoding='utf-8') as fh:
        recs = json.load(fh)['test']
    return recs, np.load(cache_dir / 'test_img.npy', mmap_mode='r')
