"""Framework-free full-resolution tiling and disjoint component conversion."""
import cv2
import numpy as np

from .metrics import encode_mask


def tile_starts(length, tile, overlap):
    if length < tile or not 0 <= overlap < tile:
        raise ValueError("Invalid tile geometry")
    starts = list(range(0, length - tile + 1, tile - overlap))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def tiled_probability(image, predict, tile=512, overlap=128, batch_size=2):
    """C,H,W input; predictor consumes N,C,tile,tile and returns N,1,tile,tile."""
    if image.ndim != 3 or batch_size < 1:
        raise ValueError("Expected C,H,W and positive batch size")
    height, width = image.shape[-2:]
    locations = [(y, x) for y in tile_starts(height, tile, overlap)
                 for x in tile_starts(width, tile, overlap)]
    hann = np.maximum(np.hanning(tile).astype(np.float32), 0.001)
    weight = hann[:, None] * hann[None, :]
    total, denominator = np.zeros((height, width), np.float32), np.zeros((height, width), np.float32)
    for start in range(0, len(locations), batch_size):
        batch_locations = locations[start:start + batch_size]
        batch = np.stack([image[:, y:y + tile, x:x + tile] for y, x in batch_locations])
        probability = np.asarray(predict(batch), dtype=np.float32)
        if probability.shape != (len(batch_locations), 1, tile, tile):
            raise ValueError("Predictor returned wrong tile shape")
        if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
            raise ValueError("Invalid predicted probability")
        for value, (y, x) in zip(probability[:, 0], batch_locations):
            total[y:y + tile, x:x + tile] += value * weight
            denominator[y:y + tile, x:x + tile] += weight
    if not np.all(denominator > 0):
        raise ValueError("Uncovered pixels")
    return np.clip(total / denominator, 0, 1)


def component_instances(probability, threshold, minimum_area=1):
    if probability.ndim != 2 or not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Invalid probability map")
    if not 0 <= threshold <= 1 or minimum_area < 1:
        raise ValueError("Invalid component parameters")
    count, labels, stats, _ = cv2.connectedComponentsWithStats((probability >= threshold).astype(np.uint8), connectivity=8)
    selected = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= minimum_area]
    return [encode_mask((labels == i).astype(np.uint8)) for i in selected], [int(stats[i, cv2.CC_STAT_AREA]) for i in selected]
