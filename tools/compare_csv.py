"""Compare two submission CSVs instance by instance.

usage: python -m tools.compare_csv <reference.csv> <candidate.csv>
Per image, instances are matched one-to-one in descending IoU (only pairs with IoU > 0.5 count, as in the metric).
Prints identical / matched / unmatched counts and the IoU distribution of the matched pairs that are not identical.
"""
import csv
import sys
from collections import defaultdict

import numpy as np
from pycocotools import mask as mu

H = W = 2048


def read(path):
    by_image = defaultdict(list)
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            stem = row["filament_id"].rsplit("_", 1)[0]
            by_image[stem].append({"size": [H, W], "counts": row["segmentation_rle"].encode()})
    return by_image


def match(ref, cand):
    """(identical, matched pairs' IoU list, unmatched ref, unmatched cand) for one image."""
    if not ref or not cand:
        return 0, [], len(ref), len(cand)
    iou = mu.iou(ref, cand, [0] * len(cand))
    ident = sum(1 for i in range(len(ref)) for j in range(len(cand)) if ref[i]["counts"] == cand[j]["counts"])
    pairs = sorted(((iou[i, j], i, j) for i in range(len(ref)) for j in range(len(cand)) if iou[i, j] > 0.5), reverse=True)
    used_r, used_c, ious = set(), set(), []
    for v, i, j in pairs:
        if i not in used_r and j not in used_c:
            used_r.add(i); used_c.add(j); ious.append(float(v))
    return ident, ious, len(ref) - len(used_r), len(cand) - len(used_c)


def main(ref_path, cand_path):
    ref, cand = read(ref_path), read(cand_path)
    tot = dict(ref=0, cand=0, identical=0, matched=0, ref_only=0, cand_only=0, same_images=0)
    changed = []
    stems = sorted(set(ref) | set(cand))
    for s in stems:
        r, c = ref.get(s, []), cand.get(s, [])
        ident, ious, ro, co = match(r, c)
        tot["ref"] += len(r); tot["cand"] += len(c); tot["identical"] += ident
        tot["matched"] += len(ious); tot["ref_only"] += ro; tot["cand_only"] += co
        tot["same_images"] += ident == len(r) == len(c)
        changed += [v for v in ious if v < 1.0]
    print(f"images {len(stems)} | instances ref {tot['ref']} cand {tot['cand']} | pixel-identical {tot['identical']} | "
          f"matched IoU>0.5 {tot['matched']} | only in ref {tot['ref_only']} | only in cand {tot['cand_only']} | "
          f"fully identical images {tot['same_images']}")
    if changed:
        a = np.array(changed)
        print(f"matched but changed: {len(a)} | IoU mean {a.mean():.4f} median {np.median(a):.4f} min {a.min():.4f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
