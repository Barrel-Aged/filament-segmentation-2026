"""G6 CPU-only object quality and recall relief. No annotation features."""
import math
import cv2
import numpy as np
from scipy.special import expit
from pycocotools import mask as coco
from .dataset import estimate_disk
from .g4_data import roi_bounds, selected_component
from .g4_assembly_fast3 import assemble, encode_mask
from .metrics import native_rle, decode_mask

FEATURES = [
    "logit_confidence", "logit_detector", "logit_mask_mean", "log_area",
    "log_width", "log_height", "log_aspect", "fill", "rho", "rho_squared",
    "raw_mean", "raw_std", "ring_contrast", "contrast_z", "mask_std",
    "mask_q10", "stable_fraction", "perimeter_over_sqrt_area",
    "small", "limb", "confidence_x_small", "confidence_x_limb",
    "contrast_x_limb", "area_squared",
]


def logit(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def candidates(raw, boxes, scores, probabilities, threshold):
    """Image-local features use predicted masks only; retain raw pixels."""
    geometry = estimate_disk(raw)
    result = []
    for i, (box, detector, probability) in enumerate(zip(boxes, scores, probabilities)):
        top, left, side = roi_bounds(box, raw.shape)
        p = cv2.resize(probability.astype(np.float32), (side, side), interpolation=cv2.INTER_LINEAR)
        hint = np.zeros((side, side), np.uint8)
        x1, y1, x2, y2 = box
        hint[max(0, int(np.floor(y1-top))):min(side, int(np.ceil(y2-top))),
             max(0, int(np.floor(x1-left))):min(side, int(np.ceil(x2-left)))] = 1
        mask = selected_component(p >= threshold, hint)
        area = int(mask.sum())
        if not area:
            continue
        yy, xx = np.nonzero(mask)
        w, h = int(xx.max()-xx.min()+1), int(yy.max()-yy.min()+1)
        rho = math.hypot(left + (xx.min()+xx.max()+1)/2 - geometry["center_x"],
                         top + (yy.min()+yy.max()+1)/2 - geometry["center_y"]) / geometry["radius"]
        crop = raw[top:top+side, left:left+side].astype(np.float32)/255
        ring = cv2.dilate(mask.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool) & ~mask
        inside, outside = crop[mask], crop[ring]
        contrast = float(np.median(outside)-inside.mean()) if len(outside) else 0.
        z = contrast/max(float(outside.std()), .02) if len(outside) else 0.
        mp = p[mask]
        confidence = float(detector)*float(mp.mean())
        lc = float(logit(confidence))
        boundary = mask & ~cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        small, limb = float(area < 256), float(rho >= .8)
        la = math.log1p(area)
        features = [lc, float(logit(detector)), float(logit(mp.mean())), la,
                    math.log1p(w), math.log1p(h), math.log(w/h), area/(w*h),
                    rho, rho*rho, float(inside.mean()), float(inside.std()), contrast,
                    float(np.clip(z, -10, 10)), float(mp.std()), float(np.quantile(mp, .1)),
                    float((mp >= min(.95, threshold+.1)).mean()),
                    float(boundary.sum()/math.sqrt(area)), small, limb, lc*small, lc*limb,
                    contrast*limb, la*la]
        full = np.zeros(raw.shape, bool)
        full[top:top+side, left:left+side] = mask
        result.append({"index": i, "score": confidence, "area": area, "rho": rho,
                       "features": features, "rle": encode_mask(full)})
    return result


def fit_logistic(x, y, weights, columns, penalty=.01):
    """Fixed ridge logistic fit with soft per-annotator targets and Newton solve."""
    x, y, weights = np.asarray(x, float)[:, columns], np.asarray(y, float), np.asarray(weights, float)
    assert len(x) and np.isfinite(x).all() and np.all((y >= 0) & (y <= 1))
    weights = weights/weights.sum()
    mean = np.average(x, axis=0, weights=weights)
    scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=weights))
    scale = np.maximum(scale, 1e-6)
    a = np.column_stack([np.ones(len(x)), np.clip((x-mean)/scale, -10, 10)])
    coef = np.zeros(a.shape[1])
    regularizer = np.r_[0., np.full(a.shape[1]-1, penalty)]
    def objective(v):
        z = a@v
        return float(np.sum(weights*(np.logaddexp(0, z)-y*z)) + .5*np.sum(regularizer*v*v))
    history = []
    for iteration in range(100):
        q = expit(a@coef)
        grad = a.T@(weights*(q-y))+regularizer*coef
        hessian = (a.T*(weights*q*(1-q)))@a+np.diag(regularizer+1e-10)
        direction = np.linalg.solve(hessian, grad)
        previous = objective(coef)
        rate = 1.
        while rate > 2**-20 and objective(coef-rate*direction) > previous:
            rate *= .5
        coef -= rate*direction
        history.append(objective(coef))
        if np.max(np.abs(grad)) < 1e-8:
            break
    assert np.max(np.abs(a.T@(weights*(expit(a@coef)-y))+regularizer*coef)) < 1e-6
    return {"columns": list(columns), "mean": mean.tolist(), "scale": scale.tolist(),
            "coef": coef.tolist(), "penalty": penalty, "iterations": iteration+1,
            "objective": history, "examples": len(x), "weighted_target": float(weights@y)}


def predict(model, features):
    if not len(features):
        return np.zeros(0)
    x = np.asarray(features, float)[:, model["columns"]]
    a = np.column_stack([np.ones(len(x)), np.clip((x-model["mean"])/model["scale"], -10, 10)])
    return expit(a@np.asarray(model["coef"]))


def decode(rows, quality, threshold, relief=False):
    """Relief changes eligibility only; quality still orders mask ownership."""
    accepted = []
    for row, q in zip(rows, quality):
        cut = max(0., threshold-.15) if relief and (row["area"] < 256 or row["rho"] >= .8) else threshold
        if q >= cut:
            if "_assembly" not in row:
                row["_assembly"] = {"index": row["index"], "area": row["area"],
                                    "mask": decode_mask(row["rle"]).astype(bool)}
            row["_assembly"]["score"] = float(q)
            accepted.append(row["_assembly"])
    return assemble(accepted, 0., 64)[0]


def targets(index, name, rows):
    """Expected supplied-annotation match, not verified physical-defect labels."""
    from .evaluation import ground_truth
    from .metrics import overlap_matrix
    labels = []
    for rid in sorted(index.images[name]["annotator_image_ids"]):
        iou = overlap_matrix(ground_truth(index, rid), [r["rle"] for r in rows])
        labels.append((iou > .5).any(axis=0).astype(float))
    return np.mean(labels, axis=0).tolist() if labels else [0.]*len(rows)


def select_inner(options, reference):
    """Recall constraints apply to all three strata; fallback is explicitly failed."""
    limits = {"small": .05, "large": .03, "limb": .03}
    for row in options:
        row["recall_feasible"] = all(row["recall"][k] >= reference[k]-drop-1e-12
                                     for k, drop in limits.items())
        row["total_violation"] = sum(max(0., reference[k]-drop-row["recall"][k]) for k, drop in limits.items())
    feasible = [r for r in options if r["recall_feasible"]]
    if feasible:
        return min(feasible, key=lambda r: (-r["pq"], r["threshold"]))
    return min(options, key=lambda r: (r["total_violation"], -r["pq"], r["threshold"]))
