"""CPU-only development calibration, pooled PQ, diagnostics and error gallery."""
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import json
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as coco_mask

from .common import read_json, sha256, write_json
from .dataset import ObservationIndex
from .metrics import PQCounts, decode_mask, native_rle, overlap_matrix, polygon_rle, score_predictions
from .tiling import component_instances


def pool_records(records):
    counts = PQCounts()
    for row in records:
        counts.tp += row["tp"]
        counts.fp += row["fp"]
        counts.fn += row["fn"]
        counts.matched_iou_sum += row["matched_iou_sum"]
    return counts.result()


def select_grid(rows):
    return min(rows, key=lambda r: (-r["pq"], r["minimum_area"], abs(r["threshold"] - 0.5), r["threshold"]))


def bootstrap_groups(records, group_by_name, samples, seed):
    grouped = defaultdict(list)
    for row in records:
        grouped[group_by_name[row["file_name"]]].append(row)
    arrays = np.array([[result[k] for k in ("tp", "fp", "fn", "matched_iou_sum")]
                       for result in (pool_records(grouped[k]) for k in sorted(grouped))])
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        tp, fp, fn, matched = arrays[rng.integers(len(arrays), size=len(arrays))].sum(axis=0)
        denominator = tp + 0.5 * (fp + fn)
        values.append(matched / denominator if denominator else 0.0)
    return {"unit": "connected_observation_group", "groups": len(grouped), "samples": samples, "seed": seed,
            "pq_percentile_95": np.quantile(values, [0.025, 0.975]).tolist(),
            "limitation": "conditional on fitted checkpoint and selected calibration; excludes selection/training uncertainty"}


def ground_truth(index, record_id):
    record = index.development_record(record_id)
    return [polygon_rle(a, record["height"], record["width"]) for a in index.annotations[record_id]]


def diagnostics(index, predictions, records):
    by_site, by_year = defaultdict(list), defaultdict(list)
    size_bins = {k: {"ground_truth_instances": 0, "matched_instances": 0} for k in ("under_256", "256_to_4095", "4096_plus")}
    details = []
    for row in records:
        name, rid = row["file_name"], row["annotator_image_id"]
        by_site[index.images[name]["site"]].append(row)
        by_year[str(index.images[name]["year"])].append(row)
        gt, pred = ground_truth(index, rid), predictions[name]
        iou = overlap_matrix(gt, pred)
        gt_area = np.array([int(coco_mask.area(native_rle(r))) for r in gt])
        pred_area = np.array([int(coco_mask.area(native_rle(r))) for r in pred])
        gt_hit, pred_hit = (iou > 0.5).any(axis=1), (iou > 0.5).any(axis=0)
        for area, hit in zip(gt_area, gt_hit):
            key = "under_256" if area < 256 else "256_to_4095" if area < 4096 else "4096_plus"
            size_bins[key]["ground_truth_instances"] += 1
            size_bins[key]["matched_instances"] += int(hit)
        intersection = iou * (gt_area[:, None] + pred_area[None, :]) / (1 + iou)
        # A diagnostic flag only: >=10% of a GT's area shared with each of 2+ predictions/GTs.
        overlap = intersection / np.maximum(gt_area[:, None], 1) >= 0.1
        details.append({**row, "ground_truth_instances": len(gt), "prediction_instances": len(pred),
                        "split_candidates": int((overlap.sum(axis=1) >= 2).sum()),
                        "merge_candidates": int((overlap.sum(axis=0) >= 2).sum()),
                        "largest_missed_area": int(gt_area[~gt_hit].max()) if (~gt_hit).any() else 0,
                        "largest_unmatched_prediction_area": int(pred_area[~pred_hit].max()) if (~pred_hit).any() else 0})
    for row in size_bins.values():
        row["recall"] = row["matched_instances"] / row["ground_truth_instances"] if row["ground_truth_instances"] else None
    return {"by_site": {k: {"records": len(v), **pool_records(v)} for k, v in sorted(by_site.items())},
            "by_year": {k: {"records": len(v), **pool_records(v)} for k, v in sorted(by_year.items())},
            "size_recall": size_bins, "records": details,
            "split_candidates": sum(r["split_candidates"] for r in details),
            "merge_candidates": sum(r["merge_candidates"] for r in details),
            "candidate_definition": "within an annotator record, at least two overlaps each covering >=10% of GT area; not verified split/merge labels"}


def error_gallery(index, predictions, diagnostic, output, experiment_id="E001"):
    rows = diagnostic["records"]
    selected, seen = [], set()
    rankings = [("Largest missed annotation", sorted(rows, key=lambda r: -r["largest_missed_area"])),
                ("Largest unmatched prediction", sorted(rows, key=lambda r: -r["largest_unmatched_prediction_area"])),
                ("Most possible fragmentation", sorted(rows, key=lambda r: -r["split_candidates"])),
                ("Most possible merging", sorted(rows, key=lambda r: -r["merge_candidates"])),
                ("Lowest record PQ", sorted(rows, key=lambda r: r["pq"])),
                ("Highest record PQ", sorted(rows, key=lambda r: -r["pq"]))]
    for title, ranking in rankings:
        row = next((r for r in ranking if r["file_name"] not in seen), ranking[0])
        selected.append((title, row))
        seen.add(row["file_name"])
    canvas = Image.new("RGB", (1500, 105 + len(selected) * 380), "#f5f7fa")
    draw = ImageDraw.Draw(canvas)
    fold = index.folds[index.validation_names[0]]
    draw.text((16, 12), f"{experiment_id} / fold {fold}: development error examples selected by metric extremes", fill="#111827", font_size=23)
    draw.text((16, 44), "Raw | supplied annotation | prediction | cropped overlay: green GT, magenta prediction, white overlap", fill="#475569", font_size=18)
    cases = []
    for i, (title, row) in enumerate(selected):
        name, rid = row["file_name"], row["annotator_image_id"]
        raw = index.load_image(name)
        gt_rles, pred_rles = ground_truth(index, rid), predictions[name]
        gt = np.zeros_like(raw, dtype=bool)
        pred = np.zeros_like(raw, dtype=bool)
        for rle in gt_rles:
            gt |= decode_mask(rle).astype(bool)
        for rle in pred_rles:
            pred |= decode_mask(rle).astype(bool)
        overlay = np.repeat(raw[..., None], 3, axis=2).astype(np.float32)
        for mask, color in ((gt & ~pred, (0, 255, 80)), (pred & ~gt, (255, 0, 180)), (pred & gt, (255, 255, 255))):
            overlay[mask] = 0.4 * overlay[mask] + 0.6 * np.array(color)
        # Focus on the largest connected error region; fall back to all foreground.
        _, labels, stats, centroids = cv2.connectedComponentsWithStats((gt ^ pred).astype(np.uint8), connectivity=8)
        if len(stats) > 1:
            x, y = centroids[1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])]
        else:
            y, x = np.array(raw.shape) / 2
        top, left = int(np.clip(y - 256, 0, raw.shape[0] - 512)), int(np.clip(x - 256, 0, raw.shape[1] - 512))
        zoom = overlay[top:top + 512, left:left + 512].astype(np.uint8)
        y0 = 88 + 380 * i
        draw.text((16, y0), f"{title}: {name} / {rid.split('-')[0]} / PQ {row['pq']:.3f}, TP {row['tp']}, FP {row['fp']}, FN {row['fn']}", fill="#111827", font_size=17)
        panels = (raw, gt.astype(np.uint8) * 255, pred.astype(np.uint8) * 255, zoom)
        for j, panel in enumerate(panels):
            resample = Image.Resampling.NEAREST if j in (1, 2) else Image.Resampling.BILINEAR
            canvas.paste(Image.fromarray(panel).convert("RGB").resize((340, 340), resample), (16 + j * 370, y0 + 28))
        cases.append({"selection": title, **row, "zoom_top": top, "zoom_left": left, "zoom_size": 512})
    canvas.save(output / "error-gallery.png")
    write_json(output / "error-cases.json", cases)


def evaluate(run_directory):
    directory = Path(run_directory)
    config, run, inference = (read_json(directory / name) for name in ("config.json", "run.json", "inference/manifest.json"))
    if run["status"] != "complete" or inference["status"] != "complete" or config["mode"] != "baseline":
        raise ValueError("Complete baseline and inference required")
    if inference["checkpoint_sha256"] != read_json(directory / "checkpoint.json")["sha256"]:
        raise ValueError("Inference/checkpoint mismatch")
    if inference["split_sha256"] != sha256("manifests/split-v2.json"):
        raise ValueError("Split changed")
    index = ObservationIndex(validation_fold=config["validation_fold"])
    if sorted(r["file_name"] for r in inference["records"]) != index.validation_names:
        raise ValueError("Inference must contain exactly validation images")
    output = directory / "evaluation"
    if output.exists():
        raise ValueError("Evaluation exists; preserve the original calibration")
    output.mkdir(parents=True)
    started = time.perf_counter()
    cv2.setNumThreads(2)
    options = config["calibration"]
    grid = {(threshold, area): [] for threshold in options["thresholds"] for area in options["minimum_areas"]}
    candidates = {threshold: {} for threshold in options["thresholds"]}
    for i, row in enumerate(inference["records"]):
        name = row["file_name"]
        path = directory / "inference" / row["path"]
        if sha256(path) != row["sha256"] or row["source_sha256"] != index.images[name]["sha256"]:
            raise ValueError("Probability/source checksum mismatch")
        with np.load(path, allow_pickle=False) as arrays:
            probability = arrays["probability"]
        if probability.shape != (index.images[name]["height"], index.images[name]["width"]) or probability.dtype != np.float32:
            raise ValueError("Probability geometry/dtype mismatch")
        records = [(rid, ground_truth(index, rid)) for rid in sorted(index.images[name]["annotator_image_ids"])]
        for threshold in options["thresholds"]:
            pred, areas = component_instances(probability, threshold, min(options["minimum_areas"]))
            candidates[threshold][name] = (pred, areas)
            for rid, gt in records:
                matrix = overlap_matrix(gt, pred)
                for area in options["minimum_areas"]:
                    counts = PQCounts()
                    counts.add_matrix(matrix[:, np.asarray(areas) >= area])
                    grid[threshold, area].append({"file_name": name, "annotator_image_id": rid, **counts.result()})
        if (i + 1) % 10 == 0 or i + 1 == len(inference["records"]):
            print(json.dumps({"calibrated_images": i + 1, "total": len(inference["records"]), "elapsed_seconds": time.perf_counter() - started}), flush=True)
    summary = [{"threshold": t, "minimum_area": a, **pool_records(rows)} for (t, a), rows in grid.items()]
    chosen = select_grid(summary)
    reference = next(r for r in summary if r["threshold"] == options["fixed_reference_threshold"] and r["minimum_area"] == options["fixed_reference_minimum_area"])
    write_json(output / "calibration-grid.json", {"predeclared_config": options, "grid": summary, "selected": chosen, "fixed_reference": reference})
    selected_score = None
    for label, setting in (("selected", chosen), ("fixed-reference", reference)):
        predictions = {name: [rle for rle, area in zip(pred, areas) if area >= setting["minimum_area"]]
                       for name, (pred, areas) in candidates[setting["threshold"]].items()}
        path = output / f"{label}-predictions.json"
        write_json(path, predictions)
        # Independent scorer path verifies disjointness, RLE, exact membership and all pooled counts.
        score = score_predictions(index.annotations_path, index.split_path, path, config["validation_fold"])
        for key in ("tp", "fp", "fn", "matched_iou_sum", "pq"):
            if not np.isclose(score[key], setting[key], rtol=0, atol=1e-10):
                raise ValueError("Grid and standalone scorer disagree")
        write_json(output / f"{label}-metrics.json", score)
        if label == "selected":
            selected_score = score
            diagnostic = diagnostics(index, predictions, score["records"])
            write_json(output / "diagnostics.json", diagnostic)
            error_gallery(index, predictions, diagnostic, output, config["experiment_id"])
    groups = {r["file_name"]: r["group_id"] for r in read_json(index.split_path)["observations"]}
    result = {"status": "complete", "utc": datetime.now(timezone.utc).isoformat(), "experiment_id": config["experiment_id"],
              "validation_fold": config["validation_fold"], "validation_observations": len(index.validation_names),
              "annotator_records": len(selected_score["records"]), "holdout_used": False,
              "scope": options["scope"], "selected": chosen, "fixed_reference": reference,
              "bootstrap": bootstrap_groups(selected_score["records"], groups, options["bootstrap_samples"], options["bootstrap_seed"]),
              "checkpoint_sha256": inference["checkpoint_sha256"], "split_sha256": inference["split_sha256"],
              "inference_manifest_sha256": sha256(directory / "inference/manifest.json"),
              "config_sha256": sha256(directory / "config.json"), "evaluation_source_sha256": sha256(__file__),
              "wall_seconds": time.perf_counter() - started,
              "artifact_sha256": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()}}
    write_json(output / "report.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result
