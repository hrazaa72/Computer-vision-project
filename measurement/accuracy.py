"""Measure held-out photos with the full pipeline and compare against the calliper values.

Covers raw vs undistorted (item 6), error against distance from the frame centre (item 8), the
refusal-threshold sweep (item 9) and timing. Photos come from the test and val splits only:
none were used for gradient updates. Val was used to pick the checkpoint, test was not.

Usage:
    python -m measurement.accuracy
"""
import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import torch

import inference.measure as pipeline
from calibration.camera import Undistorter
from models.data import read_split
from models.model import load_checkpoint

ROOT = Path(__file__).resolve().parent
RAW = ROOT.parent / "dataset" / "raw" / "Annotated images"

TRUE_WIDTH_MM, TRUE_HEIGHT_MM = 51.0, 81.0  # calliper values of the box top face
SCORE_THRESHOLDS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98]
AXIS_RATIO_THRESHOLDS = [0.0, 0.9, 0.95, 0.97, 0.98, 0.99]
TIMING_RUNS = 5


def errors(result):
    w = result["width_mm"] - TRUE_WIDTH_MM
    h = result["height_mm"] - TRUE_HEIGHT_MM
    return {"width_err_mm": w, "height_err_mm": h,
            "mean_abs_err_mm": (abs(w) + abs(h)) / 2,
            "mean_pct_err": 100 * (abs(w) / TRUE_WIDTH_MM + abs(h) / TRUE_HEIGHT_MM) / 2}


def summarise(rows, key):
    values = [r[key] for r in rows]
    return {"mean": statistics.mean(values), "median": statistics.median(values), "worst": max(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=pipeline.DEFAULT_WEIGHTS)
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True)

    # Thresholds are applied afterwards from the recorded values, so switch the in-pipeline
    # refusals off here and keep every measurement.
    pipeline.MIN_SCORE, pipeline.MIN_COIN_AXIS_RATIO = 0.0, 0.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(args.weights, device)
    undistorter = Undistorter()

    names = [(n, "test") for n in read_split("test")] + [(n, "val") for n in read_split("val")]
    rows, failed = [], []
    for name, split in names:
        photo = cv2.imread(str(RAW / f"{name}.jpeg"))
        row = {"image": name, "split": split}
        try:
            for mode, undistort in (("raw", False), ("undistorted", True)):
                result, _ = pipeline.measure(photo, undistorter, model, device, undistort=undistort)
                for key, value in {**result, **errors(result)}.items():
                    row[f"{mode}_{key}"] = value
        except pipeline.Refusal as refusal:
            failed.append({"image": name, "reason": refusal.code})
            continue
        rows.append(row)

    with open(args.out / "accuracy.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {"images_measured": len(rows), "images_failed": failed,
               "truth_mm": {"width": TRUE_WIDTH_MM, "height": TRUE_HEIGHT_MM}}
    for mode in ("raw", "undistorted"):
        summary[mode] = {
            "mean_abs_err_mm": summarise(rows, f"{mode}_mean_abs_err_mm"),
            "mean_pct_err": summarise(rows, f"{mode}_mean_pct_err"),
            "width_err_mm_signed_mean": statistics.mean(r[f"{mode}_width_err_mm"] for r in rows),
            "height_err_mm_signed_mean": statistics.mean(r[f"{mode}_height_err_mm"] for r in rows),
        }

    # Item 8: error against how far the box is from the frame centre (undistorted pipeline).
    ordered = sorted(rows, key=lambda r: r["undistorted_box_centre_offset_px"])
    third = len(ordered) // 3
    summary["error_by_centre_offset"] = [
        {"group": label, "offset_px_range": [group[0]["undistorted_box_centre_offset_px"],
                                             group[-1]["undistorted_box_centre_offset_px"]],
         "mean_abs_err_mm": statistics.mean(r["undistorted_mean_abs_err_mm"] for r in group)}
        for label, group in (("near centre", ordered[:third]), ("middle", ordered[third:2 * third]),
                             ("far from centre", ordered[2 * third:]))]

    # Item 9: refusal sweep over mask score and coin tilt proxy.
    sweep = []
    for score in SCORE_THRESHOLDS:
        for ratio in AXIS_RATIO_THRESHOLDS:
            kept = [r for r in rows if r["undistorted_confidence"] >= score
                    and r["undistorted_coin_axis_ratio"] >= ratio]
            sweep.append({"min_score": score, "min_axis_ratio": ratio, "kept": len(kept),
                          "refused_pct": 100 * (1 - len(kept) / len(rows)),
                          "mean_abs_err_mm": statistics.mean(r["undistorted_mean_abs_err_mm"] for r in kept)
                          if kept else None,
                          "worst_abs_err_mm": max(r["undistorted_mean_abs_err_mm"] for r in kept)
                          if kept else None})
    summary["refusal_sweep"] = sweep

    # Timing: full pipeline on one photo, repeated, after one warm-up.
    photo = cv2.imread(str(RAW / f"{rows[0]['image']}.jpeg"))
    pipeline.measure(photo, undistorter, model, device)
    times = []
    for _ in range(TIMING_RUNS):
        start = time.perf_counter()
        pipeline.measure(photo, undistorter, model, device)
        times.append(time.perf_counter() - start)
    summary["timing_s"] = {"runs": TIMING_RUNS, "median": statistics.median(times),
                           "min": min(times), "max": max(times), "device": str(device)}

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "refusal_sweep"}, indent=2))


if __name__ == "__main__":
    main()
