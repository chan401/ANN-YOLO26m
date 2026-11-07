#!/usr/bin/env python
"""Evaluate particle picking predictions with radius-based one-to-one matching."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Point:
    image: str
    x: float
    y: float
    conf: float = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate particle picking predictions against xy ground truth.")
    parser.add_argument("--gt", type=Path, required=True, help="Ground truth txt: image_name x y")
    parser.add_argument("--pred", type=Path, required=True, help="Prediction txt: image_name x y confidence")
    parser.add_argument("--radius", type=float, default=27.0, help="Matching radius in pixels.")
    parser.add_argument("--sweep", action="store_true", help="Sweep confidence thresholds and report best F1.")
    parser.add_argument("--out-csv", type=Path, required=True, help="Output CSV path.")
    return parser.parse_args()


def read_points(path: Path, require_conf: bool) -> list[Point]:
    points: list[Point] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 3:
            print(f"Warning: skipping malformed row {path}:{line_no}: {line}")
            continue
        try:
            image = parts[0]
            x = float(parts[1])
            y = float(parts[2])
            conf = float(parts[3]) if len(parts) >= 4 else 1.0
        except ValueError:
            print(f"Warning: skipping non-numeric row {path}:{line_no}: {line}")
            continue
        if require_conf and len(parts) < 4:
            conf = 1.0
        points.append(Point(image=image, x=x, y=y, conf=conf))
    return points


def build_gt_index(gt_points: list[Point], radius: float) -> tuple[dict[str, list[Point]], dict[str, dict[tuple[int, int], list[int]]]]:
    gt_by_image: dict[str, list[Point]] = defaultdict(list)
    for point in gt_points:
        gt_by_image[point.image].append(point)

    cell_size = max(radius, 1.0)
    grid_by_image: dict[str, dict[tuple[int, int], list[int]]] = {}
    for image, points in gt_by_image.items():
        grid: dict[tuple[int, int], list[int]] = defaultdict(list)
        for idx, point in enumerate(points):
            grid[(int(point.x // cell_size), int(point.y // cell_size))].append(idx)
        grid_by_image[image] = grid
    return gt_by_image, grid_by_image


def find_match(
    pred: Point,
    gt_points: list[Point],
    grid: dict[tuple[int, int], list[int]],
    matched: list[bool],
    radius: float,
) -> int | None:
    if not gt_points:
        return None
    cell_size = max(radius, 1.0)
    cx = int(pred.x // cell_size)
    cy = int(pred.y // cell_size)
    radius_sq = radius * radius
    best_idx: int | None = None
    best_dist = math.inf

    for gx in range(cx - 1, cx + 2):
        for gy in range(cy - 1, cy + 2):
            for idx in grid.get((gx, gy), []):
                if matched[idx]:
                    continue
                gt = gt_points[idx]
                dist = (pred.x - gt.x) ** 2 + (pred.y - gt.y) ** 2
                if dist <= radius_sq and dist < best_dist:
                    best_dist = dist
                    best_idx = idx
    return best_idx


def metrics_row(threshold: float, tp: int, fp: int, total_gt: int) -> dict[str, float | int]:
    fn = total_gt - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / total_gt if total_gt else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predictions": tp + fp,
    }


def evaluate(gt_points: list[Point], pred_points: list[Point], radius: float, sweep: bool) -> list[dict[str, float | int]]:
    gt_by_image, grid_by_image = build_gt_index(gt_points, radius)
    matched_by_image = {image: [False] * len(points) for image, points in gt_by_image.items()}
    total_gt = len(gt_points)

    preds = sorted(pred_points, key=lambda point: point.conf, reverse=True)
    rows: list[dict[str, float | int]] = []
    tp = 0
    fp = 0
    i = 0
    while i < len(preds):
        threshold = preds[i].conf
        while i < len(preds) and preds[i].conf == threshold:
            pred = preds[i]
            gt_points_for_image = gt_by_image.get(pred.image, [])
            matched = matched_by_image.get(pred.image, [])
            match_idx = find_match(
                pred,
                gt_points_for_image,
                grid_by_image.get(pred.image, {}),
                matched,
                radius,
            )
            if match_idx is None:
                fp += 1
            else:
                matched[match_idx] = True
                tp += 1
            i += 1
        if sweep:
            rows.append(metrics_row(threshold, tp, fp, total_gt))

    if not sweep:
        rows.append(metrics_row(0.0, tp, fp, total_gt))
    elif not rows:
        rows.append(metrics_row(1.0, 0, 0, total_gt))
    return rows


def write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["threshold", "tp", "fp", "fn", "precision", "recall", "f1", "predictions"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    gt_points = read_points(args.gt.resolve(), require_conf=False)
    pred_points = read_points(args.pred.resolve(), require_conf=True)
    rows = evaluate(gt_points, pred_points, args.radius, args.sweep)
    write_csv(args.out_csv.resolve(), rows)

    best = max(rows, key=lambda row: float(row["f1"]))
    print(f"gt particles: {len(gt_points)}")
    print(f"predictions: {len(pred_points)}")
    print(f"radius: {args.radius}")
    print(f"best threshold: {float(best['threshold']):.6f}")
    print(f"best precision: {float(best['precision']):.6f}")
    print(f"best recall: {float(best['recall']):.6f}")
    print(f"best f1: {float(best['f1']):.6f}")
    print(f"best tp/fp/fn: {int(best['tp'])}/{int(best['fp'])}/{int(best['fn'])}")
    print(f"out csv: {args.out_csv.resolve()}")


if __name__ == "__main__":
    main()
