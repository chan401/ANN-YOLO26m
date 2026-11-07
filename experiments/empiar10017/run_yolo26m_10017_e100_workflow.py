#!/usr/bin/env python
"""Run YOLO26m 100-epoch ANN train, predict, convert, and particle-picking eval."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(r"E:\cxj - crYOLOEM")
ULTRALYTICS_ROOT = ROOT / "SNNYOLO26" / "ultralytics-snn-yolo26"
DATA_YAML = ROOT / "datasets" / "10017_raw" / "yolo26_10017_2048" / "10017_2048.yaml"
TEST_IMAGES = ROOT / "datasets" / "10017_raw" / "yolo26_10017_2048" / "images" / "test"
GT_XY = ROOT / "datasets" / "10017_raw" / "yolo26_10017_2048" / "gt_test_xy.txt"
WEIGHTS = ULTRALYTICS_ROOT / "yolo26m.pt"
PROJECT = ROOT / "runs" / "10017"
TRAIN_NAME = "yolo26m_10017_2048_e100"
PRED_NAME = "yolo26m_10017_2048_e100_predict"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("RUN", " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd or ROOT), check=True)


def main() -> None:
    sys.path.insert(0, str(ULTRALYTICS_ROOT))
    from ultralytics import YOLO

    print(f"ultralytics root: {ULTRALYTICS_ROOT}", flush=True)
    print(f"weights: {WEIGHTS}", flush=True)
    print(f"data yaml: {DATA_YAML}", flush=True)
    print(f"project: {PROJECT}", flush=True)

    model = YOLO(str(WEIGHTS))
    train_results = model.train(
        data=str(DATA_YAML),
        imgsz=2048,
        epochs=100,
        batch=1,
        device=0,
        workers=4,
        patience=30,
        project=str(PROJECT),
        name=TRAIN_NAME,
        exist_ok=True,
    )
    train_dir = Path(train_results.save_dir)
    best_pt = train_dir / "weights" / "best.pt"
    print(f"TRAIN_DONE {train_dir}", flush=True)
    print(f"BEST_PT {best_pt}", flush=True)

    pred_model = YOLO(str(best_pt))
    pred_results = pred_model.predict(
        source=str(TEST_IMAGES),
        imgsz=2048,
        conf=0.001,
        iou=0.7,
        max_det=4000,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(PROJECT),
        name=PRED_NAME,
        exist_ok=True,
        device=0,
    )
    pred_dir = Path(pred_results[0].save_dir if pred_results else PROJECT / PRED_NAME)
    print(f"PRED_DONE {pred_dir}", flush=True)

    pred_xy = pred_dir / "pred_yolo26m_e100_test_xy.txt"
    eval_csv = pred_dir / "eval_yolo26m_e100_test.csv"
    run(
        [
            sys.executable,
            str(ROOT / "tools" / "convert_yolo_pred_to_xy.py"),
            "--label-dir",
            str(pred_dir / "labels"),
            "--image-dir",
            str(TEST_IMAGES),
            "--out",
            str(pred_xy),
            "--image-size",
            "2048",
        ],
        cwd=ROOT,
    )
    run(
        [
            sys.executable,
            str(ROOT / "tools" / "evaluate_particle_picking.py"),
            "--gt",
            str(GT_XY),
            "--pred",
            str(pred_xy),
            "--radius",
            "27",
            "--sweep",
            "--out-csv",
            str(eval_csv),
        ],
        cwd=ROOT,
    )
    print(f"WORKFLOW_DONE train={train_dir} pred={pred_dir} pred_xy={pred_xy} eval_csv={eval_csv}", flush=True)


if __name__ == "__main__":
    main()
