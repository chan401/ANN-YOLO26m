#!/usr/bin/env python
"""Convert YOLO prediction labels to particle-picking xy confidence format."""

from __future__ import annotations

import argparse
from pathlib import Path


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".mrc"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert YOLO predict labels/*.txt to: image_name x_center y_center confidence"
    )
    parser.add_argument("--label-dir", type=Path, required=True, help="YOLO predict labels directory.")
    parser.add_argument("--image-dir", type=Path, required=True, help="Corresponding test images directory.")
    parser.add_argument("--out", type=Path, required=True, help="Output txt path.")
    parser.add_argument("--image-size", type=float, default=2048.0, help="Image size used to denormalize xy centers.")
    return parser.parse_args()


def list_images(image_dir: Path) -> list[Path]:
    return sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS)


def parse_label_file(label_path: Path, image_size: float) -> list[tuple[str, float, float, float]]:
    rows: list[tuple[str, float, float, float]] = []
    image_name = label_path.stem
    for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            print(f"Warning: skipping malformed row {label_path}:{line_no}: {line}")
            continue
        try:
            cls = int(float(parts[0]))
            x_norm = float(parts[1])
            y_norm = float(parts[2])
            confidence = float(parts[5]) if len(parts) >= 6 else 1.0
        except ValueError:
            print(f"Warning: skipping non-numeric row {label_path}:{line_no}: {line}")
            continue
        if cls != 0:
            continue
        rows.append((image_name, x_norm * image_size, y_norm * image_size, confidence))
    return rows


def main() -> None:
    args = parse_args()
    label_dir = args.label_dir.resolve()
    image_dir = args.image_dir.resolve()
    out_path = args.out.resolve()

    if not label_dir.is_dir():
        raise FileNotFoundError(f"label-dir does not exist or is not a directory: {label_dir}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"image-dir does not exist or is not a directory: {image_dir}")

    label_files = sorted(label_dir.glob("*.txt"))
    images = list_images(image_dir)
    label_stems = {path.stem for path in label_files}
    missing_labels = [path for path in images if path.stem not in label_stems]

    output_rows: list[str] = []
    for label_path in label_files:
        for image_name, x_center, y_center, confidence in parse_label_file(label_path, args.image_size):
            output_rows.append(f"{image_name} {x_center:.6f} {y_center:.6f} {confidence:.6f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(output_rows) + ("\n" if output_rows else ""), encoding="utf-8")

    print(f"label files: {len(label_files)}")
    print(f"test images: {len(images)}")
    print(f"missing label files: {len(missing_labels)}")
    print(f"predictions: {len(output_rows)}")
    print(f"output: {out_path}")


if __name__ == "__main__":
    main()
