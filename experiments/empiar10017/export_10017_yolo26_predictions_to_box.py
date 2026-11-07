#!/usr/bin/env python
"""Export 10017 ANN-YOLO26 center predictions to per-micrograph .box files.

Input format is the existing whitespace-separated prediction file:
    image_id x_center y_center confidence

The script filters by a frozen confidence threshold and writes EMAN/RELION-style
.box files:
    x_top_left y_top_left width height
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--box-size", type=float, required=True)
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Multiply center coordinates and box size by this scale before export.",
    )
    parser.add_argument("--image-size", type=float, default=0.0, help="Optional square image size for clipping.")
    parser.add_argument("--decimals", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.box_size <= 0:
        raise ValueError("--box-size must be positive.")
    if args.scale <= 0:
        raise ValueError("--scale must be positive.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scaled_box = args.box_size * args.scale
    half = scaled_box / 2.0
    by_image: dict[str, list[tuple[float, float, float]]] = defaultdict(list)

    with args.predictions.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(f"Bad prediction line {line_number}: {line!r}")
            image_id, x_text, y_text, conf_text = parts
            confidence = float(conf_text)
            if confidence < args.threshold:
                continue
            by_image[image_id].append((float(x_text) * args.scale, float(y_text) * args.scale, confidence))

    fmt = f"{{:.{args.decimals}f}} {{:.{args.decimals}f}} {{:.{args.decimals}f}} {{:.{args.decimals}f}}"
    total = 0
    for image_id, points in sorted(by_image.items()):
        out_path = output_dir / f"{image_id}.box"
        with out_path.open("w", encoding="utf-8", newline="\n") as handle:
            for x_center, y_center, confidence in sorted(points, key=lambda item: -item[2]):
                x0 = x_center - half
                y0 = y_center - half
                if args.image_size > 0:
                    x0 = min(max(0.0, x0), max(0.0, args.image_size - scaled_box))
                    y0 = min(max(0.0, y0), max(0.0, args.image_size - scaled_box))
                handle.write(fmt.format(x0, y0, scaled_box, scaled_box) + "\n")
                total += 1

    print(
        f"Exported {total} boxes for {len(by_image)} images to {output_dir} "
        f"(threshold={args.threshold:g}, box_size={scaled_box:g})"
    )


if __name__ == "__main__":
    main()
