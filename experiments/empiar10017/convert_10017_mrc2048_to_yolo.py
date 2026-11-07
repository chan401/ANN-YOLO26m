#!/usr/bin/env python
"""Convert EMPIAR-10017 2048 MRC data to one-class YOLO detection format."""

from __future__ import annotations

import argparse
import random
import shutil
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from PIL import Image, ImageDraw
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit("Pillow is required. Install it with: pip install pillow") from exc


DEFAULT_SRC = Path(r"E:\cxj - crYOLOEM\datasets\10017_raw")
DEFAULT_OUT = DEFAULT_SRC / "yolo26_10017_2048"
YAML_DATASET_PATH = "E:/cxj - crYOLOEM/datasets/10017_raw/yolo26_10017_2048"
RAW_SUBDIR = "cryolo_raw_mrc_2048"
SPLITS = ("train", "val", "test")
FLAT_DEFAULT_SEED = 10017


@dataclass
class SplitStats:
    images: int = 0
    labels: int = 0
    boxes: int = 0
    skipped_boxes: int = 0
    invalid_rows: int = 0
    coords_in_01: bool = True
    mean_width: float = 0.0
    mean_height: float = 0.0
    images_without_labels: list[str] = field(default_factory=list)
    labels_without_images: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert EMPIAR-10017 2048x2048 MRC images and annotations to YOLO format."
    )
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC, help="Source dataset directory.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output YOLO dataset directory.")
    parser.add_argument("--image-size", type=int, default=2048, help="Expected square image size.")
    parser.add_argument("--box-size", type=float, default=54.0, help="Fixed test box size in pixels.")
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=None,
        help="Optional fixed split directory with train/valid/test images or micrographs txt files.",
    )
    parser.add_argument("--clean", action="store_true", help="Delete the old output directory before conversion.")
    parser.add_argument(
        "--make-vis",
        type=int,
        default=5,
        help="Number of random visual check images to create across train/val/test.",
    )
    return parser.parse_args()


def find_raw_root(src: Path) -> Path:
    src = src.resolve()
    nested = src / RAW_SUBDIR
    if nested.is_dir():
        return nested
    return src


def require_dirs(raw_root: Path) -> None:
    required = [
        "train_image",
        "train_annot",
        "valid_image",
        "valid_annot",
        "test_image",
        "gt_test",
    ]
    missing = [name for name in required if not (raw_root / name).is_dir()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"Missing required source directories under {raw_root}: {joined}")


def has_structured_source(raw_root: Path) -> bool:
    return all(
        (raw_root / name).is_dir()
        for name in ("train_image", "train_annot", "valid_image", "valid_annot", "test_image", "gt_test")
    )


def has_flat_coord_source(raw_root: Path) -> bool:
    return bool(list(raw_root.glob("*.mrc"))) and bool(list(raw_root.glob("*.coord")))


def prepare_output(out_dir: Path, raw_root: Path, clean: bool) -> None:
    out_dir = out_dir.resolve()
    raw_root = raw_root.resolve()
    if clean and out_dir.exists():
        if out_dir == raw_root or out_dir == raw_root.parent:
            raise ValueError(f"Refusing to delete source-like output directory: {out_dir}")
        shutil.rmtree(out_dir)

    for split in SPLITS:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    (out_dir / "check_vis").mkdir(parents=True, exist_ok=True)


def read_split_list(path: Path) -> list[str] | None:
    if not path.is_file():
        return None
    names: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = Path(line).name
        if Path(name).suffix.lower() != ".mrc":
            name = f"{name}.mrc"
        names.append(name)
    return names


def find_split_file(split_dir: Path | None, split_name: str) -> Path | None:
    if split_dir is None:
        return None
    candidates = {
        "train": ["train_images.txt", "train_micrographs.txt"],
        "valid": ["valid_images.txt", "val_images.txt", "valid_micrographs.txt", "val_micrographs.txt"],
        "val": ["valid_images.txt", "val_images.txt", "valid_micrographs.txt", "val_micrographs.txt"],
        "test": ["test_images.txt", "test_micrographs.txt"],
    }[split_name]
    for name in candidates:
        path = split_dir / name
        if path.is_file():
            return path
    return None


def list_mrc_files(image_dir: Path, split_file: Path | None = None) -> list[Path]:
    requested = read_split_list(split_file) if split_file else None
    if requested is None:
        return sorted(image_dir.glob("*.mrc"))

    index = {path.name.lower(): path for path in image_dir.glob("*.mrc")}
    missing = [name for name in requested if name.lower() not in index]
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"{len(missing)} images listed in {split_file} were not found. First: {preview}")
    return [index[name.lower()] for name in requested]


def read_mrc(path: Path) -> np.ndarray:
    try:
        import mrcfile  # type: ignore

        with mrcfile.open(path, permissive=True) as mrc:
            return np.asarray(mrc.data)
    except ImportError:
        return read_mrc_basic(path)
    except Exception:
        return read_mrc_basic(path)


def read_mrc_basic(path: Path) -> np.ndarray:
    mode_to_dtype = {
        0: np.int8,
        1: np.int16,
        2: np.float32,
        6: np.uint16,
        12: np.float16,
    }
    with path.open("rb") as fh:
        header = fh.read(1024)
        if len(header) != 1024:
            raise ValueError(f"{path} is too small to be an MRC file.")

        nx, ny, nz, mode = struct.unpack("<4i", header[:16])
        if nx <= 0 or ny <= 0 or nz <= 0 or mode not in mode_to_dtype:
            nx, ny, nz, mode = struct.unpack(">4i", header[:16])
        if nx <= 0 or ny <= 0 or nz <= 0 or mode not in mode_to_dtype:
            raise ValueError(f"Unsupported or invalid MRC header in {path}.")

        nsymbt = struct.unpack("<i", header[92:96])[0]
        dtype = mode_to_dtype[mode]
        fh.seek(1024 + max(nsymbt, 0))
        data = np.fromfile(fh, dtype=dtype, count=nx * ny * nz)

    expected = nx * ny * nz
    if data.size != expected:
        raise ValueError(f"{path} has {data.size} voxels, expected {expected}.")
    if nz == 1:
        return data.reshape(ny, nx)
    return data.reshape(nz, ny, nx)


def mrc_to_uint8(path: Path, image_size: int) -> tuple[np.ndarray, tuple[int, int]]:
    arr = read_mrc(path)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"{path} has unsupported shape {arr.shape}; expected (H,W) or (1,H,W).")
    source_shape = (int(arr.shape[0]), int(arr.shape[1]))
    if source_shape[0] != source_shape[1]:
        raise ValueError(f"{path} has non-square shape {arr.shape}; expected square micrograph.")

    arr = arr.astype(np.float32, copy=False)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        image = np.zeros(arr.shape, dtype=np.uint8)
        return resize_uint8_if_needed(image, image_size), source_shape

    lo, hi = np.percentile(finite, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
    if hi <= lo:
        image = np.zeros(arr.shape, dtype=np.uint8)
        return resize_uint8_if_needed(image, image_size), source_shape

    scaled = (arr - lo) / (hi - lo)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)
    image = (np.clip(scaled, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return resize_uint8_if_needed(image, image_size), source_shape


def resize_uint8_if_needed(image: np.ndarray, image_size: int) -> np.ndarray:
    if image.shape == (image_size, image_size):
        return image
    pil_image = Image.fromarray(image)
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    return np.asarray(pil_image.resize((image_size, image_size), resampling))


def save_png_from_mrc(mrc_path: Path, png_path: Path, image_size: int) -> tuple[int, int]:
    image, source_shape = mrc_to_uint8(mrc_path, image_size)
    Image.fromarray(image).save(png_path)
    return source_shape


def parse_numeric_rows(path: Path, min_cols: int) -> Iterable[list[float]]:
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < min_cols:
            raise ValueError(f"{path}:{line_no} has {len(parts)} columns, expected at least {min_cols}.")
        try:
            yield [float(value) for value in parts[:min_cols]]
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no} contains non-numeric values: {line}") from exc


def xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float, image_size: int) -> tuple[float, float, float, float] | None:
    x1 = min(max(x1, 0.0), float(image_size))
    y1 = min(max(y1, 0.0), float(image_size))
    x2 = min(max(x2, 0.0), float(image_size))
    y2 = min(max(y2, 0.0), float(image_size))
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    if w <= 0.0 or h <= 0.0:
        return None
    xc = x1 + w / 2.0
    yc = y1 + h / 2.0
    norm = np.clip(np.array([xc, yc, w, h], dtype=np.float64) / float(image_size), 0.0, 1.0)
    return tuple(float(v) for v in norm)


def convert_box_file(box_path: Path, label_path: Path, image_size: int) -> tuple[int, int]:
    rows: list[str] = []
    seen: set[str] = set()
    skipped = 0
    for x_left, y_top, width, height in parse_numeric_rows(box_path, min_cols=4):
        converted = xyxy_to_yolo(x_left, y_top, x_left + width, y_top + height, image_size)
        if converted is None:
            skipped += 1
            continue
        xc, yc, w, h = converted
        row = f"0 {xc:.8f} {yc:.8f} {w:.8f} {h:.8f}"
        if row in seen:
            skipped += 1
            continue
        seen.add(row)
        rows.append(row)
    label_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return len(rows), skipped


def convert_center_file(
    center_path: Path,
    label_path: Path,
    image_size: int,
    box_size: float,
    coord_size: int | float | None = None,
) -> tuple[int, int]:
    rows: list[str] = []
    seen: set[str] = set()
    skipped = 0
    half = box_size / 2.0
    source_size = float(coord_size if coord_size is not None else image_size)
    scale = float(image_size) / source_size
    for x_center, y_center in parse_numeric_rows(center_path, min_cols=2):
        x_scaled = x_center * scale
        y_scaled = y_center * scale
        converted = xyxy_to_yolo(
            x_scaled - half,
            y_scaled - half,
            x_scaled + half,
            y_scaled + half,
            image_size,
        )
        if converted is None:
            skipped += 1
            continue
        xc, yc, w, h = converted
        row = f"0 {xc:.8f} {yc:.8f} {w:.8f} {h:.8f}"
        if row in seen:
            skipped += 1
            continue
        seen.add(row)
        rows.append(row)
    label_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return len(rows), skipped


def convert_split(
    raw_root: Path,
    out_dir: Path,
    split: str,
    src_image_dir: str,
    src_label_dir: str,
    src_split_name: str,
    split_dir: Path | None,
    image_size: int,
    box_size: float,
) -> SplitStats:
    stats = SplitStats()
    image_dir = raw_root / src_image_dir
    label_dir = raw_root / src_label_dir
    split_file = find_split_file(split_dir, src_split_name) or raw_root / "splits" / f"{src_split_name}_images.txt"
    image_paths = list_mrc_files(image_dir, split_file)
    print(f"Converting {split}: {len(image_paths)} MRC images")

    for idx, mrc_path in enumerate(image_paths, start=1):
        png_path = out_dir / "images" / split / f"{mrc_path.stem}.png"
        save_png_from_mrc(mrc_path, png_path, image_size)

        label_path = out_dir / "labels" / split / f"{mrc_path.stem}.txt"
        if split == "test":
            src_label = label_dir / f"{mrc_path.stem}.txt"
            if src_label.is_file():
                boxes, skipped = convert_center_file(src_label, label_path, image_size, box_size)
                stats.boxes += boxes
                stats.skipped_boxes += skipped
        else:
            src_label = label_dir / f"{mrc_path.stem}.box"
            if src_label.is_file():
                boxes, skipped = convert_box_file(src_label, label_path, image_size)
                stats.boxes += boxes
                stats.skipped_boxes += skipped

        if idx % 25 == 0 or idx == len(image_paths):
            print(f"  {split}: {idx}/{len(image_paths)} images")

    return stats


def make_flat_splits(raw_root: Path, split_dir: Path | None = None) -> dict[str, list[Path]]:
    image_paths = sorted(raw_root.glob("*.mrc"))
    if not image_paths:
        raise FileNotFoundError(f"No .mrc files found under {raw_root}")

    split_dir = split_dir or raw_root / "splits"
    requested = {
        "train": read_split_list(find_split_file(split_dir, "train") or split_dir / "train_images.txt"),
        "val": read_split_list(find_split_file(split_dir, "valid") or split_dir / "valid_images.txt"),
        "test": read_split_list(find_split_file(split_dir, "test") or split_dir / "test_images.txt"),
    }
    if all(requested.values()):
        index = {path.name.lower(): path for path in image_paths}
        missing: list[str] = []
        for names in requested.values():
            missing.extend(name for name in names or [] if name.lower() not in index)
        if missing:
            preview = ", ".join(missing[:5])
            raise FileNotFoundError(f"{len(missing)} split entries were not found in {raw_root}. First: {preview}")
        return {split: [index[name.lower()] for name in names or []] for split, names in requested.items()}

    shuffled = image_paths[:]
    random.Random(FLAT_DEFAULT_SEED).shuffle(shuffled)
    n = len(shuffled)
    if n >= 84:
        val_count = 12
        test_count = 14
    else:
        val_count = max(1, round(n * 0.1))
        test_count = max(1, round(n * 0.1))
    train_count = max(0, n - val_count - test_count)
    return {
        "train": sorted(shuffled[:train_count]),
        "val": sorted(shuffled[train_count : train_count + val_count]),
        "test": sorted(shuffled[train_count + val_count :]),
    }


def convert_flat_coord_dataset(
    raw_root: Path,
    out_dir: Path,
    image_size: int,
    box_size: float,
    split_dir: Path | None = None,
) -> dict[str, SplitStats]:
    split_paths = make_flat_splits(raw_root, split_dir=split_dir)
    stats_by_split: dict[str, SplitStats] = {}
    for split, image_paths in split_paths.items():
        stats = SplitStats()
        print(f"Converting {split}: {len(image_paths)} flat MRC/.coord images")
        for idx, mrc_path in enumerate(image_paths, start=1):
            png_path = out_dir / "images" / split / f"{mrc_path.stem}.png"
            source_h, source_w = save_png_from_mrc(mrc_path, png_path, image_size)
            if source_h != source_w:
                raise ValueError(f"{mrc_path} has non-square shape {(source_h, source_w)}")

            coord_path = raw_root / f"{mrc_path.stem}.coord"
            label_path = out_dir / "labels" / split / f"{mrc_path.stem}.txt"
            if coord_path.is_file():
                boxes, skipped = convert_center_file(
                    coord_path,
                    label_path,
                    image_size=image_size,
                    box_size=box_size,
                    coord_size=source_w,
                )
                stats.boxes += boxes
                stats.skipped_boxes += skipped
            else:
                label_path.write_text("", encoding="utf-8")

            if idx % 25 == 0 or idx == len(image_paths):
                print(f"  {split}: {idx}/{len(image_paths)} images")
        stats_by_split[split] = stats

    write_flat_metadata(raw_root, out_dir, split_paths, image_size=image_size)
    return stats_by_split


def write_flat_metadata(raw_root: Path, out_dir: Path, split_paths: dict[str, list[Path]], image_size: int) -> None:
    split_dir = out_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    split_name_map = {"train": "train_images.txt", "val": "valid_images.txt", "test": "test_images.txt"}
    for split, file_name in split_name_map.items():
        lines = [path.name for path in split_paths[split]]
        (split_dir / file_name).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    gt_lines: list[str] = []
    for mrc_path in split_paths["test"]:
        coord_path = raw_root / f"{mrc_path.stem}.coord"
        if not coord_path.is_file():
            continue
        image, source_shape = mrc_to_uint8(mrc_path, image_size)
        del image
        source_size = float(source_shape[1])
        scale = float(image_size) / source_size
        for x_center, y_center in parse_numeric_rows(coord_path, min_cols=2):
            gt_lines.append(f"{mrc_path.stem} {x_center * scale:.6f} {y_center * scale:.6f}")
    (out_dir / "gt_test_xy.txt").write_text("\n".join(gt_lines) + ("\n" if gt_lines else ""), encoding="utf-8")


def copy_metadata(raw_root: Path, out_dir: Path) -> None:
    src_splits = raw_root / "splits"
    if src_splits.is_dir():
        shutil.copytree(src_splits, out_dir / "splits", dirs_exist_ok=True)
    else:
        print(f"Warning: splits directory not found: {src_splits}")

    gt_xy = raw_root / "gt_test_xy.txt"
    if gt_xy.is_file():
        shutil.copy2(gt_xy, out_dir / "gt_test_xy.txt")
    else:
        print(f"Warning: gt_test_xy.txt not found: {gt_xy}")


def write_yaml(out_dir: Path) -> Path:
    yaml_path = out_dir / "10017_2048.yaml"
    content = (
        f"path: {YAML_DATASET_PATH}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "\n"
        "nc: 1\n"
        "names:\n"
        "  0: particle\n"
    )
    yaml_path.write_text(content, encoding="utf-8")
    local_content = content.replace(YAML_DATASET_PATH, out_dir.as_posix())
    (out_dir / "10017_2048_local.yaml").write_text(local_content, encoding="utf-8")
    return yaml_path


def summarize_split(out_dir: Path, split: str, stats: SplitStats) -> SplitStats:
    image_paths = sorted((out_dir / "images" / split).glob("*.png"))
    label_paths = sorted((out_dir / "labels" / split).glob("*.txt"))
    image_stems = {path.stem for path in image_paths}
    label_stems = {path.stem for path in label_paths}

    stats.images = len(image_paths)
    stats.labels = len(label_paths)
    stats.images_without_labels = sorted(image_stems - label_stems)
    stats.labels_without_images = sorted(label_stems - image_stems)

    widths: list[float] = []
    heights: list[float] = []
    coords_ok = True
    invalid_rows = 0

    for label_path in label_paths:
        for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                invalid_rows += 1
                coords_ok = False
                print(f"Invalid label row length: {label_path}:{line_no}: {line}")
                continue
            try:
                cls, xc, yc, w, h = [float(value) for value in parts]
            except ValueError:
                invalid_rows += 1
                coords_ok = False
                print(f"Invalid numeric label row: {label_path}:{line_no}: {line}")
                continue
            if int(cls) != 0 or cls != 0.0:
                invalid_rows += 1
                coords_ok = False
                print(f"Invalid class id: {label_path}:{line_no}: {line}")
            values = [xc, yc, w, h]
            if any(value < 0.0 or value > 1.0 for value in values):
                coords_ok = False
                invalid_rows += 1
            widths.append(w)
            heights.append(h)

    stats.coords_in_01 = coords_ok
    stats.invalid_rows = invalid_rows
    stats.mean_width = float(np.mean(widths)) if widths else 0.0
    stats.mean_height = float(np.mean(heights)) if heights else 0.0
    return stats


def read_yolo_boxes(label_path: Path, image_size: int) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    if not label_path.is_file():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, xc, yc, w, h = [float(value) for value in parts]
        x1 = int(round((xc - w / 2.0) * image_size))
        y1 = int(round((yc - h / 2.0) * image_size))
        x2 = int(round((xc + w / 2.0) * image_size))
        y2 = int(round((yc + h / 2.0) * image_size))
        boxes.append((x1, y1, x2, y2))
    return boxes


def make_visual_checks(out_dir: Path, image_size: int, count: int) -> None:
    if count <= 0:
        return
    candidates: list[tuple[str, Path]] = []
    for split in SPLITS:
        candidates.extend((split, path) for path in sorted((out_dir / "images" / split).glob("*.png")))
    if not candidates:
        print("Warning: no images available for check_vis generation.")
        return

    rng = random.Random(10017)
    selected = rng.sample(candidates, k=min(count, len(candidates)))
    vis_dir = out_dir / "check_vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    for split, image_path in selected:
        label_path = out_dir / "labels" / split / f"{image_path.stem}.txt"
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        for box in read_yolo_boxes(label_path, image_size):
            draw.rectangle(box, outline=(255, 0, 0), width=3)
        image.save(vis_dir / f"{split}_{image_path.stem}.png")


def print_report(stats_by_split: dict[str, SplitStats], yaml_path: Path) -> None:
    print("\nConversion report")
    for split in SPLITS:
        stats = stats_by_split[split]
        print(f"{split} images: {stats.images}")
        print(f"{split} labels: {stats.labels}")

    print("\nIntegrity checks")
    all_coords_ok = True
    for split in SPLITS:
        stats = stats_by_split[split]
        all_coords_ok = all_coords_ok and stats.coords_in_01
        missing_labels = len(stats.images_without_labels)
        missing_images = len(stats.labels_without_images)
        print(f"{split}: images without labels: {missing_labels}{preview_names(stats.images_without_labels)}")
        print(f"{split}: labels without images: {missing_images}{preview_names(stats.labels_without_images)}")
        print(f"{split}: label coordinates in 0..1: {'yes' if stats.coords_in_01 else 'no'}")
        print(f"{split}: invalid label rows: {stats.invalid_rows}")
        print(f"{split}: skipped out-of-image boxes: {stats.skipped_boxes}")
        print(f"{split}: mean label width: {stats.mean_width:.8f}")
        print(f"{split}: mean label height: {stats.mean_height:.8f}")

    print(f"all label coordinates in 0..1: {'yes' if all_coords_ok else 'no'}")
    print(f"data yaml: {yaml_path}")


def preview_names(names: list[str], limit: int = 5) -> str:
    if not names:
        return ""
    preview = ", ".join(names[:limit])
    suffix = "..." if len(names) > limit else ""
    return f" ({preview}{suffix})"


def main() -> None:
    args = parse_args()
    raw_root = find_raw_root(args.src)
    out_dir = args.out.resolve()
    split_dir = args.split_dir.resolve() if args.split_dir else None

    print(f"source root: {raw_root}")
    print(f"output root: {out_dir}")
    if split_dir:
        print(f"split root: {split_dir}")
        if not split_dir.is_dir():
            raise FileNotFoundError(f"split-dir does not exist or is not a directory: {split_dir}")
    prepare_output(out_dir, raw_root, clean=args.clean)

    if has_structured_source(raw_root):
        require_dirs(raw_root)
        split_jobs = {
            "train": ("train_image", "train_annot", "train"),
            "val": ("valid_image", "valid_annot", "valid"),
            "test": ("test_image", "gt_test", "test"),
        }

        stats_by_split: dict[str, SplitStats] = {}
        for split, (src_image_dir, src_label_dir, src_split_name) in split_jobs.items():
            stats_by_split[split] = convert_split(
                raw_root=raw_root,
                out_dir=out_dir,
                split=split,
                src_image_dir=src_image_dir,
                src_label_dir=src_label_dir,
                src_split_name=src_split_name,
                split_dir=split_dir,
                image_size=args.image_size,
                box_size=args.box_size,
            )
        copy_metadata(raw_root, out_dir)
    elif has_flat_coord_source(raw_root):
        if split_dir:
            print("Detected flat .mrc/.coord source layout; using fixed split files from --split-dir.")
        else:
            print("Detected flat .mrc/.coord source layout; generating deterministic 58/12/14 split for 84 images.")
        stats_by_split = convert_flat_coord_dataset(
            raw_root=raw_root,
            out_dir=out_dir,
            image_size=args.image_size,
            box_size=args.box_size,
            split_dir=split_dir,
        )
    else:
        raise FileNotFoundError(
            f"{raw_root} is neither the expected cryolo_raw_mrc_2048 layout nor a flat .mrc/.coord directory."
        )

    yaml_path = write_yaml(out_dir)
    make_visual_checks(out_dir, args.image_size, args.make_vis)

    for split in SPLITS:
        stats_by_split[split] = summarize_split(out_dir, split, stats_by_split[split])
    print_report(stats_by_split, yaml_path)


if __name__ == "__main__":
    main()
