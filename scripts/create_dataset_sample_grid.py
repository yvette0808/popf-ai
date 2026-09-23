#!/usr/bin/env python3
"""Create a raw-image sample grid for manual label inspection."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.popf_dataset import (  # noqa: E402
    DEFAULT_DATASET_DIR,
    ensure_label_mapping,
    read_archive_image,
    read_csv_rows,
)


FONT_CANDIDATES = (
    Path("/System/Library/Fonts/STHeiti Medium.ttc"),
    Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
    Path("/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"),
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
)


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def create_grid(
    csv_path: Path,
    output_path: Path,
    sample_count: int,
    seed: int,
) -> None:
    rows = read_csv_rows(csv_path)
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if sample_count > len(rows):
        raise ValueError(f"sample_count {sample_count} exceeds dataset size {len(rows)}")

    ensure_label_mapping(
        DEFAULT_DATASET_DIR / "label_inventory.csv",
        DEFAULT_DATASET_DIR / "label_mapping.json",
    )
    selected = random.Random(seed).sample(rows, sample_count)

    columns = 4
    rows_count = (sample_count + columns - 1) // columns
    cell_width = 190
    cell_height = 235
    padding = 12
    caption_height = 42
    canvas = Image.new(
        "RGB",
        (
            columns * cell_width + (columns + 1) * padding,
            rows_count * cell_height + (rows_count + 1) * padding,
        ),
        "#f3f4f6",
    )
    draw = ImageDraw.Draw(canvas)
    font = load_font(17)
    small_font = load_font(13)

    for index, row in enumerate(selected):
        image = read_archive_image(row["image_path"], ROOT)
        image = ImageOps.contain(
            image,
            (cell_width - 2 * padding, cell_height - caption_height - 2 * padding),
        )
        column = index % columns
        row_index = index // columns
        cell_x = padding + column * cell_width
        cell_y = padding + row_index * cell_height
        image_x = cell_x + (cell_width - image.width) // 2
        image_y = cell_y + padding
        canvas.paste(image, (image_x, image_y))

        label = row["raw_face_type"]
        caption_y = cell_y + cell_height - caption_height + 5
        draw.text(
            (cell_x + padding, caption_y),
            f"{row['image_id']}  {label}",
            fill="#111827",
            font=font,
        )
        draw.text(
            (cell_x + padding, caption_y + 21),
            "raw image / XLS label",
            fill="#4b5563",
            font=small_font,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG")
    print(
        {
            "output": str(output_path),
            "sample_count": sample_count,
            "seed": seed,
            "source": str(csv_path),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_DATASET_DIR / "train.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/dataset_sample_grid.png",
    )
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    create_grid(args.csv, args.output, args.sample_count, args.seed)


if __name__ == "__main__":
    main()
