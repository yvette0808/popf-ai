"""Small, dependency-light reporting helpers for POPF experiments."""

from __future__ import annotations

import csv
import math
import textwrap
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont

from src.datasets.popf_dataset import read_archive_image


def write_prediction_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write top-k predictions in a stable, analysis-friendly column order."""

    if not rows:
        raise ValueError("Cannot write an empty prediction file")
    preferred = [
        "image_id",
        "image_path",
        "true_label_id",
        "true_label",
        "top1_label_id",
        "top1_label",
        "top1_score",
        "top2_label_id",
        "top2_label",
        "top2_score",
        "top3_label_id",
        "top3_label",
        "top3_score",
        "top1_correct",
        "top_k_correct",
    ]
    fieldnames = [name for name in preferred if name in rows[0]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({name: row.get(name, "") for name in fieldnames} for row in rows)


def _font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _plot_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    series: list[tuple[str, list[float], tuple[int, int, int]]],
    font: ImageFont.ImageFont,
) -> None:
    left, top, right, bottom = box
    draw.rectangle(box, outline=(150, 150, 150), width=1)
    draw.text((left + 12, top + 8), title, fill=(20, 20, 20), font=font)
    chart_left, chart_top = left + 48, top + 38
    chart_right, chart_bottom = right - 18, bottom - 34
    values = [value for _, values, _ in series for value in values]
    if not values:
        return
    lower = min(0.0, min(values))
    upper = max(1.0, max(values))
    if math.isclose(lower, upper):
        upper = lower + 1.0
    draw.line(
        (chart_left, chart_top, chart_left, chart_bottom),
        fill=(100, 100, 100),
        width=1,
    )
    draw.line(
        (chart_left, chart_bottom, chart_right, chart_bottom),
        fill=(100, 100, 100),
        width=1,
    )
    draw.text((left + 6, chart_top - 4), f"{upper:.2f}", fill=(90, 90, 90), font=font)
    draw.text((left + 6, chart_bottom - 8), f"{lower:.2f}", fill=(90, 90, 90), font=font)
    max_points = max(len(values) for _, values, _ in series)
    for name, points, color in series:
        if not points:
            continue
        coordinates = []
        for index, value in enumerate(points):
            x = chart_left if max_points == 1 else chart_left + (
                chart_right - chart_left
            ) * index / (max_points - 1)
            y = chart_bottom - (chart_bottom - chart_top) * (value - lower) / (upper - lower)
            coordinates.append((int(x), int(y)))
        if len(coordinates) > 1:
            draw.line(coordinates, fill=color, width=3)
        for x, y in coordinates:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
        legend_x = chart_left + 8 + series.index((name, points, color)) * 120
        draw.line((legend_x, bottom - 19, legend_x + 18, bottom - 19), fill=color, width=3)
        draw.text((legend_x + 23, bottom - 27), name, fill=(30, 30, 30), font=font)


def save_training_curves(
    history: list[dict[str, Any]],
    output_path: Path,
    *,
    title: str = "POPF ResNet-18 baseline training curves",
) -> None:
    """Render loss, accuracy, and macro-F1 curves without matplotlib."""

    if not history:
        raise ValueError("Training history is empty")
    epochs = [int(row["epoch"]) for row in history]
    image = Image.new("RGB", (1200, 820), "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(24)
    body_font = _font(14)
    draw.text((30, 18), title, fill=(15, 15, 15), font=title_font)
    draw.text(
        (30, 52),
        f"epochs {epochs[0]}-{epochs[-1]} | validation only during model selection",
        fill=(80, 80, 80),
        font=body_font,
    )
    colors = {
        "train": (38, 104, 176),
        "validation": (210, 92, 55),
    }
    panel_width = 555
    panel_height = 350
    panels = [
        (
            "Loss",
            [
                ("train", [float(row["train_loss"]) for row in history], colors["train"]),
                ("validation", [float(row["validation_loss"]) for row in history], colors["validation"]),
            ],
        ),
        (
            "Accuracy",
            [
                ("train", [float(row["train_accuracy"]) for row in history], colors["train"]),
                ("validation", [float(row["validation_accuracy"]) for row in history], colors["validation"]),
            ],
        ),
        (
            "Validation macro-F1",
            [
                ("macro-F1", [float(row["validation_macro_f1"]) for row in history], (57, 132, 78)),
            ],
        ),
        (
            "Validation macro precision / recall",
            [
                (
                    "precision",
                    [float(row["validation_macro_precision"]) for row in history],
                    (139, 87, 166),
                ),
                (
                    "recall",
                    [float(row["validation_macro_recall"]) for row in history],
                    (194, 142, 42),
                ),
            ],
        ),
    ]
    for index, (panel_title, series) in enumerate(panels):
        row_index, column_index = divmod(index, 2)
        x = 30 + column_index * panel_width
        y = 92 + row_index * panel_height
        _plot_panel(
            draw,
            (x, y, x + panel_width - 20, y + panel_height - 20),
            panel_title,
            series,
            body_font,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def save_test_error_grid(
    prediction_rows: list[dict[str, Any]],
    *,
    project_root: Path,
    output_path: Path,
    max_errors: int = 12,
) -> int:
    """Render a small contact sheet of top-1 test errors for manual inspection."""

    errors = [row for row in prediction_rows if not row["top1_correct"]][:max_errors]
    font = _font(13)
    small_font = _font(12)
    tile_width, tile_height = 260, 335
    columns = 3
    rows = max(1, math.ceil(len(errors) / columns))
    image = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
    draw = ImageDraw.Draw(image)
    for index, row in enumerate(errors):
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        try:
            original = read_archive_image(row["image_path"], project_root)
            original.thumbnail((220, 220))
            image.paste(
                original,
                (x + (tile_width - original.width) // 2, y + 8),
            )
        except Exception as exc:
            draw.rectangle((x + 20, y + 8, x + 240, y + 228), outline=(180, 0, 0))
            draw.text((x + 28, y + 100), f"read error: {exc}", fill=(180, 0, 0), font=small_font)
        top3_text = "top-3: " + ", ".join(
            [row["top1_label"], row["top2_label"], row["top3_label"]]
        )
        wrapped_top3 = "\n".join(
            textwrap.wrap(
                top3_text,
                width=18,
                break_long_words=True,
                break_on_hyphens=False,
            )
        )
        text = (
            f'{row["image_id"]}\n'
            f'true: {row["true_label"]}\n'
            f'pred: {row["top1_label"]}\n'
            f'{wrapped_top3}'
        )
        draw.multiline_text((x + 10, y + 238), text, fill=(20, 20, 20), font=font, spacing=2)
    if not errors:
        draw.text((30, 30), "No top-1 test errors in the evaluated test set.", fill=(20, 20, 20), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return len(errors)
