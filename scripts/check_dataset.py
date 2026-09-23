#!/usr/bin/env python3
"""Run pre-training checks for the POPF Dataset/DataLoader.

The script always performs manifest, label, image-read, and leakage checks.
Tensor/DataLoader checks require a real PyTorch installation and are reported
as blocked when PyTorch is unavailable.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.popf_dataset import (  # noqa: E402
    DEFAULT_DATASET_DIR,
    build_dataloaders,
    ensure_label_mapping,
    load_label_mapping,
    read_archive_image,
    read_csv_rows,
)


EXPECTED_COUNTS = {
    "train": 154,
    "validation": 34,
    "test": 34,
}


def class_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    return dict(sorted(Counter(row["raw_face_type"] for row in rows).items()))


def check_leakage(
    split_rows: dict[str, list[dict[str, str]]],
) -> dict[str, list[str]]:
    hashes = {
        split: {row["image_sha256"] for row in rows if row["image_sha256"]}
        for split, rows in split_rows.items()
    }
    overlaps: dict[str, list[str]] = {}
    split_names = list(hashes)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = sorted(hashes[left].intersection(hashes[right]))
            overlaps[f"{left}_vs_{right}"] = overlap
    if any(overlaps.values()):
        raise AssertionError(f"SHA-256 leakage detected: {overlaps}")
    return overlaps


def structural_checks(
    dataset_dir: Path,
    seed: int,
    sample_count: int,
) -> tuple[dict[str, list[dict[str, str]]], dict[str, int]]:
    split_rows = {
        split: read_csv_rows(dataset_dir / f"{split}.csv")
        for split in ("train", "validation", "test")
    }
    for split, rows in split_rows.items():
        expected = EXPECTED_COUNTS[split]
        if len(rows) != expected:
            raise AssertionError(f"{split} row count {len(rows)} != expected {expected}")

    mapping_path = dataset_dir / "label_mapping.json"
    mapping = ensure_label_mapping(dataset_dir / "label_inventory.csv", mapping_path)
    if mapping != load_label_mapping(mapping_path):
        raise AssertionError("label mapping changed during load")
    if set(mapping.values()) != set(range(7)):
        raise AssertionError(f"Expected label ids 0..6, got {mapping}")

    all_ids: list[str] = []
    for split, rows in split_rows.items():
        for row in rows:
            all_ids.append(row["image_id"])
            if row["raw_face_type"] not in mapping:
                raise AssertionError(
                    f"{split} contains label outside mapping: {row['raw_face_type']}"
                )
            if row.get("mapping_status") not in (
                "",
                "user_confirmed_sequence_mapping",
            ):
                raise AssertionError(
                    f"Unexpected mapping_status in {split}: {row['mapping_status']}"
                )

    if len(all_ids) != len(set(all_ids)):
        raise AssertionError("An image_id occurs in more than one split")
    check_leakage(split_rows)

    sample_pool = [row for rows in split_rows.values() for row in rows]
    selected = random.Random(seed).sample(
        sample_pool,
        min(sample_count, len(sample_pool)),
    )
    for row in selected:
        image = read_archive_image(row["image_path"], ROOT)
        if image.mode != "RGB":
            raise AssertionError(f"Image is not RGB after read: {row['image_id']}")
        if not row["image_id"] or not row["raw_face_type"]:
            raise AssertionError(f"Invalid sample metadata: {row}")

    return split_rows, mapping


def pytorch_checks(
    dataset_dir: Path,
    mapping: dict[str, int],
    image_size: tuple[int, int],
    batch_size: int,
    num_workers: int,
    seed: int,
) -> dict[str, object]:
    try:
        import torch
    except ImportError as exc:
        return {
            "status": "blocked",
            "reason": f"PyTorch is not installed: {exc}",
        }

    loaders = build_dataloaders(
        dataset_dir=dataset_dir,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )
    result: dict[str, object] = {"status": "passed", "batches": {}}
    for split, loader in zip(("train", "validation", "test"), loaders):
        batch = next(iter(loader))
        images = batch["image"]
        labels = batch["label"]
        expected_shape = (images.shape[0], 3, image_size[1], image_size[0])
        if tuple(images.shape) != expected_shape:
            raise AssertionError(
                f"{split} batch shape {tuple(images.shape)} != {expected_shape}"
            )
        if images.dtype != torch.float32:
            raise AssertionError(f"{split} images dtype is {images.dtype}")
        if labels.dtype != torch.int64:
            raise AssertionError(f"{split} labels dtype is {labels.dtype}")
        if int(labels.min()) < 0 or int(labels.max()) >= len(mapping):
            raise AssertionError(f"{split} labels outside 0..{len(mapping) - 1}")
        result["batches"][split] = {
            "shape": list(images.shape),
            "image_dtype": str(images.dtype),
            "label_dtype": str(labels.dtype),
            "label_min": int(labels.min()),
            "label_max": int(labels.max()),
            "sample_image_id": batch["image_id"][0],
            "sample_label_name": batch["label_name"][0],
            "sample_label_id": int(labels[0]),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/dataset_check.json",
    )
    args = parser.parse_args()

    split_rows, mapping = structural_checks(
        args.dataset_dir,
        args.seed,
        args.sample_count,
    )
    pytorch_result = pytorch_checks(
        args.dataset_dir,
        mapping,
        (args.image_width, args.image_height),
        args.batch_size,
        args.num_workers,
        args.seed,
    )
    result = {
        "status": (
            "passed"
            if pytorch_result["status"] == "passed"
            else "structural_checks_passed_pytorch_blocked"
        ),
        "structural_checks": "passed",
        "dataset_counts": {split: len(rows) for split, rows in split_rows.items()},
        "class_counts": {
            split: class_counts(rows) for split, rows in split_rows.items()
        },
        "label_mapping": mapping,
        "label_range": [0, len(mapping) - 1],
        "preprocess": {
            "image_size": [args.image_width, args.image_height],
            "pipeline": "JPEG -> RGB -> Resize -> float tensor -> ImageNet Normalize",
            "augmentation": "none",
        },
        "pytorch": pytorch_result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["pytorch"]["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
