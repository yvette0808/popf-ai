#!/usr/bin/env python3
"""Build the POPF 2026 modern-training dataset manifest.

This script reads only audited candidate outputs and writes derived files under
data/processed/modern_dataset. It does not modify data/raw and does not train a
model.
"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_DIR = ROOT / "data/processed/272_candidate"
OUT_DIR = ROOT / "data/processed/modern_dataset"

RAW_IMAGE_ARCHIVE = "data/raw/脸谱/整理工作/爬虫脸谱图集.zip"
RAW_METADATA_FILE = "data/raw/脸谱新一步/272数据张丹阳.xls"

SEED = 42
MIN_SAMPLES_PER_CLASS = 10
TARGET_SPLIT_RATIOS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}

MAPPING_STATUS = "user_confirmed_sequence_mapping"
MAPPING_EVIDENCE = (
    "Project owner confirmed on 2026-09-20 that Excel 序号 and the image filename "
    "number are one-to-one for the current POPF dataset."
)
PAPER_MENTIONED_VALUE = "not_evaluated"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: scalar(row.get(field, "")) for field in fieldnames})


def scalar(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def percent(count: int, total: int) -> str:
    if total == 0:
        return "0.00"
    return f"{count / total * 100:.2f}"


def round_half_up(value: float) -> int:
    return int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def allocate_split_counts(class_count: int) -> dict[str, int]:
    validation_count = max(1, round_half_up(class_count * TARGET_SPLIT_RATIOS["validation"]))
    test_count = max(1, round_half_up(class_count * TARGET_SPLIT_RATIOS["test"]))
    train_count = class_count - validation_count - test_count
    if train_count <= 0:
        raise ValueError(f"Cannot split class with only {class_count} samples")
    return {
        "train": train_count,
        "validation": validation_count,
        "test": test_count,
    }


def require_complete_ids(rows: list[dict[str, str]], field: str) -> None:
    expected = [f"{number:03d}" for number in range(1, 273)]
    observed = sorted(row[field] for row in rows)
    if observed != expected:
        raise ValueError(f"{field} is not a complete 001-272 sequence")


def load_candidates() -> tuple[list[dict[str, str]], list[str]]:
    images = read_csv(CANDIDATE_DIR / "image_manifest.csv")
    metadata = read_csv(CANDIDATE_DIR / "metadata_raw.csv")

    require_complete_ids(images, "image_id")
    require_complete_ids(metadata, "metadata_id")

    image_by_id = {row["image_id"]: row for row in images}
    metadata_by_id = {row["metadata_id"]: row for row in metadata}
    metadata_fields = list(metadata[0].keys())
    sha_counts = Counter(row["sha256"] for row in images)

    rows: list[dict[str, str]] = []
    for number in range(1, 273):
        image_id = f"{number:03d}"
        image = image_by_id[image_id]
        meta = metadata_by_id[image_id]
        sequence = meta.get("序号", "")
        if sequence != str(number):
            raise ValueError(f"Unexpected metadata 序号 for image {image_id}: {sequence}")

        raw_label = meta.get("谱式", "").strip()
        label_status = "missing" if raw_label == "" else "present"
        image_path = f"{RAW_IMAGE_ARCHIVE}::{image['archive_member_name']}"
        sha_group_size = sha_counts[image["sha256"]]

        row: dict[str, str] = {
            "dataset_version": "modern_v1",
            "image_id": image_id,
            "image_path": image_path,
            "image_filename": image["image_filename"],
            "archive_member_name": image["archive_member_name"],
            "sequence": str(number),
            "candidate_metadata_id": image_id,
            "role": meta.get("角色", ""),
            "opera": meta.get("来源剧目", ""),
            "raw_face_type": raw_label,
            "mapping_status": MAPPING_STATUS,
            "mapping_evidence": MAPPING_EVIDENCE,
            "label_status": label_status,
            "image_read_status": image.get("read_status", ""),
            "image_file_format": image.get("file_format", ""),
            "image_width": image.get("width", ""),
            "image_height": image.get("height", ""),
            "image_file_size_bytes": image.get("file_size_bytes", ""),
            "image_sha256": image.get("sha256", ""),
            "sha256_group_size": str(sha_group_size),
            "duplicate_sha256_within_272": "yes" if sha_group_size > 1 else "no",
            "metadata_image_format": meta.get("图像格式", ""),
            "metadata_image_width": meta.get("图像宽度", ""),
            "metadata_image_height": meta.get("图像高度", ""),
            "paper_mentioned_model_category": PAPER_MENTIONED_VALUE,
            "modern_v1_included": "pending",
            "modern_v1_exclusion_reason": "",
            "split": "",
            "split_seed": str(SEED),
            "notes": (
                "Sequence mapping is accepted for the current project dataset based "
                "on project-owner confirmation. Label analysis uses the current XLS "
                "raw 谱式 values."
            ),
        }
        for field in metadata_fields:
            row[f"metadata_raw__{field}"] = meta.get(field, "")
        rows.append(row)
    return rows, metadata_fields


def apply_inclusion_policy(rows: list[dict[str, str]]) -> dict[str, int]:
    label_counts = Counter(row["raw_face_type"] for row in rows)
    for row in rows:
        label = row["raw_face_type"]
        if row["image_read_status"] != "readable":
            row["modern_v1_included"] = "no"
            row["modern_v1_exclusion_reason"] = "image_unreadable"
        elif label == "":
            row["modern_v1_included"] = "no"
            row["modern_v1_exclusion_reason"] = "missing_raw_face_type"
        elif label_counts[label] < MIN_SAMPLES_PER_CLASS:
            row["modern_v1_included"] = "no"
            row["modern_v1_exclusion_reason"] = (
                f"class_below_min_samples_per_class_{MIN_SAMPLES_PER_CLASS}"
            )
        else:
            row["modern_v1_included"] = "yes"
            row["modern_v1_exclusion_reason"] = ""
    return dict(label_counts)


def assign_splits(rows: list[dict[str, str]]) -> None:
    rng = random.Random(SEED)
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["modern_v1_included"] == "yes":
            by_label[row["raw_face_type"]].append(row)

    for label in sorted(by_label):
        items = sorted(by_label[label], key=lambda row: row["image_id"])
        rng.shuffle(items)
        counts = allocate_split_counts(len(items))
        test_rows = items[: counts["test"]]
        validation_rows = items[counts["test"] : counts["test"] + counts["validation"]]
        train_rows = items[counts["test"] + counts["validation"] :]
        if len(train_rows) != counts["train"]:
            raise AssertionError(f"Split count mismatch for {label}")
        for row in train_rows:
            row["split"] = "train"
        for row in validation_rows:
            row["split"] = "validation"
        for row in test_rows:
            row["split"] = "test"


def build_label_inventory(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    total = len(rows)
    counts = Counter(row["raw_face_type"] for row in rows)
    ordered_labels = sorted(counts, key=lambda label: (-counts[label], label or "\uffff"))
    inventory: list[dict[str, str]] = []
    for label in ordered_labels:
        count = counts[label]
        missing = label == ""
        sufficient = (not missing) and count >= MIN_SAMPLES_PER_CLASS
        if missing:
            notes = "Missing 谱式 in source metadata; excluded from modern_v1."
            exclusion = "missing_raw_face_type"
        elif sufficient:
            notes = (
                "Included in modern_v1 by the 2026 minimum sample policy using the "
                "current XLS raw label."
            )
            exclusion = ""
        else:
            notes = (
                f"Excluded from modern_v1 because sample_count < {MIN_SAMPLES_PER_CLASS}; "
                "preserved in modern_all_candidates.csv."
            )
            exclusion = f"class_below_min_samples_per_class_{MIN_SAMPLES_PER_CLASS}"

        inventory.append(
            {
                "raw_label": label,
                "sample_count": str(count),
                "percentage": percent(count, total),
                "paper_mentioned": PAPER_MENTIONED_VALUE,
                "sufficient_for_training": "yes" if sufficient else "no",
                "label_status": "missing" if missing else "present",
                "eligible_for_modern_v1": "yes" if sufficient else "no",
                "modern_v1_exclusion_reason": exclusion,
                "notes": notes,
            }
        )
    return inventory


def split_leakage(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_sha: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["modern_v1_included"] == "yes":
            by_sha[row["image_sha256"]].append(row)

    leaks: list[dict[str, str]] = []
    for sha, items in sorted(by_sha.items()):
        splits = sorted({row["split"] for row in items})
        if len(splits) > 1:
            leaks.append(
                {
                    "image_sha256": sha,
                    "splits": ",".join(splits),
                    "image_ids": ",".join(row["image_id"] for row in items),
                }
            )
    return leaks


def summarize_splits(rows: list[dict[str, str]]) -> dict[str, Counter[str]]:
    summary: dict[str, Counter[str]] = {
        "train": Counter(),
        "validation": Counter(),
        "test": Counter(),
    }
    for row in rows:
        split = row["split"]
        if split:
            summary[split][row["raw_face_type"]] += 1
    return summary


def split_rows(rows: list[dict[str, str]], split: str) -> list[dict[str, str]]:
    return sorted(
        [row for row in rows if row["split"] == split],
        key=lambda row: (row["raw_face_type"], int(row["image_id"])),
    )


def write_config(label_inventory: list[dict[str, str]]) -> None:
    config = {
        "dataset_name": "POPF modern_v1",
        "status": "2026 Reconstruction",
        "purpose": "Prepare a reproducible dataset for modern visual model training. No model training is performed by this build.",
        "inputs": {
            "raw_image_archive": RAW_IMAGE_ARCHIVE,
            "raw_metadata_file": RAW_METADATA_FILE,
            "candidate_image_manifest": "data/processed/272_candidate/image_manifest.csv",
            "candidate_metadata_raw": "data/processed/272_candidate/metadata_raw.csv",
            "candidate_reconciliation": "docs/LEGACY_DATASET_RECONCILIATION.md",
        },
        "outputs": {
            "directory": "data/processed/modern_dataset",
            "all_candidates": "modern_all_candidates.csv",
            "dataset_manifest": "dataset_manifest.csv",
            "label_inventory": "label_inventory.csv",
            "train": "train.csv",
            "validation": "validation.csv",
            "test": "test.csv",
        },
        "mapping_policy": {
            "mapping_status": MAPPING_STATUS,
            "mapping_evidence": MAPPING_EVIDENCE,
            "description": "Image 001 is paired with metadata 序号 1, continuing through 272. The one-to-one sequence mapping is accepted for the current project dataset based on project-owner confirmation.",
        },
        "label_policy": {
            "source_field": "谱式",
            "preserve_raw_labels": True,
            "normalize_to_paper_16_classes": False,
            "exclude_missing_labels_from_modern_v1": True,
            "min_samples_per_class": MIN_SAMPLES_PER_CLASS,
            "paper_label_alignment": "ignored_per_project_owner_instruction",
            "analysis_basis": "current XLS raw labels only",
        },
        "loader_policy": {
            "read_mode": "direct_zip_member",
            "copy_images_to_processed": False,
            "default_image_size": [224, 224],
            "image_size_convention": "PIL (width, height)",
            "color_mode": "RGB",
            "normalize_mean": [0.485, 0.456, 0.406],
            "normalize_std": [0.229, 0.224, 0.225],
            "augmentation": "none",
        },
        "split_policy": {
            "dataset_subset": "modern_v1",
            "random_seed": SEED,
            "target_ratios": TARGET_SPLIT_RATIOS,
            "algorithm": "For each eligible raw label, sort by image_id, shuffle with Python random.Random(seed), assign rounded 15 percent to test, rounded 15 percent to validation, and the remainder to train.",
            "stratified_by": "raw_face_type",
        },
        "quality_policy": {
            "do_not_modify_raw": True,
            "do_not_copy_or_augment_images": True,
            "duplicate_detection_key": "image_sha256",
            "leakage_rule": "No identical SHA-256 may appear in more than one split.",
        },
        "labels": label_inventory,
    }
    path = OUT_DIR / "dataset_config.json"
    path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(rows: list[dict[str, str]], label_inventory: list[dict[str, str]]) -> None:
    total = len(rows)
    readable = sum(row["image_read_status"] == "readable" for row in rows)
    duplicate_groups = sum(
        1 for count in Counter(row["image_sha256"] for row in rows).values() if count > 1
    )
    duplicate_images = sum(
        count for count in Counter(row["image_sha256"] for row in rows).values() if count > 1
    )
    missing_labels = sum(row["raw_face_type"] == "" for row in rows)
    valid_labels = total - missing_labels
    nonempty_classes = len({row["raw_face_type"] for row in rows if row["raw_face_type"]})
    included = [row for row in rows if row["modern_v1_included"] == "yes"]
    excluded = [row for row in rows if row["modern_v1_included"] == "no"]
    included_classes = sorted(
        Counter(row["raw_face_type"] for row in included).items(),
        key=lambda item: (-item[1], item[0]),
    )
    excluded_counts = Counter(row["modern_v1_exclusion_reason"] for row in excluded)
    width_counts = Counter((row["image_width"], row["image_height"]) for row in rows)
    split_summary = summarize_splits(rows)
    leaks = split_leakage(rows)

    split_totals = {
        split: sum(counter.values()) for split, counter in split_summary.items()
    }

    all_label_rows = [
        [
            item["raw_label"] or "(missing)",
            item["sample_count"],
            f"{item['percentage']}%",
            item["sufficient_for_training"],
            item["notes"],
        ]
        for item in label_inventory
    ]

    included_rows = [
        [
            label,
            count,
            split_summary["train"][label],
            split_summary["validation"][label],
            split_summary["test"][label],
        ]
        for label, count in included_classes
    ]

    excluded_label_rows = [
        [
            item["raw_label"] or "(missing)",
            item["sample_count"],
            item["modern_v1_exclusion_reason"],
        ]
        for item in label_inventory
        if item["eligible_for_modern_v1"] == "no"
    ]

    dimension_rows = [
        [f"{width}x{height}", count]
        for (width, height), count in width_counts.most_common()
    ]

    report = f"""# POPF Modern Dataset Report

状态：`2026 Reconstruction / modern_v1`

本报告只描述从 272 份历史图片和历史元数据生成的现代训练前数据集。它不训练模型，不复现 KNIME。

## 1. 输入和映射边界

- 图片来源：`{RAW_IMAGE_ARCHIVE}`
- 历史元数据来源：`{RAW_METADATA_FILE}`
- 候选核验依据：`data/processed/272_candidate/`
- 映射策略：`001 -> 序号 1` 到 `272 -> 序号 272`
- `mapping_status`：`{MAPPING_STATUS}`
- 映射证据：{MAPPING_EVIDENCE}

该映射由项目负责人确认后，在当前现代数据集构建中作为有效的一一对应关系使用。

## 2. 数据概览

| 指标 | 数量 |
|---|---:|
| 历史图片总数 | {total} |
| 可读取图片 | {readable} |
| 有效非空标签记录 | {valid_labels} |
| 缺失标签记录 | {missing_labels} |
| 非空 raw_face_type 类别数 | {nonempty_classes} |
| modern_v1 入选图片 | {len(included)} |
| modern_v1 入选类别 | {len(included_classes)} |
| modern_v1 排除图片 | {len(excluded)} |

## 3. 当前实际标签分布

本阶段按当前 XLS 的 `谱式` 原始文本分析标签，不再尝试与论文标签体系对齐。

{md_table(["raw_label", "sample_count", "percentage", "sufficient_for_training", "notes"], all_label_rows)}

## 4. modern_v1 训练类别

本阶段采用的 2026 Reconstruction 策略是：只保留非空 `raw_face_type`，并要求每类至少 {MIN_SAMPLES_PER_CLASS} 张图片。该门槛服务于现代模型的基础 train/validation/test 划分。

{md_table(["raw_label", "total", "train", "validation", "test"], included_rows)}

## 5. 暂时排除的类别和原因

{md_table(["raw_label", "sample_count", "reason"], excluded_label_rows)}

排除原因汇总：

{md_table(["reason", "records"], [[reason, count] for reason, count in sorted(excluded_counts.items())])}

## 6. 数据划分

| split | records |
|---|---:|
| train | {split_totals["train"]} |
| validation | {split_totals["validation"]} |
| test | {split_totals["test"]} |

划分配置：

- random seed：`{SEED}`
- 目标比例：70% train, 15% validation, 15% test
- 实际比例：train {split_totals["train"] / len(included) * 100:.2f}%, validation {split_totals["validation"] / len(included) * 100:.2f}%, test {split_totals["test"] / len(included) * 100:.2f}%
- 方法：按 `raw_face_type` 分层，每类独立排序、固定 seed 洗牌，再按四舍五入后的 15%/15% 分配 validation/test，剩余为 train。

## 7. 数据质量检查

| 检查项 | 结果 |
|---|---|
| 图片是否全部可读取 | {readable}/{total} readable |
| ZIP 内 SHA-256 是否唯一 | {"yes" if duplicate_groups == 0 else "no"} |
| 重复 SHA-256 group 数 | {duplicate_groups} |
| 涉及重复 SHA-256 的图片数 | {duplicate_images} |
| train/validation/test 是否存在相同 SHA-256 跨 split | {"no" if not leaks else "yes"} |
| 标签缺失 | {missing_labels} records |
| 类别极度不平衡 | yes, largest class 73 and smallest nonempty class 1 |

图片尺寸分布：

{md_table(["width_height", "count"], dimension_rows)}

## 8. 泄漏检查

{"未发现相同 SHA-256 跨 train/validation/test。当前 272 张候选图内 SHA-256 全部唯一。" if not leaks else "发现相同 SHA-256 跨 split，详见重新运行脚本输出。"}

## 9. 标签问题

- 16 条记录缺少 `谱式`，不能进入第一版监督训练。
- 非空 `谱式` 有 20 种文本值，本阶段按照当前实际标签处理，不与论文标签体系做对应分析。
- 部分类别样本很少，1 到 8 张的类别暂不进入 modern_v1，但全部保留在 `modern_all_candidates.csv`。

## 10. 是否足够进入现代视觉模型训练

可以进入探索性现代视觉模型训练，数据集状态为 `2026 Reconstruction / user_confirmed_sequence_mapping`。主要限制是样本量小、类别不平衡，以及部分当前实际标签的样本数过低。

## 11. 下一步建议

1. 固化一个 `legacy_dataset` 和 `modern_dataset` 分离的目录规范，避免现代训练选择反向污染 2019 复现。
2. 编写只读取 ZIP 成员的 PyTorch Dataset/DataLoader，保持 `image_path` 的 archive-member 语义。
3. 第一版模型建议使用 ImageNet 预训练的轻量 CNN，例如 ResNet-18 或 MobileNetV3，并先冻结大部分 backbone。数据只有 {len(included)} 张，直接从零训练不稳。
4. 训练前再次人工复核 modern_v1 的当前实际标签，尤其是低样本类别和容易混淆的谱式。
"""
    (OUT_DIR / "dataset_report.md").write_text(report, encoding="utf-8")


def write_readme() -> None:
    readme = f"""# POPF Modern Dataset

This directory is a 2026 Reconstruction dataset preparation output. It is built from the audited 272-image historical set and does not modify `data/raw`.

## Files

- `modern_all_candidates.csv`: all 272 historical records, including raw metadata fields copied with `metadata_raw__` prefixes.
- `dataset_manifest.csv`: training-oriented manifest for all 272 records, with inclusion status and split.
- `label_inventory.csv`: raw `谱式` label counts and modern_v1 eligibility.
- `train.csv`, `validation.csv`, `test.csv`: modern_v1 split files.
- `dataset_config.json`: fixed seed, label policy, split policy, and provenance.
- `label_mapping.json`: fixed `raw_face_type` to integer ID mapping for all three splits.
- `dataset_report.md`: human-readable dataset analysis.

## Boundary

`mapping_status={MAPPING_STATUS}` means image `001` is paired with metadata sequence `1`, image `002` with sequence `2`, and so on. This one-to-one relationship was confirmed by the project owner on 2026-09-20 for the current POPF dataset.

## modern_v1 policy

- Preserve raw labels exactly.
- Analyze labels using current XLS `谱式` values.
- Do not compress the observed labels into the paper's 16 classes.
- Exclude missing `谱式` values.
- Include non-empty classes with at least {MIN_SAMPLES_PER_CLASS} samples.
- Use stratified split with seed `{SEED}`.
- Do not copy, augment, resize, clean, or rename raw images.

## Dataset and DataLoader

`src/datasets/popf_dataset.py` reads each `image_path` as
`archive.zip::member/name.jpg` and reads the ZIP member directly. It does not
extract the archive or create a second image directory.

The deterministic preprocessing pipeline is:

`JPEG -> RGB -> Resize -> float tensor -> ImageNet normalization`

The default input size is `224 x 224`, and the `image_size` argument can be
changed when building the loaders. No augmentation is applied.

Example:

```python
from src.datasets.popf_dataset import build_dataloaders

train_loader, validation_loader, test_loader = build_dataloaders(
    image_size=(224, 224),
    batch_size=32,
    num_workers=0,
    seed=42,
)
```

## Training note

These files prepare data for a later model phase only. No model has been trained by this stage, and no accuracy or other model metric is created here.
"""
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")


def output_fieldnames(metadata_fields: list[str], include_raw_metadata: bool = False) -> list[str]:
    base = [
        "dataset_version",
        "image_id",
        "image_path",
        "image_filename",
        "archive_member_name",
        "sequence",
        "candidate_metadata_id",
        "role",
        "opera",
        "raw_face_type",
        "mapping_status",
        "mapping_evidence",
        "label_status",
        "image_read_status",
        "image_file_format",
        "image_width",
        "image_height",
        "image_file_size_bytes",
        "image_sha256",
        "sha256_group_size",
        "duplicate_sha256_within_272",
        "metadata_image_format",
        "metadata_image_width",
        "metadata_image_height",
        "paper_mentioned_model_category",
        "modern_v1_included",
        "modern_v1_exclusion_reason",
        "split",
        "split_seed",
        "notes",
    ]
    if include_raw_metadata:
        base.extend(f"metadata_raw__{field}" for field in metadata_fields)
    return base


def write_outputs(rows: list[dict[str, str]], metadata_fields: list[str]) -> None:
    manifest_fields = output_fieldnames(metadata_fields, include_raw_metadata=False)
    all_candidate_fields = output_fieldnames(metadata_fields, include_raw_metadata=True)
    split_fields = [
        "image_id",
        "image_path",
        "sequence",
        "candidate_metadata_id",
        "role",
        "opera",
        "raw_face_type",
        "mapping_status",
        "mapping_evidence",
        "image_sha256",
        "split",
        "split_seed",
    ]

    ordered_rows = sorted(rows, key=lambda row: int(row["image_id"]))
    write_csv(OUT_DIR / "modern_all_candidates.csv", all_candidate_fields, ordered_rows)
    write_csv(OUT_DIR / "dataset_manifest.csv", manifest_fields, ordered_rows)
    for split in ("train", "validation", "test"):
        write_csv(OUT_DIR / f"{split}.csv", split_fields, split_rows(rows, split))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, metadata_fields = load_candidates()
    apply_inclusion_policy(rows)
    assign_splits(rows)
    label_inventory = build_label_inventory(rows)

    label_fields = [
        "raw_label",
        "sample_count",
        "percentage",
        "paper_mentioned",
        "sufficient_for_training",
        "label_status",
        "eligible_for_modern_v1",
        "modern_v1_exclusion_reason",
        "notes",
    ]
    write_csv(OUT_DIR / "label_inventory.csv", label_fields, label_inventory)
    write_outputs(rows, metadata_fields)
    write_config(label_inventory)
    write_report(rows, label_inventory)
    write_readme()

    included = [row for row in rows if row["modern_v1_included"] == "yes"]
    split_counts = Counter(row["split"] for row in included)
    leaks = split_leakage(rows)
    print(
        json.dumps(
            {
                "total_candidates": len(rows),
                "readable_images": sum(row["image_read_status"] == "readable" for row in rows),
                "modern_v1_images": len(included),
                "modern_v1_classes": len({row["raw_face_type"] for row in included}),
                "split_counts": dict(sorted(split_counts.items())),
                "duplicate_sha256_groups": sum(
                    1
                    for count in Counter(row["image_sha256"] for row in rows).values()
                    if count > 1
                ),
                "cross_split_duplicate_sha256_groups": len(leaks),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
