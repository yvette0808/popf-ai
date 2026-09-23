#!/usr/bin/env python3
"""Run the POPF controlled ResNet-18 layer4 fine-tuning experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torchvision

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.popf_dataset import (  # noqa: E402
    DEFAULT_DATASET_DIR,
    build_dataloaders,
    load_label_mapping,
    read_csv_rows,
)
from src.models.resnet import (  # noqa: E402
    build_resnet18,
    resnet18_trainability_audit,
)
from src.training.engine import (  # noqa: E402
    compute_class_weights,
    run_epoch,
    seed_everything,
)
from src.training.reporting import (  # noqa: E402
    save_test_error_grid,
    save_training_curves,
    write_prediction_csv,
)


EXPERIMENT_NAME = "resnet18_layer4_finetune"
BASELINE_EXPERIMENT = "resnet18_baseline"
EXPECTED_COUNTS = {"train": 154, "validation": 34, "test": 34}
EXPECTED_TRAINABLE_MODULES = ("layer4", "fc")
RAW_INPUTS = (
    ROOT / "data/raw/脸谱/整理工作/爬虫脸谱图集.zip",
    ROOT / "data/raw/脸谱新一步/272数据张丹阳.xls",
)


def parse_args() -> Any:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--layer4-learning-rate", type=float, default=1e-4)
    parser.add_argument("--fc-learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    epoch: int,
    model_config: dict[str, Any],
    label_mapping: dict[str, int],
    preprocess: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": cpu_state_dict(model),
            "epoch": epoch,
            "model_config": model_config,
            "label_mapping": label_mapping,
            "preprocess": preprocess,
            "metrics": metrics,
        },
        path,
    )


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def validate_dataset_and_baseline(
    dataset_dir: Path,
    label_mapping: dict[str, int],
) -> dict[str, Any]:
    """Verify the controlled experiment inputs without regenerating them."""

    baseline_dir = ROOT / "artifacts/experiments" / BASELINE_EXPERIMENT
    baseline_config = load_json(baseline_dir / "config.json")
    baseline_mapping = baseline_config["label_mapping"]
    if label_mapping != baseline_mapping:
        raise AssertionError("Current label_mapping.json differs from baseline mapping")

    split_rows = {
        split: read_csv_rows(dataset_dir / f"{split}.csv")
        for split in ("train", "validation", "test")
    }
    counts = {split: len(rows) for split, rows in split_rows.items()}
    if counts != EXPECTED_COUNTS:
        raise AssertionError(f"Dataset counts changed: {counts}")
    baseline_split_counts = {
        split: baseline_config["dataset_counts"][split]
        for split in ("train", "validation", "test")
    }
    if counts != baseline_split_counts:
        raise AssertionError("Current split counts differ from baseline metadata")

    all_ids = [
        row["image_id"]
        for rows in split_rows.values()
        for row in rows
    ]
    if len(all_ids) != len(set(all_ids)):
        raise AssertionError("An image_id occurs in more than one split")

    split_hashes = {
        split: {row["image_sha256"] for row in rows if row.get("image_sha256")}
        for split, rows in split_rows.items()
    }
    overlaps = {}
    split_names = list(split_hashes)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlaps[f"{left}_vs_{right}"] = sorted(
                split_hashes[left].intersection(split_hashes[right])
            )
    if any(overlaps.values()):
        raise AssertionError(f"SHA-256 leakage detected: {overlaps}")

    baseline_control = {
        "dataset_name": baseline_config["dataset_name"],
        "dataset_counts": baseline_config["dataset_counts"],
        "class_names": baseline_config["class_names"],
        "label_mapping": baseline_mapping,
        "batch_size": baseline_config["batch_size"],
        "weight_decay": baseline_config["weight_decay"],
        "epochs": baseline_config["epochs"],
        "seed": baseline_config["seed"],
        "image_size": baseline_config["image_size"],
        "model_name": baseline_config["model_name"],
        "pretrained": baseline_config["pretrained"],
        "use_class_weight": baseline_config["use_class_weight"],
        "augmentation": "none",
        "optimizer": "AdamW",
    }
    current_snapshot = {
        "split_counts": counts,
        "split_csv_sha256": {
            split: sha256_file(dataset_dir / f"{split}.csv")
            for split in ("train", "validation", "test")
        },
        "label_mapping_sha256": sha256_file(dataset_dir / "label_mapping.json"),
        "sha256_overlaps": overlaps,
    }
    return {
        "split_rows": split_rows,
        "baseline_config": baseline_config,
        "baseline_control": baseline_control,
        "current_snapshot": current_snapshot,
    }


def make_optimizer(
    model: torch.nn.Module,
    *,
    layer4_learning_rate: float,
    fc_learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    layer4_parameters = [parameter for parameter in model.layer4.parameters() if parameter.requires_grad]
    fc_parameters = [parameter for parameter in model.fc.parameters() if parameter.requires_grad]
    if not layer4_parameters or not fc_parameters:
        raise AssertionError("Expected trainable layer4 and fc parameters")
    return torch.optim.AdamW(
        [
            {"name": "layer4", "params": layer4_parameters, "lr": layer4_learning_rate},
            {"name": "fc", "params": fc_parameters, "lr": fc_learning_rate},
        ],
        weight_decay=weight_decay,
    )


def format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def prediction_rows_by_image(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            row["image_id"]: row
            for row in csv.DictReader(handle)
        }


def write_prediction_comparison(
    baseline_path: Path,
    finetune_path: Path,
    output_path: Path,
) -> None:
    baseline = prediction_rows_by_image(baseline_path)
    finetune = prediction_rows_by_image(finetune_path)
    if set(baseline) != set(finetune):
        raise AssertionError("Baseline and fine-tuning prediction IDs differ")

    fieldnames = [
        "image_id",
        "image_path",
        "true_label",
        "baseline_top1_label",
        "baseline_top2_label",
        "baseline_top3_label",
        "baseline_top1_score",
        "baseline_top2_score",
        "baseline_top3_score",
        "finetune_top1_label",
        "finetune_top2_label",
        "finetune_top3_label",
        "finetune_top1_score",
        "finetune_top2_score",
        "finetune_top3_score",
        "baseline_top1_correct",
        "finetune_top1_correct",
        "top1_prediction_changed",
        "top3_overlap_count",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for image_id in sorted(baseline):
            old = baseline[image_id]
            new = finetune[image_id]
            baseline_top3 = {
                old["top1_label"],
                old["top2_label"],
                old["top3_label"],
            }
            finetune_top3 = {
                new["top1_label"],
                new["top2_label"],
                new["top3_label"],
            }
            writer.writerow(
                {
                    "image_id": image_id,
                    "image_path": new["image_path"],
                    "true_label": new["true_label"],
                    "baseline_top1_label": old["top1_label"],
                    "baseline_top2_label": old["top2_label"],
                    "baseline_top3_label": old["top3_label"],
                    "baseline_top1_score": old["top1_score"],
                    "baseline_top2_score": old["top2_score"],
                    "baseline_top3_score": old["top3_score"],
                    "finetune_top1_label": new["top1_label"],
                    "finetune_top2_label": new["top2_label"],
                    "finetune_top3_label": new["top3_label"],
                    "finetune_top1_score": new["top1_score"],
                    "finetune_top2_score": new["top2_score"],
                    "finetune_top3_score": new["top3_score"],
                    "baseline_top1_correct": old["top1_correct"],
                    "finetune_top1_correct": new["top1_correct"],
                    "top1_prediction_changed": old["top1_label"] != new["top1_label"],
                    "top3_overlap_count": len(baseline_top3.intersection(finetune_top3)),
                }
            )


def write_finetune_report(
    path: Path,
    *,
    config: dict[str, Any],
    class_counts: list[int],
    class_weights: list[float],
    history: list[dict[str, Any]],
    selection: dict[str, Any],
    test_metrics: dict[str, Any],
    error_count: int,
    baseline_test_metrics: dict[str, Any],
    baseline_config: dict[str, Any],
) -> None:
    best_history = history[selection["selected_epoch"] - 1]
    train_f1 = float(best_history["train_accuracy"])
    validation_f1 = float(best_history["validation_macro_f1"])
    overfit_gap = train_f1 - validation_f1
    test_f1_difference = test_metrics["macro_f1"] - baseline_test_metrics["macro_f1"]
    if test_f1_difference > 0.01:
        next_step = "Fine-tuning shows a measurable test macro-F1 improvement; the next single experiment should test a more conservative fine-tuning schedule while keeping the same split and test holdout."
    elif test_f1_difference < -0.01:
        next_step = "Fine-tuning does not improve test macro-F1; the next single experiment should prioritize data quality and category-boundary analysis before adding model complexity."
    else:
        next_step = "Fine-tuning does not show a meaningful test macro-F1 change; the next single experiment should prioritize data quality and category-boundary analysis before adding model complexity."

    lines = [
        "# POPF ResNet-18 Layer4 Controlled Fine-tuning Report",
        "",
        "Status: `2026 Reconstruction` experiment. This is not a 2019 Original result.",
        "",
        f"Run timestamp (UTC): `{config['run_timestamp_utc']}`",
        "",
        "## 1. Experiment purpose",
        "",
        "This experiment tests whether unfreezing only ResNet-18 `layer4` plus the new classification head improves generalization over the frozen-backbone baseline.",
        "The experiment changes the trainable parameter set only. Dataset files, labels, split membership, preprocessing, class weights, seed, optimizer family, batch size, weight decay, epoch count, and test holdout remain controlled.",
        "",
        "## 2. Data and controls",
        "",
        f"- Dataset: `{config['dataset_name']}`",
        f"- Images: `{config['dataset_counts']['total']}` (`train={config['dataset_counts']['train']}`, `validation={config['dataset_counts']['validation']}`, `test={config['dataset_counts']['test']}`)",
        f"- Classes: `{len(config['class_names'])}`",
        "- Split files were not regenerated or modified.",
        "- Test data was not read during training or validation model selection; it was evaluated once after selecting the best validation macro-F1 checkpoint.",
        "",
        "| Condition | Frozen baseline | Layer4 fine-tuning |",
        "|---|---|---|",
        "| Model | ImageNet ResNet-18 | ImageNet ResNet-18 |",
        "| Trainable modules | `fc` | `layer4`, `fc` |",
        "| Input | RGB 224x224 | RGB 224x224 |",
        "| Augmentation | none | none |",
        "| Loss | train-only weighted CE | train-only weighted CE |",
        "| Optimizer | AdamW | AdamW |",
        "| Batch size | 16 | 16 |",
        "| Weight decay | 1e-4 | 1e-4 |",
        "| Epochs | 20 | 20 |",
        "| Seed | 42 | 42 |",
        "",
        "| Label | Total | Train | Class weight |",
        "|---|---:|---:|---:|",
    ]
    for label, total, count, weight in zip(
        config["class_names"],
        config["class_counts_total"],
        class_counts,
        class_weights,
    ):
        lines.append(f"| {label} | {total} | {count} | {weight:.6f} |")
    lines.extend(
        [
            "",
            "## 3. Model structure and trainability audit",
            "",
            "- Frozen: `conv1`, `bn1`, `layer1`, `layer2`, `layer3`",
            "- Trainable: `layer4`, `fc`",
            f"- Parameters: `{config['trainable_parameters']}` trainable / `{config['frozen_parameters']}` frozen / `{config['total_parameters']}` total",
            f"- Layer4 learning rate: `{config['layer4_learning_rate']}`",
            f"- FC learning rate: `{config['fc_learning_rate']}`",
            "",
            "| Module | Trainable | Parameters | Trainable parameters |",
            "|---|---:|---:|---:|",
        ]
    )
    for module_name, audit in config["trainability"]["modules"].items():
        lines.append(
            f"| {module_name} | {audit['trainable']} | "
            f"{audit['parameter_count']} | {audit['trainable_parameter_count']} |"
        )
    lines.extend(
        [
            "",
            "## 4. Training curves and validation selection",
            "",
            "![Training curves](resnet18_layer4_finetune_training_curves.png)",
            "",
            f"- Best epoch by validation macro-F1: `{selection['selected_epoch']}`",
            f"- Best validation accuracy: `{selection['best_validation_accuracy']:.6f}`",
            f"- Best validation macro-F1: `{selection['best_validation_macro_f1']:.6f}`",
            "",
            "| Label | Validation F1 at selected epoch |",
            "|---|---:|",
        ]
    )
    for item in best_history["validation_per_class"]:
        lines.append(f"| {item['label_name']} | {item['f1']:.6f} |")
    lines.extend(
        [
            "",
            "## 5. Final test evaluation",
            "",
            "The selected validation checkpoint was evaluated on the test split once.",
            "",
            f"- Accuracy: `{test_metrics['accuracy']:.6f}`",
            f"- Macro precision: `{test_metrics['macro_precision']:.6f}`",
            f"- Macro recall: `{test_metrics['macro_recall']:.6f}`",
            f"- Macro F1: `{test_metrics['macro_f1']:.6f}`",
            f"- Top-3 accuracy: `{test_metrics['top3_accuracy']:.6f}`",
            "",
            "### Confusion matrix",
            "",
            "Rows are true labels and columns are predicted labels, in label-mapping order.",
            "",
            "```text",
            json.dumps(test_metrics["confusion_matrix"], ensure_ascii=False),
            "```",
            "",
            "### Per-class metrics",
            "",
            "| Label | Support | Precision | Recall | F1 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for item in test_metrics["per_class"]:
        lines.append(
            f"| {item['label_name']} | {item['support']} | "
            f"{item['precision']:.6f} | {item['recall']:.6f} | {item['f1']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 6. Baseline versus fine-tuning",
            "",
            f"- Frozen baseline best validation epoch: `{baseline_config['model_selection_epoch']}`",
            "",
            "| Metric | Frozen baseline | Layer4 fine-tuning | Difference |",
            "|---|---:|---:|---:|",
            f"| Validation accuracy | {baseline_config['best_validation_accuracy']:.6f} | {selection['best_validation_accuracy']:.6f} | {selection['best_validation_accuracy'] - baseline_config['best_validation_accuracy']:+.6f} |",
            f"| Validation macro-F1 | {baseline_config['best_validation_macro_f1']:.6f} | {selection['best_validation_macro_f1']:.6f} | {selection['best_validation_macro_f1'] - baseline_config['best_validation_macro_f1']:+.6f} |",
            f"| Test accuracy | {baseline_test_metrics['accuracy']:.6f} | {test_metrics['accuracy']:.6f} | {test_metrics['accuracy'] - baseline_test_metrics['accuracy']:+.6f} |",
            f"| Test macro precision | {baseline_test_metrics['macro_precision']:.6f} | {test_metrics['macro_precision']:.6f} | {test_metrics['macro_precision'] - baseline_test_metrics['macro_precision']:+.6f} |",
            f"| Test macro recall | {baseline_test_metrics['macro_recall']:.6f} | {test_metrics['macro_recall']:.6f} | {test_metrics['macro_recall'] - baseline_test_metrics['macro_recall']:+.6f} |",
            f"| Test macro-F1 | {baseline_test_metrics['macro_f1']:.6f} | {test_metrics['macro_f1']:.6f} | {test_metrics['macro_f1'] - baseline_test_metrics['macro_f1']:+.6f} |",
            f"| Test top-3 accuracy | {baseline_test_metrics['top3_accuracy']:.6f} | {test_metrics['top3_accuracy']:.6f} | {test_metrics['top3_accuracy'] - baseline_test_metrics['top3_accuracy']:+.6f} |",
            "",
            "| Label | Baseline test F1 | Fine-tuning test F1 | Difference |",
            "|---|---:|---:|---:|",
        ]
    )
    baseline_per_class = {
        item["label_name"]: item["f1"] for item in baseline_test_metrics["per_class"]
    }
    finetune_per_class = {
        item["label_name"]: item["f1"] for item in test_metrics["per_class"]
    }
    for label in config["class_names"]:
        lines.append(
            f"| {label} | {baseline_per_class[label]:.6f} | "
            f"{finetune_per_class[label]:.6f} | "
            f"{finetune_per_class[label] - baseline_per_class[label]:+.6f} |"
        )
    lines.extend(
        [
            "",
            "Baseline test confusion matrix:",
            "",
            "```text",
            json.dumps(baseline_test_metrics["confusion_matrix"], ensure_ascii=False),
            "```",
            "",
            "Fine-tuning test confusion matrix:",
            "",
            "```text",
            json.dumps(test_metrics["confusion_matrix"], ensure_ascii=False),
            "```",
            "",
            "## 7. Error analysis and overfitting",
            "",
            f"- Test top-1 error tiles rendered (capped at 12): `{error_count}`.",
            "- Full fine-tuning top-k predictions: `../artifacts/experiments/resnet18_layer4_finetune/test_predictions_top3.csv`",
            "- Baseline versus fine-tuning prediction comparison: `../artifacts/experiments/resnet18_layer4_finetune/baseline_vs_finetune_test_predictions.csv`",
            "",
            "![Fine-tuning test errors](resnet18_layer4_finetune_test_errors.png)",
            "",
            f"- At the selected epoch, train accuracy was `{train_f1:.6f}` and validation macro-F1 was `{validation_f1:.6f}`; the raw gap is `{overfit_gap:+.6f}`.",
            "- A large train/validation gap is treated as an overfitting warning, not as evidence of better generalization.",
            "",
            "## 8. Next step",
            "",
            next_step,
            "No next experiment was executed automatically.",
            "",
            "The approximately 70% accuracy reported by the 2019 paper is historical `2019 Original` context and is not compared to this `2026 Reconstruction` result.",
            "",
            "## 9. Reproducibility",
            "",
            f"- Python: `{config['python_version']}`",
            f"- PyTorch: `{config['torch_version']}`",
            f"- torchvision: `{config['torchvision_version']}`",
            f"- Device: `{config['device']}`",
            f"- Raw input SHA-256 recorded before and after: `{config['raw_input_sha256']}`",
            f"- Dataset split SHA-256: `{json.dumps(config['dataset_snapshot']['split_csv_sha256'], ensure_ascii=False)}`",
            "",
            "No raw file or modern dataset split was modified by this experiment.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_comparison_report(
    path: Path,
    *,
    config: dict[str, Any],
    baseline_config: dict[str, Any],
    baseline_test: dict[str, Any],
    finetune_test: dict[str, Any],
    selection: dict[str, Any],
    validation_metrics: dict[str, Any],
) -> None:
    metrics = [
        ("Validation Accuracy", baseline_config["best_validation_accuracy"], selection["best_validation_accuracy"]),
        ("Validation Macro-F1", baseline_config["best_validation_macro_f1"], selection["best_validation_macro_f1"]),
        ("Test Accuracy", baseline_test["accuracy"], finetune_test["accuracy"]),
        ("Test Macro-F1", baseline_test["macro_f1"], finetune_test["macro_f1"]),
        ("Test Macro Precision", baseline_test["macro_precision"], finetune_test["macro_precision"]),
        ("Test Macro Recall", baseline_test["macro_recall"], finetune_test["macro_recall"]),
        ("Test Top-3 Accuracy", baseline_test["top3_accuracy"], finetune_test["top3_accuracy"]),
    ]
    lines = [
        "# POPF ResNet-18 Frozen Baseline vs Layer4 Fine-tuning",
        "",
        "Both experiments are `2026 Reconstruction` experiments. Neither result is a 2019 Original result.",
        "",
        "## Controlled conditions",
        "",
        "- Same modern_v1 dataset, label mapping, and fixed `154/34/34` split.",
        "- Same seed `42`, batch size `16`, epochs `20`, weight decay `1e-4`, AdamW, RGB `224x224`, ImageNet normalization, and no augmentation.",
        "- Same train-only class weights.",
        "- Test was not used for model selection and was evaluated once per experiment after validation selection.",
        "- The only intended model variable is trainability: baseline trains `fc`; fine-tuning trains `layer4 + fc` with separate learning rates.",
        "",
        "| Metric | Frozen baseline | Layer4 fine-tuning | Difference |",
        "|---|---:|---:|---:|",
    ]
    for name, baseline_value, finetune_value in metrics:
        lines.append(
            f"| {name} | {baseline_value:.6f} | {finetune_value:.6f} | "
            f"{finetune_value - baseline_value:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Validation per-class F1 at selected checkpoints",
            "",
            f"- Baseline selected epoch: `{baseline_config['model_selection_epoch']}`",
            f"- Fine-tuning selected epoch: `{selection['selected_epoch']}`",
            "",
            "| Label | Frozen baseline | Layer4 fine-tuning | Difference |",
            "|---|---:|---:|---:|",
        ]
    )
    baseline_validation = {
        item["label_name"]: item["f1"]
        for item in baseline_config["best_validation_per_class"]
    }
    validation_per_class = validation_metrics.get("per_class")
    if validation_per_class is None:
        validation_per_class = validation_metrics["validation_per_class"]
    finetune_validation = {
        item["label_name"]: item["f1"]
        for item in validation_per_class
    }
    for label in config["class_names"]:
        lines.append(
            f"| {label} | {baseline_validation[label]:.6f} | "
            f"{finetune_validation[label]:.6f} | "
            f"{finetune_validation[label] - baseline_validation[label]:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Test per-class F1",
            "",
            "| Label | Frozen baseline | Layer4 fine-tuning | Difference |",
            "|---|---:|---:|---:|",
        ]
    )
    baseline_test_f1 = {item["label_name"]: item["f1"] for item in baseline_test["per_class"]}
    finetune_test_f1 = {item["label_name"]: item["f1"] for item in finetune_test["per_class"]}
    for label in config["class_names"]:
        lines.append(
            f"| {label} | {baseline_test_f1[label]:.6f} | "
            f"{finetune_test_f1[label]:.6f} | "
            f"{finetune_test_f1[label] - baseline_test_f1[label]:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Confusion matrices",
            "",
            "Baseline test confusion matrix:",
            "",
            "```text",
            json.dumps(baseline_test["confusion_matrix"], ensure_ascii=False),
            "```",
            "",
            "Layer4 fine-tuning test confusion matrix:",
            "",
            "```text",
            json.dumps(finetune_test["confusion_matrix"], ensure_ascii=False),
            "```",
            "",
            "## Prediction-level comparison",
            "",
            "- `artifacts/experiments/resnet18_layer4_finetune/baseline_vs_finetune_test_predictions.csv` records both models' Top-3 rankings for each test image.",
            "- `reports/resnet18_layer4_finetune_test_errors.png` shows fine-tuning top-1 errors for manual inspection.",
            "",
            "No next experiment was executed automatically.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.image_size <= 0:
        raise ValueError("--image-size must be positive")
    if args.layer4_learning_rate <= 0 or args.fc_learning_rate <= 0:
        raise ValueError("Learning rates must be positive")

    seed_everything(args.seed)
    device = choose_device(args.device)
    dataset_dir = args.dataset_dir.resolve()
    output_root = args.output_root.resolve()
    experiment_dir = output_root / "experiments" / EXPERIMENT_NAME
    model_dir = output_root / "models" / EXPERIMENT_NAME
    reports_dir = ROOT / "reports"

    raw_before = {str(path.relative_to(ROOT)): sha256_file(path) for path in RAW_INPUTS}
    label_mapping = load_label_mapping(dataset_dir / "label_mapping.json")
    preflight = validate_dataset_and_baseline(dataset_dir, label_mapping)
    split_rows = preflight["split_rows"]
    baseline_config = preflight["baseline_config"]
    class_names = [
        label for label, _ in sorted(label_mapping.items(), key=lambda item: item[1])
    ]
    train_rows = split_rows["train"]
    train_class_counts, class_weights = compute_class_weights(train_rows, label_mapping)
    total_class_counts = [
        sum(
            1
            for row in split_rows["train"] + split_rows["validation"] + split_rows["test"]
            if label_mapping[row["raw_face_type"].strip()] == class_id
        )
        for class_id in range(len(class_names))
    ]

    train_loader, validation_loader, test_loader = build_dataloaders(
        dataset_dir=dataset_dir,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    model = build_resnet18(
        len(class_names),
        pretrained=True,
        freeze_backbone=False,
        trainable_modules=EXPECTED_TRAINABLE_MODULES,
    ).to(device)
    trainability = resnet18_trainability_audit(model)
    expected_frozen = {"conv1", "bn1", "layer1", "layer2", "layer3"}
    if set(trainability["trainable_modules"]) != set(EXPECTED_TRAINABLE_MODULES):
        raise AssertionError(f"Unexpected trainable modules: {trainability}")
    if set(trainability["frozen_modules"]) != expected_frozen:
        raise AssertionError(f"Unexpected frozen modules: {trainability}")
    print("trainability audit:", json.dumps(trainability, ensure_ascii=False, indent=2))

    config: dict[str, Any] = {
        "experiment_name": EXPERIMENT_NAME,
        "stage": "2026 Reconstruction",
        "dataset_name": "POPF modern_v1",
        "dataset_dir": str(dataset_dir),
        "dataset_counts": {
            "train": len(split_rows["train"]),
            "validation": len(split_rows["validation"]),
            "test": len(split_rows["test"]),
            "total": sum(len(rows) for rows in split_rows.values()),
        },
        "class_names": class_names,
        "class_counts_total": total_class_counts,
        "class_counts_train": train_class_counts,
        "class_weights": class_weights,
        "class_weights_from": "train.csv only",
        "label_mapping": label_mapping,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "layer4_learning_rate": args.layer4_learning_rate,
        "fc_learning_rate": args.fc_learning_rate,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "num_workers": args.num_workers,
        "image_size": args.image_size,
        "model_name": "resnet18",
        "pretrained": True,
        "freeze_backbone": False,
        "unfreeze_layers": list(EXPECTED_TRAINABLE_MODULES),
        "use_class_weight": True,
        "augmentation": "none",
        "optimizer": "AdamW",
        "device": str(device),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "trainability": trainability,
        "total_parameters": trainability["total_parameters"],
        "trainable_parameters": trainability["trainable_parameters"],
        "frozen_parameters": trainability["frozen_parameters"],
        "raw_input_sha256": raw_before,
        "dataset_snapshot": preflight["current_snapshot"],
        "baseline_control": preflight["baseline_control"],
    }
    experiment_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    json_dump(config, experiment_dir / "experiment_config.json")
    json_dump(config, experiment_dir / "config.json")
    json_dump(label_mapping, experiment_dir / "label_mapping.json")
    json_dump(trainability, experiment_dir / "module_trainability.json")
    json_dump(
        {
            "class_counts_train": train_class_counts,
            "class_weights": class_weights,
            "source": "train.csv only",
        },
        experiment_dir / "class_weights.json",
    )

    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = make_optimizer(
        model,
        layer4_learning_rate=args.layer4_learning_rate,
        fc_learning_rate=args.fc_learning_rate,
        weight_decay=args.weight_decay,
    )
    preprocess = {
        "image_size": [args.image_size, args.image_size],
        "color_mode": "RGB",
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "augmentation": "none",
    }

    history: list[dict[str, Any]] = []
    best_macro_f1 = float("-inf")
    best_accuracy = float("-inf")
    best_macro_path = model_dir / "best_val_macro_f1.pt"
    best_accuracy_path = model_dir / "best_val_accuracy.pt"
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            num_classes=len(class_names),
            class_names=class_names,
            optimizer=optimizer,
        )
        validation_metrics = run_epoch(
            model,
            validation_loader,
            criterion,
            device,
            num_classes=len(class_names),
            class_names=class_names,
        )
        history_row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_macro_precision": validation_metrics["macro_precision"],
            "validation_macro_recall": validation_metrics["macro_recall"],
            "validation_macro_f1": validation_metrics["macro_f1"],
            "validation_per_class": validation_metrics["per_class"],
            "validation_confusion_matrix": validation_metrics["confusion_matrix"],
            "learning_rates": {
                "layer4": float(optimizer.param_groups[0]["lr"]),
                "fc": float(optimizer.param_groups[1]["lr"]),
            },
        }
        history.append(history_row)
        if validation_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = validation_metrics["macro_f1"]
            save_checkpoint(
                best_macro_path,
                model=model,
                epoch=epoch,
                model_config=config,
                label_mapping=label_mapping,
                preprocess=preprocess,
                metrics=validation_metrics,
            )
        if validation_metrics["accuracy"] > best_accuracy:
            best_accuracy = validation_metrics["accuracy"]
            save_checkpoint(
                best_accuracy_path,
                model=model,
                epoch=epoch,
                model_config=config,
                label_mapping=label_mapping,
                preprocess=preprocess,
                metrics=validation_metrics,
            )
        print(
            f"epoch {epoch:03d}/{args.epochs:03d} | "
            f"train loss {train_metrics['loss']:.4f} acc {train_metrics['accuracy']:.4f} | "
            f"val loss {validation_metrics['loss']:.4f} "
            f"acc {validation_metrics['accuracy']:.4f} "
            f"macro-F1 {validation_metrics['macro_f1']:.4f}",
            flush=True,
        )

    final_path = model_dir / "final.pt"
    save_checkpoint(
        final_path,
        model=model,
        epoch=args.epochs,
        model_config=config,
        label_mapping=label_mapping,
        preprocess=preprocess,
        metrics=history[-1],
    )
    json_dump(history, experiment_dir / "training_history.json")
    save_training_curves(
        history,
        reports_dir / "resnet18_layer4_finetune_training_curves.png",
        title="POPF ResNet-18 layer4 fine-tuning curves",
    )

    selected = torch.load(
        best_macro_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(selected["model_state_dict"])
    test_metrics = run_epoch(
        model,
        test_loader,
        criterion,
        device,
        num_classes=len(class_names),
        class_names=class_names,
        collect_predictions=True,
        top_k=3,
    )
    prediction_rows = test_metrics.pop("prediction_rows")
    prediction_path = experiment_dir / "test_predictions_top3.csv"
    write_prediction_csv(prediction_rows, prediction_path)
    json_dump(test_metrics, experiment_dir / "test_metrics.json")

    model_selection = {
        "selected_checkpoint": str(best_macro_path),
        "selected_epoch": int(selected["epoch"]),
        "best_validation_macro_f1": best_macro_f1,
        "best_validation_accuracy": best_accuracy,
        "test_evaluations": 1,
    }
    json_dump(model_selection, experiment_dir / "model_selection.json")
    error_count = save_test_error_grid(
        prediction_rows,
        project_root=ROOT,
        output_path=reports_dir / "resnet18_layer4_finetune_test_errors.png",
    )

    baseline_test_metrics = load_json(
        ROOT / "artifacts/experiments/resnet18_baseline/test_metrics.json"
    )
    baseline_selection = load_json(
        ROOT / "artifacts/experiments/resnet18_baseline/model_selection.json"
    )
    baseline_history = json.loads(
        (ROOT / "artifacts/experiments/resnet18_baseline/training_history.json").read_text(
            encoding="utf-8"
        )
    )
    baseline_best_history = next(
        row for row in baseline_history
        if row["epoch"] == baseline_selection["selected_epoch"]
    )
    baseline_report_config = {
        "model_selection_epoch": baseline_selection["selected_epoch"],
        "best_validation_accuracy": baseline_selection["best_validation_accuracy"],
        "best_validation_macro_f1": baseline_selection["best_validation_macro_f1"],
        "best_validation_per_class": baseline_best_history["validation_per_class"],
    }

    write_prediction_comparison(
        ROOT / "artifacts/experiments/resnet18_baseline/test_predictions_top3.csv",
        prediction_path,
        experiment_dir / "baseline_vs_finetune_test_predictions.csv",
    )
    write_finetune_report(
        reports_dir / "resnet18_layer4_finetune_report.md",
        config=config,
        class_counts=train_class_counts,
        class_weights=class_weights,
        history=history,
        selection={
            "selected_epoch": int(selected["epoch"]),
            "best_validation_macro_f1": best_macro_f1,
            "best_validation_accuracy": best_accuracy,
        },
        test_metrics=test_metrics,
        error_count=error_count,
        baseline_test_metrics=baseline_test_metrics,
        baseline_config=baseline_report_config,
    )
    write_comparison_report(
        reports_dir / "resnet18_finetuning_comparison.md",
        config=config,
        baseline_config=baseline_report_config,
        baseline_test=baseline_test_metrics,
        finetune_test=test_metrics,
        selection={
            "selected_epoch": int(selected["epoch"]),
            "best_validation_macro_f1": best_macro_f1,
            "best_validation_accuracy": best_accuracy,
        },
        validation_metrics=history[int(selected["epoch"]) - 1],
    )

    raw_after = {str(path.relative_to(ROOT)): sha256_file(path) for path in RAW_INPUTS}
    if raw_before != raw_after:
        raise AssertionError(
            f"Raw input hash changed during experiment: before={raw_before}, after={raw_after}"
        )
    json_dump(
        {
            "raw_before": raw_before,
            "raw_after": raw_after,
            "unchanged": raw_before == raw_after,
        },
        experiment_dir / "raw_integrity.json",
    )
    print(json.dumps(test_metrics, ensure_ascii=False, indent=2))
    print(f"selected checkpoint: {best_macro_path}")
    print(f"test prediction CSV: {prediction_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
