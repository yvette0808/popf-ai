#!/usr/bin/env python3
"""Run POPF's third modern vision experiment.

This is a 2026 Reconstruction experiment. It keeps the modern_v1 split and
model structure from the layer4 fine-tuning run, changes only the two learning
rates, and adds validation-macro-F1 early stopping.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import statistics
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


EXPERIMENT_NAME = "resnet18_conservative_finetune"
BASELINE_NAME = "resnet18_baseline"
LAYER4_NAME = "resnet18_layer4_finetune"
EXPECTED_COUNTS = {"train": 154, "validation": 34, "test": 34}
TRAINABLE_MODULES = ("layer4", "fc")
FROZEN_MODULES = {"conv1", "bn1", "layer1", "layer2", "layer3"}
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
    parser.add_argument("--layer4-learning-rate", type=float, default=5e-5)
    parser.add_argument("--fc-learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
    )
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


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


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


def validate_inputs(
    dataset_dir: Path,
    label_mapping: dict[str, int],
) -> dict[str, Any]:
    """Validate fixed split metadata without regenerating or changing it."""

    baseline_dir = ROOT / "artifacts/experiments" / BASELINE_NAME
    layer4_dir = ROOT / "artifacts/experiments" / LAYER4_NAME
    baseline_config = load_json(baseline_dir / "config.json")
    layer4_config = load_json(layer4_dir / "experiment_config.json")

    if label_mapping != baseline_config["label_mapping"]:
        raise AssertionError("Current label mapping differs from baseline")
    if label_mapping != layer4_config["label_mapping"]:
        raise AssertionError("Current label mapping differs from layer4 experiment")

    split_rows = {
        split: read_csv_rows(dataset_dir / f"{split}.csv")
        for split in ("train", "validation", "test")
    }
    counts = {split: len(rows) for split, rows in split_rows.items()}
    if counts != EXPECTED_COUNTS:
        raise AssertionError(f"Dataset counts changed: {counts}")
    baseline_counts = {
        split: baseline_config["dataset_counts"][split]
        for split in ("train", "validation", "test")
    }
    if counts != baseline_counts:
        raise AssertionError("Current split counts differ from baseline")
    if counts != {
        split: layer4_config["dataset_counts"][split]
        for split in ("train", "validation", "test")
    }:
        raise AssertionError("Current split counts differ from layer4 experiment")

    image_ids = [
        row["image_id"]
        for rows in split_rows.values()
        for row in rows
    ]
    if len(image_ids) != len(set(image_ids)):
        raise AssertionError("An image_id occurs in more than one split")

    split_hashes = {
        split: {row["image_sha256"] for row in rows if row.get("image_sha256")}
        for split, rows in split_rows.items()
    }
    overlaps: dict[str, list[str]] = {}
    split_names = list(split_hashes)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlaps[f"{left}_vs_{right}"] = sorted(
                split_hashes[left].intersection(split_hashes[right])
            )
    if any(overlaps.values()):
        raise AssertionError(f"SHA-256 leakage detected: {overlaps}")

    snapshot = {
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
        "layer4_config": layer4_config,
        "dataset_snapshot": snapshot,
    }


def make_optimizer(
    model: torch.nn.Module,
    *,
    layer4_learning_rate: float,
    fc_learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    layer4_parameters = [
        parameter for parameter in model.layer4.parameters() if parameter.requires_grad
    ]
    fc_parameters = [
        parameter for parameter in model.fc.parameters() if parameter.requires_grad
    ]
    if not layer4_parameters or not fc_parameters:
        raise AssertionError("Expected trainable layer4 and fc parameters")
    return torch.optim.AdamW(
        [
            {
                "name": "layer4",
                "params": layer4_parameters,
                "lr": layer4_learning_rate,
            },
            {"name": "fc", "params": fc_parameters, "lr": fc_learning_rate},
        ],
        weight_decay=weight_decay,
    )


def prediction_rows_by_image(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["image_id"]: row for row in csv.DictReader(handle)}


def write_three_way_prediction_comparison(
    baseline_path: Path,
    layer4_path: Path,
    conservative_path: Path,
    output_path: Path,
) -> None:
    baseline = prediction_rows_by_image(baseline_path)
    layer4 = prediction_rows_by_image(layer4_path)
    conservative = prediction_rows_by_image(conservative_path)
    if not (set(baseline) == set(layer4) == set(conservative)):
        raise AssertionError("Three experiments have different test image IDs")

    fieldnames = [
        "image_id",
        "image_path",
        "true_label",
        "baseline_top1_label",
        "baseline_top2_label",
        "baseline_top3_label",
        "layer4_top1_label",
        "layer4_top2_label",
        "layer4_top3_label",
        "conservative_top1_label",
        "conservative_top2_label",
        "conservative_top3_label",
        "baseline_top1_correct",
        "layer4_top1_correct",
        "conservative_top1_correct",
        "baseline_top3_correct",
        "layer4_top3_correct",
        "conservative_top3_correct",
        "top1_changed_baseline_to_layer4",
        "top1_changed_layer4_to_conservative",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for image_id in sorted(baseline):
            old = baseline[image_id]
            middle = layer4[image_id]
            new = conservative[image_id]
            writer.writerow(
                {
                    "image_id": image_id,
                    "image_path": new["image_path"],
                    "true_label": new["true_label"],
                    "baseline_top1_label": old["top1_label"],
                    "baseline_top2_label": old["top2_label"],
                    "baseline_top3_label": old["top3_label"],
                    "layer4_top1_label": middle["top1_label"],
                    "layer4_top2_label": middle["top2_label"],
                    "layer4_top3_label": middle["top3_label"],
                    "conservative_top1_label": new["top1_label"],
                    "conservative_top2_label": new["top2_label"],
                    "conservative_top3_label": new["top3_label"],
                    "baseline_top1_correct": old["top1_correct"],
                    "layer4_top1_correct": middle["top1_correct"],
                    "conservative_top1_correct": new["top1_correct"],
                    "baseline_top3_correct": old["top_k_correct"],
                    "layer4_top3_correct": middle["top_k_correct"],
                    "conservative_top3_correct": new["top_k_correct"],
                    "top1_changed_baseline_to_layer4": (
                        old["top1_label"] != middle["top1_label"]
                    ),
                    "top1_changed_layer4_to_conservative": (
                        middle["top1_label"] != new["top1_label"]
                    ),
                }
            )


def selected_validation_row(history: list[dict[str, Any]], epoch: int) -> dict[str, Any]:
    for row in history:
        if int(row["epoch"]) == epoch:
            return row
    raise KeyError(f"Epoch {epoch} not found in history")


def metric_summary(
    name: str,
    *,
    history: list[dict[str, Any]],
    selection: dict[str, Any],
    test_metrics: dict[str, Any],
) -> dict[str, Any]:
    best_epoch = int(selection["selected_epoch"])
    best_row = selected_validation_row(history, best_epoch)
    validation_f1_values = [float(row["validation_macro_f1"]) for row in history]
    return {
        "name": name,
        "epochs_trained": len(history),
        "early_stopped": bool(selection.get("early_stopped", False)),
        "selected_epoch": best_epoch,
        "validation_accuracy": float(selection["best_validation_accuracy"]),
        "validation_macro_f1": float(selection["best_validation_macro_f1"]),
        "validation_per_class": best_row["validation_per_class"],
        "test": test_metrics,
        "train_accuracy_at_best": float(best_row["train_accuracy"]),
        "generalization_gap_at_best": float(
            best_row["train_accuracy"] - best_row["validation_accuracy"]
        ),
        "validation_macro_f1_range": max(validation_f1_values) - min(validation_f1_values),
        "validation_macro_f1_std": (
            statistics.pstdev(validation_f1_values)
            if len(validation_f1_values) > 1
            else 0.0
        ),
    }


def fmt(value: float) -> str:
    return f"{value:.6f}"


def choose_next_step(
    conservative: dict[str, Any],
    layer4: dict[str, Any],
) -> str:
    conservative_test = conservative["test"]
    layer4_test = layer4["test"]
    flower_f1 = next(
        item["f1"]
        for item in conservative_test["per_class"]
        if item["label_name"] == "花三块瓦脸"
    )
    if (
        conservative_test["macro_f1"] > layer4_test["macro_f1"] + 0.01
        and flower_f1 > 0
    ):
        return (
            "继续做受控 fine-tuning schedule 实验：当前 conservative schedule "
            "同时改善了 test macro-F1 和花三块瓦脸，下一轮应只改变一个 schedule "
            "变量并继续封存 test。"
        )
    if flower_f1 == 0 or conservative_test["macro_f1"] <= layer4_test["macro_f1"] + 0.01:
        return (
            "优先做数据质量 / 标签边界分析：当前实验仍未解决花三块瓦脸，且在 "
            "222 张小数据上继续改变 fine-tuning schedule 的证据不足。"
        )
    return (
        "继续做受控 fine-tuning schedule 实验：当前结果显示 schedule 仍可能影响 "
        "泛化，但下一轮必须保持同一 split、同一 test holdout，并只改变一个变量。"
    )


def write_conservative_report(
    path: Path,
    *,
    config: dict[str, Any],
    history: list[dict[str, Any]],
    selection: dict[str, Any],
    test_metrics: dict[str, Any],
    error_count: int,
    layer4_summary: dict[str, Any],
    baseline_summary: dict[str, Any],
) -> None:
    best_row = selected_validation_row(history, selection["selected_epoch"])
    val_f1_values = [float(row["validation_macro_f1"]) for row in history]
    train_acc_values = [float(row["train_accuracy"]) for row in history]
    best_gap = float(best_row["train_accuracy"] - best_row["validation_accuracy"])
    last_row = history[-1]
    last_gap = float(last_row["train_accuracy"] - last_row["validation_accuracy"])
    layer4_f1_values = [
        float(row["validation_macro_f1"]) for row in layer4_summary["history"]
    ]
    same_horizon = min(len(val_f1_values), len(layer4_f1_values))
    conservative_window = val_f1_values[:same_horizon]
    layer4_window = layer4_f1_values[:same_horizon]
    conservative_window_std = (
        statistics.pstdev(conservative_window)
        if len(conservative_window) > 1
        else 0.0
    )
    layer4_window_std = (
        statistics.pstdev(layer4_window) if len(layer4_window) > 1 else 0.0
    )
    next_step = choose_next_step(
        metric_summary(
            EXPERIMENT_NAME,
            history=history,
            selection=selection,
            test_metrics=test_metrics,
        ),
        layer4_summary,
    )

    lines = [
        "# POPF ResNet-18 Conservative Fine-tuning Report",
        "",
        "Status: `2026 Reconstruction` experiment. This is not a 2019 Original result.",
        "",
        f"Run timestamp (UTC): `{config['run_timestamp_utc']}`",
        "",
        "## 1. Experiment purpose",
        "",
        "This experiment tests whether lower learning rates and validation-macro-F1 early stopping reduce the overfitting observed in the previous `layer4 + fc` fine-tuning run.",
        "The model structure remains unchanged: `conv1`, `bn1`, `layer1`, `layer2`, and `layer3` are frozen; `layer4` and `fc` are trainable.",
        "",
        "## 2. Controlled conditions",
        "",
        f"- Dataset: `{config['dataset_name']}` with `train={config['dataset_counts']['train']}`, `validation={config['dataset_counts']['validation']}`, `test={config['dataset_counts']['test']}`",
        "- Existing modern_v1 split files and label mapping were reused without regeneration.",
        "- Cross-split SHA-256 leakage was checked and remained zero.",
        "- Test images were not iterated during training or validation model selection; the selected checkpoint was evaluated on test once.",
        f"- Layer4 learning rate: `{config['layer4_learning_rate']}`",
        f"- FC learning rate: `{config['fc_learning_rate']}`",
        f"- Batch size: `{config['batch_size']}`, weight decay: `{config['weight_decay']}`, optimizer: `{config['optimizer']}`",
        f"- Maximum epochs: `{config['max_epochs']}`, patience: `{config['early_stopping']['patience']}`, min_delta: `{config['early_stopping']['min_delta']}`",
        f"- Actual epochs trained: `{len(history)}`; early stopped: `{selection['early_stopped']}`",
        "",
        "## 3. Trainability audit",
        "",
        "- Frozen: `conv1`, `bn1`, `layer1`, `layer2`, `layer3`",
        "- Trainable: `layer4`, `fc`",
        f"- Parameters: `{config['trainable_parameters']}` trainable / `{config['frozen_parameters']}` frozen / `{config['total_parameters']}` total",
        "",
        "| Module | Trainable | Parameters | Trainable parameters |",
        "|---|---:|---:|---:|",
    ]
    for module_name, audit in config["trainability"]["modules"].items():
        lines.append(
            f"| {module_name} | {audit['trainable']} | "
            f"{audit['parameter_count']} | {audit['trainable_parameter_count']} |"
        )
    lines.extend(
        [
            "",
            "## 4. Training history",
            "",
            "![Conservative fine-tuning curves](resnet18_conservative_finetune_training_curves.png)",
            "",
            "| Epoch | Train loss | Train accuracy | Validation loss | Validation accuracy | Validation macro-F1 | Accuracy gap |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in history:
        lines.append(
            f"| {row['epoch']} | {row['train_loss']:.6f} | "
            f"{row['train_accuracy']:.6f} | {row['validation_loss']:.6f} | "
            f"{row['validation_accuracy']:.6f} | "
            f"{row['validation_macro_f1']:.6f} | "
            f"{row['train_accuracy'] - row['validation_accuracy']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## 5. Validation model selection",
            "",
            f"- Best epoch by validation macro-F1: `{selection['selected_epoch']}`",
            f"- Validation accuracy: `{selection['best_validation_accuracy']:.6f}`",
            f"- Validation macro-F1: `{selection['best_validation_macro_f1']:.6f}`",
            f"- Train accuracy at selected epoch: `{best_row['train_accuracy']:.6f}`",
            f"- Train-minus-validation accuracy gap at selected epoch: `{best_gap:+.6f}`",
            "",
            "| Label | Validation F1 at selected epoch |",
            "|---|---:|",
        ]
    )
    for item in best_row["validation_per_class"]:
        lines.append(f"| {item['label_name']} | {item['f1']:.6f} |")
    lines.extend(
        [
            "",
            "## 6. Final test evaluation",
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
            "## 7. Overfitting and stability analysis",
            "",
            f"- Train accuracy range during this run: `{min(train_acc_values):.6f}` to `{max(train_acc_values):.6f}`.",
            f"- At the selected epoch, the accuracy gap was `{best_gap:+.6f}`; at the final trained epoch it was `{last_gap:+.6f}`.",
            f"- Validation macro-F1 range during this run: `{min(val_f1_values):.6f}` to `{max(val_f1_values):.6f}`; population standard deviation: `{statistics.pstdev(val_f1_values):.6f}`.",
            f"- For the first `{same_horizon}` epochs, conservative validation macro-F1 standard deviation was `{conservative_window_std:.6f}` versus `{layer4_window_std:.6f}` for the previous layer4 run.",
            f"- The previous layer4 run selected epoch `{layer4_summary['selected_epoch']}`; this run selected epoch `{selection['selected_epoch']}`.",
            "- A smaller train/validation gap is treated as evidence of reduced fitting pressure, not proof of generalization by itself.",
            "",
            "## 8. Bad Case analysis",
            "",
            f"- Top-1 error tiles rendered for inspection (capped at 12): `{error_count}`; the prediction CSV contains all test samples.",
            "- Top-3 predictions: `../artifacts/experiments/resnet18_conservative_finetune/test_predictions_top3.csv`",
            "- Three-way prediction comparison: `../artifacts/experiments/resnet18_conservative_finetune/three_way_test_predictions.csv`",
            "",
            "![Conservative fine-tuning test errors](resnet18_conservative_finetune_test_errors.png)",
            "",
            "The comparison file is used to inspect `碎脸`/`象形脸`/`花三块瓦脸` and `三块瓦脸` transitions across all three runs.",
            "",
            "## 9. Answer to the experiment question",
            "",
            f"- Versus frozen baseline, validation macro-F1 changed by `{selection['best_validation_macro_f1'] - baseline_summary['validation_macro_f1']:+.6f}` and test macro-F1 changed by `{test_metrics['macro_f1'] - baseline_summary['test']['macro_f1']:+.6f}`.",
            f"- Versus the previous layer4 run, test macro-F1 changed by `{test_metrics['macro_f1'] - layer4_summary['test']['macro_f1']:+.6f}`, test accuracy changed by `{test_metrics['accuracy'] - layer4_summary['test']['accuracy']:+.6f}`, and test Top-3 changed by `{test_metrics['top3_accuracy'] - layer4_summary['test']['top3_accuracy']:+.6f}`.",
            f"- `花三块瓦脸` test F1 in this run: `{next(item['f1'] for item in test_metrics['per_class'] if item['label_name'] == '花三块瓦脸'):.6f}`.",
            "",
            "## 10. Next step",
            "",
            next_step,
            "No next experiment was executed automatically.",
            "",
            "The approximately 70% accuracy reported by the 2019 paper is historical `2019 Original` context and is not compared to these `2026 Reconstruction` results.",
            "",
            "## 11. Reproducibility",
            "",
            f"- Python: `{config['python_version']}`",
            f"- PyTorch: `{config['torch_version']}`",
            f"- torchvision: `{config['torchvision_version']}`",
            f"- Device: `{config['device']}`",
            f"- Raw input SHA-256 before and after: `{json.dumps(config['raw_input_sha256'], ensure_ascii=False)}`",
            f"- Dataset split SHA-256: `{json.dumps(config['dataset_snapshot']['split_csv_sha256'], ensure_ascii=False)}`",
            "",
            "No raw file or processed split was modified by this experiment.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_three_way_report(
    path: Path,
    *,
    summaries: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    lines = [
        "# POPF ResNet-18 Three-Way Fine-tuning Schedule Comparison",
        "",
        "All three entries are `2026 Reconstruction` experiments. None is a 2019 Original result.",
        "",
        "## Controlled conditions",
        "",
        "- Same `modern_v1` dataset, fixed `154/34/34` split, labels, input size, no augmentation, train-only class weights, seed `42`, batch size `16`, AdamW, and weight decay `1e-4`.",
        "- The conservative run changes only the layer4/fc learning rates and adds validation macro-F1 early stopping.",
        "- Test was not used for model selection; each test result comes from one final evaluation of the selected checkpoint.",
        "",
        "| Metric | Frozen Baseline | Layer4 Fine-tuning | Conservative Fine-tuning |",
        "|---|---:|---:|---:|",
    ]
    baseline, layer4, conservative = summaries
    metric_rows = [
        ("Validation Accuracy", "validation_accuracy"),
        ("Validation Macro-F1", "validation_macro_f1"),
        ("Test Accuracy", ("test", "accuracy")),
        ("Test Macro-F1", ("test", "macro_f1")),
        ("Test Macro Precision", ("test", "macro_precision")),
        ("Test Macro Recall", ("test", "macro_recall")),
        ("Test Top-3 Accuracy", ("test", "top3_accuracy")),
    ]
    for label, key in metric_rows:
        if isinstance(key, tuple):
            values = [summary[key[0]][key[1]] for summary in summaries]
        else:
            values = [summary[key] for summary in summaries]
        lines.append(
            f"| {label} | {values[0]:.6f} | {values[1]:.6f} | {values[2]:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Training and selection behavior",
            "",
            "| Experiment | Epochs trained | Early stopped | Best epoch | Train accuracy at best | Accuracy gap at best | Val macro-F1 range | Val macro-F1 std |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for summary in summaries:
        lines.append(
            f"| {summary['name']} | {summary['epochs_trained']} | "
            f"{summary['early_stopped']} | {summary['selected_epoch']} | "
            f"{summary['train_accuracy_at_best']:.6f} | "
            f"{summary['generalization_gap_at_best']:+.6f} | "
            f"{summary['validation_macro_f1_range']:.6f} | "
            f"{summary['validation_macro_f1_std']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Validation per-class F1 at selected checkpoints",
            "",
            "| Label | Frozen Baseline | Layer4 Fine-tuning | Conservative Fine-tuning |",
            "|---|---:|---:|---:|",
        ]
    )
    validation_by_name = [
        {item["label_name"]: item["f1"] for item in summary["validation_per_class"]}
        for summary in summaries
    ]
    for label in config["class_names"]:
        lines.append(
            f"| {label} | {validation_by_name[0][label]:.6f} | "
            f"{validation_by_name[1][label]:.6f} | "
            f"{validation_by_name[2][label]:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Test per-class F1",
            "",
            "| Label | Frozen Baseline | Layer4 Fine-tuning | Conservative Fine-tuning |",
            "|---|---:|---:|---:|",
        ]
    )
    test_by_name = [
        {item["label_name"]: item["f1"] for item in summary["test"]["per_class"]}
        for summary in summaries
    ]
    for label in config["class_names"]:
        lines.append(
            f"| {label} | {test_by_name[0][label]:.6f} | "
            f"{test_by_name[1][label]:.6f} | "
            f"{test_by_name[2][label]:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Test confusion matrices",
            "",
        ]
    )
    for summary in summaries:
        lines.extend(
            [
                f"{summary['name']}:",
                "",
                "```text",
                json.dumps(summary["test"]["confusion_matrix"], ensure_ascii=False),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "The per-class and confusion-matrix changes should be read together with the very small support of `十字门脸` and `六分脸`, and the persistent difficulty of `花三块瓦脸`.",
            "- A higher validation score alone is not treated as proof of improved generalization.",
            "- The conservative schedule was selected only by validation macro-F1; its test metrics are reported after selection and were not used to change the schedule.",
            "",
            "No next experiment was executed automatically.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.max_epochs <= 0:
        raise ValueError("--max-epochs must be positive")
    if args.patience <= 0:
        raise ValueError("--patience must be positive")
    if args.min_delta < 0:
        raise ValueError("--min-delta cannot be negative")
    if args.layer4_learning_rate <= 0 or args.fc_learning_rate <= 0:
        raise ValueError("Learning rates must be positive")

    seed_everything(args.seed)
    device = choose_device(args.device)
    dataset_dir = args.dataset_dir.resolve()
    output_root = args.output_root.resolve()
    experiment_dir = output_root / "experiments" / EXPERIMENT_NAME
    model_dir = output_root / "models" / EXPERIMENT_NAME
    reports_dir = ROOT / "reports"

    raw_before = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in RAW_INPUTS
    }
    label_mapping = load_label_mapping(dataset_dir / "label_mapping.json")
    preflight = validate_inputs(dataset_dir, label_mapping)
    split_rows = preflight["split_rows"]
    class_names = [
        label for label, _ in sorted(label_mapping.items(), key=lambda item: item[1])
    ]
    train_class_counts, class_weights = compute_class_weights(
        split_rows["train"],
        label_mapping,
    )
    total_class_counts = [
        sum(
            1
            for row in split_rows["train"]
            + split_rows["validation"]
            + split_rows["test"]
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
        trainable_modules=TRAINABLE_MODULES,
    ).to(device)
    trainability = resnet18_trainability_audit(model)
    if set(trainability["trainable_modules"]) != set(TRAINABLE_MODULES):
        raise AssertionError(f"Unexpected trainable modules: {trainability}")
    if set(trainability["frozen_modules"]) != FROZEN_MODULES:
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
        "max_epochs": args.max_epochs,
        "num_workers": args.num_workers,
        "image_size": args.image_size,
        "model_name": "resnet18",
        "pretrained": True,
        "freeze_backbone": False,
        "unfreeze_layers": list(TRAINABLE_MODULES),
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
        "early_stopping": {
            "monitor": "validation_macro_f1",
            "patience": args.patience,
            "min_delta": args.min_delta,
        },
        "raw_input_sha256": raw_before,
        "dataset_snapshot": preflight["dataset_snapshot"],
        "control_reference": {
            "baseline_experiment": BASELINE_NAME,
            "layer4_experiment": LAYER4_NAME,
            "layer4_config": {
                "layer4_learning_rate": 1e-4,
                "fc_learning_rate": 1e-3,
                "max_epochs": 20,
            },
        },
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

    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
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
    epochs_without_improvement = 0
    early_stopped = False
    best_macro_path = model_dir / "best_val_macro_f1.pt"
    best_accuracy_path = model_dir / "best_val_accuracy.pt"

    for epoch in range(1, args.max_epochs + 1):
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
        learning_rates = {
            "layer4": float(optimizer.param_groups[0]["lr"]),
            "fc": float(optimizer.param_groups[1]["lr"]),
        }
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
            "learning_rates": learning_rates,
            "train_accuracy_minus_validation_accuracy": (
                train_metrics["accuracy"] - validation_metrics["accuracy"]
            ),
            "epochs_without_improvement_before_epoch": epochs_without_improvement,
        }
        history.append(history_row)

        improved = validation_metrics["macro_f1"] > best_macro_f1 + args.min_delta
        if improved:
            best_macro_f1 = validation_metrics["macro_f1"]
            epochs_without_improvement = 0
            save_checkpoint(
                best_macro_path,
                model=model,
                epoch=epoch,
                model_config=config,
                label_mapping=label_mapping,
                preprocess=preprocess,
                metrics=validation_metrics,
            )
        else:
            epochs_without_improvement += 1

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
            f"epoch {epoch:02d}/{args.max_epochs:02d} | "
            f"train loss {train_metrics['loss']:.4f} "
            f"acc {train_metrics['accuracy']:.4f} | "
            f"val loss {validation_metrics['loss']:.4f} "
            f"acc {validation_metrics['accuracy']:.4f} "
            f"macro-F1 {validation_metrics['macro_f1']:.4f} | "
            f"no-improve {epochs_without_improvement}/{args.patience}",
            flush=True,
        )
        if epochs_without_improvement >= args.patience:
            early_stopped = True
            break

    final_path = model_dir / "final.pt"
    save_checkpoint(
        final_path,
        model=model,
        epoch=history[-1]["epoch"],
        model_config=config,
        label_mapping=label_mapping,
        preprocess=preprocess,
        metrics=history[-1],
    )
    json_dump(history, experiment_dir / "training_history.json")
    save_training_curves(
        history,
        reports_dir / "resnet18_conservative_finetune_training_curves.png",
        title="POPF ResNet-18 conservative fine-tuning curves",
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
        "early_stopped": early_stopped,
        "epochs_trained": len(history),
        "epochs_without_improvement_at_stop": epochs_without_improvement,
        "early_stopping": config["early_stopping"],
    }
    json_dump(model_selection, experiment_dir / "model_selection.json")

    error_count = save_test_error_grid(
        prediction_rows,
        project_root=ROOT,
        output_path=reports_dir / "resnet18_conservative_finetune_test_errors.png",
    )

    baseline_dir = ROOT / "artifacts/experiments" / BASELINE_NAME
    layer4_dir = ROOT / "artifacts/experiments" / LAYER4_NAME
    baseline_test = load_json(baseline_dir / "test_metrics.json")
    layer4_test = load_json(layer4_dir / "test_metrics.json")
    baseline_selection = load_json(baseline_dir / "model_selection.json")
    layer4_selection = load_json(layer4_dir / "model_selection.json")
    baseline_history = json.loads(
        (baseline_dir / "training_history.json").read_text(encoding="utf-8")
    )
    layer4_history = json.loads(
        (layer4_dir / "training_history.json").read_text(encoding="utf-8")
    )
    baseline_best = selected_validation_row(
        baseline_history,
        baseline_selection["selected_epoch"],
    )
    layer4_best = selected_validation_row(
        layer4_history,
        layer4_selection["selected_epoch"],
    )
    baseline_summary = metric_summary(
        BASELINE_NAME,
        history=baseline_history,
        selection=baseline_selection,
        test_metrics=baseline_test,
    )
    layer4_summary = metric_summary(
        LAYER4_NAME,
        history=layer4_history,
        selection=layer4_selection,
        test_metrics=layer4_test,
    )
    conservative_summary = metric_summary(
        EXPERIMENT_NAME,
        history=history,
        selection=model_selection,
        test_metrics=test_metrics,
    )
    baseline_summary["history"] = baseline_history
    layer4_summary["history"] = layer4_history

    three_way_prediction_path = (
        experiment_dir / "three_way_test_predictions.csv"
    )
    write_three_way_prediction_comparison(
        baseline_dir / "test_predictions_top3.csv",
        layer4_dir / "test_predictions_top3.csv",
        prediction_path,
        three_way_prediction_path,
    )

    report_config = dict(config)
    report_config["raw_input_sha256"] = raw_before
    write_conservative_report(
        reports_dir / "resnet18_conservative_finetune_report.md",
        config=report_config,
        history=history,
        selection=model_selection,
        test_metrics=test_metrics,
        error_count=error_count,
        layer4_summary=layer4_summary,
        baseline_summary=baseline_summary,
    )
    write_three_way_report(
        reports_dir / "resnet18_finetuning_schedule_comparison.md",
        summaries=[baseline_summary, layer4_summary, conservative_summary],
        config=config,
    )

    raw_after = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in RAW_INPUTS
    }
    if raw_before != raw_after:
        raise AssertionError(
            f"Raw input hash changed: before={raw_before}, after={raw_after}"
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
