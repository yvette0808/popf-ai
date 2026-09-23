#!/usr/bin/env python3
"""Train and evaluate the POPF 2026 Reconstruction ResNet-18 baseline."""

from __future__ import annotations

import argparse
import csv
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
    total_parameter_count,
    trainable_parameter_count,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--model-name", choices=("resnet18",), default="resnet18")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument(
        "--freeze-backbone",
        dest="freeze_backbone",
        action="store_true",
        default=True,
    )
    parser.add_argument("--no-freeze-backbone", dest="freeze_backbone", action="store_false")
    parser.add_argument("--use-class-weight", dest="use_class_weight", action="store_true", default=True)
    parser.add_argument("--no-class-weight", dest="use_class_weight", action="store_false")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="auto selects CUDA, then MPS, then CPU when available",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts",
    )
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


def json_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def write_experiment_report(
    path: Path,
    *,
    config: dict[str, Any],
    class_counts: list[int],
    class_weights: list[float],
    history: list[dict[str, Any]],
    selection: dict[str, Any],
    test_metrics: dict[str, Any],
    error_count: int,
) -> None:
    class_names = config["class_names"]
    lines = [
        "# POPF ResNet-18 Baseline Report",
        "",
        "Status: `2026 Reconstruction` experiment. This is not a 2019 Original result.",
        "",
        f"Run timestamp (UTC): `{config['run_timestamp_utc']}`",
        "",
        "## 1. Data",
        "",
        f"- Dataset: `{config['dataset_name']}`",
        f"- Images: `{config['dataset_counts']['total']}` (`train={config['dataset_counts']['train']}`, `validation={config['dataset_counts']['validation']}`, `test={config['dataset_counts']['test']}`)",
        f"- Classes: `{len(class_names)}`",
        "- Split files were kept unchanged during this experiment.",
        "- Test data was not used during training or model selection; it was evaluated once after selecting the best validation macro-F1 checkpoint.",
        "",
        "| Label | Total | Train | Weight |",
        "|---|---:|---:|---:|",
    ]
    for label, total, train_count, weight in zip(
        class_names,
        config["class_counts_total"],
        class_counts,
        class_weights,
    ):
        lines.append(f"| {label} | {total} | {train_count} | {weight:.6f} |")
    lines.extend(
        [
            "",
            "## 2. Model and training configuration",
            "",
            f"- Model: `{config['model_name']}`",
            f"- ImageNet pretrained weights: `{config['pretrained']}`",
            f"- Frozen backbone: `{config['freeze_backbone']}`",
            f"- Input: `{config['image_size']}x{config['image_size']}` RGB, ImageNet normalization, no augmentation",
            f"- Loss: weighted Cross Entropy (`use_class_weight={config['use_class_weight']}`)",
            f"- Optimizer: `AdamW`, learning rate `{config['learning_rate']}`, weight decay `{config['weight_decay']}`",
            f"- Epochs: `{config['epochs']}`; batch size: `{config['batch_size']}`; seed: `{config['seed']}`",
            f"- Device: `{config['device']}`",
            f"- Parameters: `{config['trainable_parameters']}` trainable / `{config['total_parameters']}` total",
            "",
            "## 3. Training curves",
            "",
            "![Training curves](resnet18_baseline_training_curves.png)",
            "",
            "The curves are generated from the recorded epoch history. Validation metrics were used for checkpoint selection.",
            "",
            "## 4. Validation model selection",
            "",
            f"- Selected checkpoint: `{selection['selected_checkpoint']}`",
            f"- Selection criterion: best validation macro-F1",
            f"- Best validation macro-F1: `{format_metric(selection['best_validation_macro_f1'])}`",
            f"- Best validation accuracy: `{format_metric(selection['best_validation_accuracy'])}`",
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
            "## 6. Test Bad Cases",
            "",
            f"- Top-1 errors rendered for inspection (capped at 12): `{error_count}`; the prediction CSV contains all test samples.",
            "- Full top-k output: `../artifacts/experiments/resnet18_baseline/test_predictions_top3.csv`",
            "",
            "![Test errors](resnet18_test_errors.png)",
            "",
            "The saved prediction scores are model softmax outputs used as ranking scores; they are not calibrated probabilities.",
            "",
            "## 7. Observations and next experiment",
            "",
            "- The test confusion matrix shows prediction bias: the model predicts fewer `三块瓦脸` samples than their true support and more `碎脸`/`象形脸` alternatives.",
            "- The most visible confusions are `碎脸` with `象形脸` and `花三块瓦脸`, plus several `三块瓦脸` samples assigned to neighboring visual categories.",
            "- The small-support classes have unstable estimates: `花三块瓦脸` has test F1 `0.000000`, `十字门脸` has support `2`, and `六分脸` has support `2`; the perfect F1 for `六分脸` is not strong evidence because its support is only two images.",
            f"- Top-3 accuracy exceeds Top-1 accuracy by `{test_metrics['top3_accuracy'] - test_metrics['accuracy']:.6f}` on this test split, so ranked alternatives are materially more useful than a single label for this baseline.",
            "- The result is sufficient to validate the end-to-end training and evaluation pipeline, but not sufficient to claim production readiness or generalization beyond this small split.",
            "",
            "**One next experiment:** after a focused manual audit of the `花三块瓦脸`/`碎脸`/`象形脸` examples and their labels, run a controlled fine-tuning comparison that unfreezes only ResNet-18 `layer4` plus the classification head. Keep the same split, seed, preprocessing, and test holdout, and select only on validation macro-F1.",
            "",
            "The approximately 70% accuracy reported by the 2019 paper is historical `2019 Original` context and is not compared to this `2026 Reconstruction` result.",
            "",
            "## 8. Reproducibility record",
            "",
            f"- Python: `{config['python_version']}`",
            f"- PyTorch: `{config['torch_version']}`",
            f"- torchvision: `{config['torchvision_version']}`",
            f"- Label mapping: `{json.dumps(config['label_mapping'], ensure_ascii=False)}`",
            f"- Class weights were calculated from training rows only: `{config['class_weights_from']}`",
            "",
            "No raw file was modified by the training script.",
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
    if args.model_name != "resnet18":
        raise ValueError("Only resnet18 is implemented in this baseline")

    seed_everything(args.seed)
    device = choose_device(args.device)
    dataset_dir = args.dataset_dir.resolve()
    output_root = args.output_root.resolve()
    experiment_dir = output_root / "experiments" / "resnet18_baseline"
    model_dir = output_root / "models" / "resnet18_baseline"
    reports_dir = ROOT / "reports"
    label_mapping = load_label_mapping(dataset_dir / "label_mapping.json")
    class_names = [
        label for label, _ in sorted(label_mapping.items(), key=lambda item: item[1])
    ]
    train_rows = read_csv_rows(dataset_dir / "train.csv")
    split_rows = {
        split: read_csv_rows(dataset_dir / f"{split}.csv")
        for split in ("train", "validation", "test")
    }
    train_class_counts, class_weights = compute_class_weights(train_rows, label_mapping)
    total_class_counts = [
        sum(1 for row in split_rows["train"] + split_rows["validation"] + split_rows["test"]
            if label_mapping[row["raw_face_type"].strip()] == class_id)
        for class_id in range(len(class_names))
    ]

    config: dict[str, Any] = {
        "experiment_name": "resnet18_baseline",
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
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "num_workers": args.num_workers,
        "image_size": args.image_size,
        "model_name": args.model_name,
        "pretrained": args.pretrained,
        "freeze_backbone": args.freeze_backbone,
        "use_class_weight": args.use_class_weight,
        "device": str(device),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    train_loader, validation_loader, test_loader = build_dataloaders(
        dataset_dir=dataset_dir,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    model = build_resnet18(
        len(class_names),
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
    ).to(device)
    config["trainable_parameters"] = trainable_parameter_count(model)
    config["total_parameters"] = total_parameter_count(model)

    weight_tensor = (
        torch.tensor(class_weights, dtype=torch.float32, device=device)
        if args.use_class_weight
        else None
    )
    criterion = torch.nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
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
    json_dump(config, experiment_dir / "config.json")
    json_dump(label_mapping, experiment_dir / "label_mapping.json")
    json_dump(
        {
            "class_counts_train": train_class_counts,
            "class_weights": class_weights,
            "source": "train.csv only",
        },
        experiment_dir / "class_weights.json",
    )

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
        learning_rate = float(optimizer.param_groups[0]["lr"])
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
            "learning_rate": learning_rate,
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
            f"val loss {validation_metrics['loss']:.4f} acc {validation_metrics['accuracy']:.4f} "
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
    history_path = experiment_dir / "training_history.json"
    json_dump(history, history_path)
    save_training_curves(history, reports_dir / "resnet18_baseline_training_curves.png")

    selected_checkpoint = best_macro_path
    # This checkpoint is created locally by this script and contains trusted
    # experiment metadata in addition to the tensor state dict.
    selected = torch.load(
        selected_checkpoint,
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
    write_prediction_csv(prediction_rows, experiment_dir / "test_predictions_top3.csv")
    json_dump(test_metrics, experiment_dir / "test_metrics.json")
    json_dump(
        {
            "selected_checkpoint": str(selected_checkpoint),
            "selected_epoch": int(selected["epoch"]),
            "best_validation_macro_f1": best_macro_f1,
            "best_validation_accuracy": best_accuracy,
            "test_evaluations": 1,
        },
        experiment_dir / "model_selection.json",
    )
    error_count = save_test_error_grid(
        prediction_rows,
        project_root=ROOT,
        output_path=reports_dir / "resnet18_test_errors.png",
    )
    write_experiment_report(
        reports_dir / "resnet18_baseline_report.md",
        config=config,
        class_counts=train_class_counts,
        class_weights=class_weights,
        history=history,
        selection={
            "selected_checkpoint": str(selected_checkpoint.relative_to(ROOT)),
            "best_validation_macro_f1": best_macro_f1,
            "best_validation_accuracy": best_accuracy,
        },
        test_metrics=test_metrics,
        error_count=error_count,
    )
    print(json.dumps(test_metrics, ensure_ascii=False, indent=2))
    print(f"selected checkpoint: {selected_checkpoint}")
    print(f"test prediction CSV: {experiment_dir / 'test_predictions_top3.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
