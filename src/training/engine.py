"""Training-loop utilities for the POPF modern baseline."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from torch import nn

from .metrics import classification_metrics


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch without changing project data."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(
    rows: list[dict[str, str]],
    label_mapping: dict[str, int],
) -> tuple[list[int], list[float]]:
    """Compute inverse-frequency weights from training rows only."""

    num_classes = len(label_mapping)
    counts = [0] * num_classes
    for row in rows:
        label_name = row["raw_face_type"].strip()
        if label_name not in label_mapping:
            raise ValueError(f"Unknown training label: {label_name!r}")
        counts[label_mapping[label_name]] += 1
    if any(count == 0 for count in counts):
        raise ValueError(f"Every class needs a training sample; counts={counts}")
    total = sum(counts)
    weights = [total / (num_classes * count) for count in counts]
    return counts, weights


def _move_batch(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return batch["image"].to(device), batch["label"].to(device)


def run_epoch(
    model: nn.Module,
    loader: Any,
    criterion: nn.Module,
    device: torch.device,
    *,
    num_classes: int,
    class_names: list[str],
    optimizer: torch.optim.Optimizer | None = None,
    collect_predictions: bool = False,
    top_k: int = 3,
) -> dict[str, Any]:
    """Run one train or evaluation epoch and calculate all requested metrics."""

    training = optimizer is not None
    model.train(training)
    if training:
        # Frozen BatchNorm modules must not update running statistics during a
        # partial fine-tuning experiment.
        for module_name in getattr(model, "_popf_frozen_module_names", ()):
            getattr(model, module_name).eval()
    total_loss = 0.0
    total_samples = 0
    targets: list[int] = []
    predictions: list[int] = []
    prediction_rows: list[dict[str, Any]] = []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            images, labels = _move_batch(batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.shape[0]
            total_loss += float(loss.detach().item()) * batch_size
            total_samples += batch_size
            batch_predictions = logits.argmax(dim=1)
            targets.extend(labels.detach().cpu().tolist())
            predictions.extend(batch_predictions.detach().cpu().tolist())

            if collect_predictions:
                probabilities = torch.softmax(logits.detach(), dim=1)
                scores, indices = probabilities.topk(
                    min(top_k, num_classes), dim=1
                )
                for index in range(batch_size):
                    row: dict[str, Any] = {
                        "image_id": batch["image_id"][index],
                        "image_path": batch["image_path"][index],
                        "true_label_id": int(labels[index].detach().cpu().item()),
                        "true_label": batch["label_name"][index],
                    }
                    for rank in range(indices.shape[1]):
                        label_id = int(indices[index, rank].cpu().item())
                        row[f"top{rank + 1}_label_id"] = label_id
                        row[f"top{rank + 1}_label"] = class_names[label_id]
                        row[f"top{rank + 1}_score"] = float(scores[index, rank].cpu().item())
                    row["top1_correct"] = bool(
                        row["top1_label_id"] == row["true_label_id"]
                    )
                    row["top_k_correct"] = any(
                        row[f"top{rank + 1}_label_id"] == row["true_label_id"]
                        for rank in range(indices.shape[1])
                    )
                    prediction_rows.append(row)

    metrics = classification_metrics(
        targets,
        predictions,
        num_classes,
        class_names,
    )
    metrics["loss"] = total_loss / total_samples if total_samples else 0.0
    metrics["top3_accuracy"] = None
    if collect_predictions:
        # The top-k score is reconstructed from saved rows so the same records
        # used for Bad Case analysis also define the reported top-k metric.
        metrics["top3_accuracy"] = (
            sum(row["top_k_correct"] for row in prediction_rows) / len(prediction_rows)
            if prediction_rows
            else 0.0
        )
        metrics["prediction_rows"] = prediction_rows
    return metrics
