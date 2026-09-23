"""Dependency-light classification metrics for the POPF experiments."""

from __future__ import annotations

from typing import Any, Sequence

import torch


def _as_long_tensor(values: list[int] | torch.Tensor) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.detach().to(dtype=torch.long, device="cpu").flatten()
    return torch.tensor(values, dtype=torch.long)


def confusion_matrix(
    targets: list[int] | torch.Tensor,
    predictions: list[int] | torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Build a row=true, column=predicted confusion matrix."""

    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    target_tensor = _as_long_tensor(targets)
    prediction_tensor = _as_long_tensor(predictions)
    if target_tensor.numel() != prediction_tensor.numel():
        raise ValueError("targets and predictions must have the same length")
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for target, prediction in zip(target_tensor.tolist(), prediction_tensor.tolist()):
        if not 0 <= target < num_classes:
            raise ValueError(f"target label outside range: {target}")
        if not 0 <= prediction < num_classes:
            raise ValueError(f"prediction label outside range: {prediction}")
        matrix[target, prediction] += 1
    return matrix


def classification_metrics(
    targets: list[int] | torch.Tensor,
    predictions: list[int] | torch.Tensor,
    num_classes: int,
    class_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return accuracy, macro metrics, per-class metrics, and confusion matrix."""

    target_tensor = _as_long_tensor(targets)
    prediction_tensor = _as_long_tensor(predictions)
    matrix = confusion_matrix(target_tensor, prediction_tensor, num_classes)
    true_positive = matrix.diag().to(dtype=torch.float64)
    predicted_count = matrix.sum(dim=0).to(dtype=torch.float64)
    actual_count = matrix.sum(dim=1).to(dtype=torch.float64)

    precision = torch.where(
        predicted_count > 0,
        true_positive / predicted_count,
        torch.zeros_like(true_positive),
    )
    recall = torch.where(
        actual_count > 0,
        true_positive / actual_count,
        torch.zeros_like(true_positive),
    )
    f1 = torch.where(
        precision + recall > 0,
        2 * precision * recall / (precision + recall),
        torch.zeros_like(true_positive),
    )
    accuracy = (
        float((target_tensor == prediction_tensor).to(dtype=torch.float64).mean())
        if target_tensor.numel()
        else 0.0
    )
    names = list(class_names) if class_names is not None else [
        str(index) for index in range(num_classes)
    ]
    if len(names) != num_classes:
        raise ValueError("class_names length must equal num_classes")

    per_class = []
    for index, name in enumerate(names):
        per_class.append(
            {
                "label_id": index,
                "label_name": name,
                "support": int(actual_count[index].item()),
                "precision": float(precision[index].item()),
                "recall": float(recall[index].item()),
                "f1": float(f1[index].item()),
            }
        )

    return {
        "accuracy": accuracy,
        "macro_precision": float(precision.mean().item()),
        "macro_recall": float(recall.mean().item()),
        "macro_f1": float(f1.mean().item()),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "sample_count": int(target_tensor.numel()),
    }


def top_k_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k: int = 3,
) -> float:
    """Compute top-k accuracy from a single logits batch or full tensor."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape [N, C]")
    if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        raise ValueError("targets must have shape [N]")
    if not 1 <= k <= logits.shape[1]:
        raise ValueError("k must be between 1 and the number of classes")
    top_indices = logits.topk(k, dim=1).indices
    matches = top_indices.eq(targets.view(-1, 1)).any(dim=1)
    return float(matches.to(dtype=torch.float64).mean().item())
