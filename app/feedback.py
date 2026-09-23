"""Anonymous feedback persistence for the local POPF MVP."""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEEDBACK_PATH = PROJECT_ROOT / "artifacts/feedback/feedback.jsonl"
MODEL_EXPERIMENT_NAME = "resnet18_conservative_finetune"
_WRITE_LOCK = threading.Lock()


def append_feedback(
    *,
    upload_id: str,
    top1: dict[str, Any],
    top3: list[dict[str, Any]],
    top1_correct: bool,
    corrected_label: str | None,
    allowed_labels: set[str],
) -> dict[str, Any]:
    if not upload_id or len(upload_id) > 100:
        raise ValueError("upload_id is required")
    if not isinstance(top1, dict) or not top1.get("label"):
        raise ValueError("top1 prediction is required")
    if not isinstance(top3, list) or not top3:
        raise ValueError("top3 predictions are required")
    if top1["label"] not in allowed_labels:
        raise ValueError("top1 label is not in the fixed label mapping")
    if corrected_label is not None and corrected_label not in allowed_labels:
        raise ValueError("corrected_label is not in the fixed label mapping")
    if not top1_correct and not corrected_label:
        raise ValueError("corrected_label is required when top1_correct is false")

    record = {
        "feedback_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "upload_id": upload_id,
        "top3": top3,
        "top1_label": top1["label"],
        "top1_probability": top1.get("probability"),
        "top1_correct": bool(top1_correct),
        "corrected_label": corrected_label if not top1_correct else None,
        "model_experiment": MODEL_EXPERIMENT_NAME,
        "stage": "2026 Reconstruction",
    }
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _WRITE_LOCK:
        with FEEDBACK_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {
        "status": "recorded",
        "feedback_id": record["feedback_id"],
    }

