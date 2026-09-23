"""Inference service for the POPF local Web MVP.

This module loads the existing 2026 Reconstruction checkpoint once and keeps
inference preprocessing aligned with the checkpoint metadata. Uploaded images
are decoded in memory and are never written to the raw or processed datasets.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from src.models.resnet import build_resnet18


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts/models/resnet18_conservative_finetune/best_val_macro_f1.pt"
)
MODEL_EXPERIMENT_NAME = "resnet18_conservative_finetune"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


class InferenceInputError(ValueError):
    """An input image cannot be accepted for inference."""


class UnsupportedImageFormatError(InferenceInputError):
    """The uploaded filename does not use a supported image extension."""


class InvalidImageError(InferenceInputError):
    """The uploaded bytes are not a readable image."""


@dataclass(frozen=True)
class PredictionService:
    """Load and serve one fixed checkpoint on CPU."""

    checkpoint_path: Path = DEFAULT_CHECKPOINT
    device: str = "cpu"

    def __post_init__(self) -> None:
        checkpoint_path = Path(self.checkpoint_path).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=torch.device(self.device),
            weights_only=False,
        )
        model_config = checkpoint["model_config"]
        mapping = checkpoint["label_mapping"]
        class_names = [
            label for label, _ in sorted(mapping.items(), key=lambda item: item[1])
        ]
        trainable_modules = model_config.get("unfreeze_layers")
        model = build_resnet18(
            len(class_names),
            pretrained=False,
            freeze_backbone=bool(model_config.get("freeze_backbone", False)),
            trainable_modules=trainable_modules,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(torch.device(self.device))
        model.eval()

        preprocess = checkpoint["preprocess"]
        image_size = tuple(int(value) for value in preprocess["image_size"])
        normalization = preprocess["normalization"]
        mean = torch.tensor(normalization["mean"], dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(normalization["std"], dtype=torch.float32).view(3, 1, 1)

        object.__setattr__(self, "checkpoint_path", checkpoint_path)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "class_names", class_names)
        object.__setattr__(self, "label_mapping", dict(mapping))
        object.__setattr__(self, "image_size", image_size)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)
        object.__setattr__(self, "preprocess_config", preprocess)
        object.__setattr__(
            self,
            "model_metadata",
            {
                "experiment_name": MODEL_EXPERIMENT_NAME,
                "stage": model_config.get("stage", "2026 Reconstruction"),
                "dataset_name": model_config.get("dataset_name", "POPF modern_v1"),
                "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
                "class_names": class_names,
                "preprocess": preprocess,
            },
        )

    def _decode_and_preprocess(
        self,
        image_bytes: bytes,
        filename: str,
    ) -> tuple[torch.Tensor, tuple[int, int], str]:
        suffix = Path(filename or "").suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise UnsupportedImageFormatError(
                "暂不支持该图片格式，请上传 JPG、JPEG 或 PNG 图片。"
            )
        if not image_bytes:
            raise InvalidImageError("上传的图片为空，请重新选择图片。")
        if len(image_bytes) > MAX_UPLOAD_BYTES:
            raise InvalidImageError("图片文件过大，请上传 10 MB 以内的图片。")

        try:
            with Image.open(io.BytesIO(image_bytes)) as source:
                source.load()
                original_size = source.size
                source_format = source.format or suffix.lstrip(".").upper()
                image = source.convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise InvalidImageError(
                "图片读取失败，请上传有效的 JPG、JPEG 或 PNG 图片。"
            ) from exc

        resampling = getattr(Image, "Resampling", Image).BILINEAR
        image = image.resize(self.image_size, resampling)
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())
        tensor = ((tensor - self.mean) / self.std).unsqueeze(0)
        return tensor.to(torch.device(self.device)), original_size, source_format

    def predict(self, image_bytes: bytes, filename: str, top_k: int = 3) -> dict[str, Any]:
        """Return model probabilities for an uploaded image."""

        if not 1 <= top_k <= len(self.class_names):
            raise ValueError(f"top_k must be between 1 and {len(self.class_names)}")

        tensor, original_size, source_format = self._decode_and_preprocess(
            image_bytes,
            filename,
        )
        with torch.no_grad():
            probabilities = torch.softmax(self.model(tensor), dim=1)[0]
        values, indices = probabilities.topk(top_k)
        digest = hashlib.sha256(image_bytes).hexdigest()

        top3 = []
        for rank, (value, index) in enumerate(
            zip(values.cpu().tolist(), indices.cpu().tolist()),
            start=1,
        ):
            top3.append(
                {
                    "rank": rank,
                    "label": self.class_names[index],
                    "label_id": int(index),
                    "probability": float(value),
                }
            )

        return {
            "model": MODEL_EXPERIMENT_NAME,
            "stage": "2026 Reconstruction",
            "dataset": "POPF modern_v1",
            "upload_id": f"sha256:{digest}",
            "source_format": source_format,
            "original_size": {
                "width": int(original_size[0]),
                "height": int(original_size[1]),
            },
            "top1": top3[0],
            "top3": top3,
            "disclaimer": (
                "模型置信度来自当前 2026 Reconstruction checkpoint 的 softmax 输出，"
                "不等同于真实识别准确率。"
            ),
        }

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "model": MODEL_EXPERIMENT_NAME,
            "stage": "2026 Reconstruction",
            "device": self.device,
            "class_count": len(self.class_names),
            "preprocess": {
                "color_mode": self.preprocess_config["color_mode"],
                "image_size": self.preprocess_config["image_size"],
                "normalization": self.preprocess_config["normalization"],
                "augmentation": self.preprocess_config["augmentation"],
            },
        }


def load_label_mapping() -> dict[str, int]:
    """Read the project-owned mapping without regenerating it."""

    path = PROJECT_ROOT / "data/processed/modern_dataset/label_mapping.json"
    with path.open("r", encoding="utf-8") as handle:
        mapping = json.load(handle)
    return {str(label): int(label_id) for label, label_id in mapping.items()}
