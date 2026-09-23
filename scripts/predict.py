#!/usr/bin/env python3
"""Run top-3 inference on one image stored in the historical ZIP."""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.popf_dataset import parse_archive_image_path, read_archive_image_bytes  # noqa: E402
from src.models.resnet import build_resnet18  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--image-path",
        required=True,
        help="Archive member path, e.g. data/raw/...zip::opera-01-001-s.jpg",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    return parser.parse_args()


def load_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        return torch.device("cuda")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is not available")
        return torch.device("mps")
    return torch.device("cpu")


def main() -> int:
    args = parse_args()
    if not 1 <= args.top_k:
        raise ValueError("--top-k must be positive")
    device = load_device(args.device)
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )
    model_config = checkpoint["model_config"]
    mapping = checkpoint["label_mapping"]
    class_names = [label for label, _ in sorted(mapping.items(), key=lambda item: item[1])]
    model = build_resnet18(
        len(class_names),
        pretrained=False,
        freeze_backbone=bool(model_config["freeze_backbone"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    preprocess = checkpoint["preprocess"]
    archive_bytes = read_archive_image_bytes(args.image_path, ROOT)
    with Image.open(io.BytesIO(archive_bytes)) as source:
        image = source.convert("RGB").resize(
            tuple(preprocess["image_size"]),
            getattr(Image, "Resampling", Image).BILINEAR,
        )
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())
    mean = torch.tensor(preprocess["normalization"]["mean"], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(preprocess["normalization"]["std"], dtype=torch.float32).view(3, 1, 1)
    tensor = ((tensor - mean) / std).unsqueeze(0).to(device)
    with torch.no_grad():
        scores = torch.softmax(model(tensor), dim=1)[0]
    top_k = min(args.top_k, len(class_names))
    values, indices = scores.topk(top_k)
    result = []
    for rank, (score, index) in enumerate(zip(values.cpu().tolist(), indices.cpu().tolist()), start=1):
        result.append(
            {
                "rank": rank,
                "label": class_names[index],
                "label_id": index,
                "prediction_score": score,
            }
        )
    print(json.dumps({"image_path": args.image_path, "top_k": result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
