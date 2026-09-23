"""PyTorch Dataset/DataLoader utilities for the POPF modern dataset.

The dataset reads image bytes directly from the historical ZIP archive. It
does not extract, copy, rename, or modify raw images.
"""

from __future__ import annotations

import csv
import io
import json
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # Keep label/manifest helpers usable without PyTorch.
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment,misc]
    Dataset = object  # type: ignore[assignment,misc]
    _TORCH_IMPORT_ERROR: ImportError | None = exc
else:
    _TORCH_IMPORT_ERROR = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data/processed/modern_dataset"
DEFAULT_LABEL_INVENTORY_PATH = DEFAULT_DATASET_DIR / "label_inventory.csv"
DEFAULT_LABEL_MAPPING_PATH = DEFAULT_DATASET_DIR / "label_mapping.json"

REQUIRED_SPLIT_COLUMNS = {
    "image_id",
    "image_path",
    "raw_face_type",
    "image_sha256",
}


@dataclass(frozen=True)
class ImagePreprocessConfig:
    """Deterministic preprocessing for modern pretrained vision models.

    `image_size` follows PIL's `(width, height)` convention.
    """

    image_size: tuple[int, int] = (224, 224)
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    interpolation: str = "bilinear"

    def __post_init__(self) -> None:
        width, height = self.image_size
        if width <= 0 or height <= 0:
            raise ValueError("image_size must contain positive integers")
        if len(self.mean) != 3 or len(self.std) != 3:
            raise ValueError("mean and std must each contain three values")
        if any(value <= 0 for value in self.std):
            raise ValueError("std values must be positive")
        if self.interpolation != "bilinear":
            raise ValueError("Only bilinear interpolation is supported currently")


def require_torch() -> Any:
    if torch is None:
        detail = str(_TORCH_IMPORT_ERROR) if _TORCH_IMPORT_ERROR else "unknown import error"
        raise ImportError(
            "PyTorch is required for PopfZipDataset and DataLoader operations. "
            f"Import failed with: {detail}"
        ) from _TORCH_IMPORT_ERROR
    return torch


def read_csv_rows(path: Path | str) -> list[dict[str, str]]:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV has no data rows: {csv_path}")
    return rows


def _normalise_label_inventory_rows(rows: Iterable[dict[str, str]]) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for row in rows:
        label = (row.get("raw_label") or "").strip()
        eligible = (row.get("eligible_for_modern_v1") or "").strip().lower()
        if eligible != "yes" or not label:
            continue
        if label in seen:
            raise ValueError(f"Duplicate eligible label in label_inventory.csv: {label}")
        seen.add(label)
        labels.append(label)
    if not labels:
        raise ValueError("No eligible labels found in label_inventory.csv")
    return labels


def build_label_mapping(
    label_inventory_path: Path | str = DEFAULT_LABEL_INVENTORY_PATH,
    output_path: Path | str | None = DEFAULT_LABEL_MAPPING_PATH,
) -> dict[str, int]:
    """Build a fixed mapping from the ordered label inventory.

    The inventory is already ordered by the audited class frequency. Filtering
    `eligible_for_modern_v1=yes` preserves that stable order and excludes the
    missing-label row and low-frequency classes.
    """

    labels = _normalise_label_inventory_rows(read_csv_rows(Path(label_inventory_path)))
    mapping = {label: index for index, label in enumerate(labels)}
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(mapping, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return mapping


def load_label_mapping(path: Path | str = DEFAULT_LABEL_MAPPING_PATH) -> dict[str, int]:
    mapping_path = Path(path)
    with mapping_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"Label mapping must be a non-empty JSON object: {mapping_path}")

    mapping: dict[str, int] = {}
    for label, label_id in raw.items():
        if not isinstance(label, str) or not label:
            raise ValueError(f"Invalid label name in mapping: {label!r}")
        if isinstance(label_id, bool) or not isinstance(label_id, int):
            raise ValueError(f"Label id must be an integer for {label!r}")
        mapping[label] = label_id

    expected_ids = set(range(len(mapping)))
    if set(mapping.values()) != expected_ids:
        raise ValueError(
            f"Label ids must be contiguous 0..{len(mapping) - 1}; "
            f"found {sorted(mapping.values())}"
        )
    return mapping


def ensure_label_mapping(
    label_inventory_path: Path | str = DEFAULT_LABEL_INVENTORY_PATH,
    label_mapping_path: Path | str = DEFAULT_LABEL_MAPPING_PATH,
) -> dict[str, int]:
    """Create the mapping once, then refuse silent mapping drift."""

    expected = build_label_mapping(label_inventory_path, output_path=None)
    output = Path(label_mapping_path)
    if output.exists():
        actual = load_label_mapping(output)
        if actual != expected:
            raise ValueError(
                "Existing label_mapping.json differs from label_inventory.csv. "
                "Refusing to overwrite a stable mapping."
            )
        return actual

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(expected, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return expected


def parse_archive_image_path(
    image_path: str,
    project_root: Path | str = PROJECT_ROOT,
) -> tuple[Path, str]:
    """Parse `relative/archive.zip::member/name.jpg` without extracting it."""

    if "::" not in image_path:
        raise ValueError(f"Image path is not an archive member path: {image_path}")
    archive_relative, member_name = image_path.split("::", 1)
    if not archive_relative or not member_name:
        raise ValueError(f"Archive path or member name is empty: {image_path}")

    root = Path(project_root).resolve()
    archive_path = (root / archive_relative).resolve()
    try:
        archive_path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Archive path escapes project root: {image_path}") from exc

    member_path = Path(member_name)
    if member_path.is_absolute() or ".." in member_path.parts:
        raise ValueError(f"Unsafe ZIP member path: {member_name}")
    if archive_path.suffix.lower() != ".zip":
        raise ValueError(f"Expected a ZIP archive: {archive_path}")
    return archive_path, member_name


def _open_zip(path: Path) -> zipfile.ZipFile:
    try:
        return zipfile.ZipFile(path, mode="r", metadata_encoding="gbk")
    except TypeError:
        return zipfile.ZipFile(path, mode="r")


def read_archive_image_bytes(
    image_path: str,
    project_root: Path | str = PROJECT_ROOT,
) -> bytes:
    archive_path, member_name = parse_archive_image_path(image_path, project_root)
    with _open_zip(archive_path) as archive:
        try:
            return archive.read(member_name)
        except KeyError as exc:
            raise FileNotFoundError(
                f"ZIP member not found: {member_name} in {archive_path}"
            ) from exc


def read_archive_image(
    image_path: str,
    project_root: Path | str = PROJECT_ROOT,
) -> Image.Image:
    image_bytes = read_archive_image_bytes(image_path, project_root)
    with Image.open(io.BytesIO(image_bytes)) as image:
        return image.convert("RGB")


def _validate_split_rows(
    rows: list[dict[str, str]],
    label_mapping: dict[str, int],
    csv_path: Path,
) -> None:
    missing_columns = REQUIRED_SPLIT_COLUMNS.difference(rows[0])
    if missing_columns:
        raise ValueError(
            f"{csv_path} is missing required columns: {sorted(missing_columns)}"
        )
    for row_number, row in enumerate(rows, start=2):
        image_id = row["image_id"]
        label = (row["raw_face_type"] or "").strip()
        if not image_id:
            raise ValueError(f"Missing image_id at {csv_path}:{row_number}")
        if not label:
            raise ValueError(f"Missing raw_face_type at {csv_path}:{row_number}")
        if label not in label_mapping:
            raise ValueError(
                f"Label {label!r} at {csv_path}:{row_number} is absent from label mapping"
            )
        parse_archive_image_path(row["image_path"])


class PopfZipDataset(Dataset):
    """Read one modern_v1 split directly from the historical image ZIP."""

    def __init__(
        self,
        csv_path: Path | str,
        label_mapping: dict[str, int] | Path | str,
        preprocess: ImagePreprocessConfig | None = None,
        project_root: Path | str = PROJECT_ROOT,
    ) -> None:
        require_torch()
        self.csv_path = Path(csv_path).resolve()
        self.project_root = Path(project_root).resolve()
        self.preprocess = preprocess or ImagePreprocessConfig()
        self.label_mapping = (
            load_label_mapping(label_mapping)
            if isinstance(label_mapping, (Path, str))
            else dict(label_mapping)
        )
        self.rows = read_csv_rows(self.csv_path)
        _validate_split_rows(self.rows, self.label_mapping, self.csv_path)
        self._archives: dict[Path, zipfile.ZipFile] = {}
        self._mean = torch.tensor(self.preprocess.mean, dtype=torch.float32).view(3, 1, 1)
        self._std = torch.tensor(self.preprocess.std, dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.rows)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_archives"] = {}
        return state

    def __del__(self) -> None:
        for archive in getattr(self, "_archives", {}).values():
            try:
                archive.close()
            except Exception:
                pass

    def _get_archive(self, archive_path: Path) -> zipfile.ZipFile:
        archive = self._archives.get(archive_path)
        if archive is None:
            archive = _open_zip(archive_path)
            self._archives[archive_path] = archive
        return archive

    def _read_row_image(self, row: dict[str, str]) -> Image.Image:
        archive_path, member_name = parse_archive_image_path(
            row["image_path"], self.project_root
        )
        archive = self._get_archive(archive_path)
        try:
            image_bytes = archive.read(member_name)
        except KeyError as exc:
            raise FileNotFoundError(
                f"ZIP member not found: {member_name} in {archive_path}"
            ) from exc
        with Image.open(io.BytesIO(image_bytes)) as image:
            return image.convert("RGB")

    def _preprocess_image(self, image: Image.Image) -> Any:
        resampling = getattr(Image, "Resampling", Image).BILINEAR
        image = image.resize(self.preprocess.image_size, resampling)
        array = np.asarray(image, dtype=np.float32) / 255.0
        if array.shape != (self.preprocess.image_size[1], self.preprocess.image_size[0], 3):
            raise ValueError(f"Unexpected RGB image shape after resize: {array.shape}")
        tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())
        return (tensor - self._mean) / self._std

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        label_name = row["raw_face_type"].strip()
        image = self._read_row_image(row)
        return {
            "image": self._preprocess_image(image),
            "label": torch.tensor(self.label_mapping[label_name], dtype=torch.long),
            "image_id": row["image_id"],
            "image_path": row["image_path"],
            "label_name": label_name,
        }


def _seed_worker(worker_id: int) -> None:
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_dataloaders(
    dataset_dir: Path | str = DEFAULT_DATASET_DIR,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
    label_mapping_path: Path | str | None = None,
    normalize_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    normalize_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> tuple[Any, Any, Any]:
    """Create train, validation, and test DataLoaders with shared labels."""

    runtime = require_torch()
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    mapping_path = (
        Path(label_mapping_path)
        if label_mapping_path is not None
        else Path(dataset_dir) / "label_mapping.json"
    )
    mapping = ensure_label_mapping(
        Path(dataset_dir) / "label_inventory.csv",
        mapping_path,
    )
    preprocess = ImagePreprocessConfig(
        image_size=image_size,
        mean=normalize_mean,
        std=normalize_std,
    )
    train_dataset = PopfZipDataset(
        Path(dataset_dir) / "train.csv", mapping, preprocess=preprocess
    )
    validation_dataset = PopfZipDataset(
        Path(dataset_dir) / "validation.csv", mapping, preprocess=preprocess
    )
    test_dataset = PopfZipDataset(
        Path(dataset_dir) / "test.csv", mapping, preprocess=preprocess
    )

    generator = runtime.Generator()
    generator.manual_seed(seed)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": False,
        "worker_init_fn": _seed_worker if num_workers else None,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **common,
    )
    val_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **common,
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **common,
    )
    return train_loader, val_loader, test_loader
