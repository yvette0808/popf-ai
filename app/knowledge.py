"""Data-backed knowledge and related-image lookup for the local MVP.

The MVP only renders values that already exist in the processed historical
metadata. It does not invent cultural explanations or normalize labels.
"""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path
from typing import Any

from src.datasets.popf_dataset import parse_archive_image_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = PROJECT_ROOT / "data/processed/272_candidate/metadata_raw.csv"
MANIFEST_PATH = PROJECT_ROOT / "data/processed/modern_dataset/dataset_manifest.csv"
ARCHIVE_PATH = PROJECT_ROOT / "data/raw/脸谱/整理工作/爬虫脸谱图集.zip"

DISPLAY_FIELDS = (
    ("角色", "角色"),
    ("来源剧目", "剧目"),
    ("谱式", "谱式"),
    ("绘法", "绘法"),
    ("主色", "主色"),
    ("主色寓意", "主色寓意"),
    ("性格特征", "性格特征"),
    ("历史演变", "历史演变"),
    ("角色简介", "角色简介"),
    ("来源剧目简介", "剧目简介"),
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class KnowledgeIndex:
    """Index existing metadata and modern_v1 image references."""

    def __init__(
        self,
        metadata_path: Path = METADATA_PATH,
        manifest_path: Path = MANIFEST_PATH,
    ) -> None:
        self.metadata_rows = _read_csv(metadata_path)
        self.manifest_rows = _read_csv(manifest_path)
        self.metadata_by_label: dict[str, list[dict[str, str]]] = {}
        for row in self.metadata_rows:
            label = (row.get("谱式") or "").strip()
            if label:
                self.metadata_by_label.setdefault(label, []).append(row)

        self.images_by_id: dict[str, dict[str, str]] = {}
        for row in self.manifest_rows:
            if row.get("modern_v1_included", "").strip().lower() != "yes":
                continue
            image_id = row.get("image_id", "").strip()
            if image_id:
                self.images_by_id[image_id] = row

    def _record_fields(self, row: dict[str, str]) -> list[dict[str, str]]:
        fields: list[dict[str, str]] = []
        for source_name, display_name in DISPLAY_FIELDS:
            value = (row.get(source_name) or "").strip()
            if value:
                fields.append(
                    {
                        "field": display_name,
                        "value": value,
                        "source_field": source_name,
                    }
                )
        return fields

    def knowledge_for_label(self, label: str, max_records: int = 4) -> dict[str, Any]:
        rows = self.metadata_by_label.get(label, [])
        records: list[dict[str, Any]] = []
        for row in rows[:max_records]:
            records.append(
                {
                    "metadata_id": row.get("metadata_id", ""),
                    "role": (row.get("角色") or "").strip(),
                    "opera": (row.get("来源剧目") or "").strip(),
                    "fields": self._record_fields(row),
                }
            )
        return {
            "label": label,
            "record_count": len(rows),
            "records": records,
            "source": "data/processed/272_candidate/metadata_raw.csv",
            "source_note": (
                "以下内容是当前项目历史元数据中已有字段的摘录，"
                "不代表新增文化考据，也不代表模型对角色或剧目的独立验证。"
            ),
        }

    def related_images(self, label: str, limit: int = 6) -> list[dict[str, Any]]:
        rows = [
            row
            for row in self.images_by_id.values()
            if (row.get("raw_face_type") or "").strip() == label
        ]
        rows.sort(key=lambda row: int(row.get("sequence") or 0))
        results: list[dict[str, Any]] = []
        for row in rows[:limit]:
            results.append(
                {
                    "image_id": row["image_id"],
                    "label": label,
                    "role": (row.get("role") or "").strip(),
                    "opera": (row.get("opera") or "").strip(),
                    "image_url": f"/api/reference-image/{row['image_id']}",
                }
            )
        return results

    def image_bytes(self, image_id: str) -> tuple[bytes, str]:
        row = self.images_by_id.get(image_id)
        if row is None:
            raise KeyError(image_id)
        image_path = row["image_path"]
        archive_path, member_name = parse_archive_image_path(
            image_path,
            PROJECT_ROOT,
        )
        try:
            with zipfile.ZipFile(archive_path, "r", metadata_encoding="gbk") as archive:
                data = archive.read(member_name)
        except TypeError:
            with zipfile.ZipFile(archive_path, "r") as archive:
                data = archive.read(member_name)
        suffix = Path(member_name).suffix.lower()
        media_type = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
        }.get(suffix, "application/octet-stream")
        return data, media_type

