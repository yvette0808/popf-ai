#!/usr/bin/env python3
"""Audit the historical 272-image candidate without modifying raw assets."""

from __future__ import annotations

import csv
import hashlib
import io
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import openpyxl
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RAW_XLS = ROOT / "data/raw/脸谱新一步/272数据张丹阳.xls"
RAW_ZIP = ROOT / "data/raw/脸谱/整理工作/爬虫脸谱图集.zip"
NETWORK_IMAGE_DIR = ROOT / "data/raw/脸谱新一步/网络图片汇总"
OUT_DIR = ROOT / "data/processed/272_candidate"

SIX_PAPER_CLASSES = {
    "三块瓦脸",
    "碎脸",
    "十字门脸",
    "象形脸",
    "整脸",
    "花三块瓦脸",
}


def scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: scalar(row.get(field)) for field in fieldnames})


def find_soffice() -> str:
    candidates = [
        os.environ.get("SOFFICE"),
        shutil.which("soffice"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise FileNotFoundError("LibreOffice/soffice was not found")


def convert_xls_to_xlsx(source: Path, temp_dir: Path) -> Path:
    soffice = find_soffice()
    profile = temp_dir / "lo-profile"
    profile.mkdir()
    command = [
        soffice,
        f"-env:UserInstallation={profile.as_uri()}",
        "--headless",
        "--convert-to",
        "xlsx",
        "--outdir",
        str(temp_dir),
        str(source),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "LibreOffice conversion failed:\n"
            f"stdout={completed.stdout}\n"
            f"stderr={completed.stderr}"
        )
    converted = temp_dir / f"{source.stem}.xlsx"
    if not converted.exists():
        candidates = sorted(temp_dir.glob("*.xlsx"))
        if len(candidates) != 1:
            raise FileNotFoundError(f"Converted workbook not found in {temp_dir}")
        converted = candidates[0]
    return converted


def read_metadata() -> tuple[list[str], list[dict[str, str]], dict[str, dict[str, str]]]:
    with tempfile.TemporaryDirectory(prefix="popf-272-xls-") as temp_name:
        converted = convert_xls_to_xlsx(RAW_XLS, Path(temp_name))
        workbook = openpyxl.load_workbook(converted, read_only=True, data_only=True)
        worksheet = workbook["Sheet1"]
        all_rows = list(worksheet.iter_rows(values_only=True))

    raw_header = list(all_rows[0])
    last_header_index = max(
        index for index, value in enumerate(raw_header) if value not in (None, "")
    )
    headers = [scalar(value) for value in raw_header[: last_header_index + 1]]
    records: list[dict[str, str]] = []
    by_id: dict[str, dict[str, str]] = {}

    for worksheet_row, values in enumerate(all_rows[1:], start=2):
        values = list(values[: last_header_index + 1])
        if not any(value not in (None, "") for value in values):
            continue
        record = {header: scalar(value) for header, value in zip(headers, values)}
        sequence = record.get("序号", "")
        try:
            numeric_id = int(sequence)
        except ValueError as exc:
            raise ValueError(f"Invalid 序号 at worksheet row {worksheet_row}: {sequence!r}") from exc
        metadata_id = f"{numeric_id:03d}"
        record["metadata_id"] = metadata_id
        record["source_row_number"] = str(worksheet_row)
        records.append(record)
        by_id[metadata_id] = record

    if len(records) != 272:
        raise ValueError(f"Expected 272 metadata rows, found {len(records)}")
    if set(by_id) != {f"{number:03d}" for number in range(1, 273)}:
        raise ValueError("Metadata 序号 is not a complete 1-272 bijection")
    return headers, records, by_id


def read_network_hashes() -> dict[int, tuple[str, str]]:
    result: dict[int, tuple[str, str]] = {}
    if not NETWORK_IMAGE_DIR.exists():
        return result
    for path in NETWORK_IMAGE_DIR.glob("*.jpg"):
        match = re.fullmatch(r"(\d{3})\.jpg", path.name)
        if not match:
            continue
        image_id = int(match.group(1))
        result[image_id] = (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            str(path.relative_to(ROOT)),
        )
    return result


def read_archive() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    network_hashes = read_network_hashes()
    with zipfile.ZipFile(RAW_ZIP, metadata_encoding="gbk") as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            filename = Path(member.filename).name
            match = re.search(r"-(\d+)-s\.[^.]+$", filename)
            if not match:
                raise ValueError(f"Cannot parse image id from archive member: {member.filename}")
            image_id = int(match.group(1))
            data = archive.read(member)
            sha256 = hashlib.sha256(data).hexdigest()
            width = ""
            height = ""
            aspect_ratio = ""
            file_format = ""
            read_status = "unreadable"
            try:
                with Image.open(io.BytesIO(data)) as image:
                    image.verify()
                with Image.open(io.BytesIO(data)) as image:
                    width, height = image.size
                    aspect_ratio = f"{width / height:.8f}"
                    file_format = image.format or ""
                read_status = "readable"
            except Exception:
                pass

            network_hash, network_path = network_hashes.get(image_id, ("", ""))
            if not network_hash:
                network_same_hash = "not_available"
            elif network_hash == sha256:
                network_same_hash = "yes"
            else:
                network_same_hash = "no"

            rows.append(
                {
                    "image_id": f"{image_id:03d}",
                    "image_filename": filename,
                    "archive_member_name": member.filename,
                    "file_format": file_format,
                    "width": width,
                    "height": height,
                    "aspect_ratio": aspect_ratio,
                    "file_size_bytes": member.file_size,
                    "sha256": sha256,
                    "read_status": read_status,
                    "network_expanded_path": network_path,
                    "network_expanded_same_hash": network_same_hash,
                }
            )

    rows.sort(key=lambda row: int(row["image_id"]))
    if len(rows) != 272:
        raise ValueError(f"Expected 272 archive images, found {len(rows)}")
    if [row["image_id"] for row in rows] != [f"{number:03d}" for number in range(1, 273)]:
        raise ValueError("Archive image ids are not a complete 001-272 bijection")
    return rows


def build_mapping(
    image_rows: list[dict[str, str]], metadata_by_id: dict[str, dict[str, str]]
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    mappings: list[dict[str, str]] = []
    unmatched: list[dict[str, str]] = []

    for image in image_rows:
        candidate_id = image["image_id"]
        metadata = metadata_by_id[candidate_id]
        image_width = image["width"]
        image_height = image["height"]
        metadata_width = metadata.get("图像宽度", "")
        metadata_height = metadata.get("图像高度", "")
        dimensions_match = (
            "yes"
            if image_width and image_height
            and image_width == metadata_width
            and image_height == metadata_height
            else "no"
        )
        image_aspect_ratio = ""
        metadata_aspect_ratio = ""
        aspect_ratio_delta = ""
        aspect_ratio_match = "not_computable"
        try:
            image_aspect_ratio = float(image["aspect_ratio"])
            metadata_aspect_ratio = float(metadata_width) / float(metadata_height)
            aspect_ratio_delta = abs(image_aspect_ratio - metadata_aspect_ratio)
            aspect_ratio_match = (
                "yes_within_0.01"
                if aspect_ratio_delta <= 0.01
                else "no"
            )
        except (ValueError, ZeroDivisionError):
            pass
        evidence = (
            "Archive filename suffix 001-272 forms a bijection with XLS 序号 1-272, "
            "so a sequence-only candidate can be listed. However, the filename contains "
            "no role or opera text, the exact pixel dimensions differ from the XLS "
            "image width/height fields, and this does not prove historical identity. "
            "The aspect ratios are broadly compatible with a thumbnail/downsample "
            "relationship, which is supporting evidence only. "
            f"Observed image={image_width}x{image_height}; "
            f"XLS field={metadata_width}x{metadata_height}; "
            f"aspect_ratio_delta={aspect_ratio_delta}; "
            f"network-expanded same hash={image['network_expanded_same_hash']}."
        )
        mappings.append(
            {
                "image_id": image["image_id"],
                "image_filename": image["image_filename"],
                "metadata_id": "unknown",
                "role": "",
                "opera": "",
                "original_face_type": "",
                "sequence_candidate_metadata_id": candidate_id,
                "candidate_role": metadata.get("角色", ""),
                "candidate_opera": metadata.get("来源剧目", ""),
                "candidate_original_face_type": metadata.get("谱式", ""),
                "match_method": "sequence_and_aspect_ratio_candidate_not_accepted",
                "match_confidence": "low",
                "match_status": "unknown",
                "evidence": evidence,
                "image_width": image_width,
                "image_height": image_height,
                "metadata_width": metadata_width,
                "metadata_height": metadata_height,
                "dimension_match": dimensions_match,
                "image_aspect_ratio": image_aspect_ratio,
                "metadata_aspect_ratio": metadata_aspect_ratio,
                "aspect_ratio_delta": aspect_ratio_delta,
                "aspect_ratio_match": aspect_ratio_match,
                "image_sha256": image["sha256"],
            }
        )
        unmatched.append(
            {
                "entity_type": "image",
                "image_id": image["image_id"],
                "image_filename": image["image_filename"],
                "candidate_metadata_id_by_sequence": candidate_id,
                "candidate_role": metadata.get("角色", ""),
                "candidate_opera": metadata.get("来源剧目", ""),
                "candidate_original_face_type": metadata.get("谱式", ""),
                "match_status": "unknown",
                "reason": (
                    "No independent role/opera/image-identity evidence proves this "
                    "archive member is the XLS row. Sequence-only pairing is retained "
                    "as a candidate, not accepted as a match."
                ),
            }
        )
    return mappings, unmatched


def build_face_type_inventory(records: list[dict[str, str]]) -> list[dict[str, str]]:
    counts = Counter(record.get("谱式", "") for record in records)
    result: list[dict[str, str]] = []
    for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if not label:
            result.append(
                {
                    "original_label": "",
                    "count": count,
                    "possible_normalized_label": "",
                    "normalization_status": "missing_original_label",
                    "evidence": "The 272 XLS contains empty values in 原始字段 谱式.",
                    "notes": "Do not infer a category for missing labels.",
                }
            )
        elif label in SIX_PAPER_CLASSES:
            result.append(
                {
                    "original_label": label,
                    "count": count,
                    "possible_normalized_label": label,
                    "normalization_status": "exact_text_observed_not_model_approval",
                    "evidence": (
                        "The literal text is present in the 272 XLS and is one of the "
                        "six paper-named categories. This does not prove the row was "
                        "used in the 2019 model."
                    ),
                    "notes": "Keep as an observed label; do not treat this inventory as a model split.",
                }
            )
        else:
            result.append(
                {
                    "original_label": label,
                    "count": count,
                    "possible_normalized_label": "",
                    "normalization_status": "unresolved",
                    "evidence": (
                        "The literal text is present in the 272 XLS, but no accepted "
                        "normalization to the paper's model classes was established."
                    ),
                    "notes": "Requires source-level or human-reviewed label evidence.",
                }
            )
    return result


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    headers, metadata_records, metadata_by_id = read_metadata()
    image_rows = read_archive()
    mappings, unmatched = build_mapping(image_rows, metadata_by_id)
    face_types = build_face_type_inventory(metadata_records)

    image_fields = [
        "image_id",
        "image_filename",
        "archive_member_name",
        "file_format",
        "width",
        "height",
        "aspect_ratio",
        "file_size_bytes",
        "sha256",
        "read_status",
        "network_expanded_path",
        "network_expanded_same_hash",
    ]
    write_csv(OUT_DIR / "image_manifest.csv", image_fields, image_rows)

    metadata_fields = ["metadata_id", "source_row_number", *headers]
    write_csv(OUT_DIR / "metadata_raw.csv", metadata_fields, metadata_records)

    mapping_fields = [
        "image_id",
        "image_filename",
        "metadata_id",
        "role",
        "opera",
        "original_face_type",
        "sequence_candidate_metadata_id",
        "candidate_role",
        "candidate_opera",
        "candidate_original_face_type",
        "match_method",
        "match_confidence",
        "match_status",
        "evidence",
        "image_width",
        "image_height",
        "metadata_width",
        "metadata_height",
        "dimension_match",
        "image_aspect_ratio",
        "metadata_aspect_ratio",
        "aspect_ratio_delta",
        "aspect_ratio_match",
        "image_sha256",
    ]
    write_csv(OUT_DIR / "image_metadata_mapping.csv", mapping_fields, mappings)

    unmatched_fields = [
        "entity_type",
        "image_id",
        "image_filename",
        "candidate_metadata_id_by_sequence",
        "candidate_role",
        "candidate_opera",
        "candidate_original_face_type",
        "match_status",
        "reason",
    ]
    write_csv(OUT_DIR / "unmatched.csv", unmatched_fields, unmatched)

    face_type_fields = [
        "original_label",
        "count",
        "possible_normalized_label",
        "normalization_status",
        "evidence",
        "notes",
    ]
    write_csv(OUT_DIR / "face_type_inventory.csv", face_type_fields, face_types)

    readable_images = sum(row["read_status"] == "readable" for row in image_rows)
    network_same_hash = sum(row["network_expanded_same_hash"] == "yes" for row in image_rows)
    aspect_ratio_deltas = []
    for mapping in mappings:
        try:
            aspect_ratio_deltas.append(float(mapping["aspect_ratio_delta"]))
        except ValueError:
            pass
    empty_labels = sum(record.get("谱式", "") == "" for record in metadata_records)
    six_counts = {
        label: sum(record.get("谱式", "") == label for record in metadata_records)
        for label in sorted(SIX_PAPER_CLASSES)
    }
    print(
        {
            "image_count": len(image_rows),
            "readable_images": readable_images,
            "metadata_count": len(metadata_records),
            "unique_face_type_values_including_missing": len(face_types),
            "nonempty_face_type_records": 272 - empty_labels,
            "empty_face_type_records": empty_labels,
            "network_expanded_same_hash_count": network_same_hash,
            "aspect_ratio_within_0.005_count": sum(delta <= 0.005 for delta in aspect_ratio_deltas),
            "aspect_ratio_within_0.01_count": sum(delta <= 0.01 for delta in aspect_ratio_deltas),
            "accepted_matches": 0,
            "unknown_matches": len(mappings),
            "six_paper_class_counts": six_counts,
        }
    )


if __name__ == "__main__":
    main()
