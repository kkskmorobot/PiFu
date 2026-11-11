#!/usr/bin/env python3
"""Independent read-only acceptance checks for the classification increment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


CLASS_NAMES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc", "scc")
EXPECTED_RETAINED = 14_431
EXPECTED_SPLITS = {"train": 12_979, "val": 1_452}
EXPECTED_ZIP_BYTES = 9_771_618_190


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=root / "Data" / "processed_isic2019_classification_incremental_v1",
    )
    parser.add_argument(
        "--training-zip",
        type=Path,
        default=root / "Data" / "1" / "ISIC_2019_Training_Input.zip",
    )
    parser.add_argument(
        "--ima-manifest",
        type=Path,
        default=root / "Data" / "processed_imaplusplus_binary_v1" / "manifest.csv",
    )
    parser.add_argument("--hash-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Empty CSV: {path}")
    return rows


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def assert_single_split(rows: list[dict[str, str]], key: str, *, skip_blank: bool = True) -> int:
    values: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        value = row[key].strip()
        if not value and skip_blank:
            continue
        values[value].add(row["split"])
    conflicts = {value: splits for value, splits in values.items() if len(splits) > 1}
    if conflicts:
        first = next(iter(conflicts.items()))
        raise RuntimeError(f"{key} crosses splits: {len(conflicts)} groups; first={first}")
    return len(values)


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    training_zip = args.training_zip.resolve()
    ima_manifest_path = args.ima_manifest.resolve()
    required = (
        "_SUCCESS",
        "manifest.csv",
        "excluded.csv",
        "group_manifest.csv",
        "duplicate_groups.csv",
        "hashes_sha256.csv",
        "multitask_manifest.csv",
        "audit_summary.json",
        "multitask_data.yaml",
    )
    missing = [name for name in required if not (dataset / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing published artifacts: {missing}")
    summary = json.loads((dataset / "audit_summary.json").read_text(encoding="utf-8"))
    if summary.get("classification_data_file_dedup_gate") != "PASS":
        raise RuntimeError("Published audit summary does not mark file dedup PASS")
    if summary.get("project_run10_gate") != "BLOCKED_PENDING_MODEL_AND_CLASSIFICATION_CONTRACT":
        raise RuntimeError("Project gate changed unexpectedly")
    if summary.get("test_metrics_used") is not False:
        raise RuntimeError("Published audit summary claims test metrics were used")

    manifest = read_csv(dataset / "manifest.csv")
    excluded = read_csv(dataset / "excluded.csv")
    groups = read_csv(dataset / "group_manifest.csv")
    hashes = read_csv(dataset / "hashes_sha256.csv")
    multitask = read_csv(dataset / "multitask_manifest.csv")
    ima_manifest = read_csv(ima_manifest_path)
    if len(manifest) != EXPECTED_RETAINED:
        raise RuntimeError(f"Manifest rows {len(manifest)} != {EXPECTED_RETAINED}")
    split_counts = Counter(row["split"] for row in manifest)
    if dict(split_counts) != EXPECTED_SPLITS:
        raise RuntimeError(f"Unexpected split counts: {dict(split_counts)}")
    if len({row["isic_id"] for row in manifest}) != len(manifest):
        raise RuntimeError("Duplicate isic_id in retained manifest")
    for row in manifest:
        target = int(row["class_id"])
        if target not in range(8) or row["class_name"] != CLASS_NAMES[target]:
            raise RuntimeError(f"Taxonomy mismatch: {row['isic_id']}")
        if row["has_segmentation"] != "0" or row["has_classification"] != "1":
            raise RuntimeError(f"Task availability mismatch: {row['isic_id']}")

    image_count = 0
    mask_count = 0
    for split, expected in EXPECTED_SPLITS.items():
        images = sorted((dataset / "images" / split).glob("*.jpg"))
        masks = sorted((dataset / "masks" / split).glob("*.png"))
        if len(images) != expected or len(masks) != expected:
            raise RuntimeError(f"Disk counts for {split}: images={len(images)} masks={len(masks)} expected={expected}")
        if {path.stem for path in images} != {path.stem for path in masks}:
            raise RuntimeError(f"Image/mask pairing mismatch in {split}")
        image_count += len(images)
        mask_count += len(masks)

    # Full decode, dimension, and placeholder-value validation.
    for index, row in enumerate(manifest, start=1):
        image_path = dataset / row["image_relpath"]
        mask_path = dataset / row["mask_relpath"]
        with Image.open(image_path) as image:
            image.load()
            image_size = image.size
        with Image.open(mask_path) as mask_image:
            mask_image.load()
            mask = np.asarray(mask_image.convert("L"))
            if mask_image.size != image_size or not mask.size or not np.all(mask == 255):
                raise RuntimeError(f"Invalid classification-only placeholder mask: {row['isic_id']}")
        if index % 2000 == 0 or index == len(manifest):
            print(f"Full image/mask validation: {index}/{len(manifest)}", flush=True)

    hash_by_path = {row["relative_path"]: row["sha256"] for row in hashes}
    if len(hash_by_path) != len(manifest) * 2:
        raise RuntimeError(f"Expected {len(manifest) * 2} output hashes, found {len(hash_by_path)}")
    rng = random.Random(args.seed)
    sampled = rng.sample(manifest, min(args.hash_samples, len(manifest)))
    for row in sampled:
        for field in ("image_relpath", "mask_relpath"):
            relative = row[field]
            if sha256_file(dataset / relative) != hash_by_path[relative]:
                raise RuntimeError(f"Sampled SHA-256 mismatch: {relative}")

    lesion_groups = assert_single_split(manifest, "lesion_id")
    strict_groups = assert_single_split(manifest, "group_id", skip_blank=False)
    if len({row["image_sha256"] for row in manifest}) != len(manifest):
        raise RuntimeError("Retained classification images still contain exact SHA-256 duplicates")
    signatures = {f"{row['phash64']}:{row['dhash64']}:{row['width']}x{row['height']}" for row in manifest}
    if len(signatures) != len(manifest):
        raise RuntimeError("Retained classification images still contain strict perceptual duplicates")
    group_split = {row["group_id"]: row["split"] for row in groups}
    if len(group_split) != len(groups):
        raise RuntimeError("Duplicate group_id in group_manifest.csv")

    excluded_counts = Counter(row["reason"] for row in excluded)
    expected_excluded = {
        "HAM10000_ID": 10_015,
        "IMA_PLUS_PLUS_ID": 828,
        "cross_reference_file_duplicate": 2,
        "within_candidate_strict_duplicate": 55,
    }
    if dict(excluded_counts) != expected_excluded:
        raise RuntimeError(f"Unexpected exclusion counts: {dict(excluded_counts)}")
    if len(multitask) != len(ima_manifest) + len(manifest):
        raise RuntimeError(
            f"Combined manifest rows {len(multitask)} != IMA {len(ima_manifest)} + classification {len(manifest)}"
        )
    role_counts = Counter(row["dataset_role"] for row in multitask)
    if role_counts != {"imaplusplus_binary": 11_409, "isic2019_classification_incremental": 14_431}:
        raise RuntimeError(f"Unexpected combined roles: {dict(role_counts)}")

    if training_zip.stat().st_size != EXPECTED_ZIP_BYTES or not zipfile.is_zipfile(training_zip):
        raise RuntimeError("Official training ZIP size/type validation failed")
    with zipfile.ZipFile(training_zip) as archive:
        jpeg_members = [name for name in archive.namelist() if Path(name).suffix.lower() in {".jpg", ".jpeg"}]
    if len(jpeg_members) != 25_331:
        raise RuntimeError(f"Official ZIP JPEG count is {len(jpeg_members)}, expected 25331")

    result = {
        "status": "PASS",
        "scope": "independent read-only validation; no model test metrics",
        "retained": len(manifest),
        "split_counts": dict(split_counts),
        "disk_images": image_count,
        "disk_masks": mask_count,
        "nonblank_lesion_groups": lesion_groups,
        "split_safe_groups": strict_groups,
        "sampled_hashes_recomputed": len(sampled) * 2,
        "exclusion_counts": dict(excluded_counts),
        "multitask_role_counts": dict(role_counts),
        "official_zip_bytes": training_zip.stat().st_size,
        "official_zip_jpegs": len(jpeg_members),
        "classification_data_gate": "PASS",
        "project_run10_gate": "BLOCKED_PENDING_MODEL_AND_CLASSIFICATION_CONTRACT",
        "test_metrics_used": False,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
