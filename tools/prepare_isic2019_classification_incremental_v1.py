#!/usr/bin/env python3
"""Build the strictly de-duplicated ISIC 2019 classification-only increment.

The official training ZIP is treated as immutable and is read directly without
extracting it.  HAM10000 and every IMA++ source ID are excluded first.  The
remaining images are audited against both reference pools by SHA-256 and the
same strict pHash+dHash+dimensions signature used by the accepted IMA++ build.

The output contains train/val images, all-ignore semantic placeholder masks,
an immutable manifest, a combined partial-label manifest, provenance, hashes,
and an atomic ``_SUCCESS`` marker.  It never reads an official challenge test
file and never modifies the accepted IMA++ dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import shutil
import threading
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image


BUILD_VERSION = "isic2019_classification_incremental_v1"
EXPECTED_OFFICIAL_ROWS = 25_331
EXPECTED_HAM_ROWS = 10_015
EXPECTED_IMA_ROWS = 14_967
EXPECTED_ID_CANDIDATES = 14_488
EXPECTED_ZIP_BYTES = 9_771_618_190
OFFICIAL_COLUMNS = ("MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC")
OFFICIAL_TO_CANONICAL = {
    "AK": "akiec",
    "BCC": "bcc",
    "BKL": "bkl",
    "DF": "df",
    "MEL": "mel",
    "NV": "nv",
    "VASC": "vasc",
    "SCC": "scc",
}
CLASS_NAMES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc", "scc")
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
IMAGE_SUFFIXES = {".jpg", ".jpeg"}
_ZIP_LOCAL = threading.local()


class BuildError(RuntimeError):
    """Raised when a source or derived artifact violates the locked contract."""


class UnionFind:
    def __init__(self, items: Iterable[str]) -> None:
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


@dataclass(frozen=True)
class Audit:
    isic_id: str
    source: str
    source_path: str
    sha256: str
    phash64: str
    dhash64: str
    width: int
    height: int

    @property
    def strict_signature(self) -> str:
        return f"{self.phash64}:{self.dhash64}:{self.width}x{self.height}"


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-zip",
        type=Path,
        default=project_root / "Data" / "1" / "ISIC_2019_Training_Input.zip",
    )
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--ham-groupings", type=Path, required=True)
    parser.add_argument(
        "--ima-source-metadata",
        type=Path,
        default=project_root / "Data" / "1" / "img_metadata.csv",
    )
    parser.add_argument(
        "--ima-source-images",
        type=Path,
        default=project_root / "Data" / "1" / "ISIC-images",
    )
    parser.add_argument(
        "--ima-processed-manifest",
        type=Path,
        default=project_root / "Data" / "processed_imaplusplus_binary_v1" / "manifest.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "Data" / "processed_isic2019_classification_incremental_v1",
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=project_root / "tools" / "classification_taxonomy_v1.yaml",
    )
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=max(2, min(12, os.cpu_count() or 4)))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--id-audit-only",
        action="store_true",
        help="Validate metadata and print ID-level counts without reading image payloads or writing output.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise BuildError(f"Missing required CSV: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise BuildError(f"CSV is empty: {path}")
    return rows


def require_columns(rows: list[dict[str, str]], required: set[str], label: str) -> None:
    missing = required - set(rows[0])
    if missing:
        raise BuildError(f"{label} is missing columns: {sorted(missing)}")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not rows:
            raise BuildError(f"Cannot infer columns for empty CSV: {path}")
        fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def bits_to_hex(bits: np.ndarray) -> str:
    return np.packbits(bits.astype(np.uint8).reshape(-1)).tobytes().hex()


def perceptual_hashes(rgb: np.ndarray) -> tuple[str, str]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    phash_img = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    block = cv2.dct(phash_img)[:8, :8]
    median = float(np.median(block.reshape(-1)[1:]))
    phash = bits_to_hex(block > median)
    dhash_img = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    dhash = bits_to_hex(dhash_img[:, 1:] > dhash_img[:, :-1])
    return phash, dhash


def audit_image_bytes(isic_id: str, source: str, source_path: str, data: bytes) -> Audit:
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        width, height = image.size
        rgb = np.asarray(image.convert("RGB"))
    phash64, dhash64 = perceptual_hashes(rgb)
    return Audit(isic_id, source, source_path, sha256_bytes(data), phash64, dhash64, width, height)


def zip_handle(path: Path) -> zipfile.ZipFile:
    current_path = getattr(_ZIP_LOCAL, "path", None)
    handle = getattr(_ZIP_LOCAL, "handle", None)
    if handle is None or current_path != path:
        if handle is not None:
            handle.close()
        _ZIP_LOCAL.path = path
        _ZIP_LOCAL.handle = zipfile.ZipFile(path)
    return _ZIP_LOCAL.handle


def audit_zip_member(task: tuple[Path, str, str]) -> Audit:
    zip_path, member, isic_id = task
    data = zip_handle(zip_path).read(member)
    return audit_image_bytes(isic_id, "ISIC_2019_Training_Input.zip", member, data)


def audit_file(task: tuple[Path, str]) -> Audit:
    path, isic_id = task
    return audit_image_bytes(isic_id, "IMA++", str(path), path.read_bytes())


def official_class(row: dict[str, str]) -> str:
    positives = [column for column in OFFICIAL_COLUMNS if float(row[column]) == 1.0]
    unknown = float(row.get("UNK", "0") or 0)
    total = sum(float(row[column]) for column in OFFICIAL_COLUMNS) + unknown
    if positives and len(positives) == 1 and unknown == 0.0 and total == 1.0:
        return OFFICIAL_TO_CANONICAL[positives[0]]
    raise BuildError(f"Non one-hot or UNK training label for {row.get('image')}: {row}")


def validate_id_contract(
    ground_truth: list[dict[str, str]],
    metadata: list[dict[str, str]],
    ham_rows: list[dict[str, str]],
    ima_rows: list[dict[str, str]],
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]], set[str], set[str], list[str]]:
    require_columns(ground_truth, {"image", "UNK", *OFFICIAL_COLUMNS}, "ISIC 2019 ground truth")
    require_columns(metadata, {"image", "lesion_id"}, "ISIC 2019 metadata")
    require_columns(ham_rows, {"image", "lesion_id"}, "HAM lesion groupings")
    require_columns(ima_rows, {"isic_id"}, "IMA++ source metadata")
    if len(ground_truth) != EXPECTED_OFFICIAL_ROWS or len(metadata) != EXPECTED_OFFICIAL_ROWS:
        raise BuildError(
            f"Expected {EXPECTED_OFFICIAL_ROWS} official rows, found GT={len(ground_truth)} metadata={len(metadata)}"
        )
    if len(ham_rows) != EXPECTED_HAM_ROWS:
        raise BuildError(f"Expected {EXPECTED_HAM_ROWS} HAM rows, found {len(ham_rows)}")
    if len(ima_rows) != EXPECTED_IMA_ROWS:
        raise BuildError(f"Expected {EXPECTED_IMA_ROWS} IMA rows, found {len(ima_rows)}")

    gt_by_id = {row["image"].strip(): row for row in ground_truth}
    metadata_by_id = {row["image"].strip(): row for row in metadata}
    if len(gt_by_id) != len(ground_truth) or len(metadata_by_id) != len(metadata):
        raise BuildError("Duplicate image IDs in official ISIC 2019 CSVs")
    if set(gt_by_id) != set(metadata_by_id):
        raise BuildError("ISIC 2019 ground truth and metadata image IDs differ")
    for row in ground_truth:
        official_class(row)

    ham_ids = {row["image"].strip() for row in ham_rows}
    ima_ids = {row["isic_id"].strip() for row in ima_rows}
    if len(ham_ids) != EXPECTED_HAM_ROWS or len(ima_ids) != EXPECTED_IMA_ROWS:
        raise BuildError("Duplicate or blank IDs in HAM/IMA reference metadata")
    candidates = sorted(set(gt_by_id) - ham_ids - ima_ids)
    if len(candidates) != EXPECTED_ID_CANDIDATES:
        raise BuildError(f"Expected {EXPECTED_ID_CANDIDATES} ID candidates, found {len(candidates)}")
    return gt_by_id, metadata_by_id, ham_ids, ima_ids, candidates


def index_zip_images(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise BuildError(f"Missing official training ZIP: {path}")
    if path.stat().st_size != EXPECTED_ZIP_BYTES:
        raise BuildError(f"Official ZIP size is incomplete: {path.stat().st_size} != {EXPECTED_ZIP_BYTES}")
    if not zipfile.is_zipfile(path):
        raise BuildError(f"Not a valid ZIP archive: {path}")
    result: dict[str, str] = {}
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise BuildError(f"ZIP CRC failure at member: {bad}")
        for member in archive.namelist():
            suffix = Path(member).suffix.lower()
            if suffix not in IMAGE_SUFFIXES or member.endswith("/"):
                continue
            isic_id = Path(member).stem
            if isic_id in result:
                raise BuildError(f"Duplicate image stem in ZIP: {isic_id}")
            result[isic_id] = member
    if len(result) != EXPECTED_OFFICIAL_ROWS:
        raise BuildError(f"Expected {EXPECTED_OFFICIAL_ROWS} JPEGs in ZIP, found {len(result)}")
    return result


def index_ima_images(root: Path, expected_ids: set[str]) -> dict[str, Path]:
    if not root.is_dir():
        raise BuildError(f"Missing IMA++ source image directory: {root}")
    result: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if path.stem in result:
            raise BuildError(f"Duplicate IMA image stem: {path.stem}")
        result[path.stem] = path
    missing = expected_ids - set(result)
    extra = set(result) - expected_ids
    if missing or extra:
        raise BuildError(f"IMA source image index differs from metadata: missing={len(missing)} extra={len(extra)}")
    return result


def parallel_map(tasks: list[Any], function, workers: int, label: str) -> list[Audit]:
    completed = 0
    results: list[Audit] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for audit in pool.map(function, tasks, chunksize=8):
            results.append(audit)
            completed += 1
            if completed % 1000 == 0 or completed == len(tasks):
                print(f"{label}: {completed}/{len(tasks)}", flush=True)
    return results


def strict_deduplicate(
    audits: dict[str, Audit],
    candidates: list[str],
    reference_ids: set[str],
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    reference_sha: dict[str, list[str]] = defaultdict(list)
    reference_sig: dict[str, list[str]] = defaultdict(list)
    for isic_id in sorted(reference_ids):
        audit = audits[isic_id]
        reference_sha[audit.sha256].append(isic_id)
        reference_sig[audit.strict_signature].append(isic_id)

    excluded: list[dict[str, Any]] = []
    eligible: list[str] = []
    for isic_id in candidates:
        audit = audits[isic_id]
        exact = reference_sha.get(audit.sha256, [])
        perceptual = reference_sig.get(audit.strict_signature, [])
        if exact or perceptual:
            excluded.append(
                {
                    "isic_id": isic_id,
                    "reason": "cross_reference_file_duplicate",
                    "exact_reference_ids": ";".join(exact),
                    "strict_perceptual_reference_ids": ";".join(perceptual),
                }
            )
        else:
            eligible.append(isic_id)

    uf = UnionFind(eligible)
    by_sha: dict[str, list[str]] = defaultdict(list)
    by_sig: dict[str, list[str]] = defaultdict(list)
    for isic_id in eligible:
        by_sha[audits[isic_id].sha256].append(isic_id)
        by_sig[audits[isic_id].strict_signature].append(isic_id)
    duplicate_rows: list[dict[str, Any]] = []
    for kind, groups in (("sha256", by_sha), ("strict_phash_dhash_dimensions", by_sig)):
        for signature, members in groups.items():
            if len(members) < 2:
                continue
            first = min(members)
            for member in members:
                uf.union(first, member)
            duplicate_rows.append(
                {"kind": kind, "signature": signature, "count": len(members), "members": ";".join(sorted(members))}
            )

    components: dict[str, list[str]] = defaultdict(list)
    for isic_id in eligible:
        components[uf.find(isic_id)].append(isic_id)
    retained: list[str] = []
    for members in components.values():
        canonical = min(members)
        retained.append(canonical)
        for duplicate in sorted(set(members) - {canonical}):
            excluded.append(
                {
                    "isic_id": duplicate,
                    "reason": "within_candidate_strict_duplicate",
                    "canonical_id": canonical,
                }
            )
    return sorted(retained), excluded, sorted(duplicate_rows, key=lambda row: (row["kind"], row["signature"]))


def split_groups(
    retained: list[str],
    gt_by_id: dict[str, dict[str, str]],
    metadata_by_id: dict[str, dict[str, str]],
    audits: dict[str, Audit],
    val_fraction: float,
    seed: int,
) -> tuple[dict[str, str], dict[str, str], list[dict[str, Any]]]:
    if not 0.0 < val_fraction < 0.5:
        raise BuildError(f"val_fraction must be between 0 and 0.5, found {val_fraction}")
    uf = UnionFind(retained)
    by_lesion: dict[str, list[str]] = defaultdict(list)
    by_sha: dict[str, list[str]] = defaultdict(list)
    by_sig: dict[str, list[str]] = defaultdict(list)
    for isic_id in retained:
        lesion_id = metadata_by_id[isic_id]["lesion_id"].strip()
        if lesion_id:
            by_lesion[lesion_id].append(isic_id)
        by_sha[audits[isic_id].sha256].append(isic_id)
        by_sig[audits[isic_id].strict_signature].append(isic_id)
    for groups in (by_lesion, by_sha, by_sig):
        for members in groups.values():
            for member in members[1:]:
                uf.union(members[0], member)

    members_by_root: dict[str, list[str]] = defaultdict(list)
    for isic_id in retained:
        members_by_root[uf.find(isic_id)].append(isic_id)
    group_rows: list[dict[str, Any]] = []
    groups_by_class: dict[str, list[tuple[str, list[str]]]] = defaultdict(list)
    group_id_by_image: dict[str, str] = {}
    for index, members in enumerate(sorted(members_by_root.values(), key=lambda values: min(values)), start=1):
        classes = {official_class(gt_by_id[isic_id]) for isic_id in members}
        if len(classes) != 1:
            raise BuildError(f"A leakage-safe group spans multiple classes: {sorted(members)} -> {sorted(classes)}")
        group_id = f"CLS19G{index:05d}"
        disease = classes.pop()
        for isic_id in members:
            group_id_by_image[isic_id] = group_id
        groups_by_class[disease].append((group_id, sorted(members)))

    rng = random.Random(seed)
    split_by_id: dict[str, str] = {}
    for disease in CLASS_NAMES:
        groups = list(groups_by_class[disease])
        rng.shuffle(groups)
        total = sum(len(members) for _, members in groups)
        target = max(1, round(total * val_fraction))
        val_groups: set[str] = set()
        accumulated = 0
        for group_id, members in groups:
            if accumulated >= target:
                break
            val_groups.add(group_id)
            accumulated += len(members)
        if len(val_groups) == len(groups) and len(groups) > 1:
            val_groups.remove(groups[-1][0])
        for group_id, members in groups:
            split = "val" if group_id in val_groups else "train"
            for isic_id in members:
                split_by_id[isic_id] = split
            group_rows.append(
                {
                    "group_id": group_id,
                    "class_name": disease,
                    "split": split,
                    "image_count": len(members),
                    "members": ";".join(members),
                }
            )
    return split_by_id, group_id_by_image, sorted(group_rows, key=lambda row: row["group_id"])


def ensure_output_file(data: bytes, destination: Path, expected_sha: str, resume: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not resume:
            raise BuildError(f"Unexpected existing staging file without --resume: {destination}")
        if sha256_file(destination) == expected_sha:
            return
        destination.unlink()
    destination.write_bytes(data)
    if sha256_file(destination) != expected_sha:
        raise BuildError(f"Written image hash mismatch: {destination}")


def ensure_ignore_mask(destination: Path, size: tuple[int, int], resume: bool) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not resume:
            raise BuildError(f"Unexpected existing staging mask without --resume: {destination}")
        with Image.open(destination) as image:
            array = np.asarray(image.convert("L"))
            if image.size == size and array.size and np.all(array == 255):
                return sha256_file(destination)
        destination.unlink()
    Image.new("L", size, color=255).save(destination, format="PNG", optimize=True)
    with Image.open(destination) as image:
        array = np.asarray(image.convert("L"))
        if image.size != size or not np.all(array == 255):
            raise BuildError(f"Invalid all-ignore placeholder mask: {destination}")
    return sha256_file(destination)


def combined_manifest_rows(
    ima_manifest_path: Path,
    gt_by_id: dict[str, dict[str, str]],
    classification_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ima_rows = read_csv(ima_manifest_path)
    require_columns(
        ima_rows,
        {"isic_id", "split", "group_id", "image_relpath", "mask_relpath", "lesion_id", "patient_id", "image_sha256"},
        "processed IMA++ manifest",
    )
    result: list[dict[str, Any]] = []
    for row in ima_rows:
        isic_id = row["isic_id"].strip()
        gt = gt_by_id.get(isic_id)
        class_name = official_class(gt) if gt is not None else ""
        result.append(
            {
                "isic_id": isic_id,
                "split": row["split"],
                "dataset_role": "imaplusplus_binary",
                "image_path": f"processed_imaplusplus_binary_v1/{row['image_relpath']}",
                "mask_path": f"processed_imaplusplus_binary_v1/{row['mask_relpath']}",
                "has_segmentation": 1,
                "has_classification": int(bool(class_name)),
                "classification_target": CLASS_TO_ID[class_name] if class_name else -1,
                "class_name": class_name,
                "lesion_id": row["lesion_id"],
                "patient_id": row["patient_id"],
                "group_id": row["group_id"],
                "image_sha256": row["image_sha256"],
                "phash64": row.get("phash64", ""),
                "dhash64": row.get("dhash64", ""),
                "width": row.get("width", ""),
                "height": row.get("height", ""),
            }
        )
    for row in classification_rows:
        result.append(
            {
                "isic_id": row["isic_id"],
                "split": row["split"],
                "dataset_role": "isic2019_classification_incremental",
                "image_path": f"processed_isic2019_classification_incremental_v1/{row['image_relpath']}",
                "mask_path": f"processed_isic2019_classification_incremental_v1/{row['mask_relpath']}",
                "has_segmentation": 0,
                "has_classification": 1,
                "classification_target": row["class_id"],
                "class_name": row["class_name"],
                "lesion_id": row["lesion_id"],
                "patient_id": "",
                "group_id": row["group_id"],
                "image_sha256": row["image_sha256"],
                "phash64": row["phash64"],
                "dhash64": row["dhash64"],
                "width": row["width"],
                "height": row["height"],
            }
        )
    return sorted(result, key=lambda row: (row["split"], row["dataset_role"], row["isic_id"]))


def build(args: argparse.Namespace) -> None:
    paths = [
        args.training_zip,
        args.ground_truth,
        args.metadata,
        args.ham_groupings,
        args.ima_source_metadata,
        args.ima_source_images,
        args.ima_processed_manifest,
        args.output,
        args.taxonomy,
    ]
    (
        training_zip,
        ground_truth_path,
        metadata_path,
        ham_path,
        ima_metadata_path,
        ima_images_root,
        ima_manifest_path,
        output,
        taxonomy_path,
    ) = [path.resolve() for path in paths]
    ground_truth = read_csv(ground_truth_path)
    metadata = read_csv(metadata_path)
    ham_rows = read_csv(ham_path)
    ima_rows = read_csv(ima_metadata_path)
    gt_by_id, metadata_by_id, ham_ids, ima_ids, candidates = validate_id_contract(
        ground_truth, metadata, ham_rows, ima_rows
    )
    id_counts = Counter(official_class(gt_by_id[isic_id]) for isic_id in candidates)
    print(json.dumps({"id_candidates": len(candidates), "class_counts": dict(sorted(id_counts.items()))}, indent=2))
    if args.id_audit_only:
        print("ID audit passed; image payloads and output directory were not read or modified.")
        return
    if not taxonomy_path.is_file():
        raise BuildError(f"Missing locked taxonomy file: {taxonomy_path}")
    if output.exists():
        raise BuildError(f"Refusing to overwrite published output: {output}")
    staging = output.with_name(output.name + ".building")
    if staging.exists() and not args.resume:
        raise BuildError(f"Staging exists; pass --resume after inspection: {staging}")
    staging.mkdir(parents=True, exist_ok=True)

    zip_index = index_zip_images(training_zip)
    if set(zip_index) != set(gt_by_id):
        raise BuildError("Official ZIP image IDs differ from ground truth")
    ima_index = index_ima_images(ima_images_root, ima_ids)

    official_audits = parallel_map(
        [(training_zip, zip_index[isic_id], isic_id) for isic_id in sorted(zip_index)],
        audit_zip_member,
        args.workers,
        "ISIC 2019 image audit",
    )
    ima_audits = parallel_map(
        [(ima_index[isic_id], isic_id) for isic_id in sorted(ima_index)],
        audit_file,
        args.workers,
        "IMA++ reference audit",
    )
    audits = {audit.isic_id: audit for audit in official_audits}
    if set(audits) != set(gt_by_id):
        raise BuildError("Official audit lost image IDs")

    # IMA IDs overlapping ISIC 2019 are already represented by the official
    # audit.  Non-overlapping IMA IDs extend the reference signature pool.
    reference_ids = set(ham_ids)
    for audit in ima_audits:
        synthetic_id = audit.isic_id
        if synthetic_id in audits:
            reference_ids.add(synthetic_id)
        else:
            key = f"IMA_REFERENCE::{synthetic_id}"
            audits[key] = Audit(
                key,
                audit.source,
                audit.source_path,
                audit.sha256,
                audit.phash64,
                audit.dhash64,
                audit.width,
                audit.height,
            )
            reference_ids.add(key)

    retained, file_excluded, duplicate_rows = strict_deduplicate(audits, candidates, reference_ids)
    split_by_id, group_by_id, group_rows = split_groups(
        retained, gt_by_id, metadata_by_id, audits, args.val_fraction, args.seed
    )

    manifest: list[dict[str, Any]] = []
    hashes: list[dict[str, Any]] = []
    with zipfile.ZipFile(training_zip) as archive:
        for index, isic_id in enumerate(retained, start=1):
            audit = audits[isic_id]
            split = split_by_id[isic_id]
            member = zip_index[isic_id]
            data = archive.read(member)
            image_rel = Path("images") / split / f"{isic_id}.jpg"
            mask_rel = Path("masks") / split / f"{isic_id}.png"
            ensure_output_file(data, staging / image_rel, audit.sha256, args.resume)
            mask_sha = ensure_ignore_mask(staging / mask_rel, (audit.width, audit.height), args.resume)
            class_name = official_class(gt_by_id[isic_id])
            meta = metadata_by_id[isic_id]
            manifest.append(
                {
                    "isic_id": isic_id,
                    "split": split,
                    "group_id": group_by_id[isic_id],
                    "image_relpath": image_rel.as_posix(),
                    "mask_relpath": mask_rel.as_posix(),
                    "has_segmentation": 0,
                    "has_classification": 1,
                    "class_id": CLASS_TO_ID[class_name],
                    "class_name": class_name,
                    "official_column": next(key for key, value in OFFICIAL_TO_CANONICAL.items() if value == class_name),
                    "lesion_id": meta["lesion_id"].strip(),
                    "age_approx": meta.get("age_approx", ""),
                    "anatom_site_general": meta.get("anatom_site_general", ""),
                    "sex": meta.get("sex", ""),
                    "source_zip_member": member,
                    "image_sha256": audit.sha256,
                    "ignore_mask_sha256": mask_sha,
                    "phash64": audit.phash64,
                    "dhash64": audit.dhash64,
                    "width": audit.width,
                    "height": audit.height,
                    "copyright_license": "CC-BY-NC",
                }
            )
            hashes.extend(
                [
                    {"relative_path": image_rel.as_posix(), "sha256": audit.sha256},
                    {"relative_path": mask_rel.as_posix(), "sha256": mask_sha},
                ]
            )
            if index % 1000 == 0 or index == len(retained):
                print(f"Output materialization: {index}/{len(retained)}", flush=True)

    id_excluded = [
        *({"isic_id": isic_id, "reason": "HAM10000_ID"} for isic_id in sorted(set(gt_by_id) & ham_ids)),
        *({"isic_id": isic_id, "reason": "IMA_PLUS_PLUS_ID"} for isic_id in sorted(set(gt_by_id) & ima_ids)),
    ]
    excluded = sorted([*id_excluded, *file_excluded], key=lambda row: (row["reason"], row["isic_id"]))
    write_csv(staging / "manifest.csv", manifest)
    write_csv(staging / "excluded.csv", excluded)
    write_csv(
        staging / "duplicate_groups.csv",
        duplicate_rows,
        fieldnames=["kind", "signature", "count", "members"],
    )
    write_csv(staging / "group_manifest.csv", group_rows)
    write_csv(staging / "hashes_sha256.csv", hashes)
    combined = combined_manifest_rows(ima_manifest_path, gt_by_id, manifest)
    write_csv(staging / "multitask_manifest.csv", combined)

    provenance = staging / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    for source in (ground_truth_path, metadata_path, ham_path, ima_metadata_path, taxonomy_path):
        shutil.copy2(source, provenance / source.name)
    source_hashes = [
        {"file": training_zip.name, "bytes": training_zip.stat().st_size, "sha256": sha256_file(training_zip)},
        *(
            {"file": source.name, "bytes": source.stat().st_size, "sha256": sha256_file(source)}
            for source in (ground_truth_path, metadata_path, ham_path, ima_metadata_path, taxonomy_path)
        ),
    ]
    write_csv(provenance / "source_hashes_sha256.csv", source_hashes)

    data_yaml = f"""# Partial-label binary segmentation + 8-class classification data v1
path: {output.parent.as_posix()}
train:
  - processed_imaplusplus_binary_v1/images/train
  - processed_isic2019_classification_incremental_v1/images/train
val:
  - processed_imaplusplus_binary_v1/images/val
  - processed_isic2019_classification_incremental_v1/images/val
masks_dir: masks
nc: 1
names:
  0: lesion
partial_label_manifest: processed_isic2019_classification_incremental_v1/multitask_manifest.csv
classification_taxonomy: {taxonomy_path.as_posix()}
test: null
"""
    (staging / "multitask_data.yaml").write_text(data_yaml, encoding="utf-8")

    split_counts = Counter(row["split"] for row in manifest)
    class_counts = Counter(row["class_name"] for row in manifest)
    class_split_counts: dict[str, dict[str, int]] = defaultdict(dict)
    for disease in CLASS_NAMES:
        for split in ("train", "val"):
            class_split_counts[disease][split] = sum(
                row["class_name"] == disease and row["split"] == split for row in manifest
            )
    summary = {
        "build_version": BUILD_VERSION,
        "created_local_time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "official_training_rows": len(gt_by_id),
        "ham_id_exclusions": len(set(gt_by_id) & ham_ids),
        "imaplusplus_id_exclusions": len(set(gt_by_id) & ima_ids),
        "id_candidates": len(candidates),
        "cross_reference_file_exclusions": sum(
            row["reason"] == "cross_reference_file_duplicate" for row in file_excluded
        ),
        "within_candidate_duplicate_exclusions": sum(
            row["reason"] == "within_candidate_strict_duplicate" for row in file_excluded
        ),
        "retained": len(manifest),
        "split_counts": dict(sorted(split_counts.items())),
        "class_counts": dict(sorted(class_counts.items())),
        "class_split_counts": dict(class_split_counts),
        "blank_lesion_ids": sum(not row["lesion_id"] for row in manifest),
        "classification_taxonomy_gate": "PASS",
        "classification_data_id_dedup_gate": "PASS",
        "classification_data_file_dedup_gate": "PASS",
        "project_run10_gate": "BLOCKED_PENDING_MODEL_AND_CLASSIFICATION_CONTRACT",
        "test_metrics_used": False,
    }
    (staging / "audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = f"""# ISIC 2019 官方分类增量 v1 构建报告

`BUILD_VERSION={BUILD_VERSION}`  
`CLASSIFICATION_TAXONOMY_GATE=PASS`  
`CLASSIFICATION_DATA_ID_DEDUP_GATE=PASS`  
`CLASSIFICATION_DATA_FILE_DEDUP_GATE=PASS`  
`PROJECT_RUN10_GATE=BLOCKED_PENDING_MODEL_AND_CLASSIFICATION_CONTRACT`

- 官方 training：{len(gt_by_id)} 张；没有读取官方 test。
- HAM10000 ID 排除：{len(set(gt_by_id) & ham_ids)}。
- IMA++ 原始 ID 排除：{len(set(gt_by_id) & ima_ids)}。
- ID 级候选：{len(candidates)}。
- 跨参考池文件/严格感知重复排除：{summary['cross_reference_file_exclusions']}。
- 候选内部严格重复排除：{summary['within_candidate_duplicate_exclusions']}。
- 最终保留：{len(manifest)}；train={split_counts.get('train', 0)}，val={split_counts.get('val', 0)}。
- 分类-only 的 PNG mask 全为 255，只是 SemanticDataset ignore 占位；`has_segmentation=0`，绝非病灶真值。
- split 在模型训练前按 lesion/hash/严格感知组和 seed={args.seed} 固定；空 lesion_id 作为 singleton 并保留局限性记录。
- IMA++ manifest、split、0/1 掩码和值域均未修改。
"""
    (staging / "DATASET_REPORT.md").write_text(report, encoding="utf-8")
    (staging / "_SUCCESS").write_text(
        json.dumps(
            {
                "build_version": BUILD_VERSION,
                "retained": len(manifest),
                "completed_local_time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    staging.replace(output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Published: {output}")


def main() -> None:
    args = parse_args()
    build(args)


if __name__ == "__main__":
    main()
