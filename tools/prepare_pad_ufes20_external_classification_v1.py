#!/usr/bin/env python3
"""Build PAD-UFES-20 v1 as a strictly audited external classification candidate.

This builder never creates train/val/test splits, never writes masks, and never
adds PAD-UFES-20 to a training YAML.  It validates the official four source
files, maps only the official ``diagnostic`` field to the locked eight-class
taxonomy, de-duplicates against the complete registered reference pools, and
publishes atomically only after all invariants pass.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
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
import yaml
from PIL import Image


BUILD_VERSION = "pad_ufes20_external_classification_v1"
EXPECTED_ROWS = 2_298
EXPECTED_PATIENTS = 1_373
EXPECTED_LESIONS = 1_641
EXPECTED_DIAGNOSTICS = {
    "ACK": 730,
    "BCC": 845,
    "MEL": 52,
    "NEV": 244,
    "SCC": 192,
    "SEK": 235,
}
EXPECTED_SOURCE_FILES = {
    "metadata.csv": (316_209, "14d145235cedb022548257acb0d84dcd949e2c916f65d2baa7c38ed5339e9527"),
    "imgs_part_1.zip": (1_245_184_680, "0ab44f60938bf57445e12f518a8878954cc734e6b0aec6d01194e2d26b4b2dca"),
    "imgs_part_2.zip": (1_126_646_990, "e2d9a3cbd58e823f5ae33163c48643e7d1b54ae3f9e145f01f8e9f16a363a60b"),
    "imgs_part_3.zip": (1_220_565_093, "ecc4ef10143a43e1d01cb736773148607a78b530417eb76f2c38ad24bf5d0d2c"),
}
PAD_TO_CANONICAL = {
    "ACK": "akiec",
    "BCC": "bcc",
    "MEL": "mel",
    "NEV": "nv",
    "SCC": "scc",
    "SEK": "bkl",
}
CLASS_NAMES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc", "scc")
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
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
    item_id: str
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
        "--source-dir",
        type=Path,
        default=project_root / "Data" / "external_candidates" / "pad_ufes_20_v1_official",
    )
    parser.add_argument(
        "--isic2019-zip",
        type=Path,
        default=project_root / "Data" / "1" / "ISIC_2019_Training_Input.zip",
    )
    parser.add_argument(
        "--ham-groupings",
        type=Path,
        default=project_root
        / "Data"
        / "processed_isic2019_classification_incremental_v1"
        / "provenance"
        / "ISIC2018_Task3_Training_LesionGroupings.csv",
    )
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
        "--ima-binary-manifest",
        type=Path,
        default=project_root / "Data" / "processed_imaplusplus_binary_v1" / "manifest.csv",
    )
    parser.add_argument(
        "--isic2018-external-manifest",
        type=Path,
        default=project_root / "Data" / "processed_isic2018_task1_incremental_v1" / "manifest.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "Data" / "processed_pad_ufes_20_external_classification_v1",
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=project_root / "tools" / "classification_taxonomy_v1.yaml",
    )
    parser.add_argument("--workers", type=int, default=max(2, min(12, os.cpu_count() or 4)))
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise BuildError(f"Missing CSV: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise BuildError(f"CSV is empty: {path}")
    return rows


def require_columns(rows: list[dict[str, str]], required: set[str], label: str) -> None:
    missing = required - set(rows[0])
    if missing:
        raise BuildError(f"{label} missing columns: {sorted(missing)}")


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


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
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


def audit_image_bytes(item_id: str, source: str, source_path: str, data: bytes) -> Audit:
    digest = hashlib.sha256(data).hexdigest()
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        width, height = image.size
        rgb = np.asarray(image.convert("RGB"))
    phash64, dhash64 = perceptual_hashes(rgb)
    return Audit(item_id, source, source_path, digest, phash64, dhash64, width, height)


def zip_handle(path: Path) -> zipfile.ZipFile:
    handles = getattr(_ZIP_LOCAL, "handles", None)
    if handles is None:
        handles = {}
        _ZIP_LOCAL.handles = handles
    if path not in handles:
        handles[path] = zipfile.ZipFile(path)
    return handles[path]


def audit_zip_member(task: tuple[Path, str, str, str]) -> Audit:
    zip_path, member, item_id, source = task
    return audit_image_bytes(item_id, source, member, zip_handle(zip_path).read(member))


def audit_file(task: tuple[Path, str, str]) -> Audit:
    path, item_id, source = task
    return audit_image_bytes(item_id, source, str(path), path.read_bytes())


def parallel_map(tasks: list[Any], function, workers: int, label: str) -> list[Audit]:
    results: list[Audit] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, audit in enumerate(pool.map(function, tasks, chunksize=8), start=1):
            results.append(audit)
            if index % 1000 == 0 or index == len(tasks):
                print(f"{label}: {index}/{len(tasks)}", flush=True)
    return results


def validate_taxonomy(path: Path) -> str:
    if not path.is_file():
        raise BuildError(f"Missing taxonomy: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data.get("version") != "classification_taxonomy_v1" or data.get("num_classes") != 8:
        raise BuildError("Taxonomy version/count differs from locked contract")
    names = tuple(data["classes"][index]["name"] for index in range(8))
    if names != CLASS_NAMES:
        raise BuildError(f"Taxonomy names/order differ: {names}")
    return sha256_file(path)


def validate_pad_metadata(path: Path) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    rows = read_csv(path)
    require_columns(rows, {"patient_id", "lesion_id", "diagnostic", "img_id", "biopsed"}, "PAD metadata")
    if len(rows) != EXPECTED_ROWS:
        raise BuildError(f"Expected {EXPECTED_ROWS} PAD rows, found {len(rows)}")
    ids = [row["img_id"].strip() for row in rows]
    if not all(ids) or len(set(ids)) != EXPECTED_ROWS:
        raise BuildError("PAD img_id values are blank or duplicated")
    patients = {row["patient_id"].strip() for row in rows if row["patient_id"].strip()}
    lesions = {row["lesion_id"].strip() for row in rows if row["lesion_id"].strip()}
    if len(patients) != EXPECTED_PATIENTS or len(lesions) != EXPECTED_LESIONS:
        raise BuildError(f"PAD patient/lesion counts differ: {len(patients)}/{len(lesions)}")
    counts = Counter(row["diagnostic"].strip().upper() for row in rows)
    if dict(counts) != EXPECTED_DIAGNOSTICS:
        raise BuildError(f"PAD diagnostic counts differ: {dict(counts)}")
    if any(row["diagnostic"].strip().upper() not in PAD_TO_CANONICAL for row in rows):
        raise BuildError("PAD metadata contains an unmapped official diagnostic")
    return rows, {row["img_id"].strip(): row for row in rows}


def validate_source_files(source_dir: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for name, (expected_bytes, expected_hash) in EXPECTED_SOURCE_FILES.items():
        path = source_dir / name
        if not path.is_file():
            raise BuildError(f"Missing official PAD source: {path}")
        actual_hash = sha256_file(path)
        if path.stat().st_size != expected_bytes or actual_hash != expected_hash:
            raise BuildError(
                f"Official PAD source mismatch: {name} bytes={path.stat().st_size} sha256={actual_hash}"
            )
        results.append({"file": name, "bytes": expected_bytes, "sha256": actual_hash})
    return results


def index_pad_archives(source_dir: Path) -> dict[str, tuple[Path, str]]:
    index: dict[str, tuple[Path, str]] = {}
    for name in ("imgs_part_1.zip", "imgs_part_2.zip", "imgs_part_3.zip"):
        path = source_dir / name
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            if bad is not None:
                raise BuildError(f"ZIP CRC failure in {name}: {bad}")
            for member in archive.namelist():
                member_path = Path(member)
                if member.endswith("/") or member_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                image_id = member_path.name
                if image_id in index:
                    raise BuildError(f"Duplicate PAD archive image name: {image_id}")
                index[image_id] = (path, member)
    if len(index) != EXPECTED_ROWS:
        raise BuildError(f"Expected {EXPECTED_ROWS} PAD archive images, found {len(index)}")
    return index


def index_isic2019(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise BuildError(f"Missing ISIC 2019 ZIP: {path}")
    index: dict[str, str] = {}
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            p = Path(member)
            if member.endswith("/") or p.suffix.lower() not in {".jpg", ".jpeg"}:
                continue
            item_id = p.stem
            if item_id in index:
                raise BuildError(f"Duplicate ISIC 2019 image ID in ZIP: {item_id}")
            index[item_id] = member
    if len(index) != 25_331:
        raise BuildError(f"Expected 25331 ISIC 2019 images, found {len(index)}")
    return index


def index_ima_images(root: Path, metadata_path: Path) -> dict[str, Path]:
    rows = read_csv(metadata_path)
    require_columns(rows, {"isic_id"}, "IMA++ raw metadata")
    ids = {row["isic_id"].strip() for row in rows if row["isic_id"].strip()}
    if len(rows) != 14_967 or len(ids) != 14_967:
        raise BuildError(f"Expected 14967 unique IMA++ raw IDs, found rows={len(rows)} ids={len(ids)}")
    index: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg"}:
            continue
        if path.stem in index:
            raise BuildError(f"Duplicate IMA++ raw image ID: {path.stem}")
        index[path.stem] = path
    if set(index) != ids:
        raise BuildError(f"IMA++ raw image index differs: missing={len(ids-set(index))} extra={len(set(index)-ids)}")
    return index


def stored_manifest_audits(path: Path, source: str, expected_rows: int) -> list[Audit]:
    rows = read_csv(path)
    require_columns(
        rows,
        {"isic_id", "image_sha256", "phash64", "dhash64", "width", "height"},
        source,
    )
    if len(rows) != expected_rows:
        raise BuildError(f"{source} expected {expected_rows} rows, found {len(rows)}")
    audits: list[Audit] = []
    for row in rows:
        item_id = row["isic_id"].strip()
        audits.append(
            Audit(
                item_id=item_id,
                source=source,
                source_path=str(path),
                sha256=row["image_sha256"].strip().lower(),
                phash64=row["phash64"].strip().lower(),
                dhash64=row["dhash64"].strip().lower(),
                width=int(row["width"]),
                height=int(row["height"]),
            )
        )
    if any(not audit.sha256 or not audit.phash64 or not audit.dhash64 for audit in audits):
        raise BuildError(f"{source} contains blank stored hashes")
    return audits


def reference_indices(audits: list[Audit]) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, set[str]]]:
    by_id: dict[str, set[str]] = defaultdict(set)
    by_sha: dict[str, set[str]] = defaultdict(set)
    by_sig: dict[str, set[str]] = defaultdict(set)
    for audit in audits:
        label = f"{audit.source}::{audit.item_id}"
        by_id[audit.item_id.lower()].add(label)
        by_sha[audit.sha256].add(label)
        by_sig[audit.strict_signature].add(label)
    return by_id, by_sha, by_sig


def add_exclusion(exclusions: dict[str, dict[str, Any]], image_id: str, reason: str, **details: Any) -> None:
    if image_id in exclusions:
        return
    exclusions[image_id] = {"img_id": image_id, "reason": reason, **details}


def ensure_output_image(data: bytes, destination: Path, expected_sha: str, resume: bool) -> None:
    if destination.exists():
        if resume and sha256_file(destination) == expected_sha:
            return
        raise BuildError(f"Existing output image is not an accepted resume artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(data)
    if sha256_file(temporary) != expected_sha:
        temporary.unlink(missing_ok=True)
        raise BuildError(f"Output hash mismatch before publish: {destination}")
    temporary.replace(destination)


def build(args: argparse.Namespace) -> None:
    source_dir = args.source_dir.resolve()
    output = args.output.resolve()
    taxonomy = args.taxonomy.resolve()
    taxonomy_sha = validate_taxonomy(taxonomy)
    source_hashes = validate_source_files(source_dir)
    metadata_rows, metadata_by_id = validate_pad_metadata(source_dir / "metadata.csv")
    pad_index = index_pad_archives(source_dir)
    if set(pad_index) != set(metadata_by_id):
        raise BuildError(
            f"PAD ZIP/metadata image IDs differ: missing={len(set(metadata_by_id)-set(pad_index))} "
            f"extra={len(set(pad_index)-set(metadata_by_id))}"
        )

    if output.exists():
        raise BuildError(f"Refusing to overwrite published output: {output}")
    staging = output.with_name(output.name + ".building")
    if staging.exists() and not args.resume:
        raise BuildError(f"Staging exists; inspect it and pass --resume: {staging}")
    staging.mkdir(parents=True, exist_ok=True)

    pad_audits_list = parallel_map(
        [(zip_path, member, image_id, "PAD_UFES20_V1") for image_id, (zip_path, member) in sorted(pad_index.items())],
        audit_zip_member,
        args.workers,
        "PAD image audit",
    )
    pad_audits = {audit.item_id: audit for audit in pad_audits_list}

    isic_index = index_isic2019(args.isic2019_zip.resolve())
    isic_audits = parallel_map(
        [
            (args.isic2019_zip.resolve(), member, item_id, "ISIC2019_OFFICIAL_TRAIN")
            for item_id, member in sorted(isic_index.items())
        ],
        audit_zip_member,
        args.workers,
        "ISIC 2019 reference audit",
    )
    ham_rows = read_csv(args.ham_groupings.resolve())
    require_columns(ham_rows, {"image"}, "HAM10000 groupings")
    ham_ids = {row["image"].strip() for row in ham_rows if row["image"].strip()}
    if len(ham_rows) != 10_015 or len(ham_ids) != 10_015 or not ham_ids <= set(isic_index):
        raise BuildError("HAM10000 reference IDs differ from expected ISIC 2019 subset")
    ham_aliases = [
        Audit(
            audit.item_id,
            "HAM10000_IN_ISIC2019",
            audit.source_path,
            audit.sha256,
            audit.phash64,
            audit.dhash64,
            audit.width,
            audit.height,
        )
        for audit in isic_audits
        if audit.item_id in ham_ids
    ]

    ima_index = index_ima_images(args.ima_source_images.resolve(), args.ima_source_metadata.resolve())
    ima_audits = parallel_map(
        [(path, item_id, "IMA_PLUS_PLUS_RAW") for item_id, path in sorted(ima_index.items())],
        audit_file,
        args.workers,
        "IMA++ raw reference audit",
    )
    binary_audits = stored_manifest_audits(
        args.ima_binary_manifest.resolve(), "IMA_PLUS_PLUS_BINARY_V1", 11_409
    )
    external_seg_audits = stored_manifest_audits(
        args.isic2018_external_manifest.resolve(), "ISIC2018_TASK1_EXTERNAL_CANDIDATE", 89
    )
    reference_audits = [*isic_audits, *ham_aliases, *ima_audits, *binary_audits, *external_seg_audits]
    by_ref_id, by_ref_sha, by_ref_sig = reference_indices(reference_audits)

    exclusions: dict[str, dict[str, Any]] = {}
    duplicate_rows: list[dict[str, Any]] = []

    metadata_groups: dict[str, list[str]] = defaultdict(list)
    for image_id, row in metadata_by_id.items():
        patient_id = row["patient_id"].strip()
        lesion_id = row["lesion_id"].strip()
        if not patient_id or not lesion_id:
            add_exclusion(exclusions, image_id, "blank_patient_or_lesion_id")
            continue
        metadata_groups[f"{patient_id}::{lesion_id}"].append(image_id)
    for key, members in metadata_groups.items():
        classes = {PAD_TO_CANONICAL[metadata_by_id[item]["diagnostic"].strip().upper()] for item in members}
        if len(classes) > 1:
            duplicate_rows.append(
                {
                    "kind": "patient_lesion_metadata_group",
                    "signature": key,
                    "count": len(members),
                    "members": ";".join(sorted(members)),
                    "classes": ";".join(sorted(classes)),
                    "action": "exclude_conflict_group",
                }
            )
            for image_id in members:
                add_exclusion(exclusions, image_id, "patient_lesion_conflicting_diagnosis", group_key=key)

    for image_id, audit in pad_audits.items():
        if image_id in exclusions:
            continue
        id_refs = by_ref_id.get(Path(image_id).stem.lower(), set())
        sha_refs = by_ref_sha.get(audit.sha256, set())
        sig_refs = by_ref_sig.get(audit.strict_signature, set())
        if id_refs or sha_refs or sig_refs:
            add_exclusion(
                exclusions,
                image_id,
                "cross_reference_duplicate",
                id_references=";".join(sorted(id_refs)),
                exact_references=";".join(sorted(sha_refs)),
                strict_perceptual_references=";".join(sorted(sig_refs)),
            )

    eligible = sorted(set(pad_audits) - set(exclusions))
    uf = UnionFind(eligible)
    by_sha: dict[str, list[str]] = defaultdict(list)
    by_sig: dict[str, list[str]] = defaultdict(list)
    for image_id in eligible:
        by_sha[pad_audits[image_id].sha256].append(image_id)
        by_sig[pad_audits[image_id].strict_signature].append(image_id)
    for kind, groups in (("sha256", by_sha), ("strict_phash_dhash_dimensions", by_sig)):
        for signature, members in groups.items():
            if len(members) < 2:
                continue
            for member in members[1:]:
                uf.union(members[0], member)
            classes = {
                PAD_TO_CANONICAL[metadata_by_id[member]["diagnostic"].strip().upper()] for member in members
            }
            duplicate_rows.append(
                {
                    "kind": kind,
                    "signature": signature,
                    "count": len(members),
                    "members": ";".join(sorted(members)),
                    "classes": ";".join(sorted(classes)),
                    "action": "pending_component_resolution",
                }
            )

    components: dict[str, list[str]] = defaultdict(list)
    for image_id in eligible:
        components[uf.find(image_id)].append(image_id)
    retained: list[str] = []
    for members in components.values():
        classes = {PAD_TO_CANONICAL[metadata_by_id[item]["diagnostic"].strip().upper()] for item in members}
        if len(classes) > 1:
            for image_id in members:
                add_exclusion(
                    exclusions,
                    image_id,
                    "within_candidate_conflicting_diagnosis_group",
                    component_members=";".join(sorted(members)),
                )
            continue
        canonical = min(members)
        retained.append(canonical)
        for image_id in sorted(set(members) - {canonical}):
            add_exclusion(
                exclusions,
                image_id,
                "within_candidate_strict_duplicate",
                canonical_id=canonical,
            )
    retained.sort()

    for duplicate_row in duplicate_rows:
        if duplicate_row["action"] != "pending_component_resolution":
            continue
        members = duplicate_row["members"].split(";")
        reasons = {exclusions[item]["reason"] for item in members if item in exclusions}
        if "within_candidate_conflicting_diagnosis_group" in reasons:
            duplicate_row["action"] = "exclude_conflict_component"
        elif "within_candidate_strict_duplicate" in reasons:
            duplicate_row["action"] = "retain_canonical_exclude_duplicates"
        else:
            raise BuildError(
                f"Duplicate group has no final resolution: {duplicate_row['kind']} {duplicate_row['members']}"
            )

    manifest: list[dict[str, Any]] = []
    image_hashes: list[dict[str, Any]] = []
    for index, image_id in enumerate(retained, start=1):
        row = metadata_by_id[image_id]
        audit = pad_audits[image_id]
        zip_path, member = pad_index[image_id]
        with zipfile.ZipFile(zip_path) as archive:
            data = archive.read(member)
        image_rel = Path("images") / image_id
        ensure_output_image(data, staging / image_rel, audit.sha256, args.resume)
        diagnostic = row["diagnostic"].strip().upper()
        class_name = PAD_TO_CANONICAL[diagnostic]
        patient_id = row["patient_id"].strip()
        lesion_id = row["lesion_id"].strip()
        manifest.append(
            {
                "img_id": image_id,
                "role": "external_candidate",
                "dataset_role": "pad_ufes20_external_classification",
                "image_relpath": image_rel.as_posix(),
                "has_segmentation": 0,
                "has_classification": 1,
                "class_id": CLASS_TO_ID[class_name],
                "class_name": class_name,
                "official_diagnostic": diagnostic,
                "patient_id": patient_id,
                "lesion_id": lesion_id,
                "group_key": f"{patient_id}::{lesion_id}",
                "biopsed": row["biopsed"].strip(),
                "source_archive": zip_path.name,
                "source_zip_member": member,
                "image_sha256": audit.sha256,
                "phash64": audit.phash64,
                "dhash64": audit.dhash64,
                "width": audit.width,
                "height": audit.height,
                "copyright_license": "CC BY 4.0",
            }
        )
        image_hashes.append({"relative_path": image_rel.as_posix(), "sha256": audit.sha256})
        if index % 250 == 0 or index == len(retained):
            print(f"Output materialization: {index}/{len(retained)}", flush=True)

    retained_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest:
        retained_groups[row["group_key"]].append(row)
    group_manifest: list[dict[str, Any]] = []
    group_id_by_key: dict[str, str] = {}
    for index, key in enumerate(sorted(retained_groups), start=1):
        rows = retained_groups[key]
        classes = {row["class_name"] for row in rows}
        if len(classes) != 1:
            raise BuildError(f"Published patient-lesion group spans classes: {key} -> {classes}")
        group_id = f"PADG{index:05d}"
        group_id_by_key[key] = group_id
        group_manifest.append(
            {
                "group_id": group_id,
                "group_key": key,
                "patient_id": rows[0]["patient_id"],
                "lesion_id": rows[0]["lesion_id"],
                "class_name": rows[0]["class_name"],
                "official_diagnostic": rows[0]["official_diagnostic"],
                "image_count": len(rows),
                "members": ";".join(sorted(row["img_id"] for row in rows)),
            }
        )
    for row in manifest:
        row["group_id"] = group_id_by_key[row["group_key"]]

    excluded_rows = sorted(exclusions.values(), key=lambda row: (row["reason"], row["img_id"]))
    if len(manifest) + len(excluded_rows) != EXPECTED_ROWS:
        raise BuildError("Retained/excluded partition does not cover every official PAD row exactly once")
    if len({row["img_id"] for row in manifest}) != len(manifest):
        raise BuildError("Published manifest contains duplicate img_id")

    write_csv(staging / "manifest.csv", manifest)
    write_csv(
        staging / "excluded.csv",
        excluded_rows,
        fieldnames=[
            "img_id",
            "reason",
            "group_key",
            "id_references",
            "exact_references",
            "strict_perceptual_references",
            "component_members",
            "canonical_id",
        ],
    )
    write_csv(
        staging / "duplicate_groups.csv",
        sorted(duplicate_rows, key=lambda row: (row["kind"], row["signature"])),
        fieldnames=["kind", "signature", "count", "members", "classes", "action"],
    )
    write_csv(staging / "group_manifest.csv", group_manifest)
    write_csv(staging / "hashes_sha256.csv", image_hashes)

    provenance = staging / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_dir / "metadata.csv", provenance / "metadata.csv")
    shutil.copy2(taxonomy, provenance / taxonomy.name)
    for optional in ("SOURCE_MANIFEST.json", "LICENSE_AND_PROVENANCE.md", "_DOWNLOAD_SUCCESS"):
        source = source_dir / optional
        if source.is_file():
            shutil.copy2(source, provenance / optional)
    source_hashes.extend(
        [
            {"file": taxonomy.name, "bytes": taxonomy.stat().st_size, "sha256": taxonomy_sha},
            {
                "file": args.isic2019_zip.resolve().name,
                "bytes": args.isic2019_zip.resolve().stat().st_size,
                "sha256": sha256_file(args.isic2019_zip.resolve()),
            },
            {
                "file": args.ima_source_metadata.resolve().name,
                "bytes": args.ima_source_metadata.resolve().stat().st_size,
                "sha256": sha256_file(args.ima_source_metadata.resolve()),
            },
            {
                "file": args.ham_groupings.resolve().name,
                "bytes": args.ham_groupings.resolve().stat().st_size,
                "sha256": sha256_file(args.ham_groupings.resolve()),
            },
            {
                "file": args.ima_binary_manifest.resolve().name,
                "bytes": args.ima_binary_manifest.resolve().stat().st_size,
                "sha256": sha256_file(args.ima_binary_manifest.resolve()),
            },
            {
                "file": args.isic2018_external_manifest.resolve().name,
                "bytes": args.isic2018_external_manifest.resolve().stat().st_size,
                "sha256": sha256_file(args.isic2018_external_manifest.resolve()),
            },
        ]
    )
    write_csv(provenance / "source_hashes_sha256.csv", source_hashes)

    class_counts = Counter(row["class_name"] for row in manifest)
    official_counts = Counter(row["official_diagnostic"] for row in manifest)
    summary = {
        "build_version": BUILD_VERSION,
        "created_local_time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "official_rows": EXPECTED_ROWS,
        "official_patients": EXPECTED_PATIENTS,
        "official_lesions": EXPECTED_LESIONS,
        "official_diagnostic_counts": EXPECTED_DIAGNOSTICS,
        "reference_pool_records": {
            "isic2019_official_training": len(isic_audits),
            "ham10000_subset_aliases": len(ham_aliases),
            "imaplusplus_raw": len(ima_audits),
            "imaplusplus_binary_v1": len(binary_audits),
            "isic2018_task1_external_candidate": len(external_seg_audits),
        },
        "retained": len(manifest),
        "excluded": len(excluded_rows),
        "exclusion_counts": dict(sorted(Counter(row["reason"] for row in excluded_rows).items())),
        "retained_patient_count": len({row["patient_id"] for row in manifest}),
        "retained_lesion_count": len({row["lesion_id"] for row in manifest}),
        "retained_group_count": len(group_manifest),
        "retained_official_diagnostic_counts": dict(sorted(official_counts.items())),
        "retained_class_counts": dict(sorted(class_counts.items())),
        "classification_taxonomy_gate": "PASS",
        "external_classification_source_gate": "PASS",
        "external_classification_download_gate": "PASS",
        "external_classification_strict_dedup_gate": "PASS",
        "external_classification_data_gate": "PASS",
        "external_classification_evaluation_gate": "BLOCKED_PENDING_FINAL_MODEL_AND_USER_AUTHORIZATION",
        "project_run10_gate": "BLOCKED_PENDING_MODEL_AND_CLASSIFICATION_CONTRACT",
        "entered_training_manifest": False,
        "test_metrics_used": False,
        "external_metrics_used": False,
    }
    (staging / "audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (staging / "DATASET_REPORT.md").write_text(
        "\n".join(
            [
                "# PAD-UFES-20 external classification candidate v1",
                "",
                "`EXTERNAL_CLASSIFICATION_DATA_GATE=PASS`  ",
                "`EXTERNAL_CLASSIFICATION_EVALUATION_GATE=BLOCKED_PENDING_FINAL_MODEL_AND_USER_AUTHORIZATION`  ",
                "`PROJECT_RUN10_GATE=BLOCKED_PENDING_MODEL_AND_CLASSIFICATION_CONTRACT`",
                "",
                f"- Official source rows: {EXPECTED_ROWS}; retained: {len(manifest)}; excluded: {len(excluded_rows)}.",
                f"- Retained patients/lesions/groups: {summary['retained_patient_count']}/"
                f"{summary['retained_lesion_count']}/{summary['retained_group_count']}.",
                f"- Retained classes: {json.dumps(dict(sorted(class_counts.items())), ensure_ascii=False)}.",
                "- Only the official diagnostic field was mapped; no label was inferred.",
                "- The data are external_candidate only: no training split, masks, or combined training YAML were created.",
                "- No internal blind test or external model evaluation was run.",
                "- PAD-UFES-20 lacks df and vasc; future evaluation must report supported-6 and full-8 metrics.",
                "- Clinical smartphone acquisition and partial biopsy confirmation limit interpretation.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (staging / "_SUCCESS").write_text(
        json.dumps(
            {
                "build_version": BUILD_VERSION,
                "retained": len(manifest),
                "excluded": len(excluded_rows),
                "completed_local_time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    staging.replace(output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Published: {output}")


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
