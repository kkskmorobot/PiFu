#!/usr/bin/env python3
"""Build a provenance-preserving binary semantic-segmentation dataset from IMA++.

The source tree is treated as immutable.  The build is written to a staging
directory and published only after all integrity checks pass.  Multi-annotator
images use the IMA++ STAPLE consensus mask; single-annotator images use expert
(S1) masks by default.  Official multi-annotator train/val/test assignments are
used as split anchors and expanded by patient, lesion, exact-file, and strict
perceptual-duplicate groups to prevent leakage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image


BUILD_VERSION = "imaplusplus_binary_v1"
OFFICIAL_SPLITS = ("train", "val", "test")
REQUIRED_IMAGE_COLUMNS = {
    "isic_id",
    "attribution",
    "copyright_license",
    "benign_malignant",
    "diagnosis_1",
    "diagnosis_2",
    "diagnosis_3",
    "diagnosis_4",
    "diagnosis_5",
    "lesion_id",
    "patient_id",
}
REQUIRED_SEG_COLUMNS = {
    "ISIC_id",
    "img_filename",
    "seg_filename",
    "annotator",
    "tool",
    "skill_level",
    "mskObjectID",
    "mask_md5",
}


class BuildError(RuntimeError):
    """Raised when the source or derived dataset violates the data contract."""


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
class Candidate:
    isic_id: str
    image_path: Path
    mask_path: Path
    mask_row: dict[str, str]
    image_row: dict[str, str]
    mask_policy: str
    include: bool
    exclusion_reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Extracted IMA++ directory (Data/1).")
    parser.add_argument("--output", type=Path, required=True, help="Final processed dataset directory.")
    parser.add_argument(
        "--include-single-s2",
        action="store_true",
        help="Include non-expert single-annotator S2 masks. Disabled for the v1 clean baseline.",
    )
    parser.add_argument("--workers", type=int, default=max(2, min(12, os.cpu_count() or 4)))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse verified files in an existing .building directory after interruption.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise BuildError(f"Missing required CSV: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def require_columns(rows: list[dict[str, str]], required: set[str], name: str) -> None:
    if not rows:
        raise BuildError(f"{name} is empty")
    missing = required - set(rows[0])
    if missing:
        raise BuildError(f"{name} is missing columns: {sorted(missing)}")


def indexed_files(directory: Path, suffix: str) -> dict[str, Path]:
    if not directory.is_dir():
        raise BuildError(f"Missing directory: {directory}")
    files = [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == suffix]
    result: dict[str, Path] = {}
    for path in files:
        key = path.name.casefold()
        if key in result:
            raise BuildError(f"Case-insensitive filename collision: {result[key]} and {path}")
        result[key] = path
    return result


def source_snapshot(source: Path) -> dict[str, tuple[int, int]]:
    result: dict[str, tuple[int, int]] = {}
    for path in source.rglob("*"):
        if path.is_file():
            stat = path.stat()
            result[path.relative_to(source).as_posix()] = (stat.st_size, stat.st_mtime_ns)
    return result


def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def bits_to_hex(bits: np.ndarray) -> str:
    packed = np.packbits(bits.astype(np.uint8).reshape(-1))
    return packed.tobytes().hex()


def perceptual_hashes(rgb: np.ndarray) -> tuple[str, str]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    phash_img = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(phash_img)
    block = dct[:8, :8]
    median = float(np.median(block.reshape(-1)[1:]))
    phash = bits_to_hex(block > median)

    dhash_img = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    dhash = bits_to_hex(dhash_img[:, 1:] > dhash_img[:, :-1])
    return phash, dhash


def audit_candidate(candidate: Candidate) -> dict[str, Any]:
    image_sha256 = sha256_file(candidate.image_path)
    with Image.open(candidate.image_path) as image:
        image.load()
        width, height = image.size
        rgb = np.asarray(image.convert("RGB"))
    phash64, dhash64 = perceptual_hashes(rgb)

    mask_md5 = md5_file(candidate.mask_path)
    expected_md5 = candidate.mask_row["mask_md5"].strip().lower()
    if expected_md5 and mask_md5 != expected_md5:
        raise BuildError(
            f"Mask MD5 mismatch for {candidate.isic_id}: expected {expected_md5}, got {mask_md5}"
        )
    with Image.open(candidate.mask_path) as mask_image:
        mask_image.load()
        mask_width, mask_height = mask_image.size
        mask = np.asarray(mask_image.convert("L"))
    if (mask_width, mask_height) != (width, height):
        raise BuildError(
            f"Image/mask size mismatch for {candidate.isic_id}: "
            f"image={width}x{height}, mask={mask_width}x{mask_height}"
        )

    binary = (mask > 0).astype(np.uint8)
    foreground_pixels = int(binary.sum())
    total_pixels = int(binary.size)
    invalid_reason = ""
    if foreground_pixels == 0:
        invalid_reason = "empty_mask"
    elif foreground_pixels == total_pixels:
        invalid_reason = "full_frame_mask"
    components, _ = cv2.connectedComponents(binary, connectivity=8)
    border_touch = bool(
        binary[0, :].any() or binary[-1, :].any() or binary[:, 0].any() or binary[:, -1].any()
    )
    unique_values = sorted(int(value) for value in np.unique(mask))

    return {
        "isic_id": candidate.isic_id,
        "candidate": candidate,
        "image_width": width,
        "image_height": height,
        "image_sha256": image_sha256,
        "phash64": phash64,
        "dhash64": dhash64,
        "source_mask_md5": mask_md5,
        "source_mask_values": ";".join(map(str, unique_values)),
        "foreground_pixels": foreground_pixels,
        "foreground_fraction": foreground_pixels / total_pixels,
        "connected_components": int(components - 1),
        "border_touch": border_touch,
        "mask_valid": not invalid_reason,
        "invalid_reason": invalid_reason,
    }


def select_candidates(source: Path, include_single_s2: bool) -> tuple[list[Candidate], dict[str, set[str]]]:
    image_rows = read_csv(source / "img_metadata.csv")
    seg_rows = read_csv(source / "seg_metadata.csv")
    require_columns(image_rows, REQUIRED_IMAGE_COLUMNS, "img_metadata.csv")
    require_columns(seg_rows, REQUIRED_SEG_COLUMNS, "seg_metadata.csv")

    if len(image_rows) != 14967:
        raise BuildError(f"Expected 14967 image metadata rows, found {len(image_rows)}")
    if len(seg_rows) != 22472:
        raise BuildError(f"Expected 22472 segmentation metadata rows, found {len(seg_rows)}")

    image_meta: dict[str, dict[str, str]] = {}
    for row in image_rows:
        isic_id = row["isic_id"].strip().upper()
        if isic_id in image_meta:
            raise BuildError(f"Duplicate image metadata ID: {isic_id}")
        image_meta[isic_id] = row

    masks_by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in seg_rows:
        masks_by_id[row["ISIC_id"].strip().upper()].append(row)
    if set(masks_by_id) != set(image_meta):
        raise BuildError(
            "Image/segmentation ID mismatch: "
            f"missing masks={len(set(image_meta) - set(masks_by_id))}, "
            f"orphan masks={len(set(masks_by_id) - set(image_meta))}"
        )

    image_files = indexed_files(source / "ISIC-images", ".jpg")
    mask_files = indexed_files(source / "segs", ".png")
    if len(image_files) != 14967:
        raise BuildError(f"Expected 14967 IMA++ JPG files, found {len(image_files)}")
    if len(mask_files) != 22472:
        raise BuildError(f"Expected 22472 IMA++ PNG masks, found {len(mask_files)}")

    split_ids: dict[str, set[str]] = {}
    for split in OFFICIAL_SPLITS:
        rows = read_csv(source / f"{split}.csv")
        if not rows or "image" not in rows[0]:
            raise BuildError(f"Invalid official split CSV: {split}.csv")
        split_ids[split] = {Path(row["image"]).stem.upper() for row in rows}
    for index, left in enumerate(OFFICIAL_SPLITS):
        for right in OFFICIAL_SPLITS[index + 1 :]:
            overlap = split_ids[left] & split_ids[right]
            if overlap:
                raise BuildError(f"Official {left}/{right} overlap: {len(overlap)}")
    if len(set().union(*split_ids.values())) != 2394:
        raise BuildError("Official IMA++ split files must cover exactly 2394 multi-annotator images")

    candidates: list[Candidate] = []
    for isic_id in sorted(image_meta):
        rows = masks_by_id[isic_id]
        if len(rows) == 1:
            selected = rows[0]
            skill = selected["skill_level"].strip().upper()
            if skill not in {"S1", "S2"}:
                raise BuildError(f"Unexpected single-mask skill for {isic_id}: {skill}")
            include = skill == "S1" or include_single_s2
            policy = f"single_{skill.lower()}"
            exclusion_reason = "" if include else "single_nonexpert_s2_excluded_from_clean_v1"
        else:
            staple = [row for row in rows if row["skill_level"].strip().upper() == "ST"]
            if len(staple) != 1:
                raise BuildError(f"Expected one STAPLE mask for multi-annotator image {isic_id}, found {len(staple)}")
            selected = staple[0]
            include = True
            policy = "staple_consensus"
            exclusion_reason = ""

        image_key = selected["img_filename"].casefold()
        mask_key = selected["seg_filename"].casefold()
        if image_key not in image_files:
            image_key = f"{isic_id}.jpg".casefold()
        if image_key not in image_files:
            raise BuildError(f"Missing source image for {isic_id}: {selected['img_filename']}")
        if mask_key not in mask_files:
            raise BuildError(f"Missing selected source mask for {isic_id}: {selected['seg_filename']}")
        candidates.append(
            Candidate(
                isic_id=isic_id,
                image_path=image_files[image_key],
                mask_path=mask_files[mask_key],
                mask_row=selected,
                image_row=image_meta[isic_id],
                mask_policy=policy,
                include=include,
                exclusion_reason=exclusion_reason,
            )
        )
    return candidates, split_ids


def audit_all(candidates: list[Candidate], workers: int) -> list[dict[str, Any]]:
    started = time.time()
    results: list[dict[str, Any]] = []
    print(f"Auditing {len(candidates)} canonical image/mask pairs with {workers} workers...", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(audit_candidate, item): item.isic_id for item in candidates}
        for count, future in enumerate(as_completed(futures), start=1):
            try:
                results.append(future.result())
            except Exception as exc:
                raise BuildError(f"Audit failed for {futures[future]}: {exc}") from exc
            if count % 500 == 0 or count == len(candidates):
                elapsed = time.time() - started
                print(f"  audited {count}/{len(candidates)} pairs in {elapsed:.1f}s", flush=True)
    return sorted(results, key=lambda row: row["isic_id"])


def union_equal_groups(uf: UnionFind, rows: list[dict[str, Any]], key_fn) -> None:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        key = key_fn(row)
        if key:
            grouped[str(key)].append(row["isic_id"])
    for members in grouped.values():
        first = members[0]
        for member in members[1:]:
            uf.union(first, member)


def assign_splits(
    audited: list[dict[str, Any]], official_ids: dict[str, set[str]]
) -> tuple[dict[str, str], dict[str, str], list[dict[str, Any]]]:
    active = [row for row in audited if row["candidate"].include and row["mask_valid"]]
    active_ids = {row["isic_id"] for row in active}

    uf = UnionFind(active_ids)
    union_equal_groups(uf, active, lambda row: row["candidate"].image_row["patient_id"].strip())
    union_equal_groups(uf, active, lambda row: row["candidate"].image_row["lesion_id"].strip())
    union_equal_groups(uf, active, lambda row: row["image_sha256"])
    # A strict dual-hash match plus identical dimensions is treated as a perceptual duplicate.
    union_equal_groups(
        uf,
        active,
        lambda row: (
            f"{row['phash64']}:{row['dhash64']}:{row['image_width']}x{row['image_height']}"
        ),
    )

    components: dict[str, list[str]] = defaultdict(list)
    for isic_id in active_ids:
        components[uf.find(isic_id)].append(isic_id)

    locked: dict[str, str] = {}
    for split, ids in official_ids.items():
        for isic_id in ids:
            if isic_id in active_ids:
                locked[isic_id] = split

    split_by_id: dict[str, str] = {}
    group_by_id: dict[str, str] = {}
    component_rows: list[dict[str, Any]] = []
    split_priority = {"train": 0, "val": 1, "test": 2}
    for ordinal, members in enumerate(sorted(components.values(), key=lambda values: min(values)), start=1):
        anchors = {locked[member] for member in members if member in locked}
        # When the official image-level split conflicts with stricter grouping,
        # move the whole component away from training: test > val > train.
        split = max(anchors, key=split_priority.get) if anchors else "train"
        group_id = f"G{ordinal:05d}"
        for member in members:
            split_by_id[member] = split
            group_by_id[member] = group_id
        component_rows.append(
            {
                "group_id": group_id,
                "split": split,
                "size": len(members),
                "official_anchors": ";".join(sorted(anchors, key=split_priority.get)),
                "anchor_conflict": int(len(anchors) > 1),
                "resolution": "test>val>train" if len(anchors) > 1 else "",
                "members": ";".join(sorted(members)),
            }
        )

    fields = ("patient_id", "lesion_id")
    by_id = {row["isic_id"]: row for row in active}
    for field in fields:
        values_by_split: dict[str, set[str]] = {}
        for split in OFFICIAL_SPLITS:
            values_by_split[split] = {
                by_id[isic_id]["candidate"].image_row[field].strip()
                for isic_id, assigned in split_by_id.items()
                if assigned == split and by_id[isic_id]["candidate"].image_row[field].strip()
            }
        for index, left in enumerate(OFFICIAL_SPLITS):
            for right in OFFICIAL_SPLITS[index + 1 :]:
                overlap = values_by_split[left] & values_by_split[right]
                if overlap:
                    raise BuildError(f"{field} leakage between {left}/{right}: {len(overlap)}")

    return split_by_id, group_by_id, component_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not rows:
            raise BuildError(f"Cannot infer CSV columns for empty output: {path}")
        fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def verify_or_copy(source: Path, destination: Path, expected_sha256: str, resume: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not resume:
            raise BuildError(f"Unexpected existing staging file without --resume: {destination}")
        if sha256_file(destination) == expected_sha256:
            return
        destination.unlink()
    shutil.copy2(source, destination)
    actual = sha256_file(destination)
    if actual != expected_sha256:
        raise BuildError(f"Copied file hash mismatch: {destination}")


def write_binary_mask(source: Path, destination: Path, expected_size: tuple[int, int], resume: bool) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and resume:
        with Image.open(destination) as existing:
            array = np.asarray(existing.convert("L"))
        if existing.size == expected_size and set(np.unique(array)).issubset({0, 1}) and array.any():
            return sha256_file(destination)
        destination.unlink()
    elif destination.exists():
        raise BuildError(f"Unexpected existing staging mask without --resume: {destination}")

    with Image.open(source) as mask_image:
        binary = (np.asarray(mask_image.convert("L")) > 0).astype(np.uint8)
    Image.fromarray(binary, mode="L").save(destination, format="PNG", optimize=True)
    with Image.open(destination) as check_image:
        check = np.asarray(check_image.convert("L"))
        if check_image.size != expected_size or not set(np.unique(check)).issubset({0, 1}):
            raise BuildError(f"Invalid written binary mask: {destination}")
    return sha256_file(destination)


def csv_safe(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    return value


def build_output(
    source: Path,
    output: Path,
    audited: list[dict[str, Any]],
    split_by_id: dict[str, str],
    group_by_id: dict[str, str],
    component_rows: list[dict[str, Any]],
    include_single_s2: bool,
    resume: bool,
) -> None:
    if output.exists():
        raise BuildError(f"Refusing to overwrite existing output: {output}")
    staging = output.with_name(output.name + ".building")
    if staging.exists() and not resume:
        raise BuildError(f"Staging directory exists; rerun with --resume or inspect it manually: {staging}")
    staging.mkdir(parents=True, exist_ok=True)

    active = [row for row in audited if row["candidate"].include and row["mask_valid"]]
    excluded = [row for row in audited if not row["candidate"].include or not row["mask_valid"]]
    manifest: list[dict[str, Any]] = []
    hashes: list[dict[str, Any]] = []
    print(f"Publishing {len(active)} active pairs into {staging}...", flush=True)
    started = time.time()
    for count, row in enumerate(active, start=1):
        candidate: Candidate = row["candidate"]
        split = split_by_id[row["isic_id"]]
        image_rel = Path("images") / split / f"{row['isic_id']}.jpg"
        mask_rel = Path("masks") / split / f"{row['isic_id']}.png"
        verify_or_copy(candidate.image_path, staging / image_rel, row["image_sha256"], resume)
        mask_sha256 = write_binary_mask(
            candidate.mask_path,
            staging / mask_rel,
            (row["image_width"], row["image_height"]),
            resume,
        )
        image_meta = candidate.image_row
        manifest_row = {
            "isic_id": row["isic_id"],
            "split": split,
            "group_id": group_by_id[row["isic_id"]],
            "image_relpath": image_rel.as_posix(),
            "mask_relpath": mask_rel.as_posix(),
            "source_image_relpath": candidate.image_path.relative_to(source).as_posix(),
            "source_mask_relpath": candidate.mask_path.relative_to(source).as_posix(),
            "mask_policy": candidate.mask_policy,
            "annotator": candidate.mask_row["annotator"],
            "tool": candidate.mask_row["tool"],
            "skill_level": candidate.mask_row["skill_level"],
            "patient_id": image_meta["patient_id"],
            "lesion_id": image_meta["lesion_id"],
            "benign_malignant": image_meta["benign_malignant"],
            "diagnosis_1": image_meta["diagnosis_1"],
            "diagnosis_2": image_meta["diagnosis_2"],
            "diagnosis_3": image_meta["diagnosis_3"],
            "diagnosis_4": image_meta["diagnosis_4"],
            "diagnosis_5": image_meta["diagnosis_5"],
            "attribution": image_meta["attribution"],
            "copyright_license": image_meta["copyright_license"],
            "width": row["image_width"],
            "height": row["image_height"],
            "foreground_fraction": f"{row['foreground_fraction']:.10f}",
            "connected_components": row["connected_components"],
            "border_touch": int(row["border_touch"]),
            "source_mask_values": row["source_mask_values"],
            "image_sha256": row["image_sha256"],
            "source_mask_md5": row["source_mask_md5"],
            "output_mask_sha256": mask_sha256,
            "phash64": row["phash64"],
            "dhash64": row["dhash64"],
        }
        manifest.append(manifest_row)
        hashes.extend(
            [
                {"relative_path": image_rel.as_posix(), "sha256": row["image_sha256"]},
                {"relative_path": mask_rel.as_posix(), "sha256": mask_sha256},
            ]
        )
        if count % 500 == 0 or count == len(active):
            print(f"  published {count}/{len(active)} pairs in {time.time() - started:.1f}s", flush=True)

    excluded_rows = []
    for row in excluded:
        candidate = row["candidate"]
        excluded_rows.append(
            {
                "isic_id": row["isic_id"],
                "reason": ";".join(
                    reason
                    for reason in (candidate.exclusion_reason, row["invalid_reason"])
                    if reason
                ),
                "source_image_relpath": candidate.image_path.relative_to(source).as_posix(),
                "source_mask_relpath": candidate.mask_path.relative_to(source).as_posix(),
                "mask_policy": candidate.mask_policy,
                "skill_level": candidate.mask_row["skill_level"],
                "image_sha256": row["image_sha256"],
                "source_mask_md5": row["source_mask_md5"],
            }
        )

    quality_flag_rows = []
    for row in active:
        flags = []
        if row["foreground_fraction"] < 0.001:
            flags.append("foreground_below_0.1_percent")
        if row["foreground_fraction"] > 0.95:
            flags.append("foreground_above_95_percent")
        if row["connected_components"] > 10:
            flags.append("more_than_10_components")
        if flags:
            quality_flag_rows.append(
                {
                    "isic_id": row["isic_id"],
                    "split": split_by_id[row["isic_id"]],
                    "flags": ";".join(flags),
                    "foreground_fraction": f"{row['foreground_fraction']:.10f}",
                    "connected_components": row["connected_components"],
                    "border_touch": int(row["border_touch"]),
                    "source_mask_relpath": row["candidate"].mask_path.relative_to(source).as_posix(),
                }
            )

    write_csv(staging / "manifest.csv", manifest)
    write_csv(staging / "excluded.csv", excluded_rows)
    write_csv(
        staging / "quality_flags.csv",
        quality_flag_rows,
        fieldnames=[
            "isic_id",
            "split",
            "flags",
            "foreground_fraction",
            "connected_components",
            "border_touch",
            "source_mask_relpath",
        ],
    )
    write_csv(staging / "group_manifest.csv", component_rows)
    write_csv(staging / "hashes_sha256.csv", hashes)

    split_counts = Counter(row["split"] for row in manifest)
    mask_policy_counts = Counter(row["mask_policy"] for row in manifest)
    license_counts = Counter(row["copyright_license"] or "UNKNOWN" for row in manifest)
    component_counts = Counter(row["connected_components"] for row in manifest)
    border_count = sum(int(row["border_touch"]) for row in manifest)
    anchor_conflict_count = sum(int(row["anchor_conflict"]) for row in component_rows)

    duplicate_by_sha: dict[str, list[str]] = defaultdict(list)
    perceptual_by_signature: dict[str, list[str]] = defaultdict(list)
    for row in manifest:
        duplicate_by_sha[row["image_sha256"]].append(row["isic_id"])
        signature = f"{row['phash64']}:{row['dhash64']}:{row['width']}x{row['height']}"
        perceptual_by_signature[signature].append(row["isic_id"])
    duplicate_rows = []
    for kind, groups in (
        ("sha256", duplicate_by_sha),
        ("strict_phash_dhash_dimensions", perceptual_by_signature),
    ):
        for signature, members in groups.items():
            if len(members) > 1:
                duplicate_rows.append(
                    {
                        "kind": kind,
                        "signature": signature,
                        "count": len(members),
                        "members": ";".join(sorted(members)),
                    }
                )
    write_csv(
        staging / "duplicate_groups.csv",
        duplicate_rows,
        fieldnames=["kind", "signature", "count", "members"],
    )

    provenance = staging / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    for name in (
        "img_metadata.csv",
        "seg_metadata.csv",
        "seg_metadata_multiannotator_subset.csv",
        "train.csv",
        "val.csv",
        "test.csv",
    ):
        shutil.copy2(source / name, provenance / name)
    for path in (source / "ISIC-images").rglob("*.txt"):
        destination = provenance / "ISIC-images" / path.relative_to(source / "ISIC-images")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)

    data_yaml = """# IMA++ clean binary semantic segmentation dataset\npath: .\ntrain: images/train\nval: images/val\ntest: images/test\nmasks_dir: masks\nnc: 1\nnames:\n  0: lesion\n"""
    (staging / "data.yaml").write_text(data_yaml, encoding="utf-8")

    summary = {
        "build_version": BUILD_VERSION,
        "source": str(source),
        "output": str(output),
        "created_local_time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "task": "binary semantic segmentation",
        "mask_values": {"0": "background", "1": "lesion", "255": "reserved_ignore_not_present"},
        "selection_policy": {
            "multi_annotator": "STAPLE consensus",
            "single_annotator": "S1 expert only" if not include_single_s2 else "S1 and S2",
        },
        "split_policy": (
            "Official IMA++ multi-annotator train/val/test anchors, expanded as indivisible groups by "
            "patient_id, lesion_id, exact SHA-256, and strict pHash+dHash+dimensions matches; unanchored groups train."
        ),
        "active_pairs": len(manifest),
        "excluded_pairs": len(excluded_rows),
        "invalid_mask_pairs": sum(1 for row in audited if not row["mask_valid"]),
        "quality_flag_pairs": len(quality_flag_rows),
        "split_counts": dict(sorted(split_counts.items())),
        "mask_policy_counts": dict(sorted(mask_policy_counts.items())),
        "license_counts": dict(sorted(license_counts.items())),
        "connected_component_counts": {str(k): v for k, v in sorted(component_counts.items())},
        "border_touch_count": border_count,
        "official_anchor_conflict_group_count": anchor_conflict_count,
        "exact_duplicate_group_count": sum(1 for row in duplicate_rows if row["kind"] == "sha256"),
        "strict_perceptual_duplicate_group_count": sum(
            1 for row in duplicate_rows if row["kind"] == "strict_phash_dhash_dimensions"
        ),
        "binary_segmentation_data_gate": "PASS",
        "project_run10_gate": "BLOCKED_PENDING_MODEL_AND_CLASSIFICATION_CONTRACT",
        "test_metrics_used": False,
    }
    (staging / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = f"""# IMA++ 二值语义分割数据构建报告

`BUILD_VERSION={BUILD_VERSION}`  
`BINARY_SEGMENTATION_DATA_GATE=PASS`  
`PROJECT_RUN10_GATE=BLOCKED_PENDING_MODEL_AND_CLASSIFICATION_CONTRACT`

## 数据契约

- 任务：病灶/背景二值语义分割。
- PNG 掩码值：`0=background`，`1=lesion`；没有使用 `255=ignore`。
- 多人标注图像：选择 IMA++ 提供的 STAPLE 共识掩码。
- 单人标注图像：v1 默认只使用专家级 `S1`；`S2` 保留在源目录并写入 `excluded.csv`。
- 图像级诊断字段仅原样保存在 `manifest.csv`，没有强行映射为旧项目七类像素标签。

## 数量

- 有效图像/掩码对：{len(manifest)}
- 排除但保留记录：{len(excluded_rows)}
- 客观无效（空/全图）掩码：{sum(1 for row in audited if not row['mask_valid'])}
- 需后续抽样复核的极端面积/多连通域标记：{len(quality_flag_rows)}（见 `quality_flags.csv`）
- train：{split_counts.get('train', 0)}
- val：{split_counts.get('val', 0)}
- test：{split_counts.get('test', 0)}
- STAPLE 共识：{mask_policy_counts.get('staple_consensus', 0)}
- S1 单标注：{mask_policy_counts.get('single_s1', 0)}

## 划分与泄漏控制

IMA++ 官方多标注子集划分作为锚点。所有具有相同 `patient_id`、`lesion_id`、图像 SHA-256，
或严格相同 pHash+dHash+尺寸签名的样本被绑定到同一 split；未命中官方锚点的组只进入 train。
若官方图像级锚点在严格分组后发生冲突，按 `test > val > train` 将整组移向更严格的保留集合；
本次共发现 {anchor_conflict_count} 个此类冲突组，详见 `group_manifest.csv`。
构建结束后 patient/lesion 跨 split 交集均已验证为 0。

## 完整性

- 每个拟使用掩码均通过 IMA++ `mask_md5` 校验。
- 每张图像和掩码均可解码、尺寸一致、掩码非空且非全图。
- 输出掩码重新编码为无损 PNG 类别图，并记录 SHA-256。
- `manifest.csv` 保存来源、标注策略、患者/病灶 ID、原始诊断层级、许可、尺寸和质量字段。
- `hashes_sha256.csv` 可用于以后验证处理结果未变化。
- 没有读取或计算任何 test 模型指标。

## 训练说明

`data.yaml` 符合当前 Ultralytics `SemanticDataset` 的 PNG-mask 模式，可用于 `nc=1` 的 BCE+Dice 二值语义训练。
该数据集只完成分割数据治理；旧七类分类 taxonomy、部分标签多任务 loader、新模型选择指标和外部盲测仍需单独锁定，
因此此报告不会把整个项目的 Run10 门槛标为 PASS。
"""
    (staging / "DATASET_REPORT.md").write_text(report, encoding="utf-8")

    readme = """# processed_imaplusplus_binary_v1

Canonical IMA++ binary semantic-segmentation build. Use `data.yaml` with the project Ultralytics semantic trainer.
Masks are class-ID PNGs (`0=background`, `1=lesion`), so they look almost black in ordinary image viewers.
See `DATASET_REPORT.md`, `audit_summary.json`, and `manifest.csv` before training.
The original extracted files remain under `Data/1` and were not modified.
"""
    (staging / "README.md").write_text(readme, encoding="utf-8")

    (staging / "_SUCCESS").write_text(
        json.dumps(
            {
                "build_version": BUILD_VERSION,
                "active_pairs": len(manifest),
                "completed_local_time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    os.replace(staging, output)
    print(f"Published complete dataset: {output}", flush=True)


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if source == output or source in output.parents:
        raise BuildError("Output must not be inside the immutable source directory")
    if args.workers < 1:
        raise BuildError("--workers must be >= 1")

    before = source_snapshot(source)
    candidates, official_ids = select_candidates(source, args.include_single_s2)
    audited = audit_all(candidates, args.workers)
    split_by_id, group_by_id, component_rows = assign_splits(audited, official_ids)
    build_output(
        source,
        output,
        audited,
        split_by_id,
        group_by_id,
        component_rows,
        args.include_single_s2,
        args.resume,
    )
    after = source_snapshot(source)
    if before != after:
        raise BuildError("Source tree metadata changed during build; output requires investigation")
    print("Source snapshot unchanged: PASS", flush=True)
    print("No model training or test-set evaluation was performed.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BuildError as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
