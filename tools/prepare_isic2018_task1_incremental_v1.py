#!/usr/bin/env python3
"""Build the ISIC 2018 Task-1 subset not already represented in IMA++.

This is a supplementary segmentation-only evaluation candidate, not an
independent clinical domain and not a training split.  Images are excluded by
IMA++ ISIC ID first, then by exact SHA-256 and a strict pHash+dHash+dimensions
signature against the active IMA++ manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from prepare_imaplusplus_binary_v1 import (
    BuildError,
    perceptual_hashes,
    sha256_file,
    source_snapshot,
    write_binary_mask,
    write_csv,
)


BUILD_VERSION = "isic2018_task1_incremental_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Data/1 containing extracted ISIC 2018 files.")
    parser.add_argument("--ima-processed", type=Path, required=True, help="Processed IMA++ binary v1 directory.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def audit_pair(image_path: Path, mask_path: Path) -> dict[str, Any]:
    image_sha256 = sha256_file(image_path)
    with Image.open(image_path) as image:
        image.load()
        width, height = image.size
        rgb = np.asarray(image.convert("RGB"))
    phash64, dhash64 = perceptual_hashes(rgb)

    with Image.open(mask_path) as mask_image:
        mask_image.load()
        mask_width, mask_height = mask_image.size
        mask = np.asarray(mask_image.convert("L"))
    if (mask_width, mask_height) != (width, height):
        raise BuildError(
            f"ISIC 2018 image/mask size mismatch: {image_path.name}={width}x{height}, "
            f"{mask_path.name}={mask_width}x{mask_height}"
        )
    binary = (mask > 0).astype(np.uint8)
    foreground = int(binary.sum())
    total = int(binary.size)
    invalid_reason = ""
    if foreground == 0:
        invalid_reason = "empty_mask"
    elif foreground == total:
        invalid_reason = "full_frame_mask"
    components, _ = cv2.connectedComponents(binary, connectivity=8)
    return {
        "image_sha256": image_sha256,
        "phash64": phash64,
        "dhash64": dhash64,
        "width": width,
        "height": height,
        "foreground_fraction": foreground / total,
        "connected_components": int(components - 1),
        "border_touch": bool(
            binary[0, :].any() or binary[-1, :].any() or binary[:, 0].any() or binary[:, -1].any()
        ),
        "invalid_reason": invalid_reason,
        "source_mask_values": ";".join(str(int(value)) for value in sorted(np.unique(mask))),
    }


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    ima_processed = args.ima_processed.resolve()
    output = args.output.resolve()
    if output.exists():
        raise BuildError(f"Refusing to overwrite existing output: {output}")
    staging = output.with_name(output.name + ".building")
    if staging.exists():
        raise BuildError(f"Staging directory already exists: {staging}")

    input_dir = source / "ISIC2018_Task1-2_Training_Input"
    mask_dir = source / "ISIC2018_Task1_Training_GroundTruth"
    if not input_dir.is_dir() or not mask_dir.is_dir():
        raise BuildError("Missing extracted ISIC 2018 Task-1 input or GroundTruth directory")
    if not (ima_processed / "_SUCCESS").is_file():
        raise BuildError(f"IMA++ processed dataset is incomplete: {ima_processed}")

    before = source_snapshot(source)
    isic_images = {path.stem.upper(): path for path in input_dir.glob("*.jpg")}
    task1_masks = {
        path.stem.removesuffix("_segmentation").upper(): path for path in mask_dir.glob("*.png")
    }
    if len(isic_images) != 2594 or len(task1_masks) != 2594:
        raise BuildError(
            f"Expected 2594 ISIC 2018 images/masks, found images={len(isic_images)}, masks={len(task1_masks)}"
        )
    if set(isic_images) != set(task1_masks):
        raise BuildError(
            f"ISIC 2018 pairing mismatch: missing masks={len(set(isic_images)-set(task1_masks))}, "
            f"orphan masks={len(set(task1_masks)-set(isic_images))}"
        )

    ima_ids = {path.stem.upper() for path in (source / "ISIC-images").glob("*.jpg")}
    ima_manifest = read_csv(ima_processed / "manifest.csv")
    active_sha = {row["image_sha256"] for row in ima_manifest}
    active_perceptual = {
        f"{row['phash64']}:{row['dhash64']}:{row['width']}x{row['height']}" for row in ima_manifest
    }

    excluded: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for isic_id in sorted(isic_images):
        if isic_id in ima_ids:
            excluded.append({"isic_id": isic_id, "reason": "isic_id_already_present_in_imaplusplus"})
            continue
        audit = audit_pair(isic_images[isic_id], task1_masks[isic_id])
        reason = audit["invalid_reason"]
        signature = f"{audit['phash64']}:{audit['dhash64']}:{audit['width']}x{audit['height']}"
        if not reason and audit["image_sha256"] in active_sha:
            reason = "sha256_duplicate_of_active_imaplusplus"
        if not reason and signature in active_perceptual:
            reason = "strict_perceptual_duplicate_of_active_imaplusplus"
        if reason:
            excluded.append({"isic_id": isic_id, "reason": reason, **audit})
        else:
            candidates.append(
                {
                    "isic_id": isic_id,
                    "image_path": isic_images[isic_id],
                    "mask_path": task1_masks[isic_id],
                    **audit,
                }
            )

    # De-duplicate the incremental candidates themselves, keeping the smallest ID.
    kept: list[dict[str, Any]] = []
    seen_sha: dict[str, str] = {}
    seen_perceptual: dict[str, str] = {}
    for row in candidates:
        signature = f"{row['phash64']}:{row['dhash64']}:{row['width']}x{row['height']}"
        reason = ""
        duplicate_of = ""
        if row["image_sha256"] in seen_sha:
            reason = "sha256_duplicate_within_incremental_subset"
            duplicate_of = seen_sha[row["image_sha256"]]
        elif signature in seen_perceptual:
            reason = "strict_perceptual_duplicate_within_incremental_subset"
            duplicate_of = seen_perceptual[signature]
        if reason:
            excluded.append(
                {
                    "isic_id": row["isic_id"],
                    "reason": reason,
                    "duplicate_of": duplicate_of,
                    **{key: value for key, value in row.items() if key not in {"image_path", "mask_path"}},
                }
            )
            continue
        seen_sha[row["image_sha256"]] = row["isic_id"]
        seen_perceptual[signature] = row["isic_id"]
        kept.append(row)

    staging.mkdir(parents=True)
    manifest: list[dict[str, Any]] = []
    hashes: list[dict[str, str]] = []
    for row in kept:
        isic_id = row["isic_id"]
        image_rel = Path("images") / "external_candidate" / f"{isic_id}.jpg"
        mask_rel = Path("masks") / "external_candidate" / f"{isic_id}.png"
        image_dest = staging / image_rel
        image_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row["image_path"], image_dest)
        if sha256_file(image_dest) != row["image_sha256"]:
            raise BuildError(f"Copied ISIC 2018 image hash mismatch: {isic_id}")
        mask_sha256 = write_binary_mask(
            row["mask_path"], staging / mask_rel, (row["width"], row["height"]), resume=False
        )
        manifest_row = {
            "isic_id": isic_id,
            "role": "external_candidate_not_for_training",
            "image_relpath": image_rel.as_posix(),
            "mask_relpath": mask_rel.as_posix(),
            "source_image_relpath": row["image_path"].relative_to(source).as_posix(),
            "source_mask_relpath": row["mask_path"].relative_to(source).as_posix(),
            "width": row["width"],
            "height": row["height"],
            "foreground_fraction": f"{row['foreground_fraction']:.10f}",
            "connected_components": row["connected_components"],
            "border_touch": int(row["border_touch"]),
            "source_mask_values": row["source_mask_values"],
            "image_sha256": row["image_sha256"],
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

    write_csv(staging / "manifest.csv", manifest)
    write_csv(staging / "excluded.csv", excluded, fieldnames=sorted({key for row in excluded for key in row}))
    write_csv(staging / "hashes_sha256.csv", hashes)
    shutil.copy2(input_dir / "ATTRIBUTION.txt", staging / "ATTRIBUTION.txt")
    shutil.copy2(input_dir / "LICENSE.txt", staging / "LICENSE.txt")
    shutil.copy2(mask_dir / "ATTRIBUTION.txt", staging / "GROUNDTRUTH_ATTRIBUTION.txt")

    reason_counts = Counter(row["reason"] for row in excluded)
    summary = {
        "build_version": BUILD_VERSION,
        "source_pair_count": len(isic_images),
        "isic_id_overlap_with_imaplusplus": reason_counts["isic_id_already_present_in_imaplusplus"],
        "incremental_pair_count": len(manifest),
        "excluded_reason_counts": dict(sorted(reason_counts.items())),
        "role": "segmentation_external_candidate_not_for_training",
        "independent_external_domain": False,
        "patient_or_lesion_metadata_available": False,
        "test_metrics_used": False,
    }
    (staging / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (staging / "external_eval.yaml").write_text(
        "# Evaluation-only ISIC 2018 Task-1 increment; do not use as training data.\n"
        "path: .\n"
        "val: images/external_candidate\n"
        "masks_dir: masks\n"
        "nc: 1\n"
        "names:\n  0: lesion\n",
        encoding="utf-8",
    )
    report = f"""# ISIC 2018 Task 1 增量审计报告

`BUILD_VERSION={BUILD_VERSION}`  
`ROLE=EXTERNAL_CANDIDATE_NOT_FOR_TRAINING`

- 原始 Task 1 配对：{len(isic_images)}
- 与 IMA++ 同 ISIC ID：{reason_counts['isic_id_already_present_in_imaplusplus']}
- 经 ID、SHA-256 和严格感知哈希去重后的增量：{len(manifest)}
- 图像与掩码尺寸一致，输出掩码为 `0=background, 1=lesion`。
- 没有计算模型指标，没有把该集合并入 IMA++ train/val/test。
- 该增量仍来自 ISIC 域且缺少 patient/lesion 元数据，不能声称为独立医院或独立临床外部测试。
- `ISIC2018_Task2_Training_GroundTruth_v3` 是五类皮肤镜属性掩码，本轮保持原样，未混入病灶二值真值。
"""
    (staging / "DATASET_REPORT.md").write_text(report, encoding="utf-8")
    (staging / "_SUCCESS").write_text(
        json.dumps(
            {
                "build_version": BUILD_VERSION,
                "incremental_pair_count": len(manifest),
                "completed_local_time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if before != source_snapshot(source):
        raise BuildError("Source tree metadata changed during ISIC 2018 incremental build")
    os.replace(staging, output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Published: {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BuildError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
