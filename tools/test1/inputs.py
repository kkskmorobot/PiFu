#!/usr/bin/env python3
"""Build deterministic Test1 manifests from frozen source manifests only."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


PROJECT = Path(r"E:\App\DaiMa\JetBrains\Project\PiFuV2")
DATA = PROJECT / "Data"
OUT = Path(__file__).resolve().parent
TARGET = PROJECT / "tools" / "test1"
FIELDS = [
    "isic_id", "split", "dataset_role", "image_path", "mask_path",
    "has_segmentation", "has_classification", "classification_target",
    "class_name", "lesion_id", "patient_id", "group_id", "image_sha256",
    "phash64", "dhash64", "width", "height",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in FIELDS} for row in rows])


source_internal = DATA / "processed_isic2019_classification_incremental_v1" / "multitask_manifest.csv"
source_external = DATA / "processed_isic2018_task1_incremental_v1" / "manifest.csv"
internal = [row for row in read_csv(source_internal) if row["split"] == "test"]
if len(internal) != 634:
    raise RuntimeError(f"Expected 634 frozen IMA++ test rows, found {len(internal)}")
if any(row["dataset_role"] != "imaplusplus_binary" or row["has_segmentation"] != "1" for row in internal):
    raise RuntimeError("IMA++ test role/segmentation contract changed")
classified = [row for row in internal if row["has_classification"] == "1"]
if len(classified) != 56 or {row["classification_target"] for row in classified} != {"4", "5"}:
    raise RuntimeError("Expected exactly 56 MEL/NV classification labels in IMA++ test")
if len({row["group_id"] for row in classified}) != 56:
    raise RuntimeError("The 56 labeled IMA++ test samples must remain 56 distinct groups")
if any(row["has_classification"] == "0" and (row["classification_target"] != "-1" or row["class_name"]) for row in internal):
    raise RuntimeError("Unlabeled IMA++ test row violates partial-label encoding")

external_source_rows = read_csv(source_external)
if len(external_source_rows) != 89:
    raise RuntimeError(f"Expected 89 incremental external rows, found {len(external_source_rows)}")
external: list[dict[str, str]] = []
for row in external_source_rows:
    if row["role"].lower() != "external_candidate_not_for_training":
        raise RuntimeError(f"External role changed for {row['isic_id']}")
    external.append({
        "isic_id": row["isic_id"],
        "split": "external",
        "dataset_role": "isic2018_task1_incremental",
        "image_path": "processed_isic2018_task1_incremental_v1/" + row["image_relpath"].replace("\\", "/"),
        "mask_path": "processed_isic2018_task1_incremental_v1/" + row["mask_relpath"].replace("\\", "/"),
        "has_segmentation": "1",
        "has_classification": "0",
        "classification_target": "-1",
        "class_name": "",
        "lesion_id": "",
        "patient_id": "",
        "group_id": "EXT_" + row["isic_id"],
        "image_sha256": row["image_sha256"],
        "phash64": row["phash64"],
        "dhash64": row["dhash64"],
        "width": row["width"],
        "height": row["height"],
    })

internal_ids = {row["isic_id"] for row in internal}
external_ids = {row["isic_id"] for row in external}
internal_sha = {row["image_sha256"] for row in internal}
external_sha = {row["image_sha256"] for row in external}
if internal_ids & external_ids or internal_sha & external_sha:
    raise RuntimeError("Internal test and external increment overlap by ID or SHA-256")

for rows in (internal, external):
    for row in rows:
        joined = " ".join(row.values()).lower()
        if "pad-ufes" in joined or "pad_ufes" in joined:
            raise RuntimeError("PAD-UFES-20 appeared in Test1 inputs")
        image = DATA / row["image_path"]
        mask = DATA / row["mask_path"]
        if not image.is_file() or not mask.is_file():
            raise RuntimeError(f"Missing frozen test input: {image} / {mask}")

write_csv(OUT / "manifest_test.csv", internal)
write_csv(OUT / "manifest_external.csv", external)

(OUT / "data_test.yaml").write_text(
    "# Test1 internal IMA++ test input; evaluation only.\n"
    f"path: {DATA.as_posix()}\n"
    "train: null\nval: null\n"
    "test: processed_imaplusplus_binary_v1/images/test\n"
    "masks_dir: masks\n"
    "nc: 1\nnames:\n  0: lesion\n"
    f"partial_label_manifest: {(TARGET / 'manifest_test.csv').as_posix()}\n"
    f"classification_taxonomy: {(PROJECT / 'tools' / 'classification_taxonomy_v1.yaml').as_posix()}\n",
    encoding="utf-8",
)
(OUT / "data_external.yaml").write_text(
    "# Test1 ISIC2018 incremental external segmentation input; evaluation only.\n"
    f"path: {DATA.as_posix()}\n"
    "train: null\n"
    "val: processed_isic2018_task1_incremental_v1/images/external_candidate\n"
    "test: null\n"
    "masks_dir: masks\n"
    "nc: 1\nnames:\n  0: lesion\n"
    f"partial_label_manifest: {(TARGET / 'manifest_external.csv').as_posix()}\n"
    f"classification_taxonomy: {(PROJECT / 'tools' / 'classification_taxonomy_v1.yaml').as_posix()}\n",
    encoding="utf-8",
)

report = {
    "status": "PASS",
    "internal": {"rows": 634, "segmentation": 634, "classification_mel_nv_only": 56},
    "external": {"rows": 89, "segmentation": 89, "classification": 0},
    "source_hashes": {str(source_internal): sha256(source_internal), str(source_external): sha256(source_external)},
    "generated_hashes": {
        name: sha256(OUT / name)
        for name in ("manifest_test.csv", "manifest_external.csv", "data_test.yaml", "data_external.yaml")
    },
    "id_overlap": 0,
    "sha256_overlap": 0,
    "pad_ufes_20_used": False,
    "model_inference_performed": False,
}
(OUT / "inputs.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
