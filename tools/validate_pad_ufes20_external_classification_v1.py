#!/usr/bin/env python3
"""Independently validate the published PAD-UFES-20 external candidate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


CLASS_NAMES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc", "scc")


class ValidationError(RuntimeError):
    pass


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValidationError(f"Missing artifact: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=project_root / "Data" / "processed_pad_ufes_20_external_classification_v1",
    )
    return parser.parse_args()


def validate(dataset: Path) -> dict[str, object]:
    dataset = dataset.resolve()
    required = {
        "manifest.csv",
        "excluded.csv",
        "duplicate_groups.csv",
        "group_manifest.csv",
        "hashes_sha256.csv",
        "audit_summary.json",
        "DATASET_REPORT.md",
        "_SUCCESS",
    }
    missing = sorted(name for name in required if not (dataset / name).is_file())
    if missing:
        raise ValidationError(f"Missing required artifacts: {missing}")
    if any((dataset / name).exists() for name in ("train", "val", "test", "masks", "multitask_data.yaml")):
        raise ValidationError("External candidate contains a forbidden split, mask, or training YAML artifact")

    manifest = read_csv(dataset / "manifest.csv")
    excluded = read_csv(dataset / "excluded.csv")
    duplicate_groups = read_csv(dataset / "duplicate_groups.csv")
    groups = read_csv(dataset / "group_manifest.csv")
    hashes = read_csv(dataset / "hashes_sha256.csv")
    summary = json.loads((dataset / "audit_summary.json").read_text(encoding="utf-8"))
    success = json.loads((dataset / "_SUCCESS").read_text(encoding="utf-8"))

    if len(manifest) + len(excluded) != 2_298:
        raise ValidationError("Manifest and exclusions do not partition all 2298 official rows")
    ids = [row["img_id"] for row in manifest]
    excluded_ids = [row["img_id"] for row in excluded]
    if len(ids) != len(set(ids)) or len(excluded_ids) != len(set(excluded_ids)) or set(ids) & set(excluded_ids):
        raise ValidationError("Retained/excluded image IDs are duplicated or overlap")
    if any(row["role"] != "external_candidate" for row in manifest):
        raise ValidationError("A manifest row is not external_candidate")
    if any(row["has_segmentation"] != "0" or row["has_classification"] != "1" for row in manifest):
        raise ValidationError("Partial-label flags differ from external classification-only contract")
    if any(row["class_name"] not in CLASS_NAMES or int(row["class_id"]) != CLASS_NAMES.index(row["class_name"]) for row in manifest):
        raise ValidationError("Class name/ID differs from locked taxonomy")
    if any(row["class_name"] in {"df", "vasc"} for row in manifest):
        raise ValidationError("PAD unexpectedly contains df or vasc")
    allowed_duplicate_actions = {
        "exclude_conflict_group",
        "exclude_conflict_component",
        "retain_canonical_exclude_duplicates",
    }
    invalid_actions = sorted(
        {row["action"] for row in duplicate_groups if row["action"] not in allowed_duplicate_actions}
    )
    if invalid_actions:
        raise ValidationError(f"Duplicate groups contain unresolved/unknown actions: {invalid_actions}")

    expected_hashes = {row["relative_path"]: row["sha256"] for row in hashes}
    if len(expected_hashes) != len(manifest):
        raise ValidationError("hashes_sha256.csv does not contain exactly one image per manifest row")
    for index, row in enumerate(manifest, start=1):
        path = dataset / Path(row["image_relpath"])
        if not path.is_file():
            raise ValidationError(f"Missing published image: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != row["image_sha256"] or expected_hashes.get(row["image_relpath"]) != actual_hash:
            raise ValidationError(f"Image hash mismatch: {path}")
        with Image.open(path) as image:
            image.load()
            if image.size != (int(row["width"]), int(row["height"])):
                raise ValidationError(f"Image dimensions differ from manifest: {path}")
        if index % 250 == 0 or index == len(manifest):
            print(f"Independent image validation: {index}/{len(manifest)}", flush=True)

    manifest_by_group: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in manifest:
        manifest_by_group[row["group_id"]].append(row)
    if set(manifest_by_group) != {row["group_id"] for row in groups}:
        raise ValidationError("group_manifest IDs differ from manifest")
    if len(groups) != len({(row["patient_id"], row["lesion_id"]) for row in manifest}):
        raise ValidationError("group_manifest does not match unique patient-lesion pairs")
    for group in groups:
        members = manifest_by_group[group["group_id"]]
        if len(members) != int(group["image_count"]):
            raise ValidationError(f"Group image count differs: {group['group_id']}")
        if len({row["class_name"] for row in members}) != 1:
            raise ValidationError(f"Group spans multiple diagnoses: {group['group_id']}")

    if summary.get("external_classification_data_gate") != "PASS":
        raise ValidationError("External classification data gate is not PASS")
    if summary.get("external_classification_evaluation_gate") != "BLOCKED_PENDING_FINAL_MODEL_AND_USER_AUTHORIZATION":
        raise ValidationError("External evaluation gate is not locked")
    if summary.get("project_run10_gate") != "BLOCKED_PENDING_MODEL_AND_CLASSIFICATION_CONTRACT":
        raise ValidationError("Project Run10 gate changed unexpectedly")
    if summary.get("entered_training_manifest") is not False:
        raise ValidationError("Summary claims the external candidate entered training")
    if summary.get("test_metrics_used") is not False or summary.get("external_metrics_used") is not False:
        raise ValidationError("Summary claims test or external metrics were used")
    if success.get("retained") != len(manifest) or success.get("excluded") != len(excluded):
        raise ValidationError("_SUCCESS counts differ from artifacts")

    result = {
        "status": "PASS",
        "dataset": str(dataset),
        "retained": len(manifest),
        "excluded": len(excluded),
        "patients": len({row["patient_id"] for row in manifest}),
        "lesions": len({row["lesion_id"] for row in manifest}),
        "groups": len(groups),
        "class_counts": dict(sorted(Counter(row["class_name"] for row in manifest).items())),
        "project_run10_gate": summary["project_run10_gate"],
        "test_metrics_used": False,
        "external_metrics_used": False,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    validate(parse_args().dataset)


if __name__ == "__main__":
    main()
