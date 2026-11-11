"""Build the controlled Skin7 V3 dataset for the third experiment.

V3 intentionally changes only one variable from the original processed_skin7
baseline: the number of real training NV images is reduced from 3500 to 3200.
All split logic, minority targets, offline transforms, seeds, and source
selection are inherited from prepare_skin7_dataset.py.

The script is dry-run by default. Pass --apply to create
Data/processed_skin7_v3. Existing outputs are never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from prepare_skin7_dataset import (
    CLASS_NAMES,
    build_dataset,
    cap_nv_by_lesion,
    effective_targets,
    grouped_stratified_split,
    load_and_validate,
    print_plan,
)


SEED = 42
NV_CAP = 3200
MAX_AUG_FACTOR = 4.0
EXPECTED_COUNTS = {
    "akiec": 800,
    "bcc": 1000,
    "bkl": 1400,
    "df": 320,
    "mel": 1400,
    "nv": 3200,
    "vasc": 392,
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Create the dataset after printing and validating the plan.",
    )
    return parser.parse_args()


def read_metadata(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def audit_against_v1(project_root: Path, output: Path) -> dict[str, object]:
    """Prove that V3 differs from V1 only by 300 removed training NV images."""
    baseline = project_root / "Data" / "processed_skin7"
    if not baseline.is_dir():
        raise FileNotFoundError(f"V1 baseline dataset is missing: {baseline}")

    v1_rows = {
        split: read_metadata(baseline / "metadata" / f"{split}.csv")
        for split in ("train", "val", "test")
    }
    v3_rows = {
        split: read_metadata(output / "metadata" / f"{split}.csv")
        for split in ("train", "val", "test")
    }

    for split in ("val", "test"):
        v1_ids = {row["image_id"] for row in v1_rows[split]}
        v3_ids = {row["image_id"] for row in v3_rows[split]}
        if v1_ids != v3_ids:
            raise AssertionError(f"V3 {split} differs from V1 {split}")

    def row_key(row: dict[str, str]) -> tuple[str, ...]:
        return tuple(row[key] for key in ("lesion_id", "image_id", "dx", "is_augmented", "source_image_id"))

    v1_non_nv = {row_key(row) for row in v1_rows["train"] if row["dx"] != "nv"}
    v3_non_nv = {row_key(row) for row in v3_rows["train"] if row["dx"] != "nv"}
    if v1_non_nv != v3_non_nv:
        raise AssertionError("V3 non-NV training metadata differs from V1")

    v1_nv = {row["image_id"] for row in v1_rows["train"] if row["dx"] == "nv"}
    v3_nv = {row["image_id"] for row in v3_rows["train"] if row["dx"] == "nv"}
    if not v3_nv < v1_nv or len(v1_nv - v3_nv) != 300:
        raise AssertionError("V3 NV set is not the expected 3200-image subset of V1")

    counts = {
        split: dict(Counter(row["dx"] for row in rows))
        for split, rows in v3_rows.items()
    }
    if counts["train"] != EXPECTED_COUNTS:
        raise AssertionError(f"Unexpected V3 train counts: {counts['train']}")

    for split, rows in v3_rows.items():
        image_count = len(list((output / "images" / split).glob("*.jpg")))
        label_count = len(list((output / "labels" / split).glob("*.txt")))
        if image_count != len(rows) or label_count != len(rows):
            raise AssertionError(
                f"V3 {split} file mismatch: metadata={len(rows)}, images={image_count}, labels={label_count}"
            )

    report: dict[str, object] = {
        "experiment": "single-variable NV cap",
        "seed": SEED,
        "nv_cap_v1": 3500,
        "nv_cap_v3": NV_CAP,
        "removed_nv_images": len(v1_nv - v3_nv),
        "non_nv_train_equal_v1": True,
        "val_equal_v1": True,
        "test_equal_v1": True,
        "counts": counts,
    }
    (output / "metadata" / "v3_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output = (args.output or project_root / "Data" / "processed_skin7_v3").resolve()

    samples, image_dir, mask_dir = load_and_validate(project_root)
    splits = grouped_stratified_split(samples, SEED)
    train_capped = cap_nv_by_lesion(splits["train"], NV_CAP, SEED)
    targets = effective_targets(train_capped, NV_CAP, MAX_AUG_FACTOR)
    if targets != EXPECTED_COUNTS:
        raise AssertionError(f"V3 plan drifted from its locked counts: {targets}")
    print_plan(splits, train_capped, targets)

    if not args.apply:
        print("\nV3 dry run only. Re-run with --apply to build the dataset.")
        return

    build_dataset(output, splits, train_capped, targets, image_dir, mask_dir, SEED)
    report = audit_against_v1(project_root, output)
    print(f"\nV3 dataset created and audited: {output}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
