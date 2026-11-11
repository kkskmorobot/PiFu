"""Prepare a leakage-safe, moderately balanced HAM10000 YOLO segmentation dataset.

The script never changes files under ``Data/train``. By default it performs a
dry run. Pass ``--apply`` to build ``Data/processed_skin7`` atomically.

Required packages for --apply:
    pip install numpy opencv-python
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


CLASS_NAMES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

# These are upper targets, not mandatory equal class sizes. A second limit,
# max_aug_factor, prevents a small class from being repeated excessively.
DEFAULT_TRAIN_TARGETS = {
    "akiec": 800,
    "bcc": 1000,
    "bkl": 1400,
    "df": 400,
    "mel": 1400,
    "nv": 3500,
    "vasc": 450,
}


@dataclass(frozen=True)
class Sample:
    lesion_id: str
    image_id: str
    dx: str
    dx_type: str
    age: str
    sex: str
    localization: str
    dataset: str


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nv-cap", type=int, default=3500)
    parser.add_argument("--max-aug-factor", type=float, default=4.0)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually create the processed dataset. Without this flag, only print the plan.",
    )
    return parser.parse_args()


def load_and_validate(project_root: Path) -> tuple[list[Sample], Path, Path]:
    image_dir = project_root / "Data" / "train" / "image"
    mask_dir = project_root / "Data" / "train" / "tab"
    metadata_path = project_root / "Data" / "train" / "text" / "HAM10000_metadata.tab"

    for path in (image_dir, mask_dir, metadata_path):
        if not path.exists():
            raise FileNotFoundError(f"Required input does not exist: {path}")

    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    required = {
        "lesion_id",
        "image_id",
        "dx",
        "dx_type",
        "age",
        "sex",
        "localization",
        "dataset",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Metadata columns are incomplete; required={sorted(required)}")

    samples = [Sample(**{key: row.get(key, "").strip() for key in required}) for row in rows]
    image_ids = [sample.image_id for sample in samples]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("Metadata contains duplicate image_id values")
    if any(not sample.lesion_id for sample in samples):
        raise ValueError("Metadata contains an empty lesion_id")
    unknown = sorted({sample.dx for sample in samples} - set(CLASS_NAMES))
    if unknown:
        raise ValueError(f"Unknown disease classes: {unknown}")

    disk_images = {path.stem for path in image_dir.glob("*.jpg")}
    disk_masks = {path.stem.removesuffix("_segmentation") for path in mask_dir.glob("*.png")}
    metadata_ids = set(image_ids)
    problems = {
        "images_without_metadata": sorted(disk_images - metadata_ids),
        "metadata_without_images": sorted(metadata_ids - disk_images),
        "images_without_masks": sorted(disk_images - disk_masks),
        "masks_without_images": sorted(disk_masks - disk_images),
    }
    if any(problems.values()):
        preview = {key: value[:10] for key, value in problems.items() if value}
        raise ValueError(f"Image/metadata/mask mismatch: {preview}")

    lesion_classes: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        lesion_classes[sample.lesion_id].add(sample.dx)
    mixed = {lesion: values for lesion, values in lesion_classes.items() if len(values) != 1}
    if mixed:
        raise ValueError(f"A lesion_id maps to multiple diagnoses: {dict(list(mixed.items())[:10])}")

    return samples, image_dir, mask_dir


def grouped_stratified_split(samples: list[Sample], seed: int) -> dict[str, list[Sample]]:
    """Approximately stratify by dx while keeping each lesion_id in one split."""
    rng = random.Random(seed)
    by_class_and_lesion: dict[str, dict[str, list[Sample]]] = {
        name: defaultdict(list) for name in CLASS_NAMES
    }
    for sample in samples:
        by_class_and_lesion[sample.dx][sample.lesion_id].append(sample)

    result: dict[str, list[Sample]] = {name: [] for name in SPLIT_RATIOS}
    for dx in CLASS_NAMES:
        groups = list(by_class_and_lesion[dx].values())
        rng.shuffle(groups)
        # Place larger groups first; shuffled order remains the tie breaker.
        groups.sort(key=len, reverse=True)
        total = sum(map(len, groups))
        targets = {split: total * ratio for split, ratio in SPLIT_RATIOS.items()}
        counts = {split: 0 for split in SPLIT_RATIOS}
        for group in groups:
            split = max(
                SPLIT_RATIOS,
                key=lambda name: (targets[name] - counts[name]) / max(targets[name], 1.0),
            )
            result[split].extend(group)
            counts[split] += len(group)

    for split in result:
        result[split].sort(key=lambda sample: sample.image_id)

    ownership: dict[str, str] = {}
    for split, split_samples in result.items():
        for sample in split_samples:
            previous = ownership.setdefault(sample.lesion_id, split)
            if previous != split:
                raise AssertionError(f"lesion_id leakage: {sample.lesion_id}: {previous}/{split}")
    return result


def cap_nv_by_lesion(samples: list[Sample], cap: int, seed: int) -> list[Sample]:
    if cap <= 0:
        raise ValueError("--nv-cap must be positive")
    non_nv = [sample for sample in samples if sample.dx != "nv"]
    nv_groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        if sample.dx == "nv":
            nv_groups[sample.lesion_id].append(sample)
    groups = list(nv_groups.values())
    random.Random(seed + 101).shuffle(groups)
    kept: list[Sample] = []
    for group in groups:
        if len(kept) + len(group) <= cap:
            kept.extend(group)
    return sorted(non_nv + kept, key=lambda sample: sample.image_id)


def effective_targets(train_samples: list[Sample], nv_cap: int, max_factor: float) -> dict[str, int]:
    if max_factor < 1.0:
        raise ValueError("--max-aug-factor must be at least 1.0")
    counts = Counter(sample.dx for sample in train_samples)
    targets: dict[str, int] = {}
    for dx in CLASS_NAMES:
        configured = nv_cap if dx == "nv" else DEFAULT_TRAIN_TARGETS[dx]
        targets[dx] = min(configured, max(counts[dx], int(counts[dx] * max_factor)))
    return targets


def print_plan(splits: dict[str, list[Sample]], train_capped: list[Sample], targets: dict[str, int]) -> None:
    print("\nLeakage-safe split (before balancing):")
    print("split\t" + "\t".join(CLASS_NAMES) + "\ttotal")
    for split in SPLIT_RATIOS:
        counts = Counter(sample.dx for sample in splits[split])
        values = [counts[name] for name in CLASS_NAMES]
        print(f"{split}\t" + "\t".join(map(str, values)) + f"\t{sum(values)}")

    capped = Counter(sample.dx for sample in train_capped)
    print("\nTraining balance plan:")
    print("class\tafter_nv_cap\tfinal_target\tgenerated")
    for dx in CLASS_NAMES:
        print(f"{dx}\t{capped[dx]}\t{targets[dx]}\t{targets[dx] - capped[dx]}")
    print(f"final training total: {sum(targets.values())}")


def augment_pair(image, mask, rng: random.Random):
    import cv2
    import numpy as np

    transform = rng.choice(("identity", "hflip", "vflip", "rot180"))
    if transform == "hflip":
        image, mask = cv2.flip(image, 1), cv2.flip(mask, 1)
    elif transform == "vflip":
        image, mask = cv2.flip(image, 0), cv2.flip(mask, 0)
    elif transform == "rot180":
        image, mask = cv2.flip(image, -1), cv2.flip(mask, -1)

    # Mild photometric jitter. Geometry is always applied identically to image/mask.
    alpha = rng.uniform(0.90, 1.10)
    beta = rng.uniform(-10.0, 10.0)
    image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if rng.random() < 0.65:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + rng.randint(-3, 3)) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] + rng.randint(-12, 12), 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    mask = ((mask > 127) * 255).astype(np.uint8)
    return image, mask


def mask_to_yolo_label(mask, class_id: int) -> str:
    import cv2

    binary = ((mask > 127) * 255).astype("uint8")
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("mask contains no foreground contour")
    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.001 * cv2.arcLength(contour, True)
    polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    if len(polygon) < 3:
        raise ValueError("mask contour has fewer than three polygon points")
    height, width = binary.shape
    coords: list[str] = []
    for x, y in polygon:
        coords.extend((f"{x / width:.6f}", f"{y / height:.6f}"))
    return f"{class_id} " + " ".join(coords) + "\n"


def write_one(
    sample: Sample,
    output_id: str,
    split: str,
    output_root: Path,
    image_dir: Path,
    mask_dir: Path,
    rng: random.Random | None,
) -> dict[str, str]:
    import cv2

    source_image = image_dir / f"{sample.image_id}.jpg"
    source_mask = mask_dir / f"{sample.image_id}_segmentation.png"
    output_image = output_root / "images" / split / f"{output_id}.jpg"
    output_label = output_root / "labels" / split / f"{output_id}.txt"

    if rng is None:
        shutil.copy2(source_image, output_image)
        mask = cv2.imread(str(source_mask), cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(source_mask), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise ValueError(f"Unable to decode source pair: {sample.image_id}")
        image, mask = augment_pair(image, mask, rng)
        if not cv2.imwrite(str(output_image), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(f"Unable to write {output_image}")
    if mask is None:
        raise ValueError(f"Unable to decode mask: {source_mask}")
    output_label.write_text(mask_to_yolo_label(mask, CLASS_TO_ID[sample.dx]), encoding="utf-8")

    return {
        "lesion_id": sample.lesion_id,
        "image_id": output_id,
        "dx": sample.dx,
        "is_augmented": "1" if rng is not None else "0",
        "source_image_id": sample.image_id,
    }


def build_dataset(
    output: Path,
    splits: dict[str, list[Sample]],
    train_capped: list[Sample],
    targets: dict[str, int],
    image_dir: Path,
    mask_dir: Path,
    seed: int,
) -> None:
    if output.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {output}")
    staging = output.with_name(output.name + ".building")
    if staging.exists():
        raise FileExistsError(f"Partial staging directory exists; inspect it first: {staging}")

    for split in SPLIT_RATIOS:
        (staging / "images" / split).mkdir(parents=True, exist_ok=False)
        (staging / "labels" / split).mkdir(parents=True, exist_ok=False)
        (staging / "metadata").mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed + 202)
    rows_by_split: dict[str, list[dict[str, str]]] = {split: [] for split in SPLIT_RATIOS}
    for split in ("val", "test"):
        for sample in splits[split]:
            rows_by_split[split].append(
                write_one(sample, sample.image_id, split, staging, image_dir, mask_dir, None)
            )

    by_class: dict[str, list[Sample]] = defaultdict(list)
    for sample in train_capped:
        by_class[sample.dx].append(sample)
        rows_by_split["train"].append(
            write_one(sample, sample.image_id, "train", staging, image_dir, mask_dir, None)
        )

    per_source_counter: Counter[str] = Counter()
    for dx in CLASS_NAMES:
        pool = by_class[dx]
        needed = targets[dx] - len(pool)
        for _ in range(needed):
            source = rng.choice(pool)
            per_source_counter[source.image_id] += 1
            output_id = f"{source.image_id}__aug{per_source_counter[source.image_id]:03d}"
            rows_by_split["train"].append(
                write_one(source, output_id, "train", staging, image_dir, mask_dir, rng)
            )

    fieldnames = ("lesion_id", "image_id", "dx", "is_augmented", "source_image_id")
    for split, rows in rows_by_split.items():
        with (staging / "metadata" / f"{split}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda row: row["image_id"]))

    yaml_lines = [
        f"path: {output.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    yaml_lines.extend(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    (staging / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    summary = {
        split: dict(Counter(row["dx"] for row in rows)) for split, rows in rows_by_split.items()
    }
    (staging / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    staging.rename(output)


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output = (args.output or project_root / "Data" / "processed_skin7").resolve()
    samples, image_dir, mask_dir = load_and_validate(project_root)
    splits = grouped_stratified_split(samples, args.seed)
    train_capped = cap_nv_by_lesion(splits["train"], args.nv_cap, args.seed)
    targets = effective_targets(train_capped, args.nv_cap, args.max_aug_factor)
    print_plan(splits, train_capped, targets)

    if not args.apply:
        print("\nDry run only. Re-run with --apply to create the dataset.")
        return
    build_dataset(output, splits, train_capped, targets, image_dir, mask_dir, args.seed)
    print(f"\nDataset created successfully: {output}")
    print(f"YOLO configuration: {output / 'data.yaml'}")


if __name__ == "__main__":
    main()
