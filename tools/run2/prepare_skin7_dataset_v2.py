"""Build the version-2 HAM10000 YOLO segmentation dataset.

The V2 pipeline keeps the original lesion-level train/val/test split, leaves
validation and test images untouched, uses the baseline model for train-only
hard-example mining, reduces NV dominance, and creates reproducible paired
image/mask augmentations. Existing datasets are never overwritten.

Run without ``--apply`` to print and validate the plan. Use ``--apply`` to
mine hard samples and atomically create ``Data/processed_skin7_v2``.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from prepare_skin7_dataset import (
    CLASS_NAMES,
    CLASS_TO_ID,
    Sample,
    cap_nv_by_lesion,
    grouped_stratified_split,
    load_and_validate,
)


TRAIN_TARGETS = {
    "akiec": 1000,
    "bcc": 1200,
    "bkl": 1400,
    "df": 400,
    "mel": 1800,
    "nv": 2800,
    "vasc": 450,
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nv-cap", type=int, default=TRAIN_TARGETS["nv"])
    parser.add_argument("--hard-weight", type=float, default=2.0)
    parser.add_argument("--low-confidence", type=float, default=0.35)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_hard_cache(path: Path, expected_ids: set[str]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows: list[dict[str, Any]] = read_csv(path)
    actual_ids = {row["image_id"] for row in rows}
    if actual_ids != expected_ids:
        raise ValueError(
            f"Hard-example cache does not match train split: "
            f"missing={len(expected_ids - actual_ids)}, extra={len(actual_ids - expected_ids)}"
        )
    severity = {row["image_id"]: float(row["severity"]) for row in rows}
    return rows, severity


def assert_baseline_split_is_reproducible(
    project_root: Path,
    splits: dict[str, list[Sample]],
    seed: int,
) -> None:
    """Ensure V2 preserves the exact lesion-level baseline split."""
    metadata_dir = project_root / "Data" / "processed_skin7" / "metadata"
    if not metadata_dir.is_dir():
        raise FileNotFoundError(f"Baseline metadata not found: {metadata_dir}")

    for split in ("val", "test"):
        actual = {row["source_image_id"] for row in read_csv(metadata_dir / f"{split}.csv")}
        expected = {sample.image_id for sample in splits[split]}
        if actual != expected:
            raise ValueError(
                f"Baseline {split} split differs from the deterministic lesion split: "
                f"missing={len(expected - actual)}, extra={len(actual - expected)}"
            )

    baseline_original_rows = [
        row
        for row in read_csv(metadata_dir / "train.csv")
        if row.get("is_augmented") == "0"
    ]
    actual_train = {row["source_image_id"] for row in baseline_original_rows}
    expected_baseline = {
        sample.image_id for sample in cap_nv_by_lesion(splits["train"], 3500, seed)
    }
    if actual_train != expected_baseline:
        raise ValueError(
            "Baseline train split/cap is not reproducible: "
            f"missing={len(expected_baseline - actual_train)}, "
            f"extra={len(actual_train - expected_baseline)}"
        )


def print_plan(splits: dict[str, list[Sample]], nv_cap: int) -> None:
    print("Leakage-safe original split:")
    for split in ("train", "val", "test"):
        counts = Counter(sample.dx for sample in splits[split])
        print(f"  {split}: total={sum(counts.values())}, classes={dict(counts)}")
    print("\nV2 train targets:")
    for dx in CLASS_NAMES:
        target = nv_cap if dx == "nv" else TRAIN_TARGETS[dx]
        print(f"  {dx}: {target}")


def mine_hard_examples(
    samples: list[Sample],
    image_dir: Path,
    weights: Path,
    batch: int,
    device: str,
    low_confidence: float,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Run the baseline on train-only originals and score hard examples."""
    from ultralytics import YOLO

    if not weights.is_file():
        raise FileNotFoundError(f"Baseline weights not found: {weights}")

    model = YOLO(str(weights))
    by_image = {sample.image_id: sample for sample in samples}
    image_ids = sorted(by_image)
    rows: list[dict[str, Any]] = []
    severity_by_image: dict[str, float] = {}

    chunk_size = max(batch * 8, 128)
    for start in range(0, len(image_ids), chunk_size):
        chunk_ids = image_ids[start : start + chunk_size]
        paths = [str(image_dir / f"{image_id}.jpg") for image_id in chunk_ids]
        results = model.predict(
            source=paths,
            imgsz=640,
            batch=batch,
            device=device,
            conf=0.001,
            iou=0.7,
            max_det=5,
            verbose=False,
            stream=True,
        )
        for result in results:
            image_id = Path(result.path).stem
            sample = by_image[image_id]
            pred_dx = "none"
            confidence = 0.0
            if result.boxes is not None and len(result.boxes):
                confidences = result.boxes.conf.detach().cpu().tolist()
                best_index = max(range(len(confidences)), key=confidences.__getitem__)
                confidence = float(confidences[best_index])
                class_id = int(result.boxes.cls[best_index].detach().cpu().item())
                pred_dx = CLASS_NAMES[class_id]

            if pred_dx == "none":
                status, severity = "missed", 3.0
            elif pred_dx != sample.dx:
                status = f"{sample.dx}_to_{pred_dx}"
                severity = 2.0 + confidence
            elif confidence < low_confidence:
                status, severity = "low_confidence", 1.0 + (low_confidence - confidence)
            else:
                status, severity = "correct", 0.0

            severity_by_image[image_id] = severity
            rows.append(
                {
                    "lesion_id": sample.lesion_id,
                    "image_id": image_id,
                    "true_dx": sample.dx,
                    "pred_dx": pred_dx,
                    "confidence": round(confidence, 6),
                    "status": status,
                    "severity": round(severity, 6),
                }
            )
        completed = min(start + chunk_size, len(image_ids))
        print(f"Hard-example mining: {completed}/{len(image_ids)}")

    return sorted(rows, key=lambda row: row["image_id"]), severity_by_image


def select_nv_by_lesion(
    train_samples: list[Sample],
    cap: int,
    severity_by_image: dict[str, float],
    seed: int,
) -> list[Sample]:
    """Keep difficult NV lesion groups first, then fill the cap reproducibly."""
    non_nv = [sample for sample in train_samples if sample.dx != "nv"]
    nv_groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in train_samples:
        if sample.dx == "nv":
            nv_groups[sample.lesion_id].append(sample)

    rng = random.Random(seed + 707)
    ranked: list[tuple[float, float, list[Sample]]] = []
    for group in nv_groups.values():
        severity = max(severity_by_image.get(sample.image_id, 0.0) for sample in group)
        ranked.append((-severity, rng.random(), group))
    ranked.sort(key=lambda item: (item[0], item[1]))

    kept: list[Sample] = []
    for _, _, group in ranked:
        if len(kept) + len(group) <= cap:
            kept.extend(group)
    return sorted(non_nv + kept, key=lambda sample: sample.image_id)


def source_schedule(
    pool: list[Sample],
    needed: int,
    severity_by_image: dict[str, float],
    hard_weight: float,
    rng: random.Random,
) -> list[Sample]:
    """Balanced weighted round-robin; hard sources receive more variants."""
    generated = Counter()
    tie_breaker = {sample.image_id: rng.random() for sample in pool}
    schedule: list[Sample] = []
    for _ in range(needed):
        def priority(sample: Sample) -> tuple[float, float]:
            weight = hard_weight if severity_by_image.get(sample.image_id, 0.0) > 0 else 1.0
            return generated[sample.image_id] / weight, tie_breaker[sample.image_id]

        source = min(pool, key=priority)
        schedule.append(source)
        generated[source.image_id] += 1
        tie_breaker[source.image_id] = rng.random()
    return schedule


def augment_pair_v2(image, mask, rng: random.Random):
    import cv2
    import numpy as np

    binary = ((mask > 127) * 255).astype(np.uint8)
    original_area = int(np.count_nonzero(binary))
    if original_area == 0:
        raise ValueError("source mask has no foreground")

    for _ in range(12):
        aug_image, aug_mask = image.copy(), binary.copy()
        hflip = rng.random() < 0.5
        vflip = rng.random() < 0.5
        if hflip:
            aug_image, aug_mask = cv2.flip(aug_image, 1), cv2.flip(aug_mask, 1)
        if vflip:
            aug_image, aug_mask = cv2.flip(aug_image, 0), cv2.flip(aug_mask, 0)

        height, width = aug_mask.shape
        angle = rng.uniform(-20.0, 20.0)
        scale = rng.uniform(0.92, 1.08)
        tx = rng.uniform(-0.04, 0.04) * width
        ty = rng.uniform(-0.04, 0.04) * height
        matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, scale)
        matrix[0, 2] += tx
        matrix[1, 2] += ty
        aug_image = cv2.warpAffine(
            aug_image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        aug_mask = cv2.warpAffine(
            aug_mask,
            matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        aug_mask = ((aug_mask > 127) * 255).astype(np.uint8)
        area_ratio = int(np.count_nonzero(aug_mask)) / original_area
        if 0.60 <= area_ratio <= 1.45:
            break
    else:
        raise ValueError("unable to create a valid mask-preserving transform")

    alpha = rng.uniform(0.95, 1.05)
    beta = rng.uniform(-5.0, 5.0)
    aug_image = np.clip(aug_image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    saturation = rng.uniform(0.96, 1.04)
    hsv = cv2.cvtColor(aug_image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation, 0, 255)
    aug_image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    blurred = rng.random() < 0.06
    if blurred:
        aug_image = cv2.GaussianBlur(aug_image, (3, 3), 0)

    params = {
        "hflip": hflip,
        "vflip": vflip,
        "angle": round(angle, 3),
        "scale": round(scale, 4),
        "translate_x": round(tx / width, 4),
        "translate_y": round(ty / height, 4),
        "brightness_alpha": round(alpha, 4),
        "brightness_beta": round(beta, 3),
        "saturation": round(saturation, 4),
        "blur": blurred,
        "mask_area_retained": round(area_ratio, 4),
    }
    return aug_image, aug_mask, params


def mask_to_yolo_label_v2(mask, class_id: int) -> str:
    """Convert a mask robustly after normalizing singleton/color dimensions."""
    import cv2
    import numpy as np

    array = np.asarray(mask)
    if array.ndim == 3:
        if array.shape[2] == 1:
            array = array[..., 0]
        else:
            array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"mask must be 2D after normalization; shape={array.shape}")

    binary = ((array > 127) * 255).astype(np.uint8)
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


def write_sample(
    sample: Sample,
    output_id: str,
    split: str,
    staging: Path,
    image_dir: Path,
    mask_dir: Path,
    rng: random.Random | None,
    severity: float,
) -> dict[str, str]:
    import cv2

    source_image = image_dir / f"{sample.image_id}.jpg"
    source_mask = mask_dir / f"{sample.image_id}_segmentation.png"
    output_image = staging / "images" / split / f"{output_id}.jpg"
    output_label = staging / "labels" / split / f"{output_id}.txt"
    mask = cv2.imread(str(source_mask), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Unable to decode mask: {source_mask}")

    transform: dict[str, Any] | str = "original"
    if rng is None:
        shutil.copy2(source_image, output_image)
    else:
        image = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to decode image: {source_image}")
        image, mask, transform = augment_pair_v2(image, mask, rng)
        if not cv2.imwrite(str(output_image), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(f"Unable to write image: {output_image}")

    try:
        label = mask_to_yolo_label_v2(mask, CLASS_TO_ID[sample.dx])
    except Exception as error:
        raise ValueError(
            f"Unable to convert mask for {sample.image_id} ({sample.dx}); "
            f"shape={getattr(mask, 'shape', None)}"
        ) from error
    output_label.write_text(label, encoding="utf-8")
    return {
        "lesion_id": sample.lesion_id,
        "image_id": output_id,
        "dx": sample.dx,
        "is_augmented": "1" if rng is not None else "0",
        "source_image_id": sample.image_id,
        "is_hard_source": "1" if severity > 0 else "0",
        "hard_severity": f"{severity:.6f}",
        "transform": json.dumps(transform, ensure_ascii=False, separators=(",", ":")),
    }


def audit_masks(
    splits: dict[str, list[Sample]],
    mask_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import cv2
    import numpy as np

    rows: list[dict[str, Any]] = []
    by_class: dict[str, list[float]] = defaultdict(list)
    for split, samples in splits.items():
        for sample in samples:
            mask = cv2.imread(
                str(mask_dir / f"{sample.image_id}_segmentation.png"), cv2.IMREAD_GRAYSCALE
            )
            if mask is None:
                raise ValueError(f"Unable to decode mask for audit: {sample.image_id}")
            binary = (mask > 127).astype(np.uint8)
            area_ratio = float(np.count_nonzero(binary) / binary.size)
            components, _ = cv2.connectedComponents(binary)
            touches_border = bool(
                binary[0].any() or binary[-1].any() or binary[:, 0].any() or binary[:, -1].any()
            )
            suspicious = area_ratio < 0.005 or area_ratio > 0.85 or components - 1 > 10
            by_class[sample.dx].append(area_ratio)
            rows.append(
                {
                    "split": split,
                    "lesion_id": sample.lesion_id,
                    "image_id": sample.image_id,
                    "dx": sample.dx,
                    "mask_area_ratio": round(area_ratio, 8),
                    "components": components - 1,
                    "touches_border": int(touches_border),
                    "suspicious": int(suspicious),
                }
            )

    stats: dict[str, Any] = {}
    for dx, values in by_class.items():
        array = np.asarray(values, dtype=np.float64)
        stats[dx] = {
            "count": len(values),
            "min": float(array.min()),
            "p05": float(np.quantile(array, 0.05)),
            "median": float(np.median(array)),
            "p95": float(np.quantile(array, 0.95)),
            "max": float(array.max()),
        }
    return rows, stats


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_dataset(
    output: Path,
    splits: dict[str, list[Sample]],
    train_selected: list[Sample],
    image_dir: Path,
    mask_dir: Path,
    seed: int,
    hard_rows: list[dict[str, Any]],
    severity_by_image: dict[str, float],
    hard_weight: float,
    mask_rows: list[dict[str, Any]],
    mask_stats: dict[str, Any],
) -> None:
    if output.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {output}")
    staging = output.with_name(output.name + ".building")
    if staging.exists():
        raise FileExistsError(f"Staging directory already exists: {staging}")

    for split in ("train", "val", "test"):
        (staging / "images" / split).mkdir(parents=True, exist_ok=False)
        (staging / "labels" / split).mkdir(parents=True, exist_ok=False)
    (staging / "metadata").mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed + 2026)
    rows_by_split: dict[str, list[dict[str, str]]] = {name: [] for name in ("train", "val", "test")}

    for split in ("val", "test"):
        for sample in splits[split]:
            rows_by_split[split].append(
                write_sample(sample, sample.image_id, split, staging, image_dir, mask_dir, None, 0.0)
            )

    by_class: dict[str, list[Sample]] = defaultdict(list)
    for sample in train_selected:
        by_class[sample.dx].append(sample)
        rows_by_split["train"].append(
            write_sample(
                sample,
                sample.image_id,
                "train",
                staging,
                image_dir,
                mask_dir,
                None,
                severity_by_image.get(sample.image_id, 0.0),
            )
        )

    generated_per_source: Counter[str] = Counter()
    for dx in CLASS_NAMES:
        pool = by_class[dx]
        target = len(pool) if dx == "nv" else TRAIN_TARGETS[dx]
        if target < len(pool):
            raise ValueError(f"Target is below original count for {dx}: {target} < {len(pool)}")
        schedule = source_schedule(
            pool,
            target - len(pool),
            severity_by_image,
            hard_weight,
            rng,
        )
        for source in schedule:
            generated_per_source[source.image_id] += 1
            output_id = f"{source.image_id}__v2aug{generated_per_source[source.image_id]:03d}"
            rows_by_split["train"].append(
                write_sample(
                    source,
                    output_id,
                    "train",
                    staging,
                    image_dir,
                    mask_dir,
                    rng,
                    severity_by_image.get(source.image_id, 0.0),
                )
            )

    metadata_fields = [
        "lesion_id",
        "image_id",
        "dx",
        "is_augmented",
        "source_image_id",
        "is_hard_source",
        "hard_severity",
        "transform",
    ]
    for split, rows in rows_by_split.items():
        with (staging / "metadata" / f"{split}.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=metadata_fields)
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda row: row["image_id"]))

    write_csv(staging / "metadata" / "hard_samples.csv", hard_rows)
    write_csv(staging / "metadata" / "mask_audit.csv", mask_rows)

    yaml_lines = [
        f"path: {output.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    yaml_lines.extend(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    (staging / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    hard_summary: dict[str, Counter[str]] = defaultdict(Counter)
    for row in hard_rows:
        hard_summary[str(row["true_dx"])][str(row["status"])] += 1
    summary = {
        "version": 2,
        "seed": seed,
        "train_targets": TRAIN_TARGETS,
        "actual_counts": {
            split: dict(Counter(row["dx"] for row in rows))
            for split, rows in rows_by_split.items()
        },
        "unique_lesions": {
            split: len({row["lesion_id"] for row in rows})
            for split, rows in rows_by_split.items()
        },
        "hard_sample_summary": {
            dx: dict(counter) for dx, counter in hard_summary.items()
        },
        "mask_area_stats": mask_stats,
        "augmentation": {
            "hard_weight": hard_weight,
            "max_generated_per_source": max(generated_per_source.values(), default=0),
            "generated_per_source_histogram": dict(
                sorted(Counter(generated_per_source.values()).items())
            ),
        },
    }
    (staging / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    staging.rename(output)


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output = (args.output or project_root / "Data" / "processed_skin7_v2").resolve()
    weights = (
        args.weights
        or project_root / "runs" / "segment" / "skin7_yolo11s_baseline" / "weights" / "best.pt"
    ).resolve()

    samples, image_dir, mask_dir = load_and_validate(project_root)
    splits = grouped_stratified_split(samples, args.seed)
    assert_baseline_split_is_reproducible(project_root, splits, args.seed)
    print_plan(splits, args.nv_cap)

    if not args.apply:
        print("\nDry run passed. Add --apply to mine hard samples and create V2.")
        return

    hard_cache = project_root / "Data" / "skin7_v2_hard_samples.csv"
    expected_train_ids = {sample.image_id for sample in splits["train"]}
    if hard_cache.is_file():
        print(f"Loading hard-example cache: {hard_cache}")
        hard_rows, severity_by_image = load_hard_cache(hard_cache, expected_train_ids)
    else:
        hard_rows, severity_by_image = mine_hard_examples(
            splits["train"],
            image_dir,
            weights,
            args.batch,
            args.device,
            args.low_confidence,
        )
        write_csv(hard_cache, hard_rows)
        print(f"Hard-example cache saved: {hard_cache}")
    train_selected = select_nv_by_lesion(
        splits["train"], args.nv_cap, severity_by_image, args.seed
    )
    mask_rows, mask_stats = audit_masks(splits, mask_dir)
    build_dataset(
        output,
        splits,
        train_selected,
        image_dir,
        mask_dir,
        args.seed,
        hard_rows,
        severity_by_image,
        args.hard_weight,
        mask_rows,
        mask_stats,
    )
    print(f"\nV2 dataset created: {output}")
    print(f"Data YAML: {output / 'data.yaml'}")
    print(f"Audit summary: {output / 'summary.json'}")


if __name__ == "__main__":
    main()
