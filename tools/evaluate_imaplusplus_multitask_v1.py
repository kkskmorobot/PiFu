#!/usr/bin/env python3
"""Standalone unified val evaluation for the binary+8-class partial-label model."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from imaplusplus_partial_multitask_v1 import (
    CLASS_NAMES,
    BinaryPartialMultiTaskModel,
    PartialLabelMultiTaskValidator,
    PartialLabelSemanticDataset,
    calibration_metrics,
    classification_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def resolve_weights(path: Path) -> Path:
    path = resolve(path)
    if path.is_dir():
        path = path / "trainer.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        if not rows:
            return
        fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["group_id"])].append(row)
    result = []
    for group_id, members in sorted(grouped.items()):
        targets = {int(row["target"]) for row in members}
        if len(targets) != 1:
            raise RuntimeError(f"Group {group_id} spans multiple targets: {sorted(targets)}")
        target = targets.pop()
        mean_probability = np.mean([row["probabilities"] for row in members], axis=0)
        prediction = int(mean_probability.argmax())
        result.append(
            {
                "group_id": group_id,
                "images": len(members),
                "true_id": target,
                "true_class": CLASS_NAMES[target],
                "pred_id": prediction,
                "pred_class": CLASS_NAMES[prediction],
                "correct": prediction == target,
                "probabilities": mean_probability,
            }
        )
    return result


def confusion_from_rows(rows: list[dict[str, Any]], true_key: str, pred_key: str) -> np.ndarray:
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    for row in rows:
        confusion[int(row[true_key]), int(row[pred_key])] += 1
    return confusion


def bootstrap_groups(rows: list[dict[str, Any]], samples: int, seed: int) -> dict[str, Any]:
    if samples <= 0:
        return {}
    by_class = {
        class_id: [row for row in rows if int(row["true_id"]) == class_id]
        for class_id in range(len(CLASS_NAMES))
    }
    missing = [CLASS_NAMES[class_id] for class_id, members in by_class.items() if not members]
    if missing:
        raise RuntimeError(f"Stratified group bootstrap requires every class; missing={missing}")
    rng = np.random.default_rng(seed)
    tracked: dict[str, list[float]] = defaultdict(list)
    for _ in range(samples):
        draw = []
        for members in by_class.values():
            indices = rng.integers(0, len(members), size=len(members))
            draw.extend(members[index] for index in indices)
        metric = classification_metrics(confusion_from_rows(draw, "true_id", "pred_id"))
        for key in (
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "mel_to_nv_rate",
            "malignant_to_benign_rate",
        ):
            tracked[key].append(float(metric[key]))
    return {
        key: {
            "mean": float(np.mean(values)),
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
        }
        for key, values in tracked.items()
    }


def main() -> None:
    args = parse_args()
    if args.split == "test" and not args.allow_test:
        raise ValueError("Refusing test evaluation without both --split test and --allow-test.")
    weights = resolve_weights(args.weights)
    data_yaml = resolve(args.data)
    if not data_yaml.is_file():
        raise FileNotFoundError(data_yaml)
    output_dir = resolve(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty evaluation directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import the custom model before checkpoint deserialization.
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_dataloader
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils.torch_utils import get_flops

    wrapper = YOLO(str(weights), task="semantic")
    model = wrapper.model
    if not isinstance(model, BinaryPartialMultiTaskModel):
        raise TypeError(f"Expected BinaryPartialMultiTaskModel checkpoint, found {type(model).__name__}")
    if tuple(getattr(model, "classification_names", ())) != CLASS_NAMES:
        raise ValueError("Checkpoint classification taxonomy/order does not match v1")

    data = check_det_dataset(str(data_yaml), split=args.split)
    if args.split == "test" and not data.get("test"):
        raise ValueError("Dataset contract does not expose a test path")
    manifest = Path(data.get("partial_label_manifest", ""))
    if not manifest.is_absolute():
        manifest = (Path(data["path"]) / manifest).resolve()
    cfg = get_cfg(
        overrides={
            "task": "semantic",
            "mode": "val",
            "model": str(weights),
            "data": str(data_yaml),
            "split": args.split,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "device": args.device,
            "plots": False,
            "save_json": False,
            "verbose": False,
        }
    )
    dataset = PartialLabelSemanticDataset(
        img_path=data[args.split],
        imgsz=args.imgsz,
        batch_size=args.batch,
        augment=False,
        hyp=cfg,
        rect=True,
        cache=False,
        single_cls=False,
        stride=max(int(model.stride.max()), 32),
        pad=0.0,
        prefix=f"{args.split}: ",
        task="semantic",
        classes=None,
        data=data,
        fraction=1.0,
        partial_label_manifest=manifest,
        expected_split=args.split,
    )
    loader = build_dataloader(
        dataset,
        batch=args.batch,
        workers=args.workers,
        shuffle=False,
        rank=-1,
        drop_last=False,
    )
    validator = PartialLabelMultiTaskValidator(
        dataloader=loader,
        save_dir=output_dir,
        args=cfg,
        selection={"max_mel_to_nv_rate": 0.1077844311},
    )
    flat_stats = validator(model=model)

    image_rows = []
    for row in validator.classification_rows:
        probability = row["probabilities"]
        prediction = int(probability.argmax())
        image_rows.append(
            {
                "isic_id": row["isic_id"],
                "group_id": row["group_id"],
                "true_id": int(row["target"]),
                "true_class": CLASS_NAMES[int(row["target"])],
                "pred_id": prediction,
                "pred_class": CLASS_NAMES[prediction],
                "correct": prediction == int(row["target"]),
                "probabilities": probability,
            }
        )
    group_rows = aggregate_groups(validator.classification_rows)
    image_confusion = confusion_from_rows(image_rows, "true_id", "pred_id")
    group_confusion = confusion_from_rows(group_rows, "true_id", "pred_id")
    image_metrics = classification_metrics(image_confusion)
    group_metrics = classification_metrics(group_confusion)
    group_probability = np.asarray([row["probabilities"] for row in group_rows], dtype=np.float64)
    group_target = np.asarray([row["true_id"] for row in group_rows], dtype=np.int64)
    calibration = calibration_metrics(group_probability, group_target)
    bootstrap = bootstrap_groups(group_rows, args.bootstrap_samples, args.bootstrap_seed)

    semantic_matrix = validator.metrics.matrix.detach().cpu().numpy().astype(np.int64)
    tn, fp, fn, tp = (int(semantic_matrix[0, 0]), int(semantic_matrix[0, 1]), int(semantic_matrix[1, 0]), int(semantic_matrix[1, 1]))
    lesion_iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    lesion_dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    pixel_accuracy = (tp + tn) / semantic_matrix.sum() if semantic_matrix.sum() else 0.0

    serializable_image = []
    for row in image_rows:
        probabilities = row.pop("probabilities")
        serializable_image.append(
            {**row, **{f"score_{name}": float(probabilities[index]) for index, name in enumerate(CLASS_NAMES)}}
        )
    serializable_group = []
    for row in group_rows:
        probabilities = row.pop("probabilities")
        serializable_group.append(
            {**row, **{f"score_{name}": float(probabilities[index]) for index, name in enumerate(CLASS_NAMES)}}
        )
    write_csv(output_dir / "image_predictions.csv", serializable_image)
    write_csv(output_dir / "group_predictions.csv", serializable_group)
    write_csv(output_dir / "image_per_class_metrics.csv", image_metrics["per_class"])
    write_csv(output_dir / "group_per_class_metrics.csv", group_metrics["per_class"])
    np.savetxt(output_dir / "semantic_confusion_gt_by_pred.csv", semantic_matrix, delimiter=",", fmt="%d")
    np.savetxt(output_dir / "image_confusion_gt_by_pred.csv", image_confusion, delimiter=",", fmt="%d")
    np.savetxt(output_dir / "group_confusion_gt_by_pred.csv", group_confusion, delimiter=",", fmt="%d")
    report = {
        "weights": str(weights),
        "data": str(data_yaml),
        "split": args.split,
        "test_explicitly_allowed": bool(args.allow_test),
        "binary_segmentation": {
            "lesion_iou": lesion_iou,
            "lesion_dice": lesion_dice,
            "pixel_accuracy": pixel_accuracy,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
        "image_classification": image_metrics,
        "group_classification": group_metrics,
        "group_calibration": calibration,
        "group_bootstrap": {
            "method": "stratified group resampling with replacement",
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
            "intervals": bootstrap,
        },
        "selection": {
            "joint_score": float(flat_stats["metrics/joint_score"]),
            "feasible_joint": bool(flat_stats["metrics/feasible_joint"]),
            "max_mel_to_nv_rate": 0.1077844311,
        },
        "efficiency": {
            "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
            "gflops": float(get_flops(model, imgsz=args.imgsz)),
            "mean_ms_per_image": validator.speed,
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved unified {args.split} evaluation to: {output_dir}")
    if args.split == "test":
        print("WARNING: test was evaluated with explicit --allow-test.")


if __name__ == "__main__":
    main()
