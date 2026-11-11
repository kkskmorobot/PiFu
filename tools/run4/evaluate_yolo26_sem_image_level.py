"""Evaluate a trained YOLO26 semantic model as a seven-disease image classifier.

Standard semantic validation reports pixel mIoU. This companion evaluation
excludes the appended background class and converts each predicted class map to
one disease prediction using the largest foreground class by pixel area.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = PROJECT_ROOT / "Data" / "processed_skin7_v3" / "data.yaml"
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
BACKGROUND_ID = len(CLASS_NAMES)
NO_PRED_ID = len(CLASS_NAMES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--min-foreground-ratio",
        type=float,
        default=0.01,
        help="Return no-prediction when disease pixels occupy less than this image fraction.",
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_split(data_yaml: Path, split: str) -> tuple[Path, Path]:
    with data_yaml.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    root = Path(data["path"])
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    return root / data[split], root / "labels" / split


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> None:
    args = parse_args()
    weights = resolve(args.weights)
    data_yaml = resolve(args.data)
    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")
    image_dir, label_dir = load_split(data_yaml, args.split)
    output_dir = resolve(args.output_dir) if args.output_dir else weights.parents[1] / f"{args.split}_image_level"
    output_dir.mkdir(parents=True, exist_ok=True)

    truth_by_stem: dict[str, int] = {}
    for label_path in label_dir.glob("*.txt"):
        first = label_path.read_text(encoding="utf-8").split()[0]
        truth_by_stem[label_path.stem] = int(first)

    from ultralytics import YOLO

    model = YOLO(str(weights), task="semantic")
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES) + 1), dtype=np.int64)
    rows: list[dict[str, object]] = []
    results = model.predict(
        source=str(image_dir), stream=True, imgsz=args.imgsz, batch=args.batch,
        device=args.device, verbose=False,
    )

    for result in results:
        stem = Path(result.path).stem
        true_id = truth_by_stem[stem]
        mask = result.semantic_mask.data.detach().cpu().numpy()
        counts = np.bincount(mask.reshape(-1).astype(np.int64), minlength=BACKGROUND_ID + 1)
        foreground_counts = counts[:BACKGROUND_ID]
        foreground_ratio = float(foreground_counts.sum() / mask.size)
        pred_id = int(foreground_counts.argmax()) if foreground_ratio >= args.min_foreground_ratio else NO_PRED_ID
        confusion[true_id, pred_id] += 1
        rows.append(
            {
                "image": Path(result.path).name,
                "true_id": true_id,
                "true_class": CLASS_NAMES[true_id],
                "pred_id": pred_id if pred_id != NO_PRED_ID else -1,
                "pred_class": CLASS_NAMES[pred_id] if pred_id != NO_PRED_ID else "no_prediction",
                "foreground_ratio": foreground_ratio,
                "correct": pred_id == true_id,
            }
        )

    summary: list[dict[str, object]] = []
    for class_id, name in enumerate(CLASS_NAMES):
        tp = int(confusion[class_id, class_id])
        fn = int(confusion[class_id].sum() - tp)
        fp = int(confusion[:, class_id].sum() - tp)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        summary.append(
            {"class": name, "support": int(confusion[class_id].sum()), "precision": precision,
             "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
        )

    total = int(confusion.sum())
    correct = int(sum(confusion[i, i] for i in range(len(CLASS_NAMES))))
    aggregate = {
        "split": args.split,
        "images": total,
        "top1_accuracy": safe_div(correct, total),
        "macro_precision": float(np.mean([row["precision"] for row in summary])),
        "macro_recall": float(np.mean([row["recall"] for row in summary])),
        "macro_f1": float(np.mean([row["f1"] for row in summary])),
        "no_predictions": int(confusion[:, NO_PRED_ID].sum()),
        "min_foreground_ratio": args.min_foreground_ratio,
        "per_class": summary,
    }

    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "per_class_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    np.savetxt(output_dir / "confusion_matrix.csv", confusion, delimiter=",", fmt="%d")
    (output_dir / "metrics.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    print(json.dumps(aggregate, indent=2))
    print(f"Saved image-level evaluation to: {output_dir}")
    if args.split == "test":
        print("WARNING: the internal test split was evaluated. Do this only for the final selected model.")


if __name__ == "__main__":
    main()
