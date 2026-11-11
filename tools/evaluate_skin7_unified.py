"""Unified validation for PiFuV2 semantic and image-diagnosis checkpoints.

Outputs 8-class semantic metrics, foreground seven-class mIoU, image- and
lesion-level diagnosis metrics, key clinical errors, lesion bootstrap CIs, and
model efficiency.  The default split is val. Test requires an explicit second
flag so it cannot be touched accidentally during method development.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
RUN5_DIR = TOOLS_DIR / "run5"
for module_dir in (TOOLS_DIR, RUN5_DIR):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

DEFAULT_DATA = PROJECT_ROOT / "Data" / "processed_skin7_v3" / "data.yaml"
DEFAULT_SEMANTIC_DATA = (
    PROJECT_ROOT / "experiments" / "04_YOLO26s_Semantic_C2PSA" / "data_semantic_eval_v4.yaml"
)
CLASS_NAMES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
SEMANTIC_NAMES = CLASS_NAMES + ("background",)
BACKGROUND_ID = 7
NO_PRED_ID = 7
MINORITY_IDS = (0, 1, 3)
MEL_ID = 4
NV_ID = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--semantic-data", type=Path, default=DEFAULT_SEMANTIC_DATA)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--decision", choices=("auto", "area", "classifier"), default="auto")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--min-foreground-ratio", type=float, default=0.01)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def resolve_dataset_root(data_yaml: Path, data: dict[str, Any]) -> Path:
    root = Path(data["path"])
    return root if root.is_absolute() else (data_yaml.parent / root).resolve()


def validate_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path, dict[str, Any]]:
    if args.split == "test" and not args.allow_test:
        raise ValueError("Refusing test evaluation without both --split test and --allow-test.")
    weights, data_yaml, semantic_yaml = map(resolve, (args.weights, args.data, args.semantic_data))
    for path in (weights, data_yaml, semantic_yaml):
        if not path.is_file():
            raise FileNotFoundError(path)
    semantic = load_yaml(semantic_yaml)
    names = {int(key): value for key, value in semantic.get("names", {}).items()}
    expected = {idx: name for idx, name in enumerate(SEMANTIC_NAMES)}
    if semantic.get("nc") != 8 or semantic.get("bg_class_idx") != 7 or names != expected:
        raise ValueError(
            f"Semantic evaluation YAML must explicitly define nc=8, bg_class_idx=7, names={expected}: "
            f"{semantic_yaml}"
        )
    data = load_yaml(data_yaml)
    disease_names = {int(key): value for key, value in data.get("names", {}).items()}
    if disease_names != {idx: name for idx, name in enumerate(CLASS_NAMES)}:
        raise ValueError(f"Unexpected disease mapping in {data_yaml}: {disease_names}")
    return weights, data_yaml, semantic_yaml, data


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def classification_metrics(confusion: np.ndarray) -> dict[str, Any]:
    per_class = []
    for class_id, name in enumerate(CLASS_NAMES):
        tp = int(confusion[class_id, class_id])
        fn = int(confusion[class_id].sum() - tp)
        fp = int(confusion[:, class_id].sum() - tp)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        per_class.append(
            {
                "class": name,
                "support": int(confusion[class_id].sum()),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )
    total = int(confusion.sum())
    correct = int(sum(confusion[idx, idx] for idx in range(len(CLASS_NAMES))))
    mel_support = int(confusion[MEL_ID].sum())
    return {
        "samples": total,
        "accuracy": safe_div(correct, total),
        "macro_precision": float(np.mean([row["precision"] for row in per_class])),
        "macro_recall": float(np.mean([row["recall"] for row in per_class])),
        "macro_f1": float(np.mean([row["f1"] for row in per_class])),
        "minority_recall": float(np.mean([per_class[idx]["recall"] for idx in MINORITY_IDS])),
        "mel_to_nv": int(confusion[MEL_ID, NV_ID]),
        "mel_to_nv_rate": safe_div(int(confusion[MEL_ID, NV_ID]), mel_support),
        "nv_to_mel": int(confusion[NV_ID, MEL_ID]),
        "no_predictions": int(confusion[:, NO_PRED_ID].sum()),
        "per_class": per_class,
    }


def confusion_from_rows(rows: list[dict[str, Any]]) -> np.ndarray:
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES) + 1), dtype=np.int64)
    for row in rows:
        confusion[int(row["true_id"]), int(row["pred_id"])] += 1
    return confusion


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_confusion(path: Path, confusion: np.ndarray, title: str) -> None:
    import matplotlib.pyplot as plt

    labels = list(CLASS_NAMES) + ["no_prediction"]
    fig, ax = plt.subplots(figsize=(9, 7), tight_layout=True)
    image = ax.imshow(confusion, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASS_NAMES)), labels=CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    threshold = confusion.max() / 2 if confusion.size else 0
    for row in range(confusion.shape[0]):
        for col in range(confusion.shape[1]):
            value = int(confusion[row, col])
            ax.text(col, row, value, ha="center", va="center", color="white" if value > threshold else "black")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def aggregate_lesions(image_rows: list[dict[str, Any]], metadata_path: Path) -> list[dict[str, Any]]:
    metadata: dict[str, dict[str, str]] = {}
    with metadata_path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            metadata[row["image_id"]] = row
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in image_rows:
        image_id = Path(str(row["image"])).stem
        if image_id not in metadata:
            raise KeyError(f"Missing metadata for {image_id} in {metadata_path}")
        grouped[metadata[image_id]["lesion_id"]].append(row)
    lesion_rows = []
    for lesion_id, members in sorted(grouped.items()):
        true_ids = {int(member["true_id"]) for member in members}
        if len(true_ids) != 1:
            raise ValueError(f"Lesion {lesion_id} spans multiple disease labels: {true_ids}")
        mean_scores = np.mean([member["scores"] for member in members], axis=0)
        pred_id = int(mean_scores.argmax()) if float(mean_scores.sum()) > 0 else NO_PRED_ID
        true_id = true_ids.pop()
        lesion_rows.append(
            {
                "lesion_id": lesion_id,
                "images": len(members),
                "true_id": true_id,
                "true_class": CLASS_NAMES[true_id],
                "pred_id": pred_id,
                "pred_class": CLASS_NAMES[pred_id] if pred_id != NO_PRED_ID else "no_prediction",
                "correct": pred_id == true_id,
                **{f"score_{name}": float(mean_scores[idx]) for idx, name in enumerate(CLASS_NAMES)},
            }
        )
    return lesion_rows


def lesion_bootstrap(
    rows: list[dict[str, Any]], samples: int, seed: int
) -> dict[str, dict[str, float]]:
    if samples <= 0:
        return {}
    rng = np.random.default_rng(seed)
    by_class = {
        class_id: [row for row in rows if int(row["true_id"]) == class_id]
        for class_id in range(len(CLASS_NAMES))
    }
    if any(not values for values in by_class.values()):
        raise ValueError("Stratified lesion bootstrap requires at least one lesion from every disease class.")
    tracked = defaultdict(list)
    for _ in range(samples):
        draw = []
        for values in by_class.values():
            indices = rng.integers(0, len(values), size=len(values))
            draw.extend(values[index] for index in indices)
        metric = classification_metrics(confusion_from_rows(draw))
        for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "minority_recall", "mel_to_nv_rate"):
            tracked[key].append(metric[key])
    return {
        key: {
            "estimate": float(np.mean(values)),
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
        }
        for key, values in tracked.items()
    }


def main() -> None:
    args = parse_args()
    weights, data_yaml, semantic_yaml, data = validate_inputs(args)
    dataset_root = resolve_dataset_root(data_yaml, data)
    image_dir = dataset_root / data[args.split]
    label_dir = dataset_root / "labels" / args.split
    metadata_path = dataset_root / "metadata" / f"{args.split}.csv"
    output_dir = resolve(args.output_dir) if args.output_dir else weights.parents[1] / f"{args.split}_unified"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty evaluation directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import custom classes before torch.load so runs5 checkpoints unpickle reliably.
    from skin7_multitask import Skin7MultiTaskPredictor, Skin7SemanticClassificationHead
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    model = YOLO(str(weights), task="semantic")
    model_names = tuple(model.model.names.values())
    if model_names != SEMANTIC_NAMES:
        raise ValueError(f"Checkpoint must expose {SEMANTIC_NAMES}; found {model_names}")
    has_classifier = isinstance(model.model.model[-1], Skin7SemanticClassificationHead)
    decision = "classifier" if args.decision == "auto" and has_classifier else "area" if args.decision == "auto" else args.decision
    if decision == "classifier" and not has_classifier:
        raise ValueError("--decision classifier requires a runs5 multitask checkpoint.")

    semantic_metrics = model.val(
        data=str(semantic_yaml),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        plots=False,
        save_json=False,
        project=str(output_dir),
        name="semantic_validation",
        exist_ok=False,
        verbose=False,
    )
    per_class_iou = np.asarray(semantic_metrics.per_class_iou, dtype=float)
    semantic_summary = {
        "miou_8class": float(semantic_metrics.miou),
        "foreground_miou_7class": float(per_class_iou[:BACKGROUND_ID].mean()),
        "pixel_accuracy": float(semantic_metrics.pixel_accuracy),
        "per_class_iou": {name: float(per_class_iou[idx]) for idx, name in enumerate(SEMANTIC_NAMES)},
    }
    if semantic_metrics.matrix is not None:
        np.savetxt(
            output_dir / "semantic_confusion_matrix_gt_by_pred.csv",
            semantic_metrics.matrix.detach().cpu().numpy(),
            delimiter=",",
            fmt="%.0f",
        )

    truth_by_stem = {
        path.stem: int(path.read_text(encoding="utf-8").split()[0]) for path in label_dir.glob("*.txt")
    }
    predictor = Skin7MultiTaskPredictor if decision == "classifier" else None
    started = time.perf_counter()
    results = model.predict(
        source=str(image_dir),
        stream=True,
        predictor=predictor,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        verbose=False,
    )
    image_rows: list[dict[str, Any]] = []
    speed_rows = []
    for result in results:
        stem = Path(result.path).stem
        true_id = truth_by_stem[stem]
        if decision == "classifier":
            scores = result.probs.data.detach().cpu().numpy().astype(float)
            foreground_ratio = None
            pred_id = int(scores.argmax())
        else:
            mask = result.semantic_mask.data.detach().cpu().numpy()
            counts = np.bincount(mask.reshape(-1).astype(np.int64), minlength=BACKGROUND_ID + 1)
            scores = counts[:BACKGROUND_ID].astype(float) / mask.size
            foreground_ratio = float(scores.sum())
            pred_id = int(scores.argmax()) if foreground_ratio >= args.min_foreground_ratio else NO_PRED_ID
        image_rows.append(
            {
                "image": Path(result.path).name,
                "true_id": true_id,
                "true_class": CLASS_NAMES[true_id],
                "pred_id": pred_id,
                "pred_class": CLASS_NAMES[pred_id] if pred_id != NO_PRED_ID else "no_prediction",
                "foreground_ratio": foreground_ratio,
                "correct": pred_id == true_id,
                "scores": scores,
            }
        )
        speed_rows.append(result.speed)
    elapsed = time.perf_counter() - started
    if len(image_rows) != len(truth_by_stem):
        raise RuntimeError(f"Predicted {len(image_rows)} images but expected {len(truth_by_stem)}")

    image_confusion = confusion_from_rows(image_rows)
    image_metrics = classification_metrics(image_confusion)
    lesion_rows = aggregate_lesions(image_rows, metadata_path)
    lesion_confusion = confusion_from_rows(lesion_rows)
    lesion_metrics = classification_metrics(lesion_confusion)
    bootstrap = lesion_bootstrap(lesion_rows, args.bootstrap_samples, args.bootstrap_seed)

    serializable_image_rows = []
    for row in image_rows:
        scores = row.pop("scores")
        serializable_image_rows.append(
            {**row, **{f"score_{name}": float(scores[idx]) for idx, name in enumerate(CLASS_NAMES)}}
        )
    write_csv(output_dir / "image_predictions.csv", serializable_image_rows)
    write_csv(output_dir / "image_per_class_metrics.csv", image_metrics["per_class"])
    write_csv(output_dir / "lesion_predictions.csv", lesion_rows)
    write_csv(output_dir / "lesion_per_class_metrics.csv", lesion_metrics["per_class"])
    np.savetxt(output_dir / "image_confusion_matrix.csv", image_confusion, delimiter=",", fmt="%d")
    np.savetxt(output_dir / "lesion_confusion_matrix.csv", lesion_confusion, delimiter=",", fmt="%d")
    plot_confusion(output_dir / "image_confusion_matrix.png", image_confusion, "Image-level confusion matrix")
    plot_confusion(output_dir / "lesion_confusion_matrix.png", lesion_confusion, "Lesion-level confusion matrix")

    params = int(sum(parameter.numel() for parameter in model.model.parameters()))
    speed = {
        key: float(np.mean([row.get(key, 0.0) for row in speed_rows]))
        for key in ("preprocess", "inference", "postprocess")
    }
    efficiency = {
        "parameters": params,
        "gflops": float(get_flops(model.model, imgsz=args.imgsz)),
        "mean_ms_per_image": speed,
        "end_to_end_ms_per_image": 1000.0 * elapsed / len(image_rows),
    }
    report = {
        "weights": str(weights),
        "split": args.split,
        "decision": decision,
        "semantic": semantic_summary,
        "image_level": image_metrics,
        "lesion_level": lesion_metrics,
        "lesion_bootstrap": {
            "method": "stratified lesion resampling with replacement",
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
            "intervals": bootstrap,
        },
        "efficiency": efficiency,
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved unified evaluation to: {output_dir}")
    if args.split == "test":
        print("WARNING: the internal test split was evaluated with explicit --allow-test.")


if __name__ == "__main__":
    main()
