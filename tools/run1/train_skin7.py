"""Train and optionally evaluate the PiFuV2 seven-class YOLO segmentation model.

Examples:
    python train_skin7.py --dry-run
    python train_skin7.py
    python train_skin7.py --batch 8 --device 0 --name skin7_debug
    python train_skin7.py --resume runs/segment/skin7_yolo11s_baseline/weights/last.pt
    python train_skin7.py --evaluate-test --weights runs/segment/skin7_yolo11s_baseline/weights/best.pt
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "experiments" / "03_YOLO11s_V3_NV3200" / "train_yolo11s_seg_v3.yaml"
EXPECTED_CLASSES = {0: "akiec", 1: "bcc", 2: "bkl", 3: "df", 4: "mel", 5: "nv", 6: "vasc"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--batch", type=int, default=None, help="Override YAML batch size")
    parser.add_argument("--imgsz", type=int, default=None, help="Override YAML image size")
    parser.add_argument("--device", default=None, help="Override device, e.g. 0, 0,1, or cpu")
    parser.add_argument("--name", default=None, help="Override experiment name")
    parser.add_argument("--resume", type=Path, default=None, help="Resume from a last.pt checkpoint")
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Evaluate existing weights; used with --evaluate-test and skips training",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate the untouched internal test split. Use only for a final selected model.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and print settings without training")
    return parser.parse_args()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Training configuration not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Training configuration must contain a YAML mapping: {path}")
    for required in ("model", "data", "project", "name"):
        if required not in config:
            raise ValueError(f"Missing required training setting: {required}")
    config["data"] = str(resolve_project_path(config["data"]).resolve())
    config["project"] = str(resolve_project_path(config["project"]).resolve())
    return config


def read_data_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("Dataset YAML must contain a mapping")
    names = data.get("names")
    normalized = {int(key): value for key, value in names.items()} if isinstance(names, dict) else None
    if normalized != EXPECTED_CLASSES:
        raise ValueError(f"Unexpected class mapping: {normalized}; expected: {EXPECTED_CLASSES}")
    return data


def validate_dataset(data_yaml: Path) -> dict[str, dict[int, int]]:
    data = read_data_yaml(data_yaml)
    dataset_root = Path(data["path"])
    if not dataset_root.is_absolute():
        dataset_root = data_yaml.parent / dataset_root
    counts: dict[str, dict[int, int]] = {}

    for split in ("train", "val", "test"):
        image_dir = dataset_root / data[split]
        # The generated layout uses images/<split> and labels/<split>.
        label_dir = dataset_root / "labels" / split
        images = {path.stem for path in image_dir.glob("*.jpg")}
        labels = {path.stem for path in label_dir.glob("*.txt")}
        if not images:
            raise ValueError(f"No images found for {split}: {image_dir}")
        if images != labels:
            raise ValueError(
                f"{split} image/label mismatch: missing={len(images - labels)}, orphan={len(labels - images)}"
            )

        class_counts: Counter[int] = Counter()
        for label_path in label_dir.glob("*.txt"):
            text = label_path.read_text(encoding="utf-8").strip()
            if not text:
                raise ValueError(f"Empty label: {label_path}")
            parts = text.split()
            class_id = int(parts[0])
            coords = [float(value) for value in parts[1:]]
            if class_id not in EXPECTED_CLASSES:
                raise ValueError(f"Invalid class ID in {label_path}: {class_id}")
            if len(coords) < 6 or len(coords) % 2:
                raise ValueError(f"Invalid polygon length in {label_path}: {len(coords)}")
            if any(value < 0.0 or value > 1.0 for value in coords):
                raise ValueError(f"Out-of-range polygon coordinate in {label_path}")
            class_counts[class_id] += 1
        counts[split] = dict(sorted(class_counts.items()))
    return counts


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    for key in ("batch", "imgsz", "device", "name"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value


def print_summary(config: dict[str, Any], counts: dict[str, dict[int, int]]) -> None:
    print("Dataset validation passed.")
    for split, class_counts in counts.items():
        named = {EXPECTED_CLASSES[key]: value for key, value in class_counts.items()}
        print(f"  {split}: total={sum(class_counts.values())}, classes={named}")
    print("\nTraining settings:")
    for key in ("model", "data", "epochs", "patience", "batch", "imgsz", "device", "project", "name"):
        print(f"  {key}: {config.get(key)}")


def evaluate_test(weights: Path, config: dict[str, Any]) -> None:
    from ultralytics import YOLO

    weights = weights.resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")
    model = YOLO(str(weights))
    model.val(
        data=config["data"],
        split="test",
        imgsz=config["imgsz"],
        batch=config["batch"],
        device=config["device"],
        plots=True,
        project=config["project"],
        name=f"{config['name']}_test",
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    apply_overrides(config, args)
    counts = validate_dataset(Path(config["data"]))
    print_summary(config, counts)

    if args.dry_run:
        print("\nDry run complete; training was not started.")
        return

    if args.weights:
        if not args.evaluate_test:
            raise ValueError("--weights requires --evaluate-test")
        evaluate_test(resolve_project_path(args.weights), config)
        return

    from ultralytics import YOLO

    if args.resume:
        checkpoint = resolve_project_path(args.resume).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint}")
        model = YOLO(str(checkpoint))
        model.train(resume=True)
    else:
        model_spec = str(config.pop("model"))
        model = YOLO(model_spec)
        model.train(**config)

    if args.evaluate_test:
        best = Path(model.trainer.best)
        evaluate_test(best, config)
    else:
        print("\nTraining finished. The internal test split was not touched.")
        print("After selecting the final experiment, run with --weights <best.pt> --evaluate-test.")


if __name__ == "__main__":
    main()
