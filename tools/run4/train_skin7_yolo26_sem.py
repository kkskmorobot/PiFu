"""Train the PiFuV2 seven-disease dataset with YOLO26 semantic segmentation.

The existing YOLO polygon labels are rasterized by Ultralytics at load time. For
multi-class polygon datasets, Ultralytics appends an eighth ``background`` class.
The internal test split is never evaluated unless ``--evaluate-test`` is passed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Any

import yaml
from packaging.version import Version


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "experiments" / "04_YOLO26s_Semantic_C2PSA" / "train_yolo26s_sem_v4.yaml"
DEFAULT_EVAL_DATA = (
    PROJECT_ROOT / "experiments" / "04_YOLO26s_Semantic_C2PSA" / "data_semantic_eval_v4.yaml"
)
EXPECTED_CLASSES = {0: "akiec", 1: "bcc", 2: "bkl", 3: "df", 4: "mel", 5: "nv", 6: "vasc"}
MIN_ULTRALYTICS = Version("8.4.52")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--fraction", type=float)
    parser.add_argument("--device")
    parser.add_argument("--name")
    parser.add_argument("--project", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--weights", type=Path, help="Existing weights for --evaluate-test")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one epoch on 1%% of train data at 320 px; writes outside runs4.",
    )
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    path = resolve_path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Training configuration not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Training configuration must be a YAML mapping: {path}")
    for key in ("model", "data", "project", "name"):
        if key not in config:
            raise ValueError(f"Missing required training setting: {key}")
    if "-sem" not in str(config["model"]).lower():
        raise ValueError("This runner requires a YOLO semantic model whose name contains '-sem'.")
    config["data"] = str(resolve_path(config["data"]).resolve())
    config["project"] = str(resolve_path(config["project"]).resolve())
    return config


def read_data_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    names = data.get("names") if isinstance(data, dict) else None
    normalized = {int(key): value for key, value in names.items()} if isinstance(names, dict) else None
    if normalized != EXPECTED_CLASSES:
        raise ValueError(f"Unexpected class mapping: {normalized}; expected: {EXPECTED_CLASSES}")
    return data


def validate_semantic_eval_yaml(path: Path) -> Path:
    """Require the explicit eight-class mapping used by standalone validation."""
    path = resolve_path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Semantic evaluation YAML not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    names = data.get("names") if isinstance(data, dict) else None
    normalized = {int(key): value for key, value in names.items()} if isinstance(names, dict) else None
    expected = {**EXPECTED_CLASSES, 7: "background"}
    if data.get("nc") != 8 or data.get("bg_class_idx") != 7 or normalized != expected:
        raise ValueError(
            "Standalone semantic validation requires nc=8, bg_class_idx=7 and "
            f"names={expected}; found nc={data.get('nc')}, bg_class_idx={data.get('bg_class_idx')}, "
            f"names={normalized} in {path}"
        )
    return path


def validate_dataset(data_yaml: Path) -> dict[str, dict[int, int]]:
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")
    data = read_data_yaml(data_yaml)
    dataset_root = Path(data["path"])
    if not dataset_root.is_absolute():
        dataset_root = (data_yaml.parent / dataset_root).resolve()

    # In semantic mode a root-level masks/ directory switches the loader to PNG-mask mode.
    if "masks_dir" not in data and (dataset_root / "masks").exists():
        raise ValueError(
            f"Unexpected {dataset_root / 'masks'} directory: it would disable polygon-to-semantic conversion."
        )

    counts: dict[str, dict[int, int]] = {}
    for split in ("train", "val", "test"):
        image_dir = dataset_root / data[split]
        label_dir = dataset_root / "labels" / split
        images = {p.stem for p in image_dir.glob("*.jpg")}
        labels = {p.stem for p in label_dir.glob("*.txt")}
        if not images or images != labels:
            raise ValueError(
                f"{split} image/label mismatch: images={len(images)}, labels={len(labels)}, "
                f"missing={len(images - labels)}, orphan={len(labels - images)}"
            )

        class_counts: Counter[int] = Counter()
        for label_path in label_dir.glob("*.txt"):
            parts = label_path.read_text(encoding="utf-8").strip().split()
            if not parts:
                raise ValueError(f"Empty label: {label_path}")
            class_id = int(parts[0])
            coords = [float(v) for v in parts[1:]]
            if class_id not in EXPECTED_CLASSES:
                raise ValueError(f"Invalid class ID in {label_path}: {class_id}")
            if len(coords) < 6 or len(coords) % 2 or any(v < 0.0 or v > 1.0 for v in coords):
                raise ValueError(f"Invalid polygon in {label_path}")
            class_counts[class_id] += 1
        counts[split] = dict(sorted(class_counts.items()))
    return counts


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    for key in ("batch", "imgsz", "epochs", "workers", "fraction", "device", "name"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if args.project is not None:
        config["project"] = str(resolve_path(args.project).resolve())


def print_summary(config: dict[str, Any], counts: dict[str, dict[int, int]]) -> None:
    installed = Version(version("ultralytics"))
    if installed < MIN_ULTRALYTICS:
        raise RuntimeError(f"Ultralytics {MIN_ULTRALYTICS}+ is required; found {installed}")
    print(f"Ultralytics {installed}; semantic polygon support is available.")
    print("Dataset validation passed. Ultralytics will append class 7 = background.")
    for split, class_counts in counts.items():
        named = {EXPECTED_CLASSES[k]: v for k, v in class_counts.items()}
        print(f"  {split}: total={sum(class_counts.values())}, classes={named}")
    print("\nTraining settings:")
    for key in (
        "model", "data", "epochs", "patience", "batch", "imgsz", "device", "project", "name",
        "optimizer", "lr0", "cls_pw", "mosaic", "scale", "hsv_s", "hsv_v",
    ):
        print(f"  {key}: {config.get(key)}")


def evaluate_test(weights: Path, config: dict[str, Any], eval_data: Path) -> None:
    from ultralytics import YOLO

    weights = resolve_path(weights).resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")
    model = YOLO(str(weights), task="semantic")
    model.val(
        data=str(eval_data), split="test", imgsz=config["imgsz"], batch=config["batch"],
        device=config["device"], plots=True, project=config["project"], name=f"{config['name']}_test",
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    apply_overrides(config, args)
    counts = validate_dataset(Path(config["data"]))
    eval_data = validate_semantic_eval_yaml(DEFAULT_EVAL_DATA)

    if args.smoke_test:
        config.update(
            epochs=1,
            patience=0,
            batch=min(int(config.get("batch", 4)), 4),
            imgsz=min(int(config.get("imgsz", 320)), 320),
            workers=0,
            fraction=0.01,
            val=False,
            plots=False,
            save=False,
            project=str(PROJECT_ROOT / "_smoke"),
            name="yolo26s_sem_loader_smoke",
            exist_ok=True,
        )

    print_summary(config, counts)
    if args.dry_run:
        print(f"Standalone semantic evaluation YAML: {eval_data}")
        print("\nDry run complete; training was not started and the test split was untouched.")
        return

    if args.weights:
        if not args.evaluate_test:
            raise ValueError("--weights requires --evaluate-test")
        evaluate_test(args.weights, config, eval_data)
        return

    from ultralytics import YOLO

    if args.resume:
        checkpoint = resolve_path(args.resume).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint}")
        model = YOLO(str(checkpoint), task="semantic")
        model.train(resume=True)
    else:
        model_spec = str(config.pop("model"))
        model = YOLO(model_spec, task="semantic")
        model.train(**config)

    if args.evaluate_test:
        evaluate_test(Path(model.trainer.best), config, eval_data)
    else:
        print("\nTraining finished. The internal test split was not touched.")
        print("Use --weights <best.pt> --evaluate-test only after the final experiment is selected.")


if __name__ == "__main__":
    main()
