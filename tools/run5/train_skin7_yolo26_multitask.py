"""Train runs5: YOLO26 semantic segmentation plus an auxiliary image classifier.

Run this file from PyCharm for the formal experiment.  The default configuration
uses only train/val; there is intentionally no test-evaluation option here.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "experiments"
    / "05_YOLO26s_Semantic_AuxClassifier"
    / "train_yolo26s_sem_cls_v5.yaml"
)
EXPECTED_CLASSES = {0: "akiec", 1: "bcc", 2: "bkl", 3: "df", 4: "mel", 5: "nv", 6: "vasc"}


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
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one epoch on 1%% of train data at 320 px in experiments/05 smoke output.",
    )
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = resolve_path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Training configuration not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Training configuration must be a YAML mapping: {path}")
    multitask = raw.pop("multitask", None)
    if not isinstance(multitask, dict):
        raise ValueError("The runs5 YAML requires a 'multitask' mapping.")
    for key in ("model", "data", "project", "name"):
        if key not in raw:
            raise ValueError(f"Missing required training setting: {key}")
    raw["data"] = str(resolve_path(raw["data"]).resolve())
    raw["project"] = str(resolve_path(raw["project"]).resolve())
    return raw, multitask


def validate_dataset(data_yaml: Path) -> dict[str, dict[int, int]]:
    with data_yaml.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    names = data.get("names") if isinstance(data, dict) else None
    normalized = {int(key): value for key, value in names.items()} if isinstance(names, dict) else None
    if normalized != EXPECTED_CLASSES:
        raise ValueError(f"Unexpected class mapping: {normalized}; expected: {EXPECTED_CLASSES}")
    root = Path(data["path"])
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    counts: dict[str, dict[int, int]] = {}
    for split in ("train", "val", "test"):
        image_dir = root / data[split]
        label_dir = root / "labels" / split
        images = {path.stem for path in image_dir.glob("*.jpg")}
        labels = {path.stem for path in label_dir.glob("*.txt")}
        if not images or images != labels:
            raise ValueError(
                f"{split} image/label mismatch: images={len(images)}, labels={len(labels)}, "
                f"missing={len(images - labels)}, orphan={len(labels - images)}"
            )
        class_counts: Counter[int] = Counter()
        for label_path in label_dir.glob("*.txt"):
            tokens = label_path.read_text(encoding="utf-8").split()
            if not tokens:
                raise ValueError(f"Empty label: {label_path}")
            class_id = int(tokens[0])
            if class_id not in EXPECTED_CLASSES:
                raise ValueError(f"Invalid class ID in {label_path}: {class_id}")
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


def main() -> None:
    args = parse_args()
    config, multitask = load_config(args.config)
    apply_overrides(config, args)
    counts = validate_dataset(Path(config["data"]))
    print("Dataset validation passed; train/val/test were read only for structural checks.")
    for split, class_counts in counts.items():
        print(f"  {split}: {sum(class_counts.values())} images")
    print(f"Runs5 output: {Path(config['project']) / config['name']}")
    print(f"Image classification loss weight: {multitask['image_cls_weight']}")
    print(f"Predeclared checkpoint selection: {multitask['selection']}")

    if args.dry_run:
        print("Dry run complete; training was not started and test inference was not run.")
        return

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
            project=str(PROJECT_ROOT / "experiments" / "05_YOLO26s_Semantic_AuxClassifier" / "smoke"),
            name="loader_and_loss_smoke",
            exist_ok=True,
        )

    # Import from tools explicitly so checkpoint pickle paths remain stable.
    from skin7_multitask import Skin7MultiTaskTrainer

    if args.resume:
        checkpoint = resolve_path(args.resume).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint}")
        config["resume"] = str(checkpoint)

    trainer = Skin7MultiTaskTrainer(
        overrides=config,
        image_cls_weight=float(multitask["image_cls_weight"]),
        selection=dict(multitask["selection"]),
    )
    trainer.train()
    print("Training finished. The internal test split was not evaluated.")


if __name__ == "__main__":
    main()

