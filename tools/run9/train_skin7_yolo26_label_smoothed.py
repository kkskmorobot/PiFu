"""Train Run9: fixed Run8 plus image-classification label smoothing 0.05."""

from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "experiments"
    / "09_YOLO26s_Semantic_LesionBalanced_LabelSmoothing"
    / "train_yolo26s_sem_cls_lesion_balanced_label_smoothing_v9.yaml"
)


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
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def validate_protocol(multitask: dict) -> None:
    from train_skin7_yolo26_lesion_balanced import validate_protocol as validate_run8_protocol

    validate_run8_protocol(multitask)
    regularization = multitask.get("classification_regularization")
    expected = {
        "strategy": "label_smoothing",
        "epsilon": 0.05,
        "scope": "image_classification_ce_only",
    }
    if not isinstance(regularization, dict) or dict(regularization) != expected:
        raise ValueError(f"Run9 classification_regularization must be exactly {expected}")


def main() -> None:
    args = parse_args()
    from train_skin7_yolo26_multitask import apply_overrides, load_config, validate_dataset

    config, multitask = load_config(args.config)
    config["model"] = str(resolve(Path(config["model"])))
    apply_overrides(config, args)
    counts = validate_dataset(Path(config["data"]))
    validate_protocol(multitask)
    print("Run9 controlled change: image-classification CE label smoothing epsilon=0.05.")
    print("Run8 model, sampler, class quotas, semantic loss, optimizer, augmentation and selection are unchanged.")
    for split, class_counts in counts.items():
        print(f"  {split}: {sum(class_counts.values())} images")
    print(f"Output: {Path(config['project']) / config['name']}")
    print(f"Classification regularization: {multitask['classification_regularization']}")
    print(f"Pre-registered selection: {multitask['selection']}")

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
            val=True,
            plots=False,
            amp=False,
            project=str(
                PROJECT_ROOT
                / "experiments"
                / "09_YOLO26s_Semantic_LesionBalanced_LabelSmoothing"
                / "smoke"
            ),
            name="label_smoothing_005_1pct_full_val_smoke",
            exist_ok=True,
            save_period=-1,
        )

    from skin7_label_smoothed import Skin7LabelSmoothedTrainer

    if args.resume:
        checkpoint = resolve(args.resume)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint}")
        config["resume"] = str(checkpoint)

    trainer = Skin7LabelSmoothedTrainer(
        overrides=config,
        image_cls_weight=float(multitask["image_cls_weight"]),
        selection=dict(multitask["selection"]),
        lesion_guided=dict(multitask["lesion_guided"]),
        lesion_sampling=dict(multitask["lesion_sampling"]),
        classification_regularization=dict(multitask["classification_regularization"]),
    )
    trainer.train()
    print("Run9 training finished. The internal test split was not evaluated.")


if __name__ == "__main__":
    main()
