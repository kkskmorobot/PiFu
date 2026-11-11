
"""Train runs6: runs5 multitask YOLO26 plus one standard P3/8 CBAM.

Formal training is intended to be launched visibly from PyCharm. The runner has
no test-evaluation path and never overwrites experiments 01-05.
"""

from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "experiments"
    / "06_YOLO26s_Semantic_AuxClassifier_CBAM"
    / "train_yolo26s_sem_cls_cbam_v6.yaml"
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


def validate_selection(selection: dict) -> None:
    required = {
        "macro_f1_weight",
        "foreground_miou_weight",
        "minority_recall_weight",
        "mel_to_nv_penalty",
        "max_mel_to_nv_rate",
        "min_foreground_miou",
        "min_minority_recall",
    }
    missing = required - set(selection)
    if missing:
        raise ValueError(f"Runs6 selection configuration is missing: {sorted(missing)}")


def main() -> None:
    args = parse_args()
    from train_skin7_yolo26_multitask import apply_overrides, load_config, validate_dataset

    config, multitask = load_config(args.config)
    apply_overrides(config, args)
    counts = validate_dataset(Path(config["data"]))
    validate_selection(multitask["selection"])
    print("Runs6 controlled change: one standard CBAM on the P3/8 neck feature.")
    print("Everything else (data, split, head, loss, optimizer and augmentation) matches runs5.")
    for split, class_counts in counts.items():
        print(f"  {split}: {sum(class_counts.values())} images")
    print(f"Output: {Path(config['project']) / config['name']}")
    print(f"Pre-registered selection: {multitask['selection']}")

    if args.dry_run:
        print("Dry run complete; training was not started and test was not evaluated.")
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
            amp=False,
            project=str(PROJECT_ROOT / "experiments" / "06_YOLO26s_Semantic_AuxClassifier_CBAM" / "smoke"),
            name="cbam_loader_loss_val_smoke",
            exist_ok=True,
            save_period=-1,
        )

    from skin7_cbam import Skin7CBAMTrainer

    if args.resume:
        checkpoint = resolve(args.resume)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint}")
        config["resume"] = str(checkpoint)

    trainer = Skin7CBAMTrainer(
        overrides=config,
        image_cls_weight=float(multitask["image_cls_weight"]),
        selection=dict(multitask["selection"]),
    )
    trainer.train()
    print("Runs6 training finished. The internal test split was not evaluated.")


if __name__ == "__main__":
    main()

