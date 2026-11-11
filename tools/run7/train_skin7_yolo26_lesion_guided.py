"""Train runs7: lesion-guided P3/P4/P5 dual attention on the runs5 base.

Formal training is intended to be launched visibly from PyCharm.  This runner
has no test-evaluation path and never overwrites experiments 01-06.
"""

from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "experiments"
    / "07_YOLO26s_Semantic_LesionGuidedDualAttention"
    / "train_yolo26s_sem_cls_lesion_guided_v7.yaml"
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
    selection_required = {
        "macro_f1_weight",
        "foreground_miou_weight",
        "minority_recall_weight",
        "mel_to_nv_penalty",
        "max_mel_to_nv_rate",
        "min_foreground_miou",
        "min_minority_recall",
    }
    selection = multitask.get("selection")
    if not isinstance(selection, dict) or selection_required - set(selection):
        missing = selection_required - set(selection or {})
        raise ValueError(f"Runs7 selection configuration is missing: {sorted(missing)}")
    lesion = multitask.get("lesion_guided")
    required = {"p5_layer", "eca_kernel_size", "mask_guided_mix", "detach_mask"}
    if not isinstance(lesion, dict) or required - set(lesion):
        missing = required - set(lesion or {})
        raise ValueError(f"Runs7 lesion-guided configuration is missing: {sorted(missing)}")
    if int(lesion["p5_layer"]) != 10:
        raise ValueError("Runs7 is preregistered for YOLO26-sem P5 layer 10.")
    if bool(lesion["detach_mask"]) is not True:
        raise ValueError("Runs7 must detach the semantic mask; consistency loss is reserved for runs8.")


def main() -> None:
    args = parse_args()
    from train_skin7_yolo26_multitask import apply_overrides, load_config, validate_dataset

    config, multitask = load_config(args.config)
    # Ultralytics resolves a relative model path against the process working
    # directory. Pin it to PiFuV2 so PyCharm and terminal smoke runs use the
    # exact same pretrained checkpoint without downloading a duplicate.
    config["model"] = str(resolve(Path(config["model"])))
    apply_overrides(config, args)
    counts = validate_dataset(Path(config["data"]))
    validate_protocol(multitask)
    print("Runs7 controlled change: classifier-only lesion-guided P3/P4/P5 pooling plus ECA-style channel gates.")
    print("The semantic path, data, loss, sampler, optimizer and augmentation match runs5.")
    for split, class_counts in counts.items():
        print(f"  {split}: {sum(class_counts.values())} images")
    print(f"Output: {Path(config['project']) / config['name']}")
    print(f"Lesion-guided module: {multitask['lesion_guided']}")
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
            val=False,
            plots=False,
            amp=False,
            project=str(
                PROJECT_ROOT / "experiments" / "07_YOLO26s_Semantic_LesionGuidedDualAttention" / "smoke"
            ),
            name="lesion_guided_loader_loss_val_smoke",
            exist_ok=True,
            save_period=-1,
        )

    from skin7_lesion_guided import Skin7LesionGuidedTrainer

    if args.resume:
        checkpoint = resolve(args.resume)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint}")
        config["resume"] = str(checkpoint)

    trainer = Skin7LesionGuidedTrainer(
        overrides=config,
        image_cls_weight=float(multitask["image_cls_weight"]),
        selection=dict(multitask["selection"]),
        lesion_guided=dict(multitask["lesion_guided"]),
    )
    trainer.train()
    print("Runs7 training finished. The internal test split was not evaluated.")


if __name__ == "__main__":
    main()
