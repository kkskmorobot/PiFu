"""Train Run8: fixed Run7 model plus within-class lesion-balanced sampling.

Formal training is intended to be launched visibly from PyCharm. This runner
has no test-evaluation path and never overwrites experiments 01-07.
"""

from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "experiments"
    / "08_YOLO26s_Semantic_LesionBalancedSampler"
    / "train_yolo26s_sem_cls_lesion_balanced_v8.yaml"
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
        raise ValueError(f"Run8 selection configuration is missing: {sorted(selection_required - set(selection or {}))}")
    lesion = multitask.get("lesion_guided")
    lesion_required = {"p5_layer", "eca_kernel_size", "mask_guided_mix", "detach_mask"}
    if not isinstance(lesion, dict) or lesion_required - set(lesion):
        raise ValueError(f"Run8 fixed Run7 model configuration is missing: {sorted(lesion_required - set(lesion or {}))}")
    if dict(lesion) != {"p5_layer": 10, "eca_kernel_size": 5, "mask_guided_mix": 0.5, "detach_mask": True}:
        raise ValueError("Run8 must preserve the preregistered Run7 lesion-guided architecture exactly.")
    sampling = multitask.get("lesion_sampling")
    if not isinstance(sampling, dict) or sampling.get("enabled") is not True or not sampling.get("metadata"):
        raise ValueError("Run8 requires lesion_sampling.enabled=true and a train metadata path.")
    if sampling.get("strategy") != "fixed_class_quota_uniform_lesion_then_uniform_image":
        raise ValueError("Run8 must preserve class quotas and balance lesions only within each class.")


def main() -> None:
    args = parse_args()
    from train_skin7_yolo26_multitask import apply_overrides, load_config, validate_dataset

    config, multitask = load_config(args.config)
    config["model"] = str(resolve(Path(config["model"])))
    apply_overrides(config, args)
    counts = validate_dataset(Path(config["data"]))
    validate_protocol(multitask)
    print("Run8 controlled change: lesion-balanced sampling within each fixed class quota.")
    print("Each class keeps its original image count; lesion and then image variant are selected uniformly.")
    print("Run7 model, semantic path, loss, optimizer, augmentation, epoch size and validation are unchanged.")
    for split, class_counts in counts.items():
        print(f"  {split}: {sum(class_counts.values())} images")
    print(f"Output: {Path(config['project']) / config['name']}")
    print(f"Lesion sampling: {multitask['lesion_sampling']}")
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
            project=str(PROJECT_ROOT / "experiments" / "08_YOLO26s_Semantic_LesionBalancedSampler" / "smoke"),
            name="lesion_balanced_1pct_full_val_smoke",
            exist_ok=True,
            save_period=-1,
        )

    from skin7_lesion_balanced import Skin7LesionBalancedTrainer

    if args.resume:
        checkpoint = resolve(args.resume)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint}")
        config["resume"] = str(checkpoint)

    trainer = Skin7LesionBalancedTrainer(
        overrides=config,
        image_cls_weight=float(multitask["image_cls_weight"]),
        selection=dict(multitask["selection"]),
        lesion_guided=dict(multitask["lesion_guided"]),
        lesion_sampling=dict(multitask["lesion_sampling"]),
    )
    trainer.train()
    print("Run8 training finished. The internal test split was not evaluated.")


if __name__ == "__main__":
    main()
