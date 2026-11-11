"""Resume an interrupted runs7 training job from its formal ``last.pt``.

Ultralytics restores epoch, optimizer, scheduler, scaler, EMA and global best
fitness. In 8.4.101 the EarlyStopping patience history is not serialized, so
its counter restarts after resume. Test is never evaluated here.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = (
    PROJECT_ROOT
    / "experiments"
    / "07_YOLO26s_Semantic_LesionGuidedDualAttention"
    / "results"
    / "semantic_multitask_lesion_guided"
    / "skin7_yolo26s_sem_lesion_guided_seed42"
    / "weights"
    / "last.pt"
)
CONFIG = (
    PROJECT_ROOT
    / "experiments"
    / "07_YOLO26s_Semantic_LesionGuidedDualAttention"
    / "train_yolo26s_sem_cls_lesion_guided_v7.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve() if args.checkpoint.is_absolute() else (PROJECT_ROOT / args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Runs7 resume checkpoint not found: {checkpoint}")

    import torch
    from skin7_lesion_guided import Skin7LesionGuidedMultiTaskModel

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    epoch = state.get("epoch")
    total_epochs = int((state.get("train_args") or {}).get("epochs", 0))
    if not isinstance(epoch, int) or epoch < 0 or not state.get("optimizer"):
        raise RuntimeError("Runs7 last.pt does not contain a complete resumable training state.")
    if not isinstance(state.get("ema"), Skin7LesionGuidedMultiTaskModel):
        raise RuntimeError(f"Checkpoint is not a runs7 lesion-guided model: {type(state.get('ema')).__name__}")
    if epoch + 1 >= total_epochs:
        raise RuntimeError(f"Runs7 already reached {epoch + 1}/{total_epochs} epochs.")
    print(f"Resume is valid: completed epoch {epoch + 1}; next epoch {epoch + 2}/{total_epochs}.")
    del state
    gc.collect()
    if args.dry_run:
        print("Dry run complete; training was not resumed and test was not evaluated.")
        return

    from train_skin7_yolo26_multitask import load_config, validate_dataset
    from skin7_lesion_guided import Skin7LesionGuidedTrainer

    config, multitask = load_config(CONFIG)
    validate_dataset(Path(config["data"]))
    config["resume"] = str(checkpoint)
    trainer = Skin7LesionGuidedTrainer(
        overrides=config,
        image_cls_weight=float(multitask["image_cls_weight"]),
        selection=dict(multitask["selection"]),
        lesion_guided=dict(multitask["lesion_guided"]),
    )
    trainer.train()
    print("Resumed runs7 training finished. Test was not evaluated.")


if __name__ == "__main__":
    main()
