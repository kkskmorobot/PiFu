"""Resume an interrupted formal Run8 job from its complete ``last.pt``.

Ultralytics 8.4.101 restores model/EMA, epoch, optimizer, scheduler and scaler.
Its EarlyStopping patience counter is not serialized and restarts on resume;
the Run8 sampler itself is reconstructed deterministically from start_epoch.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_NAME = "skin7_yolo26s_sem_lesion_balanced_seed42"
CHECKPOINT = (
    PROJECT_ROOT
    / "experiments"
    / "08_YOLO26s_Semantic_LesionBalancedSampler"
    / "results"
    / "semantic_multitask_lesion_balanced"
    / RUN_NAME
    / "weights"
    / "last.pt"
)
CONFIG = (
    PROJECT_ROOT
    / "experiments"
    / "08_YOLO26s_Semantic_LesionBalancedSampler"
    / "train_yolo26s_sem_cls_lesion_balanced_v8.yaml"
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
        raise FileNotFoundError(f"Run8 resume checkpoint not found: {checkpoint}")

    import torch
    from skin7_lesion_guided import Skin7LesionGuidedMultiTaskModel

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    epoch = state.get("epoch")
    train_args = state.get("train_args") or {}
    total_epochs = int(train_args.get("epochs", 0))
    if not isinstance(epoch, int) or epoch < 0 or not state.get("optimizer"):
        raise RuntimeError("Run8 last.pt does not contain a complete resumable training state.")
    if not isinstance(state.get("ema"), Skin7LesionGuidedMultiTaskModel):
        raise RuntimeError(f"Checkpoint is not the fixed Run7 architecture: {type(state.get('ema')).__name__}")
    if train_args.get("name") != RUN_NAME:
        raise RuntimeError(f"Checkpoint is not the formal Run8 run: name={train_args.get('name')!r}")
    if epoch + 1 >= total_epochs:
        raise RuntimeError(f"Run8 already reached {epoch + 1}/{total_epochs} epochs.")
    print(f"Resume is valid: completed epoch {epoch + 1}; sampler restarts deterministically at epoch {epoch + 2}/{total_epochs}.")
    del state
    gc.collect()
    if args.dry_run:
        print("Dry run complete; training was not resumed and test was not evaluated.")
        return

    from train_skin7_yolo26_multitask import load_config, validate_dataset
    from train_skin7_yolo26_lesion_balanced import validate_protocol
    from skin7_lesion_balanced import Skin7LesionBalancedTrainer

    config, multitask = load_config(CONFIG)
    validate_dataset(Path(config["data"]))
    validate_protocol(multitask)
    config["resume"] = str(checkpoint)
    trainer = Skin7LesionBalancedTrainer(
        overrides=config,
        image_cls_weight=float(multitask["image_cls_weight"]),
        selection=dict(multitask["selection"]),
        lesion_guided=dict(multitask["lesion_guided"]),
        lesion_sampling=dict(multitask["lesion_sampling"]),
    )
    trainer.train()
    print("Resumed Run8 training finished. Test was not evaluated.")


if __name__ == "__main__":
    main()
