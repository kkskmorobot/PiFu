"""Resume interrupted runs6 CBAM training from its formal ``last.pt``.

Ultralytics restores epoch, optimizer, scheduler, scaler, EMA and global best
fitness. In 8.4.101 the EarlyStopping best-epoch counter is not serialized, so
patience counting restarts after resume. Test is never evaluated here.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = (
    PROJECT_ROOT
    / "experiments"
    / "06_YOLO26s_Semantic_AuxClassifier_CBAM"
    / "results"
    / "semantic_multitask_cbam"
    / "skin7_yolo26s_sem_auxcls_cbam_seed42"
    / "weights"
    / "last.pt"
)
CONFIG = (
    PROJECT_ROOT
    / "experiments"
    / "06_YOLO26s_Semantic_AuxClassifier_CBAM"
    / "train_yolo26s_sem_cls_cbam_v6.yaml"
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
        raise FileNotFoundError(f"Runs6 resume checkpoint not found: {checkpoint}")

    import torch
    from skin7_cbam import Skin7CBAMMultiTaskModel

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    epoch = state.get("epoch")
    total_epochs = int((state.get("train_args") or {}).get("epochs", 0))
    if not isinstance(epoch, int) or epoch < 0 or not state.get("optimizer"):
        raise RuntimeError("Runs6 last.pt does not contain a complete resumable training state.")
    if not isinstance(state.get("ema"), Skin7CBAMMultiTaskModel):
        raise RuntimeError(f"Checkpoint is not a runs6 CBAM model: {type(state.get('ema')).__name__}")
    if epoch + 1 >= total_epochs:
        raise RuntimeError(f"Runs6 already reached {epoch + 1}/{total_epochs} epochs.")
    print(f"Resume is valid: completed epoch {epoch + 1}; next epoch {epoch + 2}/{total_epochs}.")
    del state
    gc.collect()
    if args.dry_run:
        print("Dry run complete; training was not resumed and test was not evaluated.")
        return

    from train_skin7_yolo26_multitask import load_config, validate_dataset
    from skin7_cbam import Skin7CBAMTrainer

    config, multitask = load_config(CONFIG)
    validate_dataset(Path(config["data"]))
    config["resume"] = str(checkpoint)
    trainer = Skin7CBAMTrainer(
        overrides=config,
        image_cls_weight=float(multitask["image_cls_weight"]),
        selection=dict(multitask["selection"]),
    )
    trainer.train()
    print("Resumed runs6 training finished. Test was not evaluated.")


if __name__ == "__main__":
    main()

