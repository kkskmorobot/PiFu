"""Resume the interrupted runs5 multitask training from ``last.pt``.

Right-click this file in PyCharm and choose Run.  This is a true checkpoint
resume: Ultralytics restores the completed epoch, optimizer, LR scheduler,
scaler, EMA and global best fitness. Ultralytics 8.4.101 does not serialize the
EarlyStopping ``best_epoch`` counter, so patience counting restarts after a
resume. It does not start a new experiment or evaluate the test split.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "experiments"
    / "05_YOLO26s_Semantic_AuxClassifier"
    / "results"
    / "semantic_multitask"
    / "skin7_yolo26s_sem_auxcls_seed42"
    / "weights"
    / "last.pt"
)
CONFIG = (
    PROJECT_ROOT
    / "experiments"
    / "05_YOLO26s_Semantic_AuxClassifier"
    / "train_yolo26s_sem_cls_v5.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect resumability and exit without creating a trainer or changing files.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def inspect_checkpoint(path: Path) -> dict[str, Any]:
    """Fail early when last.pt was stripped or belongs to a different model."""
    import torch

    # Import before torch.load so the project-local pickle classes are known.
    from skin7_multitask import Skin7MultiTaskModel

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    epoch = checkpoint.get("epoch")
    optimizer = checkpoint.get("optimizer")
    ema = checkpoint.get("ema")
    train_args = checkpoint.get("train_args") or {}
    total_epochs = int(train_args.get("epochs", 0))

    if not isinstance(epoch, int) or epoch < 0:
        raise RuntimeError(f"Checkpoint has no resumable epoch state: epoch={epoch!r}")
    if not isinstance(optimizer, dict) or not optimizer:
        raise RuntimeError(
            "Checkpoint optimizer state is missing. It may be a stripped final weight; "
            "use a full last.pt/epoch*.pt checkpoint for resume."
        )
    if not isinstance(ema, Skin7MultiTaskModel):
        raise RuntimeError(
            "Checkpoint is not a runs5 Skin7MultiTaskModel EMA: "
            f"found {type(ema).__module__}.{type(ema).__name__}"
        )
    if total_epochs <= epoch + 1:
        raise RuntimeError(
            f"Checkpoint already reached its configured limit: completed={epoch + 1}, epochs={total_epochs}."
        )

    info = {
        "completed_epochs": epoch + 1,
        "next_epoch": epoch + 2,
        "total_epochs": total_epochs,
        "best_fitness": checkpoint.get("best_fitness"),
        "optimizer_present": True,
        "ema_present": True,
    }
    del checkpoint, optimizer, ema
    gc.collect()
    return info


def main() -> None:
    args = parse_args()
    checkpoint = resolve(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint}")

    info = inspect_checkpoint(checkpoint)
    print(f"Checkpoint: {checkpoint}")
    print(
        f"Resume is valid: completed epoch {info['completed_epochs']}; "
        f"next epoch {info['next_epoch']} of {info['total_epochs']}."
    )
    print(f"Optimizer/EMA present; previous best fitness={info['best_fitness']}")
    if args.dry_run:
        print("Dry run complete; training was not resumed and test was not evaluated.")
        return

    # Reuse exactly the runs5 config and custom trainer/selection rule. Passing
    # resume=<last.pt> lets Ultralytics replace ordinary launch arguments with
    # the checkpoint's original training arguments and restore runtime state.
    from train_skin7_yolo26_multitask import load_config, validate_dataset
    from skin7_multitask import Skin7MultiTaskTrainer

    config, multitask = load_config(CONFIG)
    validate_dataset(Path(config["data"]))
    config["resume"] = str(checkpoint)
    trainer = Skin7MultiTaskTrainer(
        overrides=config,
        image_cls_weight=float(multitask["image_cls_weight"]),
        selection=dict(multitask["selection"]),
    )
    trainer.train()
    print("Resumed runs5 training finished. The internal test split was not evaluated.")


if __name__ == "__main__":
    main()
