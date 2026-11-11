"""Resume an interrupted formal Run9 job from its complete ``last.pt``.

Ultralytics 8.4.101 restores model/EMA, epoch, optimizer, scheduler and scaler.
Its EarlyStopping patience counter is not serialized and restarts on resume;
the inherited Run8 sampler is reconstructed deterministically from start_epoch.
"""

from __future__ import annotations

import argparse
import gc
import pickletools
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_NAME = "skin7_yolo26s_sem_lesion_balanced_ls005_seed42"
CHECKPOINT = (
    PROJECT_ROOT
    / "experiments"
    / "09_YOLO26s_Semantic_LesionBalanced_LabelSmoothing"
    / "results"
    / "semantic_multitask_lesion_balanced_label_smoothing"
    / RUN_NAME
    / "weights"
    / "last.pt"
)
CONFIG = (
    PROJECT_ROOT
    / "experiments"
    / "09_YOLO26s_Semantic_LesionBalanced_LabelSmoothing"
    / "train_yolo26s_sem_cls_lesion_balanced_label_smoothing_v9.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def safe_checkpoint_audit(checkpoint: Path) -> dict:
    """Inspect checkpoint structure without executing its pickle payload."""
    import torch

    unsafe_globals = torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint)
    with zipfile.ZipFile(checkpoint) as archive:
        members = archive.namelist()
        pickle_member = next((name for name in members if name.endswith("data.pkl")), None)
        if pickle_member is None:
            raise RuntimeError("Run9 checkpoint archive has no data.pkl payload.")
        operations = list(pickletools.genops(archive.read(pickle_member)))
    strings = [argument for _, argument, _ in operations if isinstance(argument, str)]
    required = {"epoch", "best_fitness", "model", "ema", "optimizer", "train_args"}
    missing = sorted(required - set(strings))
    if missing:
        raise RuntimeError(f"Run9 checkpoint is missing structural keys: {missing}")
    expected_global = "skin7_label_smoothed.Skin7LabelSmoothedMultiTaskModel"
    if expected_global not in unsafe_globals:
        raise RuntimeError(f"Run9 checkpoint does not identify {expected_global}")
    if RUN_NAME not in strings:
        raise RuntimeError(f"Checkpoint does not contain formal Run9 name {RUN_NAME!r}")

    def next_value(index: int):
        memo_ops = {"BINPUT", "LONG_BINPUT", "MEMOIZE"}
        for operation, argument, _ in operations[index + 1 :]:
            if operation.name not in memo_ops:
                return operation.name, argument
        return None, None

    epoch_index = next(index for index, (_, argument, _) in enumerate(operations) if argument == "epoch")
    epoch_opcode, epoch = next_value(epoch_index)
    if epoch_opcode not in {"BININT", "BININT1", "BININT2", "LONG", "LONG1", "LONG4"} or int(epoch) < 0:
        raise RuntimeError(f"Run9 checkpoint has an invalid structural epoch: {epoch_opcode} {epoch}")

    optimizer_occurrences = [
        index for index, (_, argument, _) in enumerate(operations) if argument == "optimizer"
    ]
    optimizer_nonempty = False
    for index in optimizer_occurrences:
        opcode, _ = next_value(index)
        nearby_strings = {
            argument
            for _, argument, _ in operations[index : index + 30]
            if isinstance(argument, str)
        }
        if opcode == "EMPTY_DICT" and "state" in nearby_strings and "param_groups" in strings:
            optimizer_nonempty = True
            break
    if not optimizer_nonempty:
        raise RuntimeError("Run9 checkpoint optimizer structure is absent or stripped.")

    data_members = sum("/data/" in name for name in members)
    if data_members == 0:
        raise RuntimeError("Run9 checkpoint contains no tensor storage members.")
    return {
        "epoch": int(epoch),
        "archive_members": len(members),
        "tensor_storage_members": data_members,
        "optimizer_structure": "nonempty",
        "model_global": expected_global,
        "formal_name": RUN_NAME,
    }


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve() if args.checkpoint.is_absolute() else (PROJECT_ROOT / args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Run9 resume checkpoint not found: {checkpoint}")

    if args.dry_run:
        audit = safe_checkpoint_audit(checkpoint)
        print(f"Safe Run9 checkpoint audit passed: {audit}")
        print("Dry run did not deserialize the pickle payload, resume training, or evaluate test.")
        return

    import torch
    from skin7_label_smoothed import Skin7LabelSmoothedMultiTaskModel

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    epoch = state.get("epoch")
    train_args = state.get("train_args") or {}
    total_epochs = int(train_args.get("epochs", 0))
    if not isinstance(epoch, int) or epoch < 0 or not state.get("optimizer"):
        raise RuntimeError("Run9 last.pt does not contain a complete resumable training state.")
    if not isinstance(state.get("ema"), Skin7LabelSmoothedMultiTaskModel):
        raise RuntimeError(f"Checkpoint is not a Run9 label-smoothed model: {type(state.get('ema')).__name__}")
    if train_args.get("name") != RUN_NAME:
        raise RuntimeError(f"Checkpoint is not the formal Run9 run: name={train_args.get('name')!r}")
    if float(getattr(state["ema"], "image_cls_label_smoothing", -1.0)) != 0.05:
        raise RuntimeError("Checkpoint does not carry Run9 label smoothing epsilon=0.05")
    if epoch + 1 >= total_epochs:
        raise RuntimeError(f"Run9 already reached {epoch + 1}/{total_epochs} epochs.")
    print(f"Resume is valid: completed epoch {epoch + 1}; sampler restarts deterministically at epoch {epoch + 2}/{total_epochs}.")
    del state
    gc.collect()
    from train_skin7_yolo26_multitask import load_config, validate_dataset
    from train_skin7_yolo26_label_smoothed import validate_protocol
    from skin7_label_smoothed import Skin7LabelSmoothedTrainer

    config, multitask = load_config(CONFIG)
    validate_dataset(Path(config["data"]))
    validate_protocol(multitask)
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
    print("Resumed Run9 training finished. Test was not evaluated.")


if __name__ == "__main__":
    main()
