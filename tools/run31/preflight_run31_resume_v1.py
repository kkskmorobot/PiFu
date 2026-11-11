#!/usr/bin/env python3
"""End-to-end interruption/resume preflight for the Run31 delayed-stop trainer.

This is not a formal Run31 experiment. It creates a tiny synthetic train/val dataset,
never defines or reads a test split, and compares a continuous two-epoch run
with a run interrupted at the first completed epoch and resumed from the newest
complete checkpoint bundle.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from imaplusplus_partial_multitask_plain_mel_nv_aux_head_seed43_v1 import (
    MELNVAuxHeadDelayedUtilityGatedTrainer,
    RUN31_MEL_NV_AUX_PARAMETERS,
    RUN31_MEL_NV_AUX_WEIGHT,
)
from imaplusplus_partial_multitask_v1 import PartialLabelSemanticDataset
from multitask_checkpoint_contract import CheckpointBundleManager, stable_fingerprint


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "yolo26m-sem.pt"
TAXONOMY = PROJECT_ROOT / "tools/classification_taxonomy_v1.yaml"


class PreflightTracingTrainer(MELNVAuxHeadDelayedUtilityGatedTrainer):
    """Record synthetic train sample order without changing batch contents."""

    def __init__(self, *args, **kwargs) -> None:
        self.preflight_batch_trace: list[dict[str, Any]] = []
        super().__init__(*args, **kwargs)

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        self.preflight_batch_trace.append(
            {"epoch": int(self.epoch), "isic_ids": [str(value) for value in batch.get("isic_id", [])]}
        )
        return super().preprocess_batch(batch)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or CUDA index such as 0")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--imgsz", type=int, default=64)
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true", help="Exercise CUDA GradScaler/autocast; CUDA only.")
    parser.add_argument(
        "--output-root",
        type=Path,
        help="New audit directory. Defaults to preflight/run31_checkpoint_resume_v1/<timestamp>.",
    )
    parser.add_argument("--atol", type=float, default=None, help="Model/optimizer tensor absolute tolerance.")
    return parser.parse_args()


def resolve_device(value: str) -> str:
    normalized = value.strip().lower()
    if normalized == "cuda":
        normalized = "0"
    if normalized != "cpu" and not normalized.isdigit():
        raise ValueError("--device must be cpu, cuda, or a CUDA index such as 0")
    if normalized != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA preflight requested but torch.cuda.is_available() is false")
    return normalized


def new_output_root(value: Path | None) -> Path:
    if value is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        value = PROJECT_ROOT / "preflight" / "run31_checkpoint_resume_v1" / stamp
    root = value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()
    experiments = (PROJECT_ROOT / "experiments").resolve()
    protected = (
        (PROJECT_ROOT / "Data" / "processed_imaplusplus_binary_v1").resolve(),
        (PROJECT_ROOT / "Data" / "processed_isic2019_classification_incremental_v1").resolve(),
    )
    if root == experiments or experiments in root.parents:
        raise ValueError("Preflight output must not be placed under experiments")
    if any(root == item or item in root.parents for item in protected):
        raise ValueError("Preflight output must not be placed inside a protected dataset")
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite existing preflight directory: {root}")
    root.mkdir(parents=True)
    return root


def synthetic_image(index: int, size: int) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    image = np.empty((size, size, 3), dtype=np.uint8)
    image[..., 0] = (xx * (index + 3) + 17 * index) % 256
    image[..., 1] = (yy * (index + 5) + 29) % 256
    image[..., 2] = ((xx + yy) * (index + 1) + 43) % 256
    return image


def synthetic_mask(index: int, size: int, has_segmentation: bool) -> np.ndarray:
    if not has_segmentation:
        return np.full((size, size), 255, dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)
    margin = size // 4 + index % max(size // 16, 1)
    mask[margin : size - margin, margin : size - margin] = 1
    return mask


def build_synthetic_dataset(root: Path, size: int) -> tuple[Path, Path]:
    data_root = root / "synthetic_data"
    rows: list[dict[str, Any]] = []
    roles = (
        (True, False, -1, ""),
        (False, True, 5, "nv"),
        (True, True, 4, "mel"),
    )
    for split, repeats in (("train", 2), ("val", 1)):
        for folder in (data_root / "images" / split, data_root / "masks" / split):
            folder.mkdir(parents=True, exist_ok=True)
        for repeat in range(repeats):
            for role_index, (has_seg, has_cls, target, class_name) in enumerate(roles):
                index = repeat * len(roles) + role_index + (0 if split == "train" else 100)
                sample_id = f"PREFLIGHT_{split.upper()}_{index:03d}"
                image_rel = Path("images") / split / f"{sample_id}.png"
                mask_rel = Path("masks") / split / f"{sample_id}.png"
                Image.fromarray(synthetic_image(index, size)).save(data_root / image_rel)
                Image.fromarray(synthetic_mask(index, size, has_seg)).save(data_root / mask_rel)
                rows.append(
                    {
                        "isic_id": sample_id,
                        "split": split,
                        "dataset_role": "synthetic_preflight",
                        "image_path": image_rel.as_posix(),
                        "mask_path": mask_rel.as_posix(),
                        "has_segmentation": int(has_seg),
                        "has_classification": int(has_cls),
                        "classification_target": target,
                        "class_name": class_name,
                        "group_id": sample_id,
                    }
                )
    manifest = data_root / "partial_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    data_yaml = data_root / "multitask_data.yaml"
    data_yaml.write_text(
        "\n".join(
            (
                f"path: {data_root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: null",
                "masks_dir: masks",
                "nc: 1",
                "names:",
                "  0: lesion",
                f"partial_label_manifest: {manifest.as_posix()}",
                f"classification_taxonomy: {TAXONOMY.as_posix()}",
                "",
            )
        ),
        encoding="utf-8",
    )
    return data_yaml, manifest


def prewarm_dataset_cache(data_yaml: Path, manifest: Path, imgsz: int, batch: int) -> None:
    """Make both comparison paths start from the same already-built cache state."""
    from ultralytics.cfg import get_cfg
    from ultralytics.data.utils import check_det_dataset

    data = check_det_dataset(str(data_yaml))
    cfg = get_cfg(overrides={"task": "semantic", "mode": "val", "imgsz": imgsz, "batch": batch})
    for name in ("mosaic", "mixup", "cutmix", "copy_paste"):
        setattr(cfg, name, 0.0)
    for split in ("train", "val"):
        PartialLabelSemanticDataset(
            img_path=data[split],
            imgsz=imgsz,
            batch_size=batch,
            augment=False,
            hyp=cfg,
            rect=False,
            cache=False,
            single_cls=False,
            stride=32,
            pad=0.0,
            prefix=f"prewarm-{split}: ",
            task="semantic",
            classes=None,
            data=data,
            fraction=1.0,
            partial_label_manifest=manifest,
            expected_split=split,
        )


def checkpoint_fingerprint(weights: Path, data_yaml: Path, manifest: Path, args: argparse.Namespace) -> str:
    config = {
        "purpose": "run31_plain_mel_nv_aux_seed43_preflight_only",
        "model_contract": "binary_semantic_plus_plain_p3_p4_image_head_and_training_only_mel_nv_aux_v1",
        "mel_nv_aux_weight": RUN31_MEL_NV_AUX_WEIGHT,
        "mel_nv_aux_classes": ["nv", "mel"],
        "mel_nv_aux_parameters": RUN31_MEL_NV_AUX_PARAMETERS,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "seed": args.seed,
        "optimizer": "SGD",
        "amp": bool(args.amp),
        "workers": args.workers,
        "augmentations": "all_disabled",
        "epoch_rng_policy": "seed_plus_rank_plus_epoch_times_1000003_v1",
        "test_split": "absent",
    }
    return stable_fingerprint(config, [weights, data_yaml, manifest, TAXONOMY])


def trainer_overrides(
    *,
    weights: Path,
    data_yaml: Path,
    save_parent: Path,
    name: str,
    device: str,
    args: argparse.Namespace,
    resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    overrides: dict[str, Any] = {
        "task": "semantic",
        "mode": "train",
        "model": str(weights),
        "data": str(data_yaml),
        "project": str(save_parent),
        "name": name,
        "exist_ok": True,
        "epochs": args.epochs,
        "patience": 100,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": device,
        "workers": args.workers,
        "cache": False,
        "save": True,
        "save_period": 1,
        "val": True,
        "plots": False,
        "verbose": False,
        "amp": bool(args.amp),
        "deterministic": True,
        "seed": args.seed,
        "optimizer": "SGD",
        "lr0": 0.001,
        "lrf": 0.1,
        "momentum": 0.9,
        "weight_decay": 0.0,
        "warmup_epochs": 0.0,
        "warmup_momentum": 0.0,
        "warmup_bias_lr": 0.0,
        "nbs": args.batch,
        "close_mosaic": 0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "bgr": 0.0,
        "rect": False,
        "fraction": 1.0,
    }
    if resume_checkpoint is not None:
        overrides["resume"] = str(resume_checkpoint)
    return overrides


def run_trainer(
    *,
    weights: Path,
    data_yaml: Path,
    run_root: Path,
    name: str,
    device: str,
    fingerprint: str,
    args: argparse.Namespace,
    resume_bundle: Path | None = None,
    interrupt_after_epoch: int | None = None,
) -> PreflightTracingTrainer:
    bundle_root = run_root / name / "checkpoint_bundles"
    resume_checkpoint = None
    if resume_bundle is not None:
        manager = CheckpointBundleManager(bundle_root, fingerprint=fingerprint, keep_last=args.epochs + 1)
        resolved = manager.resolve(resume_bundle)
        resume_checkpoint = resolved.trainer_checkpoint
        resume_bundle = resolved.path
    contract = {
        "enabled": True,
        "fingerprint": fingerprint,
        "root": bundle_root,
        "keep_last": args.epochs + 1,
        "resume_bundle": str(resume_bundle) if resume_bundle is not None else None,
    }
    trainer = PreflightTracingTrainer(
        overrides=trainer_overrides(
            weights=weights,
            data_yaml=data_yaml,
            save_parent=run_root,
            name=name,
            device=device,
            args=args,
            resume_checkpoint=resume_checkpoint,
        ),
        checkpoint_contract=contract,
        image_cls_weight=0.5,
        mel_nv_aux_weight=RUN31_MEL_NV_AUX_WEIGHT,
        selection={
            "max_mel_to_nv_rate": 0.1077844311,
            "min_binary_iou": 0.70,
            "min_group_macro_f1": 0.50,
        },
    )
    if interrupt_after_epoch is not None:
        def request_safe_stop(current: MELNVAuxHeadDelayedUtilityGatedTrainer) -> None:
            if int(current.epoch) == interrupt_after_epoch:
                current._signal_latch.handler(signal.SIGINT, None)

        trainer.add_callback("on_train_epoch_end", request_safe_stop)
    trainer.train()
    return trainer


def load_bundle_payload(manager: CheckpointBundleManager) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    bundle = manager.resolve("auto")
    checkpoint = torch.load(bundle.trainer_checkpoint, map_location="cpu", weights_only=False)
    runtime = torch.load(bundle.runtime_state, map_location="cpu", weights_only=False)
    return bundle, checkpoint, runtime


def compare_nested(left: Any, right: Any, *, atol: float, path: str = "root") -> dict[str, Any]:
    result = {"tensor_count": 0, "max_abs_diff": 0.0, "mismatches": []}

    def visit(a: Any, b: Any, key: str) -> None:
        if len(result["mismatches"]) >= 20:
            return
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            result["tensor_count"] += 1
            if a.shape != b.shape or a.dtype != b.dtype:
                result["mismatches"].append(f"{key}: tensor shape/dtype differ")
                return
            if a.numel() == 0:
                return
            if a.is_floating_point() or a.is_complex():
                diff = float((a - b).abs().max().item())
                result["max_abs_diff"] = max(result["max_abs_diff"], diff)
                if not torch.allclose(a, b, rtol=0.0, atol=atol, equal_nan=True):
                    result["mismatches"].append(f"{key}: max_abs_diff={diff}")
            elif not torch.equal(a, b):
                result["mismatches"].append(f"{key}: integer/bool tensor differs")
            return
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            if a.shape != b.shape or a.dtype != b.dtype or not np.array_equal(a, b, equal_nan=True):
                result["mismatches"].append(f"{key}: NumPy array differs")
            return
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                result["mismatches"].append(f"{key}: mapping keys differ")
                return
            for child in sorted(a, key=str):
                visit(a[child], b[child], f"{key}.{child}")
            return
        if isinstance(a, (list, tuple)) and isinstance(b, type(a)):
            if len(a) != len(b):
                result["mismatches"].append(f"{key}: sequence lengths differ")
                return
            for index, (child_a, child_b) in enumerate(zip(a, b)):
                visit(child_a, child_b, f"{key}[{index}]")
            return
        if isinstance(a, float) and isinstance(b, float):
            if math.isnan(a) and math.isnan(b):
                return
            if abs(a - b) > atol:
                result["mismatches"].append(f"{key}: {a} != {b}")
            return
        if a != b:
            result["mismatches"].append(f"{key}: {a!r} != {b!r}")

    visit(left, right, path)
    return result


def model_state(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    ema = checkpoint.get("ema")
    if ema is None:
        raise RuntimeError("Checkpoint has no EMA model")
    return ema.float().state_dict()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.epochs < 2:
        raise ValueError("--epochs must be at least 2 to test interruption and resume")
    if args.imgsz < 64 or args.imgsz % 32:
        raise ValueError("--imgsz must be a multiple of 32 and at least 64")
    if args.batch != 3:
        raise ValueError("v1 synthetic fixture contains three label roles; use --batch 3")
    device = resolve_device(args.device)
    if args.amp and device == "cpu":
        raise ValueError("--amp requires a CUDA device")
    if args.workers < 0:
        raise ValueError("--workers must be non-negative")
    weights = args.weights.resolve() if args.weights.is_absolute() else (PROJECT_ROOT / args.weights).resolve()
    if not weights.is_file() or not TAXONOMY.is_file():
        raise FileNotFoundError(f"Missing weights or taxonomy: weights={weights}, taxonomy={TAXONOMY}")
    output_root = new_output_root(args.output_root)
    data_yaml, manifest = build_synthetic_dataset(output_root, args.imgsz)
    prewarm_dataset_cache(data_yaml, manifest, args.imgsz, args.batch)
    fingerprint = checkpoint_fingerprint(weights, data_yaml, manifest, args)
    runs_root = output_root / "runs"

    continuous = run_trainer(
        weights=weights,
        data_yaml=data_yaml,
        run_root=runs_root,
        name="continuous",
        device=device,
        fingerprint=fingerprint,
        args=args,
    )
    interrupted = run_trainer(
        weights=weights,
        data_yaml=data_yaml,
        run_root=runs_root,
        name="interrupted_resume",
        device=device,
        fingerprint=fingerprint,
        args=args,
        interrupt_after_epoch=0,
    )
    interrupted_manager = CheckpointBundleManager(
        runs_root / "interrupted_resume" / "checkpoint_bundles",
        fingerprint=fingerprint,
        keep_last=args.epochs + 1,
    )
    auto_bundle = interrupted_manager.resolve("auto")
    if auto_bundle.epoch != 0:
        raise RuntimeError(f"Expected safe stop after epoch 0, auto selected epoch {auto_bundle.epoch}")
    resumed = run_trainer(
        weights=weights,
        data_yaml=data_yaml,
        run_root=runs_root,
        name="interrupted_resume",
        device=device,
        fingerprint=fingerprint,
        args=args,
        resume_bundle=auto_bundle.path,
    )

    continuous_manager = CheckpointBundleManager(
        runs_root / "continuous" / "checkpoint_bundles",
        fingerprint=fingerprint,
        keep_last=args.epochs + 1,
    )
    continuous_bundle, continuous_ckpt, continuous_runtime = load_bundle_payload(continuous_manager)
    resumed_bundle, resumed_ckpt, resumed_runtime = load_bundle_payload(interrupted_manager)
    expected_epoch = args.epochs - 1
    if continuous_bundle.epoch != expected_epoch or resumed_bundle.epoch != expected_epoch:
        raise RuntimeError(
            f"Final epoch mismatch: continuous={continuous_bundle.epoch}, resumed={resumed_bundle.epoch}, "
            f"expected={expected_epoch}"
        )
    atol = args.atol if args.atol is not None else (0.0 if device == "cpu" else 1e-6)
    continuous_training = continuous_runtime["training_state"]
    resumed_training = resumed_runtime["training_state"]
    comparisons = {
        "live_model": compare_nested(
            continuous_training["model"], resumed_training["model"], atol=atol, path="live_model"
        ),
        "ema_model": compare_nested(
            continuous_training["ema"], resumed_training["ema"], atol=atol, path="ema_model"
        ),
        "optimizer": compare_nested(
            continuous_training["optimizer"], resumed_training["optimizer"], atol=atol, path="optimizer"
        ),
        "scaler": compare_nested(
            continuous_training["scaler"], resumed_training["scaler"], atol=atol, path="scaler"
        ),
        "ema_updates": compare_nested(
            continuous_training["ema_updates"],
            resumed_training["ema_updates"],
            atol=atol,
            path="ema_updates",
        ),
        "scheduler": compare_nested(
            continuous_runtime["scheduler"], resumed_runtime["scheduler"], atol=atol, path="scheduler"
        ),
        "stopper": compare_nested(
            continuous_runtime["stopper"], resumed_runtime["stopper"], atol=atol, path="stopper"
        ),
        "selection_state": compare_nested(
            continuous_runtime["selection_state"],
            resumed_runtime["selection_state"],
            atol=atol,
            path="selection_state",
        ),
        "rng": compare_nested(
            continuous_runtime["rng"], resumed_runtime["rng"], atol=atol, path="rng"
        ),
        "second_epoch_batch_order": compare_nested(
            [row for row in continuous.preflight_batch_trace if row["epoch"] == 1],
            [row for row in resumed.preflight_batch_trace if row["epoch"] == 1],
            atol=0.0,
            path="second_epoch_batch_order",
        ),
    }
    failures = {name: value["mismatches"] for name, value in comparisons.items() if value["mismatches"]}
    report = {
        "status": "PASS" if not failures else "FAIL",
        "scope": "synthetic train/val end-to-end trainer checkpoint preflight; no test split or test metrics",
        "device": device,
        "gpu_name": torch.cuda.get_device_name(int(device)) if device != "cpu" else None,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "weights": str(weights),
        "weights_sha256": sha256(weights),
        "output_root": str(output_root),
        "fingerprint": fingerprint,
        "epochs": args.epochs,
        "amp": bool(args.amp),
        "workers": args.workers,
        "safe_interrupt_bundle": str(auto_bundle.path),
        "safe_interrupt_epoch": auto_bundle.epoch,
        "safe_interrupt_metadata": torch.load(
            auto_bundle.runtime_state, map_location="cpu", weights_only=False
        )["interruption"],
        "continuous_final_bundle": str(continuous_bundle.path),
        "resumed_final_bundle": str(resumed_bundle.path),
        "comparison_atol": atol,
        "comparisons": comparisons,
        "batch_traces": {
            "continuous": continuous.preflight_batch_trace,
            "interrupted": interrupted.preflight_batch_trace,
            "resumed": resumed.preflight_batch_trace,
        },
        "failures": failures,
        "test_metrics_used": False,
        "formal_run31_experiment_created": False,
    }
    report_path = output_root / "PREFLIGHT_REPORT.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise RuntimeError(f"Continuous/resumed state mismatch; see {report_path}")


if __name__ == "__main__":
    main()

