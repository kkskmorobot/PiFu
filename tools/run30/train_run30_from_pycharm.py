#!/usr/bin/env python3
"""Canonical Run30 PyCharm entrypoint.

Run this file with no arguments to start or safely auto-resume formal Run30.
Codex uses ``--prepare-only`` to perform the final no-training release check.
There is deliberately no test or external-evaluation option in this runner.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = PROJECT_ROOT / "tools"
RUN30_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "30_YOLO26m_Binary8Class_Plain_MELNVFocalRisk_Seed43"
CONFIG_PATH = RUN30_DIR / "run30_preregistered_config_v1.yaml"
PROTOCOL_PATH = RUN30_DIR / "RUN30_PROTOCOL_2026-09-06.md"
DEVELOPMENT_MANIFEST = RUN30_DIR / "run30_development_manifest.csv"
DEVELOPMENT_DATA = RUN30_DIR / "run30_development_data.yaml"
PREFLIGHT_REPORT = RUN30_DIR / "RUN30_PREFLIGHT_REPORT.json"
READY_MARKER = RUN30_DIR / "RUN30_READY_TO_TRAIN.json"
RELEASE_DOCUMENT = RUN30_DIR / "PROJECT_RUN30_RELEASE_2026-09-06.md"
LOCKED_CONFIG_SHA256 = "cae5b6571e724529c36ea92fbff4dca73a27002130232db2343d4931e0179ce0"
EXPECTED_DEVELOPMENT_MANIFEST_SHA256 = "3544f94e0cfd4aa3b15a7f4798ea29cbedd8ab7e5d313b72a1f94835542f6c1b"
EXPECTED_DEVELOPMENT_COUNTS = {
    "train": {"total": 23_462, "seg": 10_483, "cls": 13_715, "both": 736},
    "val": {"total": 1_744, "seg": 292, "cls": 1_488, "both": 36},
}
PROTECTED_SHA256 = {
    "models/yolo26m-sem.pt": "3de52574cf1b18e38d32d9e51fc57135abc4f8dda37ff13ccbf44dcf99986233",
    "tools/yolo26n.pt": "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef",
    "tools/classification_taxonomy_v1.yaml": "ffd14ba2e73c8df5d3df5af67007db122d4397adbaf9ef4475fdb9858f802b8e",
    "Data/processed_imaplusplus_binary_v1/manifest.csv": "47c3763f210f36d9428a0d0b90427451eb6d5f43b43163bcb7141c191614396e",
    "Data/processed_isic2019_classification_incremental_v1/manifest.csv": (
        "b2aee37dee5dd983ee4edadc03f570a70ba791f380ab1485415a42ce4365a17a"
    ),
}
REQUIRED_DEVELOPMENT_COLUMNS = {
    "isic_id",
    "split",
    "dataset_role",
    "image_path",
    "mask_path",
    "has_segmentation",
    "has_classification",
    "classification_target",
    "class_name",
    "group_id",
}


class ReleaseError(RuntimeError):
    """Raised when the frozen Run30 release contract is not satisfied."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Run final data/CUDA/model forward checks and write the ready marker without training.",
    )
    parser.add_argument(
        "--resume",
        default="auto",
        help="auto (default), none for a new run, or an explicit complete checkpoint-bundle directory.",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8"))


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_locked_config() -> dict[str, Any]:
    if not CONFIG_PATH.is_file():
        raise ReleaseError(f"Frozen Run30 config is missing: {CONFIG_PATH}")
    actual_hash = sha256_file(CONFIG_PATH)
    if actual_hash != LOCKED_CONFIG_SHA256:
        raise ReleaseError(
            f"Run30 config SHA-256 changed: {actual_hash} != {LOCKED_CONFIG_SHA256}. "
            "A new preregistration and release review are required."
        )
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["contract"]["version"] != "run30_yolo26m_plain_mel_nv_focal_risk_seed43_v1":
        raise ReleaseError("Run30 contract version changed")
    if config["contract"]["single_preregistered_factor"] != (
        "add_true_mel_to_nv_posterior_focal_risk_gamma_2_weight_0_5_to_run22"
    ):
        raise ReleaseError("Run30 single preregistered factor changed")
    ultra = config["ultralytics"]
    expected = {
        "epochs": 100,
        "patience": 30,
        "batch": 16,
        "imgsz": 640,
        "device": 0,
        "workers": 8,
        "optimizer": "AdamW",
        "seed": 43,
        "amp": True,
        "val": True,
        "fraction": 1.0,
    }
    for key, value in expected.items():
        if ultra.get(key) != value:
            raise ReleaseError(f"Frozen Run30 ultralytics.{key} changed: {ultra.get(key)!r} != {value!r}")
    if ultra.get("model") != "models/yolo26m-sem.pt" or ultra.get("pretrained") is not True:
        raise ReleaseError("Run30 must fresh-start from the frozen yolo26m-sem pretrained asset")
    for key in ("mosaic", "mixup", "cutmix", "copy_paste", "erasing"):
        if float(config["augmentations"].get(key, -1.0)) != 0.0:
            raise ReleaseError(f"Run30 requires augmentations.{key}=0.0")
    if config["augmentations"].get("auto_augment") is not None:
        raise ReleaseError("Run30 requires augmentations.auto_augment=null")
    if float(ultra.get("dropout", -1.0)) != 0.0:
        raise ReleaseError("Run30 requires framework dropout=0.0; the custom image-head dropout stays code-locked at 0.20")
    if config["multitask"] != {
        "semantic_classes": 1,
        "classification_classes": 8,
        "image_cls_weight": 0.5,
        "image_head": "plain_p3_p4_global_average_pool_mlp_dropout020",
        "classification_loss": "unweighted_cross_entropy_plus_true_mel_to_nv_posterior_focal_risk",
        "classification_label_smoothing": 0.0,
        "sampler": "standard_seeded_epoch_shuffle_without_replacement",
        "epoch_size": "all_23462_train_samples_once",
        "segmentation_loss_scope": "has_segmentation_only",
        "classification_loss_scope": "has_official_classification_only",
    }:
        raise ReleaseError("Frozen Run30 multitask block changed")
    if config["risk_loss"] != {
        "enabled": True,
        "name": "true_mel_to_nv_posterior_focal_risk",
        "weight_inside_classification_objective": 0.5,
        "gamma": 2.0,
        "true_class_id": 4,
        "risk_class_id": 5,
        "formula": "mean_true_mel_p_nv_pow_gamma_times_negative_log_one_minus_p_nv",
        "normalization": "true_mel_rows_only",
        "non_mel_auxiliary_gradient": "zero",
        "application_scope": "labeled_samples_in_training_objective_and_val_loss_reporting",
        "optimization_scope": "training_only",
        "new_parameters": 0,
        "validation_loss_does_not_change_metrics_or_selection": True,
        "inference_rule": "unchanged_eight_class_argmax",
    }:
        raise ReleaseError("Frozen Run30 risk_loss block changed")
    if config["attention_ablation"] != {
        "enabled": False,
        "same_as_run22": "plain_p3_p4_global_average_pool_mlp_dropout020",
        "plain_descriptor_scales": ["P3", "P4"],
        "pooling": "global_average",
        "semantic_branch_modified": False,
    }:
        raise ReleaseError("Frozen Run30 attention_ablation block changed")
    if config["amp_check"] != {
        "asset": "tools/yolo26n.pt",
        "sha256": PROTECTED_SHA256["tools/yolo26n.pt"],
        "working_directory": "tools",
        "runtime_download": "forbidden_local_asset_required",
    }:
        raise ReleaseError("Run30 local AMP-check asset contract changed")
    if config["early_stopping"] != {
        "policy": "delayed_utility_gated_v1",
        "start_epoch": 40,
        "patience_after_start": 30,
        "earliest_possible_stop_epoch": 70,
        "ineligible_score": "joint_score_minus_two",
        "eligible_score": "original_joint_minus_mel_to_nv_excess_fitness",
        "utility_eligible_requires": {"min_binary_iou": 0.70, "min_group_macro_f1": 0.50},
        "applies_to": ["ultralytics_stopper", "ultralytics_best_pt"],
        "does_not_change": ["validation_metrics", "checkpoint_selection_aliases"],
    }:
        raise ReleaseError("Run30 delayed utility-gated early-stop policy changed")
    if config["selection"]["primary_alias"] != "best_feasible_joint":
        raise ReleaseError("Run30 primary selection alias changed")
    if float(config["selection"]["max_mel_to_nv_rate"]) != 0.1077844311:
        raise ReleaseError("Run30 mel->nv feasibility constraint changed")
    if float(config["selection"].get("min_binary_iou", -1.0)) != 0.70:
        raise ReleaseError("Run30 minimum binary IoU eligibility floor changed")
    if float(config["selection"].get("min_group_macro_f1", -1.0)) != 0.50:
        raise ReleaseError("Run30 minimum group Macro-F1 eligibility floor changed")
    if config["selection"].get("fitness_policy") != "unchanged_joint_minus_mel_to_nv_excess_only":
        raise ReleaseError("Run30 fitness policy changed")
    if config["checkpoint"]["bundle_version"] != "pifuv2_multitask_checkpoint_v2":
        raise ReleaseError("Run30 checkpoint bundle version changed")
    return config


def verify_protected_files() -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for relative, expected_hash in PROTECTED_SHA256.items():
        path = PROJECT_ROOT / Path(relative)
        if not path.is_file():
            raise ReleaseError(f"Protected Run30 input is missing: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ReleaseError(f"Protected Run30 input changed: {relative}: {actual_hash} != {expected_hash}")
        report[relative] = {"bytes": path.stat().st_size, "sha256": actual_hash}
    return report


def validate_development_artifacts() -> dict[str, Any]:
    if not DEVELOPMENT_MANIFEST.is_file():
        raise ReleaseError("Run30 frozen development manifest is missing")
    actual_manifest_hash = sha256_file(DEVELOPMENT_MANIFEST)
    if actual_manifest_hash != EXPECTED_DEVELOPMENT_MANIFEST_SHA256:
        raise ReleaseError(
            f"Run30 development manifest changed: {actual_manifest_hash} != "
            f"{EXPECTED_DEVELOPMENT_MANIFEST_SHA256}"
        )
    with DEVELOPMENT_MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = set(reader.fieldnames or [])
    missing = REQUIRED_DEVELOPMENT_COLUMNS - fieldnames
    if missing:
        raise ReleaseError(f"Run30 development manifest is missing columns: {sorted(missing)}")
    if any(row["split"] not in {"train", "val"} for row in rows):
        raise ReleaseError("Run30 manifest contains a non-development split")
    if any(row["dataset_role"] not in {"imaplusplus_binary", "isic2019_classification_incremental"} for row in rows):
        raise ReleaseError("Run30 manifest contains an unregistered dataset role")
    ids = [row["isic_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ReleaseError("Run30 development manifest contains duplicate isic_id")
    counts: dict[str, dict[str, int]] = {}
    for split in ("train", "val"):
        selected = [row for row in rows if row["split"] == split]
        counts[split] = {
            "total": len(selected),
            "seg": sum(row["has_segmentation"] == "1" for row in selected),
            "cls": sum(row["has_classification"] == "1" for row in selected),
            "both": sum(row["has_segmentation"] == "1" and row["has_classification"] == "1" for row in selected),
        }
    if counts != EXPECTED_DEVELOPMENT_COUNTS:
        raise ReleaseError(f"Run30 development supervision counts changed: {counts}")
    if not DEVELOPMENT_DATA.is_file():
        raise ReleaseError("Run30 development data YAML is missing")
    data = yaml.safe_load(DEVELOPMENT_DATA.read_text(encoding="utf-8"))
    if data.get("test", "unexpected") is not None:
        raise ReleaseError("Run30 development YAML must use test: null")
    serialized = json.dumps(data, ensure_ascii=False).lower()
    if "pad_ufes" in serialized or "isic2018_task1" in serialized or "/test" in serialized or "\\test" in serialized:
        raise ReleaseError("Run30 development YAML references test or an external candidate")
    return {
        "counts": counts,
        "development_manifest_sha256": actual_manifest_hash,
        "development_data_sha256": sha256_file(DEVELOPMENT_DATA),
        "test_key": None,
        "external_candidates_present": False,
    }


def lock_input_paths() -> list[Path]:
    return [
        CONFIG_PATH,
        PROTOCOL_PATH,
        DEVELOPMENT_MANIFEST,
        DEVELOPMENT_DATA,
        PROJECT_ROOT / "models" / "yolo26m-sem.pt",
        TOOLS_DIR / "yolo26n.pt",
        PROJECT_ROOT / "tools/classification_taxonomy_v1.yaml",
        PROJECT_ROOT / "Data" / "processed_imaplusplus_binary_v1" / "manifest.csv",
        PROJECT_ROOT / "Data" / "processed_isic2019_classification_incremental_v1" / "manifest.csv",
        TOOLS_DIR / "imaplusplus_partial_multitask_v1.py",
        TOOLS_DIR / "imaplusplus_partial_multitask_utility_selection_v1.py",
        RUN30_DIR / "imaplusplus_partial_multitask_plain_mel_nv_focal_risk_seed43_v1.py",
        TOOLS_DIR / "multitask_checkpoint_contract.py",
        TOOLS_DIR / "evaluate_imaplusplus_multitask_v1.py",
        Path(__file__).resolve(),
    ]


def compute_fingerprint(config: dict[str, Any]) -> tuple[str, dict[str, dict[str, Any]]]:
    if str(TOOLS_DIR) not in sys.path:
        sys.path.insert(0, str(TOOLS_DIR))
    from multitask_checkpoint_contract import stable_fingerprint

    paths = [path.resolve() for path in lock_input_paths()]
    inputs = {str(path): {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in paths}
    behavior = {
        "contract": config["contract"],
        "ultralytics": config["ultralytics"],
        "augmentations": config["augmentations"],
        "multitask": config["multitask"],
        "risk_loss": config["risk_loss"],
        "attention_ablation": config["attention_ablation"],
        "amp_check": config["amp_check"],
        "early_stopping": config["early_stopping"],
        "checkpoint": config["checkpoint"],
        "selection": config["selection"],
        "development_counts": EXPECTED_DEVELOPMENT_COUNTS,
        "test_split": "absent_from_development_manifest_and_yaml_null",
    }
    return stable_fingerprint(behavior, paths), inputs


def cuda_status() -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ReleaseError(
            f"Run30 requires the verified single CUDA device; available={torch.cuda.is_available()} "
            f"count={torch.cuda.device_count()}"
        )
    name = torch.cuda.get_device_name(0)
    if "RTX 5080" not in name:
        raise ReleaseError(f"Run30 expected NVIDIA GeForce RTX 5080, found {name}")
    free, total = torch.cuda.mem_get_info(0)
    return {
        "available": True,
        "count": 1,
        "name": name,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "free_bytes_before_preflight": int(free),
        "total_bytes": int(total),
    }


def expand_role_batch(batch: dict[str, Any], target_batch: int) -> dict[str, Any]:
    source_batch = int(batch["img"].shape[0])
    indices = torch.arange(target_batch, dtype=torch.long) % source_batch
    expanded: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == source_batch:
            expanded[key] = value.index_select(0, indices)
        elif isinstance(value, (list, tuple)) and len(value) == source_batch:
            expanded[key] = [value[int(index)] for index in indices.tolist()]
        else:
            expanded[key] = value
    return expanded


def no_training_gpu_forward_preflight(config: dict[str, Any]) -> dict[str, Any]:
    for path in (TOOLS_DIR, RUN30_DIR):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from imaplusplus_partial_multitask_plain_mel_nv_focal_risk_seed43_v1 import (
        MELNVFocalRiskBinaryPartialMultiTaskModel,
        MELNVFocalRiskPartialLabelMultiTaskLoss,
        RUN30_MEL_NV_FOCAL_GAMMA,
        RUN30_MEL_NV_FOCAL_RISK_WEIGHT,
        run30_mel_nv_focal_risk,
    )
    from ultralytics.nn.tasks import load_checkpoint
    from verify_partial_multitask_v1 import real_partial_batch

    ultra = config["ultralytics"]
    weights = resolve_project_path(ultra["model"])
    batch, metadata = real_partial_batch(DEVELOPMENT_DATA, "train", int(ultra["imgsz"]))
    expanded = expand_role_batch(batch, int(ultra["batch"]))
    device = torch.device("cuda:0")
    pretrained, _ = load_checkpoint(weights, device="cpu")
    model = MELNVFocalRiskBinaryPartialMultiTaskModel(
        cfg=pretrained.yaml,
        nc=1,
        num_diseases=8,
        image_cls_weight=float(config["multitask"]["image_cls_weight"]),
        mel_nv_focal_risk_weight=float(config["risk_loss"]["weight_inside_classification_objective"]),
        verbose=False,
    )
    transferred = model.load(pretrained)
    model.args = SimpleNamespace(data=str(DEVELOPMENT_DATA))
    model.to(device).train()
    for module in model.modules():
        if isinstance(module, (nn.modules.batchnorm._BatchNorm, nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.eval()
    criterion = MELNVFocalRiskPartialLabelMultiTaskLoss(model)
    before = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    expanded = {
        key: value.to(device, non_blocking=False) if isinstance(value, torch.Tensor) else value
        for key, value in expanded.items()
    }
    images = expanded["img"].float() / 255.0
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=bool(ultra["amp"])):
        predictions = model(images)
        total, items = criterion(predictions, expanded)
    if not torch.isfinite(total) or not torch.isfinite(items).all():
        raise ReleaseError(f"Run30 forward/loss preflight produced non-finite values: total={total}, items={items}")
    if items.numel() != 5:
        raise ReleaseError(f"Run30 loss item count changed: {items}")
    if model.mel_nv_focal_risk_weight != RUN30_MEL_NV_FOCAL_RISK_WEIGHT:
        raise ReleaseError("Run30 GPU preflight found a drifted MEL-to-NV focal risk weight")
    labeled_mel_count = int(
        ((expanded["classification_target"] == 4) & expanded["has_classification"].bool()).sum().item()
    )
    probe_logits = torch.zeros((2, 8), device=device)
    probe_logits[0, 5] = 2.0
    probe_loss = run30_mel_nv_focal_risk(
        probe_logits, torch.tensor([4, 5], device=device), RUN30_MEL_NV_FOCAL_GAMMA
    )
    if not torch.isfinite(probe_loss) or float(probe_loss) <= 0.0:
        raise ReleaseError(f"Run30 synthetic focal-risk probe failed: {probe_loss}")
    changed = [key for key, value in model.state_dict().items() if not torch.equal(before[key], value.detach().cpu())]
    if changed:
        raise ReleaseError(f"No-training preflight changed model state: {changed[:5]}")
    head = model.model[-1]
    if list(head.f) != [16, 13]:
        raise ReleaseError(f"Run30 plain baseline head inputs changed: f={head.f}")
    forbidden_attention_attributes = [
        name for name in ("channel_attention", "mask_guided_mix", "detach_mask", "p5_layer") if hasattr(head, name)
    ]
    if forbidden_attention_attributes:
        raise ReleaseError(f"Run30 unexpectedly contains attention attributes: {forbidden_attention_attributes}")
    report = {
        "status": "PASS",
        "scope": "real development role batch expanded to frozen batch size; forward and loss only",
        "source_role_ids": metadata.get("isic_ids", []),
        "batch": int(images.shape[0]),
        "imgsz": int(images.shape[-1]),
        "amp": bool(ultra["amp"]),
        "semantic_shape": list(predictions[0].shape),
        "classification_shape": list(predictions[-1].shape),
        "loss_items": [float(value) for value in items.detach().cpu()],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "classification_head_parameters": sum(parameter.numel() for parameter in head.image_classifier.parameters()),
        "new_risk_loss_parameters": 0,
        "mel_nv_focal_risk_weight": float(model.mel_nv_focal_risk_weight),
        "mel_nv_focal_gamma": RUN30_MEL_NV_FOCAL_GAMMA,
        "labeled_mel_rows": labeled_mel_count,
        "synthetic_focal_risk_probe": float(probe_loss),
        "plain_head_inputs": list(head.f),
        "attention_modules_present": False,
        "descriptor": "p3_p4_global_average_pool",
        "pretrained_load_result": str(transferred),
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "model_state_changed": False,
        "optimizer_created": False,
        "backward_called": False,
        "optimizer_step_called": False,
        "trainer_train_called": False,
        "test_or_external_loaded": False,
    }
    del criterion, model, pretrained, expanded, images, predictions, total, items, before, probe_logits, probe_loss
    torch.cuda.empty_cache()
    return report


def prepare_only(config: dict[str, Any]) -> None:
    run_dir = resolve_project_path(config["ultralytics"]["project"]) / config["ultralytics"]["name"]
    if run_dir.exists():
        raise ReleaseError(f"Run30 result directory already exists; preparation refuses to touch it: {run_dir}")
    verify_protected_files()
    development = validate_development_artifacts()
    cuda = cuda_status()
    forward = no_training_gpu_forward_preflight(config)
    fingerprint, lock_inputs = compute_fingerprint(config)
    report = {
        "status": "PASS",
        "scope": "final Run30 preparation; no optimizer, backward, training, test, or external evaluation",
        "created_local_time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "config_sha256": sha256_file(CONFIG_PATH),
        "fingerprint": fingerprint,
        "protected_files": verify_protected_files(),
        "lock_inputs": lock_inputs,
        "development": development,
        "test_and_external_accessed": False,
        "cuda": cuda,
        "forward_only_preflight": forward,
        "formal_training_started": False,
        "formal_result_directory_created": run_dir.exists(),
        "amp_check_asset": str(TOOLS_DIR / "yolo26n.pt"),
        "amp_check_working_directory": str(TOOLS_DIR),
        "amp_check_runtime_download": "forbidden_local_asset_required",
        "test_metrics_used": False,
        "external_metrics_used": False,
    }
    atomic_json(PREFLIGHT_REPORT, report)
    ready = {
        "status": "PASS",
        "ready_to_click_pycharm": True,
        "entrypoint": str(Path(__file__).resolve()),
        "config": str(CONFIG_PATH),
        "config_sha256": report["config_sha256"],
        "fingerprint": fingerprint,
        "lock_inputs": lock_inputs,
        "preflight_report": str(PREFLIGHT_REPORT),
        "formal_training_started": False,
        "test_metrics_used": False,
        "external_metrics_used": False,
    }
    atomic_json(READY_MARKER, ready)
    print(json.dumps(ready, ensure_ascii=False, indent=2))
    print("PREPARE_ONLY_PASS: trainer.train() was not called.")


def validate_ready(config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    if not READY_MARKER.is_file() or not PREFLIGHT_REPORT.is_file():
        raise ReleaseError("Run30 ready/preflight marker is missing; do not train")
    ready = json.loads(READY_MARKER.read_text(encoding="utf-8"))
    if ready.get("status") != "PASS" or ready.get("ready_to_click_pycharm") is not True:
        raise ReleaseError("Run30 ready marker is not PASS")
    if ready.get("formal_training_started") is not False:
        raise ReleaseError("Run30 ready marker does not represent an untouched pre-training state")
    validate_development_artifacts()
    verify_protected_files()
    fingerprint, lock_inputs = compute_fingerprint(config)
    if fingerprint != ready.get("fingerprint") or lock_inputs != ready.get("lock_inputs"):
        raise ReleaseError("Run30 lock inputs changed after final preflight")
    return ready, fingerprint


def validate_release_document() -> None:
    if not RELEASE_DOCUMENT.is_file():
        raise ReleaseError(f"Latest Run30 release document is missing: {RELEASE_DOCUMENT}")
    text = RELEASE_DOCUMENT.read_text(encoding="utf-8")
    required = {
        "`PROJECT_RUN30_GATE=PASS`",
        "`RUN30_READY_TO_CLICK_GATE=PASS`",
        "`USER_CONFIRMATION_RECORDED=TRUE`",
        "`TEST_METRICS_USED=FALSE`",
        "`EXTERNAL_METRICS_USED=FALSE`",
    }
    missing = sorted(value for value in required if value not in text)
    if missing:
        raise ReleaseError(f"Latest Run30 release document is not a valid PASS: {missing}")


def read_aliases(bundle_root: Path) -> dict[str, Any]:
    aliases: dict[str, Any] = {}
    alias_dir = bundle_root / "aliases"
    if alias_dir.is_dir():
        for path in sorted(alias_dir.glob("*.json")):
            aliases[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return aliases


def train_formal(config: dict[str, Any], resume_value: str) -> None:
    validate_release_document()
    ready, fingerprint = validate_ready(config)
    for path in (TOOLS_DIR, RUN30_DIR):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from imaplusplus_partial_multitask_plain_mel_nv_focal_risk_seed43_v1 import (
        MELNVFocalRiskDelayedUtilityGatedTrainer,
    )
    from multitask_checkpoint_contract import CheckpointBundleManager, CheckpointContractError

    ultra = dict(config["ultralytics"])
    ultra.update(config["augmentations"])
    ultra["model"] = str(resolve_project_path(ultra["model"]))
    ultra["data"] = str(DEVELOPMENT_DATA.resolve())
    ultra["project"] = str(resolve_project_path(ultra["project"]))
    run_dir = Path(ultra["project"]) / ultra["name"]
    bundle_root = run_dir / "checkpoint_bundles"
    complete_marker = run_dir / "RUN30_TRAINING_COMPLETE.json"
    if complete_marker.exists():
        raise ReleaseError(f"Run30 already completed; refusing to train again: {complete_marker}")

    manager = CheckpointBundleManager(
        bundle_root,
        fingerprint=fingerprint,
        keep_last=int(config["checkpoint"]["keep_last"]),
    )
    resume_bundle = None
    normalized_resume = str(resume_value).strip()
    if normalized_resume.lower() == "auto":
        if bundle_root.is_dir():
            try:
                resume_bundle = manager.resolve("auto")
            except CheckpointContractError as error:
                raise ReleaseError(
                    f"Run30 output exists but no complete compatible auto-resume bundle was found: {error}"
                ) from error
        elif run_dir.exists():
            raise ReleaseError(
                "Run30 directory exists without a complete epoch bundle. Refusing to overwrite a first-epoch "
                f"incomplete run: {run_dir}"
            )
    elif normalized_resume.lower() == "none":
        if run_dir.exists():
            raise ReleaseError(f"--resume none requires an absent Run30 output directory: {run_dir}")
    else:
        candidate = Path(normalized_resume)
        if not candidate.is_absolute():
            candidate = (PROJECT_ROOT / candidate).resolve()
        resume_bundle = manager.resolve(candidate)

    if resume_bundle is not None:
        ultra["resume"] = str(resume_bundle.trainer_checkpoint)
        ultra["exist_ok"] = True
        print(f"Run30 resume bundle: {resume_bundle.path} (completed epoch {resume_bundle.epoch})")
    else:
        ultra["exist_ok"] = False
        print("Run30 fresh start: no prior result directory or complete bundle was found.")

    checkpoint_contract = {
        "enabled": True,
        "fingerprint": fingerprint,
        "root": bundle_root,
        "keep_last": int(config["checkpoint"]["keep_last"]),
        "resume_bundle": str(resume_bundle.path) if resume_bundle is not None else None,
    }
    trainer = MELNVFocalRiskDelayedUtilityGatedTrainer(
        overrides=ultra,
        checkpoint_contract=checkpoint_contract,
        image_cls_weight=float(config["multitask"]["image_cls_weight"]),
        mel_nv_focal_risk_weight=float(config["risk_loss"]["weight_inside_classification_objective"]),
        selection={
            "max_mel_to_nv_rate": float(config["selection"]["max_mel_to_nv_rate"]),
            "min_binary_iou": float(config["selection"]["min_binary_iou"]),
            "min_group_macro_f1": float(config["selection"]["min_group_macro_f1"]),
        },
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(
        run_dir / "RUN30_LAUNCH_RECORD.json",
        {
            "status": "STARTING_OR_RESUMING",
            "started_local_time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "entrypoint": str(Path(__file__).resolve()),
            "config_sha256": ready["config_sha256"],
            "fingerprint": fingerprint,
            "resume_bundle": str(resume_bundle.path) if resume_bundle is not None else None,
            "test_metrics_used": False,
            "external_metrics_used": False,
        },
    )
    try:
        trainer.train()
    except BaseException as error:
        atomic_json(
            run_dir / "RUN30_EXCEPTION_NOT_COMPLETE.json",
            {
                "status": "NOT_COMPLETE",
                "error_type": type(error).__name__,
                "error": str(error),
                "recorded_local_time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
                "auto_resume_requires_complete_bundle": True,
            },
        )
        raise

    aliases = read_aliases(bundle_root)
    if trainer._signal_latch.requested:
        atomic_json(
            run_dir / "RUN30_INTERRUPTED_SAFE.json",
            {
                "status": "INTERRUPTED_AT_SAFE_COMPLETED_EPOCH",
                "signal": trainer._signal_latch.metadata(),
                "aliases": aliases,
                "resume_instruction": "Run the same train_run30_from_pycharm.py again; default resume=auto.",
            },
        )
        print("Run30 stopped at a safe completed-epoch boundary. Run this same script again to auto-resume.")
        return

    atomic_json(
        complete_marker,
        {
            "status": "TRAINING_COMPLETE_OR_NORMAL_EARLY_STOP",
            "completed_local_time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "last_completed_epoch": int(trainer.epoch),
            "aliases": aliases,
            "primary_selection_alias": config["selection"]["primary_alias"],
            "test_metrics_used": False,
            "external_metrics_used": False,
            "next_action": "Return to Codex for val-only model lock; do not run test or external evaluation.",
        },
    )
    print("Run30 training finished or stopped normally by early stopping. No test/external evaluation was run.")
    print("Return to Codex for val-only checkpoint lock. Do not run any test script.")


def main() -> None:
    os.chdir(TOOLS_DIR)
    args = parse_args()
    config = load_locked_config()
    if args.prepare_only:
        prepare_only(config)
    else:
        train_formal(config, args.resume)


if __name__ == "__main__":
    main()

