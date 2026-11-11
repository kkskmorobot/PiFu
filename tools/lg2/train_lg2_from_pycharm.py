#!/usr/bin/env python3
"""Canonical PyCharm entrypoint for the lg2 LGLA v1 screening experiment.

Run this file with no arguments to start or safely auto-resume formal lg2.
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
LG2_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "lg2_YOLO26m_Binary8Class_LGLAv1_GateBalance_Seed42"
CONFIG_PATH = LG2_DIR / "lg2_preregistered_config_v1.yaml"
PROTOCOL_PATH = LG2_DIR / "LG2_PROTOCOL_2026-09-13.md"
DEVELOPMENT_MANIFEST = LG2_DIR / "lg2_development_manifest.csv"
DEVELOPMENT_DATA = LG2_DIR / "lg2_development_data.yaml"
PREFLIGHT_REPORT = LG2_DIR / "LG2_PREFLIGHT_REPORT.json"
READY_MARKER = LG2_DIR / "LG2_READY_TO_TRAIN.json"
RELEASE_DOCUMENT = LG2_DIR / "PROJECT_LG2_RELEASE_2026-09-13.md"
SOURCE_DEVELOPMENT_MANIFEST = PROJECT_ROOT / "tools" / "run22" / "run22_development_manifest.csv"
LOCKED_CONFIG_SHA256 = "ef019674fe95ae80560285abfc74fe046b72f4b3a473fb05334cae4aef19f170"
EXPECTED_DEVELOPMENT_COUNTS = {
    "train": {"total": 23_462, "seg": 10_483, "cls": 13_715, "both": 736},
    "val": {"total": 1_744, "seg": 292, "cls": 1_488, "both": 36},
}
PROTECTED_SHA256 = {
    "models/yolo26m-sem.pt": "3de52574cf1b18e38d32d9e51fc57135abc4f8dda37ff13ccbf44dcf99986233",
    "tools/yolo26n.pt": "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef",
    "tools/classification_taxonomy_v1.yaml": "ffd14ba2e73c8df5d3df5af67007db122d4397adbaf9ef4475fdb9858f802b8e",
    "tools/run22/run22_development_manifest.csv": (
        "3544f94e0cfd4aa3b15a7f4798ea29cbedd8ab7e5d313b72a1f94835542f6c1b"
    ),
    "tools/run22/run22_development_data.yaml": (
        "3e06dad3dc6769502ad41ec6f5ed52a97f3ae4c001c75e70c3cd5529d87aeff9"
    ),
}
REQUIRED_SOURCE_COLUMNS = {
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
    """Raised when the frozen lg2 release contract is not satisfied."""


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
        raise ReleaseError(f"Frozen lg2 config is missing: {CONFIG_PATH}")
    actual_hash = sha256_file(CONFIG_PATH)
    if actual_hash != LOCKED_CONFIG_SHA256:
        raise ReleaseError(
            f"lg2 config SHA-256 changed: {actual_hash} != {LOCKED_CONFIG_SHA256}. "
            "A new preregistration and release review are required."
        )
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["contract"]["version"] != "lg2_yolo26m_p3p4_lgla_v1_gate_balance_seed42":
        raise ReleaseError("lg2 contract version changed")
    if config["contract"]["single_preregistered_factor"] != (
        "replace_plain_p3_p4_global_descriptor_with_locked_lgx_gate_balanced_lgla_v1"
    ):
        raise ReleaseError("lg2 single preregistered factor changed")
    ultra = config["ultralytics"]
    expected = {
        "epochs": 100,
        "patience": 30,
        "batch": 16,
        "imgsz": 640,
        "device": 0,
        "workers": 8,
        "optimizer": "AdamW",
        "seed": 42,
        "amp": True,
        "val": True,
        "fraction": 1.0,
    }
    for key, value in expected.items():
        if ultra.get(key) != value:
            raise ReleaseError(f"Frozen lg2 ultralytics.{key} changed: {ultra.get(key)!r} != {value!r}")
    if ultra.get("project") != "experiments/lg2_YOLO26m_Binary8Class_LGLAv1_GateBalance_Seed42/results":
        raise ReleaseError("lg2 experiment project path changed")
    if ultra.get("name") != "lg2_yolo26m_lgla_v1_gate_balance_seed42":
        raise ReleaseError("lg2 experiment name changed")
    if ultra.get("model") != "models/yolo26m-sem.pt" or ultra.get("pretrained") is not True:
        raise ReleaseError("lg2 must fresh-start from the frozen yolo26m-sem pretrained asset")
    for key in ("mosaic", "mixup", "cutmix", "copy_paste", "erasing"):
        if float(config["augmentations"].get(key, -1.0)) != 0.0:
            raise ReleaseError(f"lg2 requires augmentations.{key}=0.0")
    if config["augmentations"].get("auto_augment") is not None:
        raise ReleaseError("lg2 requires augmentations.auto_augment=null")
    if float(ultra.get("dropout", -1.0)) != 0.0:
        raise ReleaseError("lg2 requires framework dropout=0.0; the custom image-head dropout stays code-locked at 0.20")
    if config["multitask"] != {
        "semantic_classes": 1,
        "classification_classes": 8,
        "image_cls_weight": 0.5,
        "image_head": "lgla_v1_p3_p4_predicted_lesion_global_local_softmax_gate_mlp_dropout020",
        "classification_loss": "unweighted_cross_entropy_plus_gate_balance_only",
        "classification_label_smoothing": 0.0,
        "sampler": "standard_seeded_epoch_shuffle_without_replacement",
        "epoch_size": "all_23462_train_samples_once",
        "segmentation_loss_scope": "has_segmentation_only",
        "classification_loss_scope": "has_official_classification_only",
    }:
        raise ReleaseError("Frozen lg2 multitask block changed")
    if config["lgla_v1"] != {
        "enabled": True,
        "baseline_run18": "p3_p4_plain_global_average_pool_mlp_dropout020",
        "scales": ["P3", "P4"],
        "attention_source": "sigmoid_main_predicted_semantic_logit",
        "attention_stop_gradient": True,
        "ground_truth_mask_as_classifier_input": False,
        "resize": "bilinear_align_corners_false",
        "global_pooling": "global_average",
        "local_pooling": "lesion_probability_weighted_average",
        "projection_dim": 128,
        "projection_sharing": "same_scale_projection_shared_between_global_and_local",
        "scale_fusion": "concatenate_projected_p3_p4",
        "gate": "two_way_softmax_mlp",
        "gate_hidden_dim": 128,
        "gate_initialization": "equal_half_weights",
        "gate_balance_regularizer": "classified_batch_mean_local_minus_global_logit_squared",
        "gate_balance_weight": 0.01,
        "gate_balance_scope": "official_classification_labels_only",
        "gate_balance_intent": "prevent_systematic_global_or_local_gate_collapse_without_penalizing_sample_variation",
        "attention_eps": 0.000001,
        "min_attention_mass_fraction": 0.0001,
        "empty_or_tiny_attention_fallback": "global_average",
        "classifier_only": True,
        "p5_descriptor": False,
        "semantic_branch_modified": False,
    }:
        raise ReleaseError("Frozen lg2 LGLA block changed")
    if config["amp_check"] != {
        "asset": "tools/yolo26n.pt",
        "sha256": PROTECTED_SHA256["tools/yolo26n.pt"],
        "working_directory": "tools",
        "runtime_download": "forbidden_local_asset_required",
    }:
        raise ReleaseError("lg2 local AMP-check asset contract changed")
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
        raise ReleaseError("lg2 delayed utility-gated early-stop policy changed")
    if config["selection"]["primary_alias"] != "best_feasible_joint":
        raise ReleaseError("lg2 primary selection alias changed")
    if float(config["selection"]["max_mel_to_nv_rate"]) != 0.1077844311:
        raise ReleaseError("lg2 mel->nv feasibility constraint changed")
    if float(config["selection"].get("min_binary_iou", -1.0)) != 0.70:
        raise ReleaseError("lg2 minimum binary IoU eligibility floor changed")
    if float(config["selection"].get("min_group_macro_f1", -1.0)) != 0.50:
        raise ReleaseError("lg2 minimum group Macro-F1 eligibility floor changed")
    if config["selection"].get("fitness_policy") != "unchanged_joint_minus_mel_to_nv_excess_only":
        raise ReleaseError("lg2 fitness policy changed")
    if config["screening_gate"] != {
        "comparison_row": "each_runs_best_joint_score_bundle_row",
        "baseline_run18": {
            "binary_iou": 0.76323,
            "image_macro_f1": 0.54417,
            "group_macro_f1": 0.61023,
            "group_recall": 0.59915,
            "mel_to_nv": 0.38776,
            "malignant_to_benign": 0.27536,
            "joint": 0.68246,
        },
        "required_deltas": {
            "min_group_macro_f1": 0.005,
            "min_joint": 0.003,
            "min_image_macro_f1": 0.0,
            "min_binary_iou": -0.005,
        },
        "attention_diagnostics": {
            "all_finite": True,
            "local_gate_mean_min": 0.05,
            "local_gate_mean_max": 0.95,
            "local_gate_std_min": 0.001,
        },
        "action_if_pass": "stop_and_return_complete_deltas_to_user_before_seed44_or_any_ablation",
        "action_if_borderline": "stop_and_return_complete_deltas_and_tradeoffs_before_seed44",
        "action_if_fail": "stop_and_return_complete_deltas_before_deciding_whether_seed44_is_informative",
    }:
        raise ReleaseError("lg2 screening gate changed")
    if config["checkpoint"]["bundle_version"] != "pifuv2_multitask_checkpoint_v2":
        raise ReleaseError("lg2 checkpoint bundle version changed")
    return config


def verify_protected_files() -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for relative, expected_hash in PROTECTED_SHA256.items():
        path = PROJECT_ROOT / Path(relative)
        if not path.is_file():
            raise ReleaseError(f"Protected lg2 input is missing: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ReleaseError(f"Protected lg2 input changed: {relative}: {actual_hash} != {expected_hash}")
        report[relative] = {"bytes": path.stat().st_size, "sha256": actual_hash}
    return report


def read_source_manifest() -> tuple[list[dict[str, str]], list[str]]:
    with SOURCE_DEVELOPMENT_MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    missing = REQUIRED_SOURCE_COLUMNS - set(fieldnames)
    if missing:
        raise ReleaseError(f"Accepted combined manifest is missing columns: {sorted(missing)}")
    return rows, fieldnames


def development_manifest_payload() -> tuple[bytes, dict[str, dict[str, int]]]:
    rows, fieldnames = read_source_manifest()
    source_split_counts = {split: sum(row["split"] == split for row in rows) for split in ("train", "val")}
    if source_split_counts != {"train": 23_462, "val": 1_744} or any(
        row["split"] not in {"train", "val"} for row in rows
    ):
        raise ReleaseError(f"Frozen Run22 development-only manifest split counts changed: {source_split_counts}")
    development_rows = rows
    ids = [row["isic_id"] for row in development_rows]
    if len(ids) != len(set(ids)):
        raise ReleaseError("Development projection contains duplicate isic_id")
    if any("pad_ufes" in json.dumps(row).lower() for row in development_rows):
        raise ReleaseError("PAD-UFES-20 appeared in the lg2 development manifest")
    if any(
        token in json.dumps(row, ensure_ascii=False).lower()
        for row in development_rows
        for token in ("isic2018_task1", "/test/", "\\test\\")
    ):
        raise ReleaseError("lg2 development manifest contains test/external paths")
    allowed_roles = {"imaplusplus_binary", "isic2019_classification_incremental"}
    if {row["dataset_role"] for row in development_rows} - allowed_roles:
        raise ReleaseError("Development projection contains an unregistered dataset role")
    counts: dict[str, dict[str, int]] = {}
    for split in ("train", "val"):
        selected = [row for row in development_rows if row["split"] == split]
        counts[split] = {
            "total": len(selected),
            "seg": sum(row["has_segmentation"] == "1" for row in selected),
            "cls": sum(row["has_classification"] == "1" for row in selected),
            "both": sum(row["has_segmentation"] == "1" and row["has_classification"] == "1" for row in selected),
        }
    if counts != EXPECTED_DEVELOPMENT_COUNTS:
        raise ReleaseError(f"Development supervision counts changed: {counts}")

    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(development_rows)
    return buffer.getvalue().encode("utf-8-sig"), counts


def development_data_payload() -> bytes:
    root = (PROJECT_ROOT / "Data").resolve().as_posix()
    manifest = DEVELOPMENT_MANIFEST.resolve().as_posix()
    taxonomy = (PROJECT_ROOT / "tools/classification_taxonomy_v1.yaml").resolve().as_posix()
    value = f"""# lg2 development-only partial-label data. Test and external candidates are absent.
path: {root}
train:
  - processed_imaplusplus_binary_v1/images/train
  - processed_isic2019_classification_incremental_v1/images/train
val:
  - processed_imaplusplus_binary_v1/images/val
  - processed_isic2019_classification_incremental_v1/images/val
masks_dir: masks
nc: 1
names:
  0: lesion
partial_label_manifest: {manifest}
classification_taxonomy: {taxonomy}
test: null
"""
    return value.encode("utf-8")


def validate_development_artifacts() -> dict[str, Any]:
    expected_manifest, counts = development_manifest_payload()
    expected_data = development_data_payload()
    if not DEVELOPMENT_MANIFEST.is_file() or DEVELOPMENT_MANIFEST.read_bytes() != expected_manifest:
        raise ReleaseError("lg2 development manifest is missing or differs from the deterministic projection")
    if not DEVELOPMENT_DATA.is_file() or DEVELOPMENT_DATA.read_bytes() != expected_data:
        raise ReleaseError("lg2 development data YAML is missing or differs from the locked payload")
    data = yaml.safe_load(DEVELOPMENT_DATA.read_text(encoding="utf-8"))
    if data.get("test", "unexpected") is not None:
        raise ReleaseError("lg2 development YAML must use test: null")
    serialized = json.dumps(data, ensure_ascii=False).lower()
    if "pad_ufes" in serialized or "isic2018_task1" in serialized or "/test" in serialized or "\\test" in serialized:
        raise ReleaseError("lg2 development YAML references test or an external candidate")
    return {
        "counts": counts,
        "development_manifest_sha256": sha256_file(DEVELOPMENT_MANIFEST),
        "development_data_sha256": sha256_file(DEVELOPMENT_DATA),
        "source_manifest_role": "frozen_run22_development_only",
        "source_rows_all_development": True,
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
        SOURCE_DEVELOPMENT_MANIFEST,
        PROJECT_ROOT / "tools" / "run22" / "run22_development_data.yaml",
        TOOLS_DIR / "imaplusplus_partial_multitask_v1.py",
        TOOLS_DIR / "imaplusplus_partial_multitask_utility_selection_v1.py",
        LG2_DIR / "imaplusplus_partial_multitask_lgla_v1_gate_balance_seed42.py",
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
        "lgla_v1": config["lgla_v1"],
        "amp_check": config["amp_check"],
        "early_stopping": config["early_stopping"],
        "checkpoint": config["checkpoint"],
        "selection": config["selection"],
        "screening_gate": config["screening_gate"],
        "development_counts": EXPECTED_DEVELOPMENT_COUNTS,
        "test_split": "absent_from_development_manifest_and_yaml_null",
    }
    return stable_fingerprint(behavior, paths), inputs


def cuda_status() -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ReleaseError(
            f"lg2 requires the verified single CUDA device; available={torch.cuda.is_available()} "
            f"count={torch.cuda.device_count()}"
        )
    name = torch.cuda.get_device_name(0)
    if "RTX 5080" not in name:
        raise ReleaseError(f"lg2 expected NVIDIA GeForce RTX 5080, found {name}")
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
    if str(TOOLS_DIR) not in sys.path:
        sys.path.insert(0, str(TOOLS_DIR))
    from imaplusplus_partial_multitask_lgla_v1_gate_balance_seed42 import (
        GateBalancedPartialLabelMultiTaskLoss,
        LGLAV1MultiTaskModel,
        LesionGuidedGlobalLocalClassificationHead,
    )
    from ultralytics.nn.tasks import load_checkpoint
    from verify_partial_multitask_v1 import real_partial_batch

    ultra = config["ultralytics"]
    weights = resolve_project_path(ultra["model"])
    batch, metadata = real_partial_batch(DEVELOPMENT_DATA, "train", int(ultra["imgsz"]))
    expanded = expand_role_batch(batch, int(ultra["batch"]))
    device = torch.device("cuda:0")
    pretrained, _ = load_checkpoint(weights, device="cpu")
    model = LGLAV1MultiTaskModel(
        cfg=pretrained.yaml,
        nc=1,
        num_diseases=8,
        image_cls_weight=float(config["multitask"]["image_cls_weight"]),
        projection_dim=int(config["lgla_v1"]["projection_dim"]),
        gate_hidden_dim=int(config["lgla_v1"]["gate_hidden_dim"]),
        attention_eps=float(config["lgla_v1"]["attention_eps"]),
        min_attention_mass_fraction=float(config["lgla_v1"]["min_attention_mass_fraction"]),
        gate_balance_weight=float(config["lgla_v1"]["gate_balance_weight"]),
        verbose=False,
    )
    transferred = model.load(pretrained)
    model.args = SimpleNamespace(data=str(DEVELOPMENT_DATA))
    model.to(device).train()
    for module in model.modules():
        if isinstance(module, (nn.modules.batchnorm._BatchNorm, nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.eval()
    criterion = GateBalancedPartialLabelMultiTaskLoss(model)
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
        raise ReleaseError(f"lg2 forward/loss preflight produced non-finite values: total={total}, items={items}")
    changed = [key for key, value in model.state_dict().items() if not torch.equal(before[key], value.detach().cpu())]
    if changed:
        raise ReleaseError(f"No-training preflight changed model state: {changed[:5]}")
    head = model.model[-1]
    if list(head.f) != [16, 13]:
        raise ReleaseError(f"lg2 P3/P4 head inputs changed: f={head.f}")
    if not isinstance(head, LesionGuidedGlobalLocalClassificationHead):
        raise ReleaseError(f"lg2 head class changed: {type(head).__name__}")
    if len(head.scale_projections) != 2 or head.projection_dim != 128 or head.gate_hidden_dim != 128:
        raise ReleaseError("lg2 projection or gate structure changed")
    if head.last_gate_weights is None or not torch.isfinite(head.last_gate_weights).all():
        raise ReleaseError("lg2 did not expose finite detached gate weights")
    if not torch.allclose(head.last_gate_weights.sum(1), torch.ones_like(head.last_gate_weights[:, 0])):
        raise ReleaseError("lg2 gate weights do not sum to one")
    if head.last_attention_mass_fraction is None or not torch.isfinite(head.last_attention_mass_fraction).all():
        raise ReleaseError("lg2 attention mass diagnostics are missing or non-finite")
    forbidden = [name for name in ("cbam", "channel_attention", "p5_layer") if hasattr(head, name)]
    if forbidden:
        raise ReleaseError(f"lg2 unexpectedly contains non-LGLA/P5 attributes: {forbidden}")
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
        "gate_balance_weight": float(config["lgla_v1"]["gate_balance_weight"]),
        "gate_balance_loss": float(items[-1].detach().cpu()),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "classification_head_parameters": sum(parameter.numel() for parameter in head.image_classifier.parameters()),
        "attention_head_inputs": list(head.f),
        "attention_modules_present": True,
        "descriptor": "p3_p4_detached_predicted_lesion_global_local_softmax_gate",
        "projection_dim": int(head.projection_dim),
        "gate_hidden_dim": int(head.gate_hidden_dim),
        "attention_stop_gradient": True,
        "ground_truth_mask_as_classifier_input": False,
        "empty_attention_fallback": "global_average",
        "gate_weights_shape": list(head.last_gate_weights.shape),
        "gate_weights_finite": bool(torch.isfinite(head.last_gate_weights).all()),
        "gate_weight_sums": [float(value) for value in head.last_gate_weights.sum(1).cpu()],
        "attention_mass_fraction_shape": list(head.last_attention_mass_fraction.shape),
        "attention_mass_fraction_finite": bool(torch.isfinite(head.last_attention_mass_fraction).all()),
        "p5_descriptor": False,
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
    del criterion, model, pretrained, expanded, images, predictions, total, items, before
    torch.cuda.empty_cache()
    return report


def prepare_only(config: dict[str, Any]) -> None:
    run_dir = resolve_project_path(config["ultralytics"]["project"]) / config["ultralytics"]["name"]
    if run_dir.exists():
        raise ReleaseError(f"lg2 result directory already exists; preparation refuses to touch it: {run_dir}")
    verify_protected_files()
    manifest_payload, _ = development_manifest_payload()
    atomic_write(DEVELOPMENT_MANIFEST, manifest_payload)
    atomic_write(DEVELOPMENT_DATA, development_data_payload())
    development = validate_development_artifacts()
    cuda = cuda_status()
    forward = no_training_gpu_forward_preflight(config)
    fingerprint, lock_inputs = compute_fingerprint(config)
    report = {
        "status": "PASS",
        "scope": "final lg2 preparation; no optimizer, backward, training, test, or external evaluation",
        "created_local_time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "config_sha256": sha256_file(CONFIG_PATH),
        "fingerprint": fingerprint,
        "protected_files": verify_protected_files(),
        "lock_inputs": lock_inputs,
        "development": development,
        "forbidden_data_access": {
            "imaplusplus_test": False,
            "isic2018_external": False,
            "pad_ufes_20": False,
            "source": "frozen Run22 development-only manifest",
        },
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
        raise ReleaseError("lg2 ready/preflight marker is missing; do not train")
    ready = json.loads(READY_MARKER.read_text(encoding="utf-8"))
    if ready.get("status") != "PASS" or ready.get("ready_to_click_pycharm") is not True:
        raise ReleaseError("lg2 ready marker is not PASS")
    if ready.get("formal_training_started") is not False:
        raise ReleaseError("lg2 ready marker does not represent an untouched pre-training state")
    validate_development_artifacts()
    verify_protected_files()
    fingerprint, lock_inputs = compute_fingerprint(config)
    if fingerprint != ready.get("fingerprint") or lock_inputs != ready.get("lock_inputs"):
        raise ReleaseError("lg2 lock inputs changed after final preflight")
    return ready, fingerprint


def validate_release_document() -> None:
    if not RELEASE_DOCUMENT.is_file():
        raise ReleaseError(f"Latest lg2 release document is missing: {RELEASE_DOCUMENT}")
    text = RELEASE_DOCUMENT.read_text(encoding="utf-8")
    required = {
        "`PROJECT_LG2_GATE=PASS`",
        "`LG2_READY_TO_CLICK_GATE=PASS`",
        "`USER_CONFIRMATION_RECORDED=TRUE`",
        "`TEST_METRICS_USED=FALSE`",
        "`EXTERNAL_METRICS_USED=FALSE`",
    }
    missing = sorted(value for value in required if value not in text)
    if missing:
        raise ReleaseError(f"Latest lg2 release document is not a valid PASS: {missing}")


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
    for path in (TOOLS_DIR, LG2_DIR):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from imaplusplus_partial_multitask_lgla_v1_gate_balance_seed42 import (
        DelayedUtilityGatedLGLAV1Trainer,
    )
    from multitask_checkpoint_contract import CheckpointBundleManager, CheckpointContractError

    ultra = dict(config["ultralytics"])
    ultra.update(config["augmentations"])
    ultra["model"] = str(resolve_project_path(ultra["model"]))
    ultra["data"] = str(DEVELOPMENT_DATA.resolve())
    ultra["project"] = str(resolve_project_path(ultra["project"]))
    run_dir = Path(ultra["project"]) / ultra["name"]
    bundle_root = run_dir / "checkpoint_bundles"
    complete_marker = run_dir / "LG2_TRAINING_COMPLETE.json"
    if complete_marker.exists():
        raise ReleaseError(f"lg2 already completed; refusing to train again: {complete_marker}")

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
                    f"lg2 output exists but no complete compatible auto-resume bundle was found: {error}"
                ) from error
        elif run_dir.exists():
            raise ReleaseError(
                "lg2 directory exists without a complete epoch bundle. Refusing to overwrite a first-epoch "
                f"incomplete run: {run_dir}"
            )
    elif normalized_resume.lower() == "none":
        if run_dir.exists():
            raise ReleaseError(f"--resume none requires an absent lg2 output directory: {run_dir}")
    else:
        candidate = Path(normalized_resume)
        if not candidate.is_absolute():
            candidate = (PROJECT_ROOT / candidate).resolve()
        resume_bundle = manager.resolve(candidate)

    if resume_bundle is not None:
        ultra["resume"] = str(resume_bundle.trainer_checkpoint)
        ultra["exist_ok"] = True
        print(f"lg2 resume bundle: {resume_bundle.path} (completed epoch {resume_bundle.epoch})")
    else:
        ultra["exist_ok"] = False
        print("lg2 fresh start: no prior result directory or complete bundle was found.")

    checkpoint_contract = {
        "enabled": True,
        "fingerprint": fingerprint,
        "root": bundle_root,
        "keep_last": int(config["checkpoint"]["keep_last"]),
        "resume_bundle": str(resume_bundle.path) if resume_bundle is not None else None,
    }
    trainer = DelayedUtilityGatedLGLAV1Trainer(
        overrides=ultra,
        checkpoint_contract=checkpoint_contract,
        image_cls_weight=float(config["multitask"]["image_cls_weight"]),
        lgla_v1={
            "projection_dim": int(config["lgla_v1"]["projection_dim"]),
            "gate_hidden_dim": int(config["lgla_v1"]["gate_hidden_dim"]),
            "attention_eps": float(config["lgla_v1"]["attention_eps"]),
            "min_attention_mass_fraction": float(config["lgla_v1"]["min_attention_mass_fraction"]),
            "gate_balance_weight": float(config["lgla_v1"]["gate_balance_weight"]),
        },
        selection={
            "max_mel_to_nv_rate": float(config["selection"]["max_mel_to_nv_rate"]),
            "min_binary_iou": float(config["selection"]["min_binary_iou"]),
            "min_group_macro_f1": float(config["selection"]["min_group_macro_f1"]),
        },
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(
        run_dir / "LG2_LAUNCH_RECORD.json",
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
            run_dir / "LG2_EXCEPTION_NOT_COMPLETE.json",
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
            run_dir / "LG2_INTERRUPTED_SAFE.json",
            {
                "status": "INTERRUPTED_AT_SAFE_COMPLETED_EPOCH",
                "signal": trainer._signal_latch.metadata(),
                "aliases": aliases,
                "resume_instruction": "Run the same train_lg2_from_pycharm.py again; default resume=auto.",
            },
        )
        print("lg2 stopped at a safe completed-epoch boundary. Run this same script again to auto-resume.")
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
    print("lg2 training finished or stopped normally by early stopping. No test/external evaluation was run.")
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
    # Portable copy: reruns use new outputs; historical release gates are audit records.
    import sys as _repro_sys
    from pathlib import Path as _ReproPath
    _repro_sys.path.insert(0, str(_ReproPath(__file__).resolve().parents[2] / "reproduction"))
    from run import main as _repro_main
    _repro_args = ["train", "--run", "lg2"] + [("--prepare" if a == '--prepare-only' else a) for a in _repro_sys.argv[1:]]
    _repro_main(_repro_args)



