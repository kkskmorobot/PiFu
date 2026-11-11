#!/usr/bin/env python3
"""Canonical PyCharm entry for Run34 development-only fixed-cost evaluation.

Running this file without arguments performs validation inference only. It can
never train, access a test split, or evaluate an external/PAD dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = PROJECT_ROOT / "tools"
RUN34_DIR = Path(__file__).resolve().parent
CONFIG_PATH = RUN34_DIR / "run34_preregistered_config_v1.yaml"
PROTOCOL_PATH = RUN34_DIR / "RUN34_PROTOCOL_2026-09-10.md"
REVIEW_PATH = RUN34_DIR / "RUN33_RESULT_REVIEW_AND_RUN34_DECISION_2026-09-10.md"
TEST_PATH = RUN34_DIR / "test_run34_fixed_cost_decision_v1.py"
COMPAT_ENTRY_PATH = RUN34_DIR / "run_run34_from_pycharm.py"
EXACT_REPRO_PREFLIGHT_PATH = RUN34_DIR / "preflight_run34_exact_ema_reproduction_v1.py"
RELEASE_PATH = RUN34_DIR / "PROJECT_RUN34_RELEASE_2026-09-10.md"
PREFLIGHT_REPORT = RUN34_DIR / "RUN34_PREFLIGHT_REPORT.json"
READY_MARKER = RUN34_DIR / "RUN34_READY_TO_RUN.json"
LOCKED_CONFIG_SHA256 = "2952bb706ee8eacd24d75348be2d0dc1a8e8df6d7204a64fd0cc932ee15dbf0d"
CLASS_NAMES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc", "scc")
MALIGNANT_IDS = frozenset({1, 4, 7})
MEL_ID = 4
NV_ID = 5

# The script is launched from tools/run34, while the shared model modules live
# directly in tools. Changing cwd later does not retroactively update sys.path.
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))


class ReleaseError(RuntimeError):
    """Raised when the immutable Run34 contract is not satisfied."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate frozen inputs and perform one no-training GPU forward pass.",
    )
    return parser.parse_args()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def load_config() -> dict[str, Any]:
    actual = sha256_file(CONFIG_PATH)
    if actual != LOCKED_CONFIG_SHA256:
        raise ReleaseError(f"Run34 config hash changed: {actual}")
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    contract = config["contract"]
    if contract["training_permitted"] is not False or contract["split"] != "val":
        raise ReleaseError("Run34 must remain val-only with training_permitted=false")
    if contract["test_policy"] != "forbidden_absent_from_entrypoint":
        raise ReleaseError("Run34 test boundary changed")
    return config


def build_cost_matrix(config: dict[str, Any]) -> np.ndarray:
    rule = config["decision_rule"]
    if tuple(rule["classes"]) != CLASS_NAMES:
        raise ReleaseError("Run34 class order changed")
    if tuple(rule["malignant_classes"]) != ("bcc", "mel", "scc"):
        raise ReleaseError("Run34 malignant class set changed")
    matrix = np.full((len(CLASS_NAMES), len(CLASS_NAMES)), float(rule["ordinary_error_cost"]))
    np.fill_diagonal(matrix, float(rule["correct_cost"]))
    for truth in MALIGNANT_IDS:
        for prediction in range(len(CLASS_NAMES)):
            if prediction not in MALIGNANT_IDS:
                matrix[truth, prediction] = float(rule["malignant_to_benign_cost"])
    matrix[MEL_ID, NV_ID] = float(rule["mel_to_nv_cost"])
    return matrix


def cost_predictions(probabilities: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if probabilities.ndim != 2 or probabilities.shape[1] != len(CLASS_NAMES):
        raise ReleaseError(f"Expected probabilities [N,8], found {probabilities.shape}")
    return np.argmin(probabilities @ matrix, axis=1).astype(np.int64)


def validate_data_contract(config: dict[str, Any]) -> dict[str, Any]:
    data_config = config["data"]
    data_yaml = resolve_project_path(data_config["yaml"])
    manifest = resolve_project_path(data_config["manifest"])
    taxonomy = resolve_project_path(data_config["taxonomy"])
    for path, expected in (
        (data_yaml, data_config["yaml_sha256"]),
        (manifest, data_config["manifest_sha256"]),
        (taxonomy, data_config["taxonomy_sha256"]),
    ):
        if not path.is_file() or sha256_file(path) != expected:
            raise ReleaseError(f"Frozen Run34 data input changed: {path}")
    parsed = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if parsed.get("test") is not None:
        raise ReleaseError("Run34 data YAML exposes a test split")
    if Path(parsed["partial_label_manifest"]).resolve() != manifest:
        raise ReleaseError("Run34 manifest path differs from frozen contract")
    if Path(parsed["classification_taxonomy"]).resolve() != taxonomy:
        raise ReleaseError("Run34 taxonomy path differs from frozen contract")
    allowed_roles = set(data_config["allowed_roles"])
    allowed_splits = set(data_config["allowed_splits"])
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            role = row["dataset_role"]
            split = row["split"]
            if role not in allowed_roles or split not in allowed_splits:
                raise ReleaseError(f"Forbidden manifest row: role={role}, split={split}")
            joined = " ".join(str(value) for value in row.values()).lower()
            if "pad-ufes" in joined or "pad_ufes" in joined:
                raise ReleaseError("PAD-UFES-20 appeared in Run34 manifest")
            counts[split]["total"] += 1
            counts[split]["classification"] += int(row["has_classification"] == "1")
            counts[split]["segmentation"] += int(row["has_segmentation"] == "1")
    normalized = {split: dict(values) for split, values in counts.items()}
    if normalized.get("val", {}).get("total") != 1744:
        raise ReleaseError(f"Run34 val count changed: {normalized}")
    return {
        "yaml": str(data_yaml),
        "manifest": str(manifest),
        "taxonomy": str(taxonomy),
        "counts": normalized,
        "test_present": False,
        "external_present": False,
        "pad_ufes_20_present": False,
    }


def validate_checkpoints(config: dict[str, Any]) -> list[dict[str, Any]]:
    reports = []
    for spec in config["checkpoints"]:
        bundle = resolve_project_path(spec["bundle"])
        manifest_path = bundle / "manifest.json"
        success_path = bundle / "_SUCCESS"
        trainer = bundle / "trainer.pt"
        runtime = bundle / "runtime_state.pt"
        if not all(path.is_file() for path in (manifest_path, success_path, trainer, runtime)):
            raise ReleaseError(f"Incomplete Run34 source bundle: {bundle}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        success = json.loads(success_path.read_text(encoding="utf-8"))
        expected_epoch = int(spec["zero_based_epoch"])
        expected_fingerprint = str(spec["fingerprint"])
        if manifest["epoch"] != expected_epoch or success["epoch"] != expected_epoch:
            raise ReleaseError(f"Bundle epoch changed: {bundle}")
        if manifest["fingerprint"] != expected_fingerprint or success["fingerprint"] != expected_fingerprint:
            raise ReleaseError(f"Bundle fingerprint changed: {bundle}")
        actual = {
            "trainer.pt": (trainer.stat().st_size, sha256_file(trainer)),
            "runtime_state.pt": (runtime.stat().st_size, sha256_file(runtime)),
        }
        expected = {
            "trainer.pt": (int(spec["trainer_bytes"]), str(spec["trainer_sha256"])),
            "runtime_state.pt": (int(spec["runtime_bytes"]), str(spec["runtime_sha256"])),
        }
        if actual != expected:
            raise ReleaseError(f"Bundle file size/hash changed: {bundle}")
        for name, (size, digest) in actual.items():
            recorded = manifest["files"][name]
            if int(recorded["bytes"]) != size or str(recorded["sha256"]) != digest:
                raise ReleaseError(f"Bundle manifest mismatch: {bundle / name}")
        reports.append(
            {
                "run": int(spec["run"]),
                "seed": int(spec["seed"]),
                "bundle": str(bundle),
                "fingerprint": expected_fingerprint,
                "trainer_sha256": actual["trainer.pt"][1],
                "runtime_sha256": actual["runtime_state.pt"][1],
            }
        )
    return reports


def lock_fingerprint(config: dict[str, Any]) -> str:
    payload = {
        "config_sha256": sha256_file(CONFIG_PATH),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "review_sha256": sha256_file(REVIEW_PATH),
        "test_sha256": sha256_file(TEST_PATH),
        "compat_entry_sha256": sha256_file(COMPAT_ENTRY_PATH),
        "exact_repro_preflight_sha256": sha256_file(EXACT_REPRO_PREFLIGHT_PATH),
        "data": config["data"],
        "checkpoints": config["checkpoints"],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def tensors_from(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        return [tensor for item in value for tensor in tensors_from(item)]
    if isinstance(value, dict):
        return [tensor for item in value.values() for tensor in tensors_from(item)]
    return []


def load_exact_ema_model(bundle: Path):
    """Load structure from trainer.pt, then restore the lossless fp32 EMA."""
    from ultralytics import YOLO

    bundle = Path(bundle).resolve()
    wrapper = YOLO(str(bundle / "trainer.pt"), task="semantic")
    model = wrapper.model.float().cpu()
    serialized_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    runtime = torch.load(bundle / "runtime_state.pt", map_location="cpu", weights_only=False)
    exact_state = runtime.get("training_state", {}).get("ema")
    if not isinstance(exact_state, dict) or set(exact_state) != set(serialized_state):
        raise ReleaseError(f"Lossless EMA state is missing or incompatible: {bundle}")
    differing = 0
    max_abs_difference = 0.0
    for name, serialized in serialized_state.items():
        exact = exact_state[name].detach().cpu().to(serialized.dtype)
        if serialized.is_floating_point() and not torch.equal(serialized, exact):
            differing += 1
            max_abs_difference = max(max_abs_difference, float((serialized - exact).abs().max()))
    model.load_state_dict(exact_state, strict=True)
    loaded = model.state_dict()
    if any(not torch.equal(loaded[name].detach().cpu(), exact_state[name].detach().cpu()) for name in loaded):
        raise ReleaseError(f"Lossless EMA did not load exactly: {bundle}")
    model.eval()
    return model, {
        "structure_source": str(bundle / "trainer.pt"),
        "state_source": str(bundle / "runtime_state.pt") + "::training_state.ema",
        "serialized_fp16_differing_float_tensors": differing,
        "serialized_fp16_max_abs_difference": max_abs_difference,
        "exact_ema_loaded": True,
    }


def no_training_model_preflight(config: dict[str, Any]) -> dict[str, Any]:
    for path in (TOOLS_DIR, RUN34_DIR):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from imaplusplus_partial_multitask_v1 import BinaryPartialMultiTaskModel
    bundle = resolve_project_path(config["checkpoints"][0]["bundle"])
    model, weight_source = load_exact_ema_model(bundle)
    if not isinstance(model, BinaryPartialMultiTaskModel):
        raise ReleaseError(f"Expected frozen plain model, found {type(model).__name__}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    before = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    model.to(device).eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 640, 640, device=device))
    tensors = tensors_from(output)
    if not tensors or any(not torch.isfinite(tensor).all().item() for tensor in tensors):
        raise ReleaseError("Run34 no-training forward produced non-finite output")
    after = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    if any(not torch.equal(before[name], after[name]) for name in before):
        raise ReleaseError("Run34 preflight modified model state")
    return {
        "device": str(device),
        "model_class": type(model).__name__,
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "output_shapes": [list(tensor.shape) for tensor in tensors],
        "all_finite": True,
        "model_state_unchanged": True,
        "weight_source": weight_source,
        "optimizer_created": False,
        "backward_called": False,
        "training_called": False,
    }


def prepare_only(config: dict[str, Any]) -> None:
    output = resolve_project_path(config["evaluation"]["output"])
    if output.exists():
        raise ReleaseError(f"Run34 formal output already exists: {output}")
    report = {
        "status": "PASS",
        "prepared_local_time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "config_sha256": sha256_file(CONFIG_PATH),
        "fingerprint": lock_fingerprint(config),
        "data": validate_data_contract(config),
        "checkpoints": validate_checkpoints(config),
        "model_forward": no_training_model_preflight(config),
        "formal_output_exists": False,
        "training_performed": False,
        "test_metrics_used": False,
        "external_metrics_used": False,
        "pad_ufes_20_used": False,
    }
    atomic_json(PREFLIGHT_REPORT, report)
    locked_files = {
        path.name: sha256_file(path)
        for path in (
            CONFIG_PATH,
            Path(__file__).resolve(),
            COMPAT_ENTRY_PATH,
            EXACT_REPRO_PREFLIGHT_PATH,
            PROTOCOL_PATH,
            REVIEW_PATH,
            TEST_PATH,
        )
    }
    ready = {
        "status": "READY_TO_RUN_DEVELOPMENT_ONLY_EVALUATION",
        "prepared_local_time": report["prepared_local_time"],
        "config_sha256": report["config_sha256"],
        "fingerprint": report["fingerprint"],
        "locked_files": locked_files,
        "formal_output_exists": False,
        "training_permitted": False,
        "test_metrics_used": False,
        "external_metrics_used": False,
        "pad_ufes_20_used": False,
    }
    atomic_json(READY_MARKER, ready)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("RUN34_PREPARE_ONLY=PASS; no training/test/external evaluation was run.")


def validate_ready(config: dict[str, Any]) -> dict[str, Any]:
    if not READY_MARKER.is_file() or not PREFLIGHT_REPORT.is_file():
        raise ReleaseError("Run34 ready/preflight marker is missing")
    ready = json.loads(READY_MARKER.read_text(encoding="utf-8"))
    if ready.get("status") != "READY_TO_RUN_DEVELOPMENT_ONLY_EVALUATION":
        raise ReleaseError("Run34 ready marker is not PASS")
    current = {
        path.name: sha256_file(path)
        for path in (
            CONFIG_PATH,
            Path(__file__).resolve(),
            COMPAT_ENTRY_PATH,
            EXACT_REPRO_PREFLIGHT_PATH,
            PROTOCOL_PATH,
            REVIEW_PATH,
            TEST_PATH,
        )
    }
    if current != ready.get("locked_files"):
        raise ReleaseError("Run34 locked files changed after final preflight")
    if ready.get("config_sha256") != sha256_file(CONFIG_PATH):
        raise ReleaseError("Run34 config changed after final preflight")
    if ready.get("fingerprint") != lock_fingerprint(config):
        raise ReleaseError("Run34 fingerprint changed after final preflight")
    validate_data_contract(config)
    validate_checkpoints(config)
    return ready


def validate_release_document() -> None:
    if not RELEASE_PATH.is_file():
        raise ReleaseError("Run34 release document is missing")
    text = RELEASE_PATH.read_text(encoding="utf-8")
    required = {
        "`PROJECT_RUN34_GATE=PASS`",
        "`RUN34_READY_TO_CLICK_GATE=PASS`",
        "`USER_CONFIRMATION_RECORDED=TRUE`",
        "`TRAINING_PERFORMED=FALSE`",
        "`TEST_METRICS_USED=FALSE`",
        "`EXTERNAL_METRICS_USED=FALSE`",
        "`PAD_UFES_20_USED=FALSE`",
    }
    missing = sorted(required.difference(text.splitlines()))
    if missing:
        raise ReleaseError(f"Run34 release document is not a valid PASS: {missing}")


def confusion(targets: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    for truth, prediction in zip(targets.tolist(), predictions.tolist()):
        matrix[int(truth), int(prediction)] += 1
    return matrix


def metric_view(matrix: np.ndarray) -> dict[str, float]:
    from imaplusplus_partial_multitask_v1 import classification_metrics

    raw = classification_metrics(matrix)
    return {
        "accuracy": float(raw["accuracy"]),
        "macro_f1": float(raw["macro_f1"]),
        "macro_recall": float(raw["macro_recall"]),
        "mel_to_nv": int(raw["mel_to_nv"]),
        "mel_to_nv_rate": float(raw["mel_to_nv_rate"]),
        "malignant_to_benign": int(raw["malignant_to_benign"]),
        "malignant_to_benign_rate": float(raw["malignant_to_benign_rate"]),
    }


def aggregate_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["group_id"])].append(row)
    result = []
    for group_id, members in sorted(grouped.items()):
        targets = {int(row["target"]) for row in members}
        if len(targets) != 1:
            raise ReleaseError(f"Validation group spans multiple targets: {group_id}")
        result.append(
            {
                "group_id": group_id,
                "target": targets.pop(),
                "probabilities": np.mean([row["probabilities"] for row in members], axis=0),
                "images": len(members),
            }
        )
    return result


def write_prediction_csv(path: Path, rows: list[dict[str, Any]], matrix: np.ndarray) -> None:
    serializable = []
    for row in rows:
        probabilities = np.asarray(row["probabilities"], dtype=np.float64)
        argmax_id = int(probabilities.argmax())
        cost_id = int(cost_predictions(probabilities[None, :], matrix)[0])
        item = {key: value for key, value in row.items() if key != "probabilities"}
        item.update(
            {
                "true_class": CLASS_NAMES[int(row["target"])],
                "argmax_id": argmax_id,
                "argmax_class": CLASS_NAMES[argmax_id],
                "fixed_cost_id": cost_id,
                "fixed_cost_class": CLASS_NAMES[cost_id],
            }
        )
        item.update({f"score_{name}": float(probabilities[index]) for index, name in enumerate(CLASS_NAMES)})
        serializable.append(item)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(serializable[0]))
        writer.writeheader()
        writer.writerows(serializable)


def validate_one_checkpoint(spec: dict[str, Any], config: dict[str, Any], save_dir: Path) -> dict[str, Any]:
    from imaplusplus_partial_multitask_v1 import (
        BinaryPartialMultiTaskModel,
        PartialLabelMultiTaskValidator,
        PartialLabelSemanticDataset,
    )
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_dataloader
    from ultralytics.data.utils import check_det_dataset

    bundle = resolve_project_path(spec["bundle"])
    weights = bundle / "trainer.pt"
    data_yaml = resolve_project_path(config["data"]["yaml"])
    settings = config["evaluation"]
    model, weight_source = load_exact_ema_model(bundle)
    if not isinstance(model, BinaryPartialMultiTaskModel):
        raise ReleaseError(f"Run{spec['run']} is not the frozen Plain model")
    if tuple(getattr(model, "classification_names", ())) != CLASS_NAMES:
        raise ReleaseError(f"Run{spec['run']} taxonomy/order mismatch")
    data = check_det_dataset(str(data_yaml), split="val")
    if data.get("test") is not None:
        raise ReleaseError("Run34 loader unexpectedly exposes test")
    manifest = Path(data["partial_label_manifest"]).resolve()
    cfg = get_cfg(
        overrides={
            "task": "semantic",
            "mode": "val",
            "model": str(weights),
            "data": str(data_yaml),
            "split": "val",
            "imgsz": int(settings["imgsz"]),
            "batch": int(settings["batch"]),
            "workers": int(settings["workers"]),
            "device": str(settings["device"]),
            "plots": False,
            "save_json": False,
            "verbose": False,
        }
    )
    dataset = PartialLabelSemanticDataset(
        img_path=data["val"],
        imgsz=int(settings["imgsz"]),
        batch_size=int(settings["batch"]),
        augment=False,
        hyp=cfg,
        rect=True,
        cache=False,
        single_cls=False,
        stride=max(int(model.stride.max()), 32),
        pad=0.0,
        prefix=f"Run{spec['run']} val: ",
        task="semantic",
        classes=None,
        data=data,
        fraction=1.0,
        partial_label_manifest=manifest,
        expected_split="val",
    )
    loader = build_dataloader(
        dataset,
        batch=int(settings["batch"]),
        workers=int(settings["workers"]),
        shuffle=False,
        rank=-1,
        drop_last=False,
    )
    validator = PartialLabelMultiTaskValidator(
        dataloader=loader,
        save_dir=save_dir,
        args=cfg,
        selection={"max_mel_to_nv_rate": float(config["selection"]["max_mel_to_nv_rate"])},
    )
    device = torch.device(f"cuda:{settings['device']}" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    # Reproduce the original training-time validation path exactly: unfused fp32
    # EMA weights with CUDA autocast. Passing model= would invoke standalone
    # AutoBackend, fuse Conv+BN, and change boundary-adjacent group decisions.
    frozen_trainer = SimpleNamespace(
        device=device,
        data=data,
        amp=device.type == "cuda",
        ema=SimpleNamespace(ema=model),
        model=model,
        args=SimpleNamespace(compile=False),
        loss_items=torch.zeros(4, device=device),
        stopper=SimpleNamespace(possible_stop=False),
        epoch=0,
        epochs=1,
        world_size=1,
        label_loss_items=lambda _loss, prefix="val": {},
    )
    flat = validator(frozen_trainer)
    image_rows = [
        {
            "isic_id": str(row["isic_id"]),
            "group_id": str(row["group_id"]),
            "target": int(row["target"]),
            "probabilities": np.asarray(row["probabilities"], dtype=np.float64),
        }
        for row in validator.classification_rows
    ]
    group_rows = aggregate_groups(image_rows)
    matrix = build_cost_matrix(config)
    image_targets = np.asarray([row["target"] for row in image_rows], dtype=np.int64)
    image_probabilities = np.asarray([row["probabilities"] for row in image_rows], dtype=np.float64)
    group_targets = np.asarray([row["target"] for row in group_rows], dtype=np.int64)
    group_probabilities = np.asarray([row["probabilities"] for row in group_rows], dtype=np.float64)
    image_argmax = metric_view(confusion(image_targets, image_probabilities.argmax(axis=1)))
    image_cost = metric_view(confusion(image_targets, cost_predictions(image_probabilities, matrix)))
    group_argmax = metric_view(confusion(group_targets, group_probabilities.argmax(axis=1)))
    group_cost = metric_view(confusion(group_targets, cost_predictions(group_probabilities, matrix)))
    binary_iou = float(flat["metrics/binary_lesion_iou"])
    argmax_joint = math.sqrt(max(binary_iou, 0.0) * max(group_argmax["macro_f1"], 0.0))
    cost_joint = math.sqrt(max(binary_iou, 0.0) * max(group_cost["macro_f1"], 0.0))
    actual_argmax = {
        "binary_iou": binary_iou,
        "image_macro_f1": image_argmax["macro_f1"],
        "group_macro_f1": group_argmax["macro_f1"],
        "group_recall": group_argmax["macro_recall"],
        "mel_to_nv": group_argmax["mel_to_nv_rate"],
        "malignant_to_benign": group_argmax["malignant_to_benign_rate"],
        "joint": argmax_joint,
    }
    differences = {
        name: actual_argmax[name] - float(expected)
        for name, expected in spec["expected_argmax"].items()
    }
    if any(abs(value) > 5e-4 for value in differences.values()):
        raise ReleaseError(f"Run{spec['run']} argmax reproduction mismatch: {differences}")
    write_prediction_csv(save_dir / "image_predictions.csv", image_rows, matrix)
    write_prediction_csv(save_dir / "group_predictions.csv", group_rows, matrix)
    return {
        "run": int(spec["run"]),
        "seed": int(spec["seed"]),
        "visible_epoch": int(spec["visible_epoch"]),
        "bundle": str(bundle),
        "weight_source": weight_source,
        "validation_mode": "training_path_unfused_fp32_ema_cuda_autocast",
        "binary_iou": binary_iou,
        "argmax": {"image": image_argmax, "group": group_argmax, "joint": argmax_joint},
        "fixed_cost": {"image": image_cost, "group": group_cost, "joint": cost_joint},
        "delta_fixed_cost_minus_argmax": {
            "group_macro_f1": group_cost["macro_f1"] - group_argmax["macro_f1"],
            "group_recall": group_cost["macro_recall"] - group_argmax["macro_recall"],
            "mel_to_nv": group_cost["mel_to_nv_rate"] - group_argmax["mel_to_nv_rate"],
            "malignant_to_benign": group_cost["malignant_to_benign_rate"] - group_argmax["malignant_to_benign_rate"],
            "joint": cost_joint - argmax_joint,
        },
        "argmax_reproduction_delta": differences,
        "training_performed": False,
        "test_metrics_used": False,
        "external_metrics_used": False,
    }


def mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def acceptance(results: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    deltas = [row["delta_fixed_cost_minus_argmax"] for row in results]
    mean_delta = {name: mean([float(row[name]) for row in deltas]) for name in deltas[0]}
    gate = config["acceptance_gate"]
    checks = {
        "each_seed_non_increasing_mel_to_nv": all(row["mel_to_nv"] <= 1e-12 for row in deltas),
        "each_seed_non_increasing_malignant_to_benign": all(
            row["malignant_to_benign"] <= 1e-12 for row in deltas
        ),
        "mean_delta_mel_to_nv": mean_delta["mel_to_nv"] <= float(gate["min_mean_delta_mel_to_nv"]),
        "mean_delta_malignant_to_benign": mean_delta["malignant_to_benign"]
        <= float(gate["min_mean_delta_malignant_to_benign"]),
        "mean_delta_group_macro_f1": mean_delta["group_macro_f1"]
        >= float(gate["min_mean_delta_group_macro_f1"]),
        "mean_delta_joint": mean_delta["joint"] >= float(gate["min_mean_delta_joint"]),
        "each_seed_delta_group_macro_f1": all(
            row["group_macro_f1"] >= float(gate["min_each_seed_delta_group_macro_f1"]) for row in deltas
        ),
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "checks": checks,
        "mean_delta": mean_delta,
        "action": gate["action_if_pass"] if passed else gate["action_if_fail"],
    }


def run_formal(config: dict[str, Any]) -> None:
    validate_release_document()
    ready = validate_ready(config)
    output = resolve_project_path(config["evaluation"]["output"])
    partial = output.with_name(output.name + f".partial-{os.getpid()}")
    if output.exists() or partial.exists():
        raise ReleaseError(f"Refusing to overwrite Run34 output: {output}")
    partial.mkdir(parents=True)
    try:
        results = []
        for spec in config["checkpoints"]:
            seed_dir = partial / f"seed{int(spec['seed'])}_run{int(spec['run'])}"
            seed_dir.mkdir()
            results.append(validate_one_checkpoint(spec, config, seed_dir))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        decision = acceptance(results, config)
        report = {
            "status": "COMPLETE",
            "completed_local_time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "config_sha256": sha256_file(CONFIG_PATH),
            "fingerprint": ready["fingerprint"],
            "architecture": "Plain YOLO26m P3/P4 partial-label multitask",
            "decision_rule": config["decision_rule"],
            "per_seed": results,
            "acceptance_gate": decision,
            "training_performed": False,
            "test_metrics_used": False,
            "external_metrics_used": False,
            "pad_ufes_20_used": False,
        }
        atomic_json(partial / "RUN34_RESULT.json", report)
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(partial, output)
    except BaseException:
        if partial.exists():
            failure = partial / "RUN34_FAILED_NOT_COMPLETE.txt"
            failure.write_text("Run34 did not complete; no result is eligible for model selection.\n", encoding="utf-8")
        raise
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("Run34 development-only evaluation complete. No training/test/external/PAD evaluation was run.")


def main() -> None:
    os.chdir(TOOLS_DIR)
    args = parse_args()
    config = load_config()
    if args.prepare_only:
        prepare_only(config)
    else:
        run_formal(config)


if __name__ == "__main__":
    main()
