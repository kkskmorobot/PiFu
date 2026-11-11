#!/usr/bin/env python3
"""Test2 frozen LGLA-v1 evaluation entry. Use --prepare before authorization."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import yaml


PROJECT = Path(__file__).resolve().parents[2]
TOOLS = PROJECT / "tools"
HERE = Path(__file__).resolve().parent
CONFIG = HERE / "test2.yaml"
PROTOCOL = HERE / "protocol.md"
UNIT = HERE / "unit.py"
INPUT_BUILDER = HERE / "inputs.py"
MODEL_LOCK = PROJECT / "tools" / "sum2" / "PROJECT_LGLA_FINAL_MODEL_LOCK_2026-09-16.md"
PREFLIGHT = HERE / "preflight.json"
READY = HERE / "ready.json"
RELEASE = HERE / "release.md"
CLASS_NAMES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc", "scc")
MEL_ID = 4
NV_ID = 5

if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
for model_dir in (TOOLS / "lg2", TOOLS / "lgx", TOOLS / "lg3"):
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))


class GateError(RuntimeError):
    pass


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="Run static/GPU preflight only; never access test images.")
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def load_config() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    contract = config["contract"]
    if contract["training_permitted"] or contract["model_selection_permitted"]:
        raise GateError("Test1 must remain evaluation-only with no model selection")
    if contract["threshold_tuning_permitted"] or contract["decision_rule_tuning_permitted"]:
        raise GateError("Test1 tuning boundary changed")
    if contract["decision_rule"] != "eight_class_argmax" or float(contract["calibration_temperature"]) != 1.0:
        raise GateError("Locked classification decision changed")
    if tuple(config["taxonomy"]["classes"]) != CLASS_NAMES:
        raise GateError("Taxonomy order changed")
    if config["contract"]["pad_ufes_20_policy"] != "excluded":
        raise GateError("PAD-UFES-20 exclusion changed")
    return config


def validate_manifest(path: Path, expected_hash: str, split: str, role: str, total: int, cls_count: int) -> dict[str, Any]:
    if not path.is_file() or sha256(path) != expected_hash:
        raise GateError(f"Frozen Test1 manifest changed: {path}")
    counts = {"total": 0, "segmentation": 0, "classification": 0, "mel": 0, "nv": 0}
    ids: set[str] = set()
    hashes: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            joined = " ".join(row.values()).lower()
            if "pad-ufes" in joined or "pad_ufes" in joined:
                raise GateError("PAD-UFES-20 appeared in Test1 manifest")
            if row["split"] != split or row["dataset_role"] != role:
                raise GateError(f"Unexpected Test1 role/split in {path}: {row['dataset_role']}/{row['split']}")
            if row["isic_id"] in ids or row["image_sha256"] in hashes:
                raise GateError(f"Duplicate ID or SHA-256 in {path}")
            ids.add(row["isic_id"])
            hashes.add(row["image_sha256"])
            counts["total"] += 1
            counts["segmentation"] += int(row["has_segmentation"] == "1")
            counts["classification"] += int(row["has_classification"] == "1")
            counts["mel"] += int(row["classification_target"] == str(MEL_ID))
            counts["nv"] += int(row["classification_target"] == str(NV_ID))
    if counts["total"] != total or counts["segmentation"] != total or counts["classification"] != cls_count:
        raise GateError(f"Test1 manifest counts changed: {path} {counts}")
    if split == "test" and (counts["mel"], counts["nv"]) != (16, 40):
        raise GateError(f"Restricted classification support changed: {counts}")
    if split == "external" and (counts["mel"] or counts["nv"]):
        raise GateError("External manifest unexpectedly contains classification labels")
    return {"path": str(path), "sha256": expected_hash, "counts": counts, "ids": ids, "hashes": hashes}


def validate_inputs(config: dict[str, Any]) -> dict[str, Any]:
    reports = []
    for section, split, role in (
        ("internal_test", "test", "imaplusplus_binary"),
        ("external", "external", "isic2018_task1_incremental"),
    ):
        spec = config[section]
        yaml_path = resolve(spec["yaml"])
        manifest_path = resolve(spec["manifest"])
        source_path = resolve(spec["source_manifest"])
        if not yaml_path.is_file() or sha256(yaml_path) != spec["yaml_sha256"]:
            raise GateError(f"Frozen Test1 YAML changed: {yaml_path}")
        if not source_path.is_file() or sha256(source_path) != spec["source_manifest_sha256"]:
            raise GateError(f"Frozen source manifest changed: {source_path}")
        parsed = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        if Path(parsed["partial_label_manifest"]).resolve() != manifest_path:
            raise GateError(f"Manifest path mismatch in {yaml_path}")
        report = validate_manifest(
            manifest_path, spec["manifest_sha256"], split, role,
            int(spec["total_samples"]), int(spec["classification_samples"]),
        )
        report["yaml"] = str(yaml_path)
        report["yaml_sha256"] = spec["yaml_sha256"]
        report["source_manifest"] = str(source_path)
        reports.append(report)
    if reports[0]["ids"] & reports[1]["ids"] or reports[0]["hashes"] & reports[1]["hashes"]:
        raise GateError("Internal test and external overlap")
    taxonomy = resolve(config["taxonomy"]["path"])
    if not taxonomy.is_file() or sha256(taxonomy) != config["taxonomy"]["sha256"]:
        raise GateError("Frozen taxonomy changed")
    return {
        "internal": {k: v for k, v in reports[0].items() if k not in {"ids", "hashes"}},
        "external": {k: v for k, v in reports[1].items() if k not in {"ids", "hashes"}},
        "id_overlap": 0,
        "sha256_overlap": 0,
        "taxonomy": str(taxonomy),
        "pad_ufes_20_present": False,
    }


def validate_checkpoints(config: dict[str, Any]) -> list[dict[str, Any]]:
    reports = []
    for spec in config["checkpoints"]:
        bundle = resolve(spec["bundle"])
        files = {name: bundle / name for name in ("manifest.json", "_SUCCESS", "trainer.pt", "runtime_state.pt")}
        if not all(path.is_file() for path in files.values()):
            raise GateError(f"Incomplete frozen bundle: {bundle}")
        manifest = json.loads(files["manifest.json"].read_text(encoding="utf-8"))
        success = json.loads(files["_SUCCESS"].read_text(encoding="utf-8"))
        epoch = int(spec["zero_based_epoch"])
        fingerprint = str(spec["fingerprint"])
        if manifest["epoch"] != epoch or success["epoch"] != epoch:
            raise GateError(f"Bundle epoch mismatch: {bundle}")
        if manifest["fingerprint"] != fingerprint or success["fingerprint"] != fingerprint:
            raise GateError(f"Bundle fingerprint mismatch: {bundle}")
        actual_trainer = (files["trainer.pt"].stat().st_size, sha256(files["trainer.pt"]))
        actual_runtime = (files["runtime_state.pt"].stat().st_size, sha256(files["runtime_state.pt"]))
        if actual_trainer != (int(spec["trainer_bytes"]), spec["trainer_sha256"]):
            raise GateError(f"trainer.pt changed: {bundle}")
        if actual_runtime != (int(spec["runtime_bytes"]), spec["runtime_sha256"]):
            raise GateError(f"runtime_state.pt changed: {bundle}")
        reports.append({"run": spec["run"], "seed": spec["seed"], "bundle": str(bundle), "fingerprint": fingerprint})
    return reports


def locked_files() -> dict[str, str]:
    paths = [CONFIG, PROTOCOL, UNIT, INPUT_BUILDER, MODEL_LOCK, HERE / "data_test.yaml", HERE / "data_external.yaml", HERE / "manifest_test.csv", HERE / "manifest_external.csv", Path(__file__).resolve()]
    return {str(path.resolve()): sha256(path) for path in paths}


def fingerprint(config: dict[str, Any]) -> str:
    payload = {"locked_files": locked_files(), "checkpoints": config["checkpoints"]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_exact_ema(bundle: Path):
    from ultralytics import YOLO

    wrapper = YOLO(str(bundle / "trainer.pt"), task="semantic")
    model = wrapper.model.float().cpu()
    serialized = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    runtime = torch.load(bundle / "runtime_state.pt", map_location="cpu", weights_only=False)
    exact = runtime.get("training_state", {}).get("ema")
    if not isinstance(exact, dict) or set(exact) != set(serialized):
        raise GateError(f"Lossless EMA state missing: {bundle}")
    model.load_state_dict(exact, strict=True)
    if any(not torch.equal(model.state_dict()[name].detach().cpu(), exact[name].detach().cpu()) for name in exact):
        raise GateError(f"Lossless EMA load mismatch: {bundle}")
    model.eval()
    return model, {
        "structure_source": str(bundle / "trainer.pt"),
        "state_source": str(bundle / "runtime_state.pt") + "::training_state.ema",
        "exact_ema_loaded": True,
    }


def tensors(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        return [tensor for item in value for tensor in tensors(item)]
    if isinstance(value, dict):
        return [tensor for item in value.values() for tensor in tensors(item)]
    return []


def gpu_smoke(config: dict[str, Any]) -> dict[str, Any]:
    from imaplusplus_partial_multitask_lgla_v1_gate_balance_seed42 import LGLAV1MultiTaskModel

    model, source = load_exact_ema(resolve(config["checkpoints"][0]["bundle"]))
    if not isinstance(model, LGLAV1MultiTaskModel):
        raise GateError(f"Expected frozen LGLA-v1 model, found {type(model).__name__}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    before = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    model.to(device).eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 640, 640, device=device))
    output_tensors = tensors(output)
    if not output_tensors or any(not torch.isfinite(t).all().item() for t in output_tensors):
        raise GateError("Test1 GPU smoke produced non-finite output")
    after = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    if any(not torch.equal(before[name], after[name]) for name in before):
        raise GateError("Test1 GPU smoke modified model state")
    return {
        "device": str(device), "model_class": type(model).__name__,
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "output_shapes": [list(t.shape) for t in output_tensors],
        "all_finite": True, "model_state_unchanged": True, "weight_source": source,
        "optimizer_created": False, "backward_called": False, "test_images_loaded": False,
    }


def prepare(config: dict[str, Any]) -> None:
    output = resolve(config["evaluation"]["output"])
    if output.exists():
        raise GateError(f"Formal Test1 output already exists: {output}")
    partials = sorted(output.parent.glob(output.name + ".partial-*")) if output.parent.exists() else []
    if partials:
        raise GateError(f"Incomplete prior Test1 output requires audit: {partials}")
    report = {
        "status": "PASS",
        "prepared_local_time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "config_sha256": sha256(CONFIG),
        "fingerprint": fingerprint(config),
        "inputs": validate_inputs(config),
        "checkpoints": validate_checkpoints(config),
        "gpu_smoke": gpu_smoke(config),
        "formal_output_exists": False,
        "test_inference_performed": False,
        "external_inference_performed": False,
        "training_performed": False,
        "pad_ufes_20_used": False,
    }
    atomic_json(PREFLIGHT, report)
    ready = {
        "status": "PREFLIGHT_PASS_PENDING_FRESH_USER_CONFIRMATION",
        "prepared_local_time": report["prepared_local_time"],
        "config_sha256": report["config_sha256"],
        "fingerprint": report["fingerprint"],
        "locked_files": locked_files(),
        "formal_output_exists": False,
        "test_inference_performed": False,
        "external_inference_performed": False,
        "training_performed": False,
    }
    atomic_json(READY, ready)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("TEST2_PREFLIGHT=PASS_PENDING_FRESH_USER_CONFIRMATION")


def validate_ready(config: dict[str, Any]) -> dict[str, Any]:
    if not READY.is_file() or not PREFLIGHT.is_file():
        raise GateError("Test1 preflight/ready marker missing")
    ready = json.loads(READY.read_text(encoding="utf-8"))
    if ready.get("status") != "PREFLIGHT_PASS_PENDING_FRESH_USER_CONFIRMATION":
        raise GateError("Test1 ready status changed")
    if ready.get("locked_files") != locked_files():
        raise GateError("Test1 locked files changed after preflight")
    if ready.get("config_sha256") != sha256(CONFIG) or ready.get("fingerprint") != fingerprint(config):
        raise GateError("Test1 config/fingerprint changed after preflight")
    validate_inputs(config)
    validate_checkpoints(config)
    return ready


def validate_release() -> None:
    if not RELEASE.is_file():
        raise GateError("Test1 release document missing")
    required = {
        "`PROJECT_TEST2_GATE=PASS`",
        "`TEST2_READY_GATE=PASS`",
        "`USER_CONFIRMATION_RECORDED=TRUE`",
        "`ONE_TIME_FINAL_EVALUATION_AUTHORIZED=TRUE`",
        "`TRAINING_PERFORMED=FALSE`",
    }
    lines = set(RELEASE.read_text(encoding="utf-8").splitlines())
    missing = sorted(required - lines)
    if missing:
        raise GateError(f"Test1 formal execution remains blocked: {missing}")


def aggregate_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["group_id"]].append(row)
    result = []
    for group_id, members in sorted(grouped.items()):
        targets = {int(row["target"]) for row in members}
        if len(targets) != 1:
            raise GateError(f"Classification group spans targets: {group_id}")
        result.append({
            "group_id": group_id,
            "target": targets.pop(),
            "probabilities": np.mean([row["probabilities"] for row in members], axis=0),
            "images": len(members),
        })
    return result


def restricted_metrics(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predictions = probabilities.argmax(axis=1)
    mel = targets == MEL_ID
    nv = targets == NV_ID
    if int(mel.sum()) != 16 or int(nv.sum()) != 40:
        raise GateError("Restricted MEL/NV support changed")
    mel_recall = float((predictions[mel] == MEL_ID).mean())
    nv_recall = float((predictions[nv] == NV_ID).mean())
    clipped = np.clip(probabilities.astype(np.float64), 1e-12, 1.0)
    one_hot = np.eye(len(CLASS_NAMES), dtype=np.float64)[targets]
    confidence = clipped.max(axis=1)
    correct = predictions == targets
    ece = 0.0
    edges = np.linspace(0.0, 1.0, 16)
    for index in range(15):
        selected = (confidence > edges[index]) & (confidence <= edges[index + 1]) if index else (confidence >= 0) & (confidence <= edges[1])
        if selected.any():
            ece += float(selected.mean()) * abs(float(correct[selected].mean()) - float(confidence[selected].mean()))
    return {
        "support": {"total": len(targets), "mel": int(mel.sum()), "nv": int(nv.sum())},
        "accuracy": float(correct.mean()),
        "mel_recall": mel_recall,
        "nv_recall": nv_recall,
        "balanced_recall": 0.5 * (mel_recall + nv_recall),
        "mel_to_nv_count": int(((targets == MEL_ID) & (predictions == NV_ID)).sum()),
        "mel_to_nv_rate": float(((targets == MEL_ID) & (predictions == NV_ID)).sum() / mel.sum()),
        "nv_to_mel_count": int(((targets == NV_ID) & (predictions == MEL_ID)).sum()),
        "nv_to_mel_rate": float(((targets == NV_ID) & (predictions == MEL_ID)).sum() / nv.sum()),
        "nll": float(-np.log(clipped[np.arange(len(targets)), targets]).mean()),
        "brier": float(np.square(clipped - one_hot).sum(axis=1).mean()),
        "ece": ece,
        "predicted_class_counts": {name: int((predictions == index).sum()) for index, name in enumerate(CLASS_NAMES)},
    }


def bootstrap_ci(targets: np.ndarray, probabilities: np.ndarray, seed: int, repetitions: int) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    mel_idx = np.where(targets == MEL_ID)[0]
    nv_idx = np.where(targets == NV_ID)[0]
    names = ("accuracy", "mel_recall", "nv_recall", "balanced_recall", "mel_to_nv_rate", "nv_to_mel_rate")
    samples = {name: np.empty(repetitions, dtype=np.float64) for name in names}
    for index in range(repetitions):
        chosen = np.concatenate((rng.choice(mel_idx, len(mel_idx), replace=True), rng.choice(nv_idx, len(nv_idx), replace=True)))
        metrics = restricted_metrics(targets[chosen], probabilities[chosen])
        for name in names:
            samples[name][index] = metrics[name]
    return {name: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))] for name, values in samples.items()}


def save_classification(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["group_id", "target", "true_class", "images", "argmax_id", "argmax_class", *[f"score_{name}" for name in CLASS_NAMES]]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            probs = np.asarray(row["probabilities"], dtype=np.float64)
            pred = int(probs.argmax())
            writer.writerow({
                "group_id": row["group_id"], "target": row["target"], "true_class": CLASS_NAMES[row["target"]],
                "images": row["images"], "argmax_id": pred, "argmax_class": CLASS_NAMES[pred],
                **{f"score_{name}": float(probs[i]) for i, name in enumerate(CLASS_NAMES)},
            })


def evaluate_dataset(spec: dict[str, Any], config: dict[str, Any], dataset_name: str, save_dir: Path) -> dict[str, Any]:
    from imaplusplus_partial_multitask_v1 import BinaryPartialMultiTaskModel, PartialLabelMultiTaskValidator, PartialLabelSemanticDataset
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_dataloader
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.models.yolo.semantic import SemanticSegmentationValidator

    model, weight_source = load_exact_ema(resolve(spec["bundle"]))
    if not isinstance(model, BinaryPartialMultiTaskModel) or tuple(model.classification_names) != CLASS_NAMES:
        raise GateError(f"Run{spec['run']} model/taxonomy mismatch")
    settings = config["evaluation"]
    data_spec = config["internal_test"] if dataset_name == "internal_test" else config["external"]
    data_yaml = resolve(data_spec["yaml"])
    split_arg = "test" if dataset_name == "internal_test" else "val"
    expected_split = "test" if dataset_name == "internal_test" else "external"
    data = check_det_dataset(str(data_yaml), split=split_arg)
    image_path = data["test"] if dataset_name == "internal_test" else data["val"]
    cfg = get_cfg(overrides={
        "task": "semantic", "mode": "val", "model": str(resolve(spec["bundle"]) / "trainer.pt"),
        "data": str(data_yaml), "split": split_arg, "imgsz": int(settings["imgsz"]),
        "batch": int(settings["batch"]), "workers": int(settings["workers"]), "device": str(settings["device"]),
        "plots": False, "save_json": False, "verbose": False,
    })
    dataset = PartialLabelSemanticDataset(
        img_path=image_path, imgsz=int(settings["imgsz"]), batch_size=int(settings["batch"]), augment=False,
        hyp=cfg, rect=True, cache=False, single_cls=False, stride=max(int(model.stride.max()), 32), pad=0.0,
        prefix=f"Test2 {spec['run']} {dataset_name}: ", task="semantic", classes=None, data=data, fraction=1.0,
        partial_label_manifest=resolve(data_spec["manifest"]), expected_split=expected_split,
    )
    loader = build_dataloader(dataset, batch=int(settings["batch"]), workers=int(settings["workers"]), shuffle=False, rank=-1, drop_last=False)
    validator_class = PartialLabelMultiTaskValidator
    if dataset_name == "external":
        class ExternalSegmentationValidator(PartialLabelMultiTaskValidator):
            """Keep multi-task output unpacking but skip classification statistics for label-free external data."""

            def get_stats(self):
                stats = SemanticSegmentationValidator.get_stats(self)
                binary_iou = float(self.metrics.miou)
                stats.update({
                    "metrics/binary_lesion_iou": binary_iou,
                    "metrics/binary_lesion_dice": 2.0 * binary_iou / (1.0 + binary_iou),
                })
                return stats

        validator_class = ExternalSegmentationValidator
    validator = validator_class(dataloader=loader, save_dir=save_dir, args=cfg, selection={"max_mel_to_nv_rate": 0.1077844311})
    device = torch.device(f"cuda:{settings['device']}" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    frozen = SimpleNamespace(
        device=device, data=data, amp=device.type == "cuda", ema=SimpleNamespace(ema=model), model=model,
        args=SimpleNamespace(compile=False), loss_items=torch.zeros(5, device=device),
        stopper=SimpleNamespace(possible_stop=False), epoch=0, epochs=1, world_size=1,
        label_loss_items=lambda _loss, prefix="val": {},
    )
    flat = validator(frozen)
    segmentation = {
        "binary_iou": float(flat["metrics/binary_lesion_iou"]),
        "binary_dice": float(flat["metrics/binary_lesion_dice"]),
        "pixel_accuracy": float(flat["metrics/pixel_acc"]),
        "samples": int(data_spec["segmentation_samples"]),
    }
    result = {"segmentation": segmentation, "weight_source": weight_source, "validation_mode": "training_path_unfused_fp32_ema_cuda_autocast"}
    if dataset_name == "internal_test":
        image_rows = [{
            "isic_id": str(row["isic_id"]), "group_id": str(row["group_id"]), "target": int(row["target"]),
            "probabilities": np.asarray(row["probabilities"], dtype=np.float64),
        } for row in validator.classification_rows]
        group_rows = aggregate_groups(image_rows)
        targets = np.asarray([row["target"] for row in group_rows], dtype=np.int64)
        probabilities = np.asarray([row["probabilities"] for row in group_rows], dtype=np.float64)
        classification = restricted_metrics(targets, probabilities)
        classification["bootstrap_95ci"] = bootstrap_ci(
            targets, probabilities, int(settings["classification_bootstrap_seed"]),
            int(settings["classification_bootstrap_repetitions"]),
        )
        classification["scope"] = "IMA++ test MEL/NV labeled subset only; not eight-class test Macro-F1"
        result["classification"] = classification
        save_classification(save_dir / "classification.csv", group_rows)
    elif validator.classification_rows:
        raise GateError("External evaluation unexpectedly produced classification rows")
    return result


def aggregate_seed_metrics(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    def mean_sd(values: list[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        return {"mean": float(array.mean()), "sample_sd": float(array.std(ddof=1))}
    output: dict[str, Any] = {"internal_segmentation": {}, "external_segmentation": {}, "restricted_classification": {}}
    for metric in ("binary_iou", "binary_dice", "pixel_accuracy"):
        output["internal_segmentation"][metric] = mean_sd([row["internal_test"]["segmentation"][metric] for row in per_seed])
        output["external_segmentation"][metric] = mean_sd([row["external"]["segmentation"][metric] for row in per_seed])
    for metric in ("accuracy", "mel_recall", "nv_recall", "balanced_recall", "mel_to_nv_rate", "nv_to_mel_rate", "nll", "brier", "ece"):
        output["restricted_classification"][metric] = mean_sd([row["internal_test"]["classification"][metric] for row in per_seed])
    return output


def formal(config: dict[str, Any]) -> None:
    validate_release()
    ready = validate_ready(config)
    output = resolve(config["evaluation"]["output"])
    partial = output.with_name(output.name + f".partial-{os.getpid()}")
    existing_partials = sorted(output.parent.glob(output.name + ".partial-*")) if output.parent.exists() else []
    if output.exists() or existing_partials:
        raise GateError(f"Refusing to overwrite or bypass existing Test1 output: {output}; partials={existing_partials}")
    partial.mkdir(parents=True)
    try:
        per_seed = []
        for spec in config["checkpoints"]:
            seed_dir = partial / f"seed{spec['seed']}"
            seed_dir.mkdir()
            internal_dir = seed_dir / "internal"
            external_dir = seed_dir / "external"
            internal_dir.mkdir()
            external_dir.mkdir()
            per_seed.append({
                "run": str(spec["run"]), "seed": int(spec["seed"]), "visible_epoch": int(spec["visible_epoch"]),
                "bundle": str(resolve(spec["bundle"])),
                "internal_test": evaluate_dataset(spec, config, "internal_test", internal_dir),
                "external": evaluate_dataset(spec, config, "external", external_dir),
            })
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        report = {
            "status": "COMPLETE",
            "completed_local_time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "config_sha256": sha256(CONFIG), "fingerprint": ready["fingerprint"],
            "architecture": "LGLA-v1 YOLO26m P3/P4 partial-label multitask",
            "decision_rule": "eight_class_argmax", "representative_seed": 43,
            "per_seed": per_seed, "aggregate": aggregate_seed_metrics(per_seed),
            "selection_performed": False, "training_performed": False,
            "test_metrics_used": True, "external_metrics_used": True, "pad_ufes_20_used": False,
            "post_test_development_changes_permitted": False,
        }
        atomic_json(partial / "result.json", report)
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(partial, output)
    except BaseException:
        if partial.exists():
            (partial / "FAILED.txt").write_text("Test1 did not complete; this partial output is not an eligible final result.\n", encoding="utf-8")
        raise
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("TEST2_COMPLETE; no training or post-test selection was performed")


def main() -> None:
    os.chdir(TOOLS)
    parsed = args()
    config = load_config()
    if parsed.prepare:
        prepare(config)
    else:
        formal(config)


if __name__ == "__main__":
    # Portable copy: reruns use new outputs; historical release gates are audit records.
    import sys as _repro_sys
    from pathlib import Path as _ReproPath
    _repro_sys.path.insert(0, str(_ReproPath(__file__).resolve().parents[2] / "reproduction"))
    from run import main as _repro_main
    _repro_args = ["evaluate", "--suite", "test2"] + [("--prepare" if a == '--prepare' else a) for a in _repro_sys.argv[1:]]
    _repro_main(_repro_args)
