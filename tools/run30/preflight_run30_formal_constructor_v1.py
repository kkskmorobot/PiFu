#!/usr/bin/env python3
"""Construct the frozen Run30 Trainer without creating an optimizer or training."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = PROJECT_ROOT / "tools"
RUN30_DIR = Path(__file__).resolve().parent
for path in (TOOLS_DIR, RUN30_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_run30_from_pycharm as entry
from imaplusplus_partial_multitask_plain_mel_nv_focal_risk_seed43_v1 import (
    DelayedEarlyStopping,
    MELNVFocalRiskDelayedUtilityGatedTrainer,
    RUN30_MEL_NV_FOCAL_GAMMA,
    RUN30_MEL_NV_FOCAL_RISK_WEIGHT,
)
from imaplusplus_partial_multitask_v1 import BinaryPartialClassificationHead
from ultralytics.nn.tasks import load_checkpoint


OUTPUT_PARENT = PROJECT_ROOT / "preflight" / "run30_formal_trainer_constructor_v1"
NAME = "plain_mel_nv_focal_risk_seed43_constructor_20260906_v1"


def main() -> None:
    output = OUTPUT_PARENT / NAME
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite constructor preflight: {output}")
    config = entry.load_locked_config()
    ready, fingerprint = entry.validate_ready(config)
    ultra = dict(config["ultralytics"])
    ultra.update(config["augmentations"])
    ultra["model"] = str(entry.resolve_project_path(ultra["model"]))
    ultra["data"] = str(entry.DEVELOPMENT_DATA.resolve())
    ultra["project"] = str(OUTPUT_PARENT.resolve())
    ultra["name"] = NAME
    ultra["exist_ok"] = False
    trainer = MELNVFocalRiskDelayedUtilityGatedTrainer(
        overrides=ultra,
        checkpoint_contract={
            "enabled": True,
            "fingerprint": fingerprint,
            "root": output / "checkpoint_bundles",
            "keep_last": int(config["checkpoint"]["keep_last"]),
            "resume_bundle": None,
        },
        image_cls_weight=float(config["multitask"]["image_cls_weight"]),
        mel_nv_focal_risk_weight=float(config["risk_loss"]["weight_inside_classification_objective"]),
        selection={
            "max_mel_to_nv_rate": float(config["selection"]["max_mel_to_nv_rate"]),
            "min_binary_iou": float(config["selection"]["min_binary_iou"]),
            "min_group_macro_f1": float(config["selection"]["min_group_macro_f1"]),
        },
    )
    expected_args = {
        "epochs": 100,
        "patience": 30,
        "batch": 16,
        "imgsz": 640,
        "workers": 8,
        "optimizer": "AdamW",
        "amp": True,
        "device": "0",
        "seed": 43,
    }
    actual_args = {name: getattr(trainer.args, name) for name in expected_args}
    if actual_args != expected_args:
        raise AssertionError(f"Formal Trainer args differ: {actual_args} != {expected_args}")
    if trainer.selection != {
        "max_mel_to_nv_rate": 0.1077844311,
        "min_binary_iou": 0.70,
        "min_group_macro_f1": 0.50,
    }:
        raise AssertionError(f"Formal Trainer selection protocol changed: {trainer.selection}")
    if trainer.mel_nv_focal_risk_weight != RUN30_MEL_NV_FOCAL_RISK_WEIGHT:
        raise AssertionError("Formal Trainer MEL-to-NV focal risk weight drifted")
    pretrained, _ = load_checkpoint(entry.resolve_project_path(config["ultralytics"]["model"]), device="cpu")
    formal_model = trainer.get_model(cfg=pretrained.yaml, weights=pretrained, verbose=False)
    if not isinstance(formal_model.model[-1], BinaryPartialClassificationHead):
        raise AssertionError(f"Run30 did not construct the plain image head: {type(formal_model.model[-1]).__name__}")
    forbidden = [
        name for name in ("channel_attention", "mask_guided_mix", "detach_mask", "p5_layer")
        if hasattr(formal_model.model[-1], name)
    ]
    if forbidden:
        raise AssertionError(f"Run30 unexpectedly contains attention attributes: {forbidden}")
    if getattr(trainer, "optimizer", None) is not None:
        raise AssertionError("Constructor preflight unexpectedly created an optimizer")
    formal_run = entry.resolve_project_path(config["ultralytics"]["project"]) / config["ultralytics"]["name"]
    report = {
        "status": "PASS",
        "scope": "formal Run30 Trainer constructor only; trainer.train not called",
        "args": actual_args,
        "model_class": type(formal_model).__name__,
        "head_class": type(formal_model.model[-1]).__name__,
        "attention_modules_present": False,
        "trainer_class": type(trainer).__name__,
        "stopper_class_after_setup_expected": DelayedEarlyStopping.__name__,
        "early_stop_start_epoch": 40,
        "patience_after_start": 30,
        "classification_loss": "unweighted_cross_entropy_plus_true_mel_to_nv_posterior_focal_risk",
        "mel_nv_focal_risk_weight": trainer.mel_nv_focal_risk_weight,
        "mel_nv_focal_gamma": RUN30_MEL_NV_FOCAL_GAMMA,
        "new_risk_loss_parameters": 0,
        "selection": trainer.selection,
        "fingerprint": fingerprint,
        "config_sha256": ready["config_sha256"],
        "optimizer_created": False,
        "trainer_train_called": False,
        "formal_result_directory_created": formal_run.exists(),
        "test_metrics_used": False,
        "external_metrics_used": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "CONSTRUCTOR_REPORT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

