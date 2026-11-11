#!/usr/bin/env python3
"""Construct the frozen lgx Trainer without creating an optimizer or training."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = PROJECT_ROOT / "tools"
LGX_DIR = Path(__file__).resolve().parent
for path in (TOOLS_DIR, LGX_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_lgx_from_pycharm as entry
from imaplusplus_partial_multitask_lgla_v1_gate_balance_seed43 import (
    DelayedEarlyStopping,
    DelayedUtilityGatedLGLAV1Trainer,
    LesionGuidedGlobalLocalClassificationHead,
)
from ultralytics.nn.tasks import load_checkpoint


OUTPUT_PARENT = PROJECT_ROOT / "preflight" / "lgx_formal_trainer_constructor_v1"
NAME = "lgla_v1_gate_balance_constructor_20260913_v1"


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
    trainer = DelayedUtilityGatedLGLAV1Trainer(
        overrides=ultra,
        checkpoint_contract={
            "enabled": True,
            "fingerprint": fingerprint,
            "root": output / "checkpoint_bundles",
            "keep_last": int(config["checkpoint"]["keep_last"]),
            "resume_bundle": None,
        },
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
    pretrained, _ = load_checkpoint(entry.resolve_project_path(config["ultralytics"]["model"]), device="cpu")
    formal_model = trainer.get_model(cfg=pretrained.yaml, weights=pretrained, verbose=False)
    head = formal_model.model[-1]
    if not isinstance(head, LesionGuidedGlobalLocalClassificationHead):
        raise AssertionError(
            f"lgx did not construct the LGLA v1 image head: {type(head).__name__}"
        )
    forbidden = [
        name for name in ("mask_guided_mix", "detach_mask", "p5_layer")
        if hasattr(head, name)
    ]
    if forbidden:
        raise AssertionError(f"lgx unexpectedly contains legacy mask-guided/P5 attributes: {forbidden}")
    if len(head.scale_projections) != 2:
        raise AssertionError("lgx must contain exactly two P3/P4 scale projections")
    if head.projection_dim != 128 or head.gate_hidden_dim != 128:
        raise AssertionError("lgx projection/gate dimensions changed")
    if formal_model.gate_balance_weight != 0.01:
        raise AssertionError("lgx gate-balance weight changed")
    if not torch.count_nonzero(head.global_local_gate[-1].weight).item() == 0:
        raise AssertionError("lgx gate output weights must start at zero for 0.5/0.5 fusion")
    if getattr(trainer, "optimizer", None) is not None:
        raise AssertionError("Constructor preflight unexpectedly created an optimizer")
    formal_run = entry.resolve_project_path(config["ultralytics"]["project"]) / config["ultralytics"]["name"]
    report = {
        "status": "PASS",
        "scope": "formal lgx Trainer constructor only; trainer.train not called",
        "args": actual_args,
        "model_class": type(formal_model).__name__,
        "head_class": type(head).__name__,
        "attention_modules_present": True,
        "attention_head_inputs": list(head.f),
        "scale_projections": len(head.scale_projections),
        "projection_dim": head.projection_dim,
        "gate_hidden_dim": head.gate_hidden_dim,
        "attention_stop_gradient": True,
        "ground_truth_mask_as_classifier_input": False,
        "gate_initialization": "equal_half_weights",
        "empty_attention_fallback": "global_average",
        "p5_descriptor": False,
        "trainer_class": type(trainer).__name__,
        "stopper_class_after_setup_expected": DelayedEarlyStopping.__name__,
        "early_stop_start_epoch": 40,
        "patience_after_start": 30,
        "classification_loss": "unweighted_cross_entropy_plus_gate_balance_only",
        "gate_balance_regularizer": config["lgla_v1"]["gate_balance_regularizer"],
        "gate_balance_weight": float(config["lgla_v1"]["gate_balance_weight"]),
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



