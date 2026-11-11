#!/usr/bin/env python3
"""Construct the frozen Run19 Trainer without creating an optimizer or training."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = PROJECT_ROOT / "tools"
RUN19_DIR = Path(__file__).resolve().parent
for path in (TOOLS_DIR, RUN19_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_run19_from_pycharm as entry
from imaplusplus_partial_multitask_attention_delayed_early_stop_v1 import (
    DelayedEarlyStopping,
    DelayedUtilityGatedBinaryLesionGuidedAttentionTrainer,
)
from imaplusplus_partial_multitask_lesion_attention_v1 import (
    BinaryLesionGuidedAttentionHead,
    IdentityInitializedECA,
)
from ultralytics.nn.tasks import load_checkpoint


OUTPUT_PARENT = PROJECT_ROOT / "preflight" / "run19_formal_trainer_constructor_v1"
NAME = "blg_attention_constructor_20260811_v1"


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
    attention = config["lesion_attention"]
    trainer = DelayedUtilityGatedBinaryLesionGuidedAttentionTrainer(
        overrides=ultra,
        checkpoint_contract={
            "enabled": True,
            "fingerprint": fingerprint,
            "root": output / "checkpoint_bundles",
            "keep_last": int(config["checkpoint"]["keep_last"]),
            "resume_bundle": None,
        },
        image_cls_weight=float(config["multitask"]["image_cls_weight"]),
        lesion_attention={
            "p5_layer": int(attention["p5_layer"]),
            "eca_kernel_size": int(attention["eca_kernel_size"]),
            "mask_guided_mix": float(attention["mask_guided_mix"]),
            "detach_mask": bool(attention["detach_mask"]),
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
    if not isinstance(head, BinaryLesionGuidedAttentionHead):
        raise AssertionError(f"Run19 did not construct BLG attention head: {type(head).__name__}")
    eca_modules = [module for module in head.channel_attention if isinstance(module, IdentityInitializedECA)]
    if list(head.f) != [16, 13, 10] or len(eca_modules) != 3:
        raise AssertionError(f"Run19 attention wiring changed: f={head.f}, eca={len(eca_modules)}")
    if trainer.lesion_attention != {
        "p5_layer": 10,
        "eca_kernel_size": 5,
        "mask_guided_mix": 0.5,
        "detach_mask": True,
    }:
        raise AssertionError(f"Run19 attention config changed: {trainer.lesion_attention}")
    if getattr(trainer, "optimizer", None) is not None:
        raise AssertionError("Constructor preflight unexpectedly created an optimizer")
    formal_run = entry.resolve_project_path(config["ultralytics"]["project"]) / config["ultralytics"]["name"]
    report = {
        "status": "PASS",
        "scope": "formal Run19 Trainer constructor only; trainer.train not called",
        "args": actual_args,
        "model_class": type(formal_model).__name__,
        "head_class": type(formal_model.model[-1]).__name__,
        "attention_modules_present": True,
        "attention_head_inputs": list(head.f),
        "identity_initialized_eca_modules": len(eca_modules),
        "lesion_attention": trainer.lesion_attention,
        "trainer_class": type(trainer).__name__,
        "stopper_class_after_setup_expected": DelayedEarlyStopping.__name__,
        "early_stop_start_epoch": 40,
        "patience_after_start": 30,
        "classification_loss": "unweighted_cross_entropy",
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


