#!/usr/bin/env python3
"""Construct the frozen Run17 Trainer without creating an optimizer or training."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = PROJECT_ROOT / "tools"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import train_run17_from_pycharm as entry
from imaplusplus_partial_multitask_utility_selection_v1 import UtilityEligiblePartialLabelMultiTaskTrainer
from imaplusplus_partial_multitask_v1 import BinaryPartialClassificationHead
from ultralytics.nn.tasks import load_checkpoint


OUTPUT_PARENT = PROJECT_ROOT / "preflight" / "run17_formal_trainer_constructor_v1"
NAME = "baseline_constructor_20260810_v2"


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
    trainer = UtilityEligiblePartialLabelMultiTaskTrainer(
        overrides=ultra,
        checkpoint_contract={
            "enabled": True,
            "fingerprint": fingerprint,
            "root": output / "checkpoint_bundles",
            "keep_last": int(config["checkpoint"]["keep_last"]),
            "resume_bundle": None,
        },
        image_cls_weight=float(config["multitask"]["image_cls_weight"]),
        selection={
            "max_mel_to_nv_rate": float(config["selection"]["max_mel_to_nv_rate"]),
            "min_binary_iou": float(config["selection"]["min_binary_iou"]),
            "min_group_macro_f1": float(config["selection"]["min_group_macro_f1"]),
        },
    )
    expected_args = {
        "epochs": 100,
        "patience": 20,
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
    if not isinstance(formal_model.model[-1], BinaryPartialClassificationHead):
        raise AssertionError(f"Run17 did not construct the plain image head: {type(formal_model.model[-1]).__name__}")
    forbidden = [
        name for name in ("channel_attention", "mask_guided_mix", "detach_mask", "p5_layer")
        if hasattr(formal_model.model[-1], name)
    ]
    if forbidden:
        raise AssertionError(f"Run17 unexpectedly contains attention attributes: {forbidden}")
    if getattr(trainer, "optimizer", None) is not None:
        raise AssertionError("Constructor preflight unexpectedly created an optimizer")
    formal_run = entry.resolve_project_path(config["ultralytics"]["project"]) / config["ultralytics"]["name"]
    report = {
        "status": "PASS",
        "scope": "formal Run17 Trainer constructor only; trainer.train not called",
        "args": actual_args,
        "model_class": type(formal_model).__name__,
        "head_class": type(formal_model.model[-1]).__name__,
        "attention_modules_present": False,
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
