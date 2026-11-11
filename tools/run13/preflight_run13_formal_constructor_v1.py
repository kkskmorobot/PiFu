#!/usr/bin/env python3
"""Construct the frozen Run13 Trainer without creating an optimizer or training."""

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

import train_run13_from_pycharm as entry
from imaplusplus_partial_multitask_lesion_attention_mel_weight_v1 import (
    MelWeightedBinaryLesionGuidedAttentionTrainer,
    RUN11_CLASSIFICATION_CLASS_WEIGHTS,
)


OUTPUT_PARENT = PROJECT_ROOT / "preflight" / "run13_formal_trainer_constructor_v1"
NAME = "constructor_20260808_v2"


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
    class_names = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc", "scc")
    class_weights = tuple(
        float(config["multitask"]["classification_class_weights"][name]) for name in class_names
    )
    trainer = MelWeightedBinaryLesionGuidedAttentionTrainer(
        overrides=ultra,
        checkpoint_contract={
            "enabled": True,
            "fingerprint": fingerprint,
            "root": output / "checkpoint_bundles",
            "keep_last": int(config["checkpoint"]["keep_last"]),
            "resume_bundle": None,
        },
        image_cls_weight=float(config["multitask"]["image_cls_weight"]),
        classification_class_weights=class_weights,
        lesion_attention={
            "p5_layer": int(attention["p5_layer"]),
            "eca_kernel_size": int(attention["eca_kernel_size"]),
            "mask_guided_mix": float(attention["mask_guided_mix"]),
            "detach_mask": bool(attention["detach_mask"]),
        },
        selection={"max_mel_to_nv_rate": float(config["selection"]["max_mel_to_nv_rate"])},
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
    if tuple(trainer.classification_class_weights) != RUN11_CLASSIFICATION_CLASS_WEIGHTS:
        raise AssertionError("Formal Trainer MEL weights differ from Run11")
    if trainer.lesion_attention != {
        "p5_layer": 10,
        "eca_kernel_size": 5,
        "mask_guided_mix": 0.5,
        "detach_mask": True,
    }:
        raise AssertionError(f"Formal Trainer attention changed: {trainer.lesion_attention}")
    if getattr(trainer, "optimizer", None) is not None:
        raise AssertionError("Constructor preflight unexpectedly created an optimizer")
    formal_run = entry.resolve_project_path(config["ultralytics"]["project"]) / config["ultralytics"]["name"]
    report = {
        "status": "PASS",
        "scope": "formal Run13 Trainer constructor only; trainer.train not called",
        "args": actual_args,
        "classification_class_weights": list(trainer.classification_class_weights),
        "lesion_attention": trainer.lesion_attention,
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
