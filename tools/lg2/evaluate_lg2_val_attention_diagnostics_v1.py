#!/usr/bin/env python3
"""Frozen-development validation and gate diagnostics for completed lg2.

This script has no test/external option. It evaluates the preregistered
best-Joint bundle and records aggregate LGLA gate/attention diagnostics.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS = PROJECT_ROOT / "tools"
LG2 = TOOLS / "lg2"
RUN = (
    PROJECT_ROOT
    / "experiments"
    / "lg2_YOLO26m_Binary8Class_LGLAv1_GateBalance_Seed42"
    / "results"
    / "lg2_yolo26m_lgla_v1_gate_balance_seed42"
)
WEIGHTS = RUN / "checkpoint_bundles" / "epoch_0073" / "trainer.pt"
DATA = LG2 / "lg2_development_data.yaml"
OUTPUT = RUN / "analysis" / "val_attention_diagnostics_v1"

for path in (TOOLS, LG2):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from imaplusplus_partial_multitask_lgla_v1_gate_balance_seed42 import (  # noqa: E402
    LGLAV1MultiTaskModel,
    LesionGuidedGlobalLocalClassificationHead,
)
from imaplusplus_partial_multitask_v1 import (  # noqa: E402
    CLASS_NAMES,
    PartialLabelMultiTaskValidator,
    PartialLabelSemanticDataset,
)


PLAIN_RUN18 = {
    "binary_lesion_iou": 0.76323,
    "image_macro_f1": 0.54417,
    "group_macro_f1": 0.61023,
    "group_macro_recall": 0.59915,
    "group_mel_to_nv_rate": 0.38776,
    "joint_score": 0.68246,
}


def describe(values: torch.Tensor) -> dict[str, float | int]:
    values = values.float().reshape(-1)
    quantiles = torch.quantile(values, torch.tensor([0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]))
    return {
        "count": int(values.numel()),
        "mean": float(values.mean()),
        "std_population": float(values.std(unbiased=False)),
        "min": float(quantiles[0]),
        "p05": float(quantiles[1]),
        "p25": float(quantiles[2]),
        "median": float(quantiles[3]),
        "p75": float(quantiles[4]),
        "p95": float(quantiles[5]),
        "max": float(quantiles[6]),
    }


def main() -> None:
    if not WEIGHTS.is_file() or not DATA.is_file():
        raise FileNotFoundError((WEIGHTS, DATA))
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite diagnostics: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    raw_data = yaml.safe_load(DATA.read_text(encoding="utf-8"))
    if raw_data.get("test") is not None:
        raise RuntimeError("Development YAML exposes a test path")
    if any(key in raw_data for key in ("external", "pad_ufes_20", "pad-ufes-20")):
        raise RuntimeError("Development YAML exposes an external-data key")

    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_dataloader
    from ultralytics.data.utils import check_det_dataset

    wrapper = YOLO(str(WEIGHTS), task="semantic")
    model = wrapper.model
    if not isinstance(model, LGLAV1MultiTaskModel):
        raise TypeError(f"Expected LGLAV1MultiTaskModel, found {type(model).__name__}")
    if tuple(model.classification_names) != CLASS_NAMES:
        raise RuntimeError("Classification taxonomy changed")
    head = model.model[-1]
    if not isinstance(head, LesionGuidedGlobalLocalClassificationHead):
        raise TypeError(type(head).__name__)

    gates: list[torch.Tensor] = []
    masses: list[torch.Tensor] = []
    fallbacks: list[torch.Tensor] = []
    descriptor_cosines: list[torch.Tensor] = []
    descriptor_relative_l2: list[torch.Tensor] = []

    def capture(module, inputs, output):
        gates.append(module.last_gate_weights.detach().float().cpu())
        masses.append(module.last_attention_mass_fraction.detach().float().cpu())
        fallbacks.append(module.last_attention_fallback.detach().cpu())
        with torch.no_grad():
            features = inputs[0]
            semantic = output[0]
            semantic_main = semantic[0] if isinstance(semantic, tuple) else semantic
            attention = torch.sigmoid(semantic_main.detach().float())
            global_parts = []
            local_parts = []
            for projection, feature in zip(module.scale_projections, features):
                global_raw, local_raw, _, _ = module._global_local_pool(feature, attention)
                dtype = projection.weight.dtype
                global_parts.append(projection(global_raw.to(dtype=dtype)).float())
                local_parts.append(projection(local_raw.to(dtype=dtype)).float())
            global_descriptor = torch.cat(global_parts, dim=1)
            local_descriptor = torch.cat(local_parts, dim=1)
            descriptor_cosines.append(
                F.cosine_similarity(global_descriptor, local_descriptor, dim=1).detach().cpu()
            )
            descriptor_relative_l2.append(
                (
                    (local_descriptor - global_descriptor).norm(dim=1)
                    / global_descriptor.norm(dim=1).clamp_min(1.0e-12)
                )
                .detach()
                .cpu()
            )

    hook = head.register_forward_hook(capture)
    try:
        data = check_det_dataset(str(DATA), split="val")
        if data.get("test"):
            raise RuntimeError("Development-only data unexpectedly exposes test")
        manifest = Path(data["partial_label_manifest"])
        if not manifest.is_absolute():
            manifest = (Path(data["path"]) / manifest).resolve()
        cfg = get_cfg(
            overrides={
                "task": "semantic",
                "mode": "val",
                "model": str(WEIGHTS),
                "data": str(DATA),
                "split": "val",
                "imgsz": 640,
                "batch": 16,
                "workers": 8,
                "device": "0",
                "plots": False,
                "save_json": False,
                "verbose": False,
            }
        )
        dataset = PartialLabelSemanticDataset(
            img_path=data["val"],
            imgsz=640,
            batch_size=16,
            augment=False,
            hyp=cfg,
            rect=True,
            cache=False,
            single_cls=False,
            stride=max(int(model.stride.max()), 32),
            pad=0.0,
            prefix="lg2 frozen val diagnostics: ",
            task="semantic",
            classes=None,
            data=data,
            fraction=1.0,
            partial_label_manifest=manifest,
            expected_split="val",
        )
        loader = build_dataloader(
            dataset, batch=16, workers=8, shuffle=False, rank=-1, drop_last=False
        )
        validator = PartialLabelMultiTaskValidator(
            dataloader=loader,
            save_dir=OUTPUT,
            args=cfg,
            selection={"max_mel_to_nv_rate": 0.1077844311},
        )
        metrics = validator(model=model)
    finally:
        hook.remove()

    gate = torch.cat(gates)
    mass = torch.cat(masses)
    fallback = torch.cat(fallbacks)
    cosine = torch.cat(descriptor_cosines)
    relative_l2 = torch.cat(descriptor_relative_l2)
    if not all(torch.isfinite(value).all() for value in (gate, mass, cosine, relative_l2)):
        raise RuntimeError("Non-finite LGLA diagnostics")

    observed = {
        "binary_lesion_iou": float(metrics["metrics/binary_lesion_iou"]),
        "image_macro_f1": float(metrics["metrics/image_macro_f1"]),
        "group_macro_f1": float(metrics["metrics/group_macro_f1"]),
        "group_macro_recall": float(metrics["metrics/group_macro_recall"]),
        "group_mel_to_nv_rate": float(metrics["metrics/group_mel_to_nv_rate"]),
        "joint_score": float(metrics["metrics/joint_score"]),
    }
    delta = {key: observed[key] - PLAIN_RUN18[key] for key in observed}
    local_gate = describe(gate[:, 1])
    diagnostic_gate = {
        "local_gate_mean_in_0p05_0p95": 0.05 <= local_gate["mean"] <= 0.95,
        "local_gate_std_at_least_0p001": local_gate["std_population"] >= 0.001,
        "all_finite": True,
    }
    screening_gate = {
        "group_macro_f1_delta_at_least_0p005": delta["group_macro_f1"] >= 0.005,
        "joint_delta_at_least_0p003": delta["joint_score"] >= 0.003,
        "image_macro_f1_non_decreasing": delta["image_macro_f1"] >= 0.0,
        "binary_iou_delta_at_least_minus_0p005": delta["binary_lesion_iou"] >= -0.005,
    }
    report = {
        "status": "PASS",
        "scope": "frozen development validation only; no test, external, or PAD-UFES-20",
        "weights": str(WEIGHTS),
        "data": str(DATA),
        "sample_count_all_forwarded": int(gate.shape[0]),
        "classification_rows": len(validator.classification_rows),
        "plain_run18": PLAIN_RUN18,
        "lg2_best_joint_validation": observed,
        "delta_lg2_minus_run18": delta,
        "screening_gate": screening_gate,
        "screening_gate_pass": all(screening_gate.values()),
        "diagnostics": {
            "global_gate": describe(gate[:, 0]),
            "local_gate": local_gate,
            "attention_mass_fraction_p3": describe(mass[:, 0]),
            "attention_mass_fraction_p4": describe(mass[:, 1]),
            "fallback_rate_p3": float(fallback[:, 0].float().mean()),
            "fallback_rate_p4": float(fallback[:, 1].float().mean()),
            "global_local_descriptor_cosine": describe(cosine),
            "global_local_descriptor_relative_l2": describe(relative_l2),
        },
        "diagnostic_gate": diagnostic_gate,
        "diagnostic_gate_pass": all(diagnostic_gate.values()),
        "forbidden_data_access": {
            "imaplusplus_test": False,
            "isic2018_external": False,
            "pad_ufes_20": False,
        },
    }
    destination = OUTPUT / "LG2_VAL_ATTENTION_DIAGNOSTICS.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved: {destination}")


if __name__ == "__main__":
    main()
