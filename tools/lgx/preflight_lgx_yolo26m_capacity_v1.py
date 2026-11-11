#!/usr/bin/env python3
"""One full batch=16 AMP forward/backward/AdamW capacity step for lgx LGLA v1."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = PROJECT_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

LGX_DIR = Path(__file__).resolve().parent
if str(LGX_DIR) not in sys.path:
    sys.path.insert(0, str(LGX_DIR))

from imaplusplus_partial_multitask_lgla_v1_gate_balance_seed43 import (
    LGLAV1MultiTaskModel,
)
from ultralytics.nn.tasks import load_checkpoint


OUTPUT = PROJECT_ROOT / "preflight" / "lgx_yolo26m_capacity_v1" / "batch16_img640_amp_20260913_v1.json"
WEIGHTS = PROJECT_ROOT / "models" / "yolo26m-sem.pt"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite capacity report: {OUTPUT}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    pretrained, _ = load_checkpoint(WEIGHTS, device="cpu")
    model = LGLAV1MultiTaskModel(
        cfg=pretrained.yaml,
        nc=1,
        num_diseases=8,
        image_cls_weight=0.5,
        projection_dim=128,
        gate_hidden_dim=128,
        attention_eps=1.0e-6,
        min_attention_mass_fraction=1.0e-4,
        gate_balance_weight=0.01,
        verbose=False,
    )
    transferred = model.load(pretrained)
    model.to(device).train()
    model.args = SimpleNamespace(data="synthetic_capacity_preflight_without_test")
    criterion = model.init_criterion()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    batch_size = 16
    imgsz = 640
    images = torch.rand(batch_size, 3, imgsz, imgsz, device=device)
    masks = torch.full((batch_size, imgsz, imgsz), 255, dtype=torch.int32, device=device)
    masks[:8] = 0
    masks[:8, imgsz // 4 : 3 * imgsz // 4, imgsz // 4 : 3 * imgsz // 4] = 1
    batch = {
        "semantic_mask": masks,
        "has_segmentation": torch.tensor([True] * 8 + [False] * 8, device=device),
        "has_classification": torch.ones(batch_size, dtype=torch.bool, device=device),
        "classification_target": torch.arange(batch_size, device=device) % 8,
    }

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        detach_probe = model(images)[-1].square().mean()
    detach_probe.backward()
    head = model.model[-1]
    if any(parameter.grad is not None for parameter in head.classifier.parameters()):
        raise RuntimeError("lgx classification loss reached the main semantic classifier through attention")
    if any(parameter.grad is not None for parameter in head.aux_head.parameters()):
        raise RuntimeError("lgx classification loss reached the auxiliary semantic classifier")
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        predictions = model(images)
        total, items = criterion(predictions, batch)
    if not torch.isfinite(total) or not torch.isfinite(items).all():
        raise RuntimeError(f"Non-finite capacity preflight: total={total}, items={items}")
    if items.numel() != 5:
        raise RuntimeError(f"Expected five lgx loss items, found {items.numel()}")
    scaler.scale(total).backward()
    projection_gradients = [module.weight.grad for module in head.scale_projections]
    gate_gradient = head.global_local_gate[-1].weight.grad
    if any(gradient is None or not torch.isfinite(gradient).all() for gradient in projection_gradients):
        raise RuntimeError("lgx projection gradients are absent or non-finite")
    if gate_gradient is None or not torch.isfinite(gate_gradient).all():
        raise RuntimeError("lgx fusion-gate gradient is absent or non-finite")
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()

    total_device = int(torch.cuda.get_device_properties(device).total_memory)
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    report = {
        "status": "PASS",
        "scope": "lgx LGLA v1 YOLO26m synthetic one-step capacity; no train/val/test data",
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "batch": batch_size,
        "imgsz": imgsz,
        "amp": True,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "classification_head_parameters": sum(
            parameter.numel() for parameter in model.model[-1].image_classifier.parameters()
        ),
        "lgla_parameters": sum(parameter.numel() for parameter in head.scale_projections.parameters())
        + sum(parameter.numel() for parameter in head.global_local_gate.parameters()),
        "projection_parameters": sum(parameter.numel() for parameter in head.scale_projections.parameters()),
        "gate_parameters": sum(parameter.numel() for parameter in head.global_local_gate.parameters()),
        "gate_balance_weight": 0.01,
        "gate_balance_loss": float(items[-1].detach().cpu()),
        "projection_dim": head.projection_dim,
        "gate_hidden_dim": head.gate_hidden_dim,
        "gate_initialization": "equal_half_weights",
        "attention_stop_gradient": True,
        "gpu_classification_only_semantic_head_gradients": False,
        "ground_truth_mask_as_classifier_input": False,
        "empty_attention_fallback": "global_average",
        "attention_head_inputs": list(head.f),
        "gate_weights_shape": list(head.last_gate_weights.shape),
        "gate_weights_finite": bool(torch.isfinite(head.last_gate_weights).all()),
        "attention_mass_fraction_finite": bool(torch.isfinite(head.last_attention_mass_fraction).all()),
        "projection_gradients_finite": True,
        "gate_gradient_finite": True,
        "p5_descriptor": False,
        "semantic_shape": list(predictions[0].shape),
        "classification_shape": list(predictions[-1].shape),
        "loss_items": [float(value) for value in items.detach().cpu()],
        "peak_cuda_allocated_bytes": peak_allocated,
        "peak_cuda_reserved_bytes": peak_reserved,
        "peak_reserved_fraction_of_device": peak_reserved / total_device,
        "total_device_bytes": total_device,
        "pretrained_load_result": str(transferred),
        "optimizer": "AdamW",
        "optimizer_step_completed": True,
        "formal_training_started": False,
        "test_metrics_used": False,
        "external_metrics_used": False,
        "pretrained_weights_used": True,
        "download_performed": False,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()



