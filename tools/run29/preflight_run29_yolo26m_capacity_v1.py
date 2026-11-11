#!/usr/bin/env python3
"""One full batch=16 AMP forward/backward/AdamW capacity step for Run29."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = PROJECT_ROOT / "tools"
RUN29_DIR = Path(__file__).resolve().parent
for path in (TOOLS_DIR, RUN29_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from imaplusplus_partial_multitask_plain_malignant_group_aux_seed43_v1 import (
    MalignantGroupAuxBinaryPartialMultiTaskModel,
    RUN29_MALIGNANT_GROUP_AUX_WEIGHT,
)
from ultralytics.nn.tasks import load_checkpoint


OUTPUT = PROJECT_ROOT / "preflight" / "run29_yolo26m_capacity_v1" / "batch16_img640_amp_20260906_v1.json"
WEIGHTS = PROJECT_ROOT / "models" / "yolo26m-sem.pt"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite capacity report: {OUTPUT}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    pretrained, _ = load_checkpoint(WEIGHTS, device="cpu")
    model = MalignantGroupAuxBinaryPartialMultiTaskModel(
        cfg=pretrained.yaml,
        nc=1,
        num_diseases=8,
        image_cls_weight=0.5,
        malignant_group_aux_weight=RUN29_MALIGNANT_GROUP_AUX_WEIGHT,
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
        predictions = model(images)
        total, items = criterion(predictions, batch)
    if not torch.isfinite(total) or not torch.isfinite(items).all():
        raise RuntimeError(f"Non-finite capacity preflight: total={total}, items={items}")
    if items.numel() != 5 or float(items[-1]) <= 0.0:
        raise RuntimeError(f"Run29 group auxiliary loss was not exercised: {items}")
    scaler.scale(total).backward()
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()

    total_device = int(torch.cuda.get_device_properties(device).total_memory)
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    report = {
        "status": "PASS",
        "scope": "Run29 plain YOLO26m plus criterion-only risk loss synthetic one-step capacity; no dataset loaded",
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
        "new_risk_loss_parameters": 0,
        "malignant_group_aux_weight": model.malignant_group_aux_weight,
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

