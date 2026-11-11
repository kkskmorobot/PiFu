#!/usr/bin/env python3
"""Synthetic forward/backward verification for the partial-label v1 model.

This command never opens train/val/test images.  It constructs the real
YOLO26s semantic backbone, replaces the head with binary segmentation plus the
plain 8-class classifier, and runs one optimizer step on synthetic partial
labels.  It is an implementation smoke test, not an experiment result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from imaplusplus_partial_multitask_v1 import (
    BinaryPartialMultiTaskModel,
    PartialLabelMultiTaskLoss,
    PartialLabelSemanticDataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=PROJECT_ROOT / "models" / "yolo26s-sem.pt")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or CUDA index such as 0")
    parser.add_argument("--imgsz", type=int, default=128)
    parser.add_argument("--data", type=Path, help="Optional combined multitask_data.yaml for a real train/val batch.")
    parser.add_argument("--split", choices=("train", "val"), default="train")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    normalized = value.strip().lower()
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized == "cuda":
        normalized = "cuda:0"
    elif normalized.isdigit():
        normalized = f"cuda:{normalized}"
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke requested but torch.cuda.is_available() is false")
    return device


def grad_l1(parameters) -> float:
    return float(sum(parameter.grad.detach().abs().sum().item() for parameter in parameters if parameter.grad is not None))


def real_partial_batch(data_yaml: Path, split: str, imgsz: int) -> tuple[dict, dict]:
    from ultralytics.cfg import get_cfg
    from ultralytics.data.utils import check_det_dataset

    data = check_det_dataset(str(data_yaml), split=split)
    manifest = Path(data["partial_label_manifest"])
    if not manifest.is_absolute():
        manifest = (Path(data["path"]) / manifest).resolve()
    cfg = get_cfg(overrides={"task": "semantic", "mode": "val", "imgsz": imgsz, "batch": 3})
    for name in ("mosaic", "mixup", "cutmix", "copy_paste"):
        setattr(cfg, name, 0.0)
    dataset = PartialLabelSemanticDataset(
        img_path=data[split],
        imgsz=imgsz,
        batch_size=3,
        augment=False,
        hyp=cfg,
        rect=False,
        cache=False,
        single_cls=False,
        stride=32,
        pad=0.0,
        prefix=f"{split}: ",
        task="semantic",
        classes=None,
        data=data,
        fraction=1.0,
        partial_label_manifest=manifest,
        expected_split=split,
    )
    role_indices = {}
    for index, path in enumerate(dataset.im_files):
        record = dataset.partial_records[Path(path).resolve()]
        has_seg, has_cls = record["has_segmentation"], record["has_classification"]
        role = "joint" if has_seg and has_cls else "segmentation_only" if has_seg else "classification_only"
        role_indices.setdefault(role, index)
        if len(role_indices) == 3:
            break
    if set(role_indices) != {"joint", "segmentation_only", "classification_only"}:
        raise RuntimeError(f"Unable to find all three partial-label roles in {split}: {sorted(role_indices)}")
    batch = PartialLabelSemanticDataset.collate_fn([dataset[role_indices[key]] for key in sorted(role_indices)])
    metadata = {"data": str(data_yaml), "split": split, "isic_ids": list(batch["isic_id"])}
    return batch, metadata


def main() -> None:
    args = parse_args()
    if args.imgsz < 64 or args.imgsz % 32:
        raise ValueError("--imgsz must be a multiple of 32 and at least 64")
    weights = args.weights.resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)
    device = resolve_device(args.device)

    from ultralytics.nn.tasks import load_checkpoint

    pretrained, _ = load_checkpoint(weights, device="cpu")
    model = BinaryPartialMultiTaskModel(
        cfg=pretrained.yaml,
        nc=1,
        num_diseases=8,
        image_cls_weight=1.0,
        verbose=False,
    )
    transferred = model.load(pretrained)
    model.args = SimpleNamespace(data="synthetic_partial_label_smoke")
    model.to(device).train()
    criterion = PartialLabelMultiTaskLoss(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    batch_size = 3
    if args.data:
        data_yaml = args.data.resolve() if args.data.is_absolute() else (PROJECT_ROOT / args.data).resolve()
        if not data_yaml.is_file():
            raise FileNotFoundError(data_yaml)
        batch, batch_metadata = real_partial_batch(data_yaml, args.split, args.imgsz)
        images = batch["img"].to(device).float() / 255.0
        batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        scope = "real combined loader batch implementation smoke; no test metrics"
    else:
        images = torch.rand(batch_size, 3, args.imgsz, args.imgsz, device=device)
        masks = torch.full((batch_size, args.imgsz, args.imgsz), 255, dtype=torch.int32, device=device)
        masks[0] = 0
        masks[0, args.imgsz // 4 : 3 * args.imgsz // 4, args.imgsz // 4 : 3 * args.imgsz // 4] = 1
        masks[2] = 0
        masks[2, args.imgsz // 3 : 2 * args.imgsz // 3, args.imgsz // 3 : 2 * args.imgsz // 3] = 1
        batch = {
            "semantic_mask": masks,
            "has_segmentation": torch.tensor([True, False, True], device=device),
            "has_classification": torch.tensor([False, True, True], device=device),
            "classification_target": torch.tensor([-1, 4, 7], device=device),
        }
        batch_metadata = {"data": None, "split": None, "isic_ids": []}
        scope = "synthetic implementation smoke; no dataset or test metrics"
    optimizer.zero_grad(set_to_none=True)
    predictions = model(images)
    total, items = criterion(predictions, batch)
    if not torch.isfinite(total) or not torch.isfinite(items).all():
        raise RuntimeError(f"Non-finite synthetic loss: total={total} items={items}")
    total.backward()
    head = model.model[-1]
    semantic_grad = grad_l1(head.classifier.parameters())
    classification_grad = grad_l1(head.image_classifier.parameters())
    if semantic_grad <= 0 or classification_grad <= 0:
        raise RuntimeError(
            f"Expected non-zero gradients for both heads, found semantic={semantic_grad}, classification={classification_grad}"
        )
    optimizer.step()
    report = {
        "status": "PASS",
        "scope": scope,
        "batch_metadata": batch_metadata,
        "device": str(device),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "semantic_shape": list(predictions[0].shape),
        "classification_shape": list(predictions[-1].shape),
        "loss_items": [float(value) for value in items.detach().cpu()],
        "semantic_grad_l1": semantic_grad,
        "classification_grad_l1": classification_grad,
        "pretrained_load_result": str(transferred),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
