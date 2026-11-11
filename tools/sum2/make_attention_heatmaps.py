#!/usr/bin/env python3
"""Create qualitative LGLA attention overlays from frozen seed43 development validation only."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


PROJECT = Path(r"E:\App\DaiMa\JetBrains\Project\PiFuV2")
TOOLS = PROJECT / "tools"
LGX = TOOLS / "lgx"
BUNDLE = PROJECT / "experiments/lgx_YOLO26m_Binary8Class_LGLAv1_GateBalance_Seed43/results/lgx_yolo26m_lgla_v1_gate_balance_seed43/checkpoint_bundles/epoch_0098"
MANIFEST = LGX / "lgx_development_manifest.csv"
OUT = Path(r"C:\Users\ROBOT\Documents\皮肤识别\lgla_finalize\generated")

for p in (TOOLS, LGX):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from imaplusplus_partial_multitask_lgla_v1_gate_balance_seed43 import LGLAV1MultiTaskModel  # noqa: E402


def load_model():
    from ultralytics import YOLO
    wrapper = YOLO(str(BUNDLE / "trainer.pt"), task="semantic")
    model = wrapper.model.float().cpu()
    runtime = torch.load(BUNDLE / "runtime_state.pt", map_location="cpu", weights_only=False)
    exact = runtime["training_state"]["ema"]
    model.load_state_dict(exact, strict=True)
    if not isinstance(model, LGLAV1MultiTaskModel):
        raise TypeError(type(model).__name__)
    return model.cuda().eval()


with MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
    rows = [r for r in csv.DictReader(handle) if r["split"] == "val"]
selected = [
    next(r for r in rows if r["has_segmentation"] == "1" and r["class_name"] == "mel"),
    next(r for r in rows if r["has_segmentation"] == "1" and r["class_name"] == "nv"),
    *[r for r in rows if r["has_segmentation"] == "1" and r["has_classification"] == "0"][:2],
]

model = load_model()
fig, axes = plt.subplots(len(selected), 4, figsize=(12, 11))
metadata = []
for index, row in enumerate(selected):
    image_path = PROJECT / "Data" / row["image_path"]
    mask_path = PROJECT / "Data" / row["mask_path"]
    image = Image.open(image_path).convert("RGB")
    original = np.asarray(image)
    resized = image.resize((640, 640), Image.Resampling.BILINEAR)
    tensor = torch.from_numpy(np.asarray(resized).copy()).permute(2, 0, 1).unsqueeze(0).float().cuda() / 255.0
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=True):
        semantic, logits = model(tensor)
    if isinstance(semantic, tuple):
        semantic = semantic[0]
    attention = torch.sigmoid(semantic.detach().float())
    attention = F.interpolate(attention, size=(original.shape[0], original.shape[1]), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    pred = int(logits.detach().float().argmax(dim=1).item())
    gate = model.model[-1].last_gate_weights[0].detach().float().cpu().tolist()
    mask = np.asarray(Image.open(mask_path).convert("L"))
    class_label = row["class_name"] or "unlabeled"
    axes[index, 0].imshow(original); axes[index, 0].set_title(f"{row['isic_id']} / true {class_label}")
    axes[index, 1].imshow(mask, cmap="gray", vmin=0, vmax=255 if mask.max() > 1 else 1); axes[index, 1].set_title("Ground-truth mask")
    axes[index, 2].imshow(attention, cmap="magma", vmin=0, vmax=1); axes[index, 2].set_title("Predicted lesion attention")
    axes[index, 3].imshow(original); axes[index, 3].imshow(attention, cmap="jet", alpha=0.42, vmin=0, vmax=1)
    axes[index, 3].set_title(f"Overlay / gates G={gate[0]:.2f}, L={gate[1]:.2f}")
    for ax in axes[index]: ax.axis("off")
    metadata.append({"isic_id": row["isic_id"], "true_class": class_label, "predicted_class_id": pred, "global_gate": gate[0], "local_gate": gate[1], "image_path": str(image_path), "mask_path": str(mask_path), "scope": "development validation only"})
fig.suptitle("LGLA v1 qualitative lesion-attention examples (frozen seed43 validation)", y=0.995, fontsize=14)
fig.tight_layout(); fig.savefig(OUT / "figure_attention_heatmaps_seed43.png", dpi=180, bbox_inches="tight"); plt.close(fig)
(OUT / "attention_heatmaps_seed43.json").write_text(json.dumps({"status": "PASS", "weights": str(BUNDLE), "samples": metadata, "test_or_external_used": False}, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"status": "PASS", "figure": str(OUT / 'figure_attention_heatmaps_seed43.png'), "samples": len(metadata)}, ensure_ascii=False))
