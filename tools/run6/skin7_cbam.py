"""Runs6: runs5 multitask YOLO26 with one standard CBAM on the P3 feature.

The CBAM is applied inside the final head, immediately after the existing P3/8
neck fusion output and before both the semantic classifier and the P3/P4 image
classifier. This is functionally the planned single insertion without changing
Ultralytics ``site-packages`` or the YOLO layer indices.
"""

from __future__ import annotations

import csv
import json
import shutil
from copy import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils import RANK

from skin7_multitask import (
    DISEASE_NAMES,
    Skin7MultiTaskModel,
    Skin7MultiTaskTrainer,
    Skin7MultiTaskValidator,
    Skin7SemanticClassificationHead,
)


class ChannelAttention(nn.Module):
    """Standard CBAM channel attention using shared avg/max-pool MLP."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.shared_mlp(F.adaptive_avg_pool2d(x, 1))
        # ``adaptive_max_pool2d_backward_cuda`` is nondeterministic in the
        # pinned PyTorch build. Global amax is mathematically identical here
        # and preserves the standard CBAM max-pooling descriptor.
        maximum = self.shared_mlp(x.amax(dim=(2, 3), keepdim=True))
        return x * torch.sigmoid(avg + maximum)


class SpatialAttention(nn.Module):
    """Standard CBAM spatial attention over channel mean/max maps."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError(f"CBAM spatial kernel must be 3 or 7, found {kernel_size}")
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        descriptors = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.conv(descriptors))


class CBAM(nn.Module):
    """Convolutional Block Attention Module: channel then spatial attention."""

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction=reduction)
        self.spatial_attention = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial_attention(self.channel_attention(x))


class Skin7CBAMSemanticClassificationHead(Skin7SemanticClassificationHead):
    """Runs5 head with exactly one CBAM applied to its P3/8 input."""

    def __init__(
        self,
        nc: int,
        ch: tuple[int, int],
        num_diseases: int = 7,
        hidden_channels: int = 128,
        reduction: int = 16,
        spatial_kernel: int = 7,
    ):
        super().__init__(nc=nc, ch=ch, num_diseases=num_diseases, hidden_channels=hidden_channels)
        self.p3_cbam = CBAM(ch[0], reduction=reduction, spatial_kernel=spatial_kernel)

    def forward(self, x: list[torch.Tensor]):
        return super().forward([self.p3_cbam(x[0]), x[1]])


class Skin7CBAMMultiTaskModel(Skin7MultiTaskModel):
    """Runs6 model; all runs5 behavior is retained except the single P3 CBAM."""

    def __init__(
        self,
        cfg="yolo26s-sem.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        num_diseases: int = 7,
        image_cls_weight: float = 0.5,
    ):
        super().__init__(
            cfg=cfg,
            ch=ch,
            nc=nc,
            verbose=False,
            num_diseases=num_diseases,
            image_cls_weight=image_cls_weight,
        )
        old_head = self.model[-1]
        p3_channels = old_head.classifier[0].conv.in_channels
        p4_channels = old_head.aux_head[0].conv.in_channels
        new_head = Skin7CBAMSemanticClassificationHead(
            nc=old_head.nc,
            ch=(p3_channels, p4_channels),
            num_diseases=num_diseases,
            hidden_channels=p3_channels,
            reduction=16,
            spatial_kernel=7,
        )
        # Preserve the exact runs5 semantic/classification-head initialization;
        # only CBAM introduces additional randomly initialized parameters.
        new_head.classifier.load_state_dict(old_head.classifier.state_dict())
        new_head.aux_head.load_state_dict(old_head.aux_head.state_dict())
        new_head.image_classifier.load_state_dict(old_head.image_classifier.state_dict())
        for attr in ("f", "i", "type"):
            setattr(new_head, attr, getattr(old_head, attr))
        new_head.type = f"{type(new_head).__module__}.{type(new_head).__name__}"
        new_head.np = sum(parameter.numel() for parameter in new_head.parameters())
        new_head.stride = old_head.stride
        self.model[-1] = new_head
        if verbose:
            self.info()


class Skin7CBAMTrainer(Skin7MultiTaskTrainer):
    """Runs6 trainer with pre-registered auxiliary checkpoint selection files."""

    def __init__(self, *args, **kwargs):
        self.best_macro_f1 = float("-inf")
        self.best_feasible_macro_f1 = float("-inf")
        self.best_macro_epoch = None
        self.best_feasible_epoch = None
        super().__init__(*args, **kwargs)
        self._restore_selection_state()

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        model = Skin7CBAMMultiTaskModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            num_diseases=len(DISEASE_NAMES),
            image_cls_weight=self.image_cls_weight,
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        self.loss_names = "ce_loss", "dice_loss", "aux_loss", "image_cls_loss"
        return Skin7MultiTaskValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
            selection=self.selection,
        )

    def _is_feasible(self, row: dict[str, float]) -> bool:
        return (
            row["foreground_miou"] >= self.selection["min_foreground_miou"]
            and row["minority_recall"] >= self.selection["min_minority_recall"]
            and row["mel_to_nv_rate"] <= self.selection["max_mel_to_nv_rate"]
        )

    @staticmethod
    def _row_from_mapping(mapping: dict, epoch: int) -> dict[str, float]:
        return {
            "epoch": int(epoch),
            "macro_f1": float(mapping["metrics/macro_f1"]),
            "foreground_miou": float(mapping["metrics/foreground_mIoU"]),
            "minority_recall": float(mapping["metrics/minority_recall"]),
            "mel_to_nv_rate": float(mapping["metrics/mel_to_nv_rate"]),
        }

    def _restore_selection_state(self) -> None:
        """Reconstruct auxiliary selection state from results.csv after resume."""
        if not Path(self.csv).is_file():
            return
        with Path(self.csv).open("r", newline="", encoding="utf-8-sig") as handle:
            for raw in csv.DictReader(handle):
                row = self._row_from_mapping(raw, int(float(raw["epoch"])))
                if row["macro_f1"] > self.best_macro_f1:
                    self.best_macro_f1, self.best_macro_epoch = row["macro_f1"], row["epoch"]
                if self._is_feasible(row) and row["macro_f1"] > self.best_feasible_macro_f1:
                    self.best_feasible_macro_f1 = row["macro_f1"]
                    self.best_feasible_epoch = row["epoch"]

    def save_model(self):
        """Keep Ultralytics checkpoints plus best-Macro-F1 and feasible-best files."""
        saved = super().save_model()
        row = self._row_from_mapping(self.metrics, self.epoch + 1)
        if row["macro_f1"] > self.best_macro_f1:
            self.best_macro_f1, self.best_macro_epoch = row["macro_f1"], row["epoch"]
            shutil.copy2(self.last, self.wdir / "best_macro_f1.pt")
        if self._is_feasible(row) and row["macro_f1"] > self.best_feasible_macro_f1:
            self.best_feasible_macro_f1 = row["macro_f1"]
            self.best_feasible_epoch = row["epoch"]
            shutil.copy2(self.last, self.wdir / "best_feasible_macro_f1.pt")
        state = {
            "protocol": "primary Macro-F1 with pre-registered segmentation, minority-recall and mel-to-nv constraints",
            "constraints": {
                "min_foreground_miou": self.selection["min_foreground_miou"],
                "min_minority_recall": self.selection["min_minority_recall"],
                "max_mel_to_nv_rate": self.selection["max_mel_to_nv_rate"],
            },
            "best_macro_f1": self.best_macro_f1,
            "best_macro_epoch": self.best_macro_epoch,
            "best_feasible_macro_f1": (
                None if self.best_feasible_macro_f1 == float("-inf") else self.best_feasible_macro_f1
            ),
            "best_feasible_epoch": self.best_feasible_epoch,
        }
        (self.save_dir / "checkpoint_selection.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
        return saved
