"""Run12 binary-lesion-guided multi-scale dual attention.

This module preserves the accepted Run10 partial-label dataset, losses,
validator, sampling, and checkpoint contract.  Its only training-behavior
change is the image-classification feature aggregator:

* the unchanged binary semantic head produces a detached soft lesion map;
* identity-initialized ECA gates P3, P4, and P5 channels;
* each scale mixes global average pooling with lesion-guided pooling;
* the semantic main/auxiliary paths remain unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import SemanticSegment
from ultralytics.utils import RANK

from imaplusplus_partial_multitask_v1 import (
    CLASS_NAMES,
    BinaryPartialClassificationHead,
    BinaryPartialMultiTaskModel,
    ContractError,
    PartialLabelMultiTaskTrainer,
    PartialLabelSemanticDataset,
)


class IdentityInitializedECA(nn.Module):
    """Local cross-channel gate whose initial multiplicative value is one."""

    def __init__(self, kernel_size: int = 5) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ContractError(f"eca_kernel_size must be a positive odd integer, found {kernel_size}")
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        nn.init.zeros_(self.conv.weight)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        descriptor = F.adaptive_avg_pool2d(feature, 1).flatten(2).transpose(1, 2)
        gate = 2.0 * torch.sigmoid(self.conv(descriptor)).transpose(1, 2).unsqueeze(-1)
        return feature * gate


class BinaryLesionGuidedAttentionHead(BinaryPartialClassificationHead):
    """Unchanged binary semantic head plus P3/P4/P5 dual-attention pooling."""

    def __init__(
        self,
        nc: int,
        ch: tuple[int, int, int],
        num_diseases: int = 8,
        hidden_channels: int = 128,
        eca_kernel_size: int = 5,
        mask_guided_mix: float = 0.5,
        detach_mask: bool = True,
    ) -> None:
        if len(ch) != 3:
            raise ContractError(f"Run12 requires P3/P4/P5 channels, found {ch}")
        if not 0.0 <= mask_guided_mix <= 1.0:
            raise ContractError(f"mask_guided_mix must be in [0, 1], found {mask_guided_mix}")
        super().__init__(
            nc=nc,
            ch=(ch[0], ch[1]),
            num_diseases=num_diseases,
            hidden_channels=hidden_channels,
        )
        self.scale_channels = tuple(int(value) for value in ch)
        self.mask_guided_mix = float(mask_guided_mix)
        self.detach_mask = bool(detach_mask)
        self.channel_attention = nn.ModuleList(
            [IdentityInitializedECA(kernel_size=eca_kernel_size) for _ in self.scale_channels]
        )
        self.image_classifier = nn.Sequential(
            nn.Linear(sum(self.scale_channels), hidden_channels),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.20),
            nn.Linear(hidden_channels, num_diseases),
        )

    def _foreground_probability(self, semantic_logits: torch.Tensor) -> torch.Tensor:
        if semantic_logits.ndim != 4 or semantic_logits.shape[1] != 1:
            raise ContractError(
                f"Run12 requires one-channel semantic logits, found {tuple(semantic_logits.shape)}"
            )
        foreground = semantic_logits.float().sigmoid()
        return foreground.detach() if self.detach_mask else foreground

    def _pool_scale(
        self,
        feature: torch.Tensor,
        foreground: torch.Tensor,
        scale_index: int,
    ) -> torch.Tensor:
        attended = self.channel_attention[scale_index](feature)
        global_pool = F.adaptive_avg_pool2d(attended, 1).flatten(1)
        mask = F.interpolate(foreground, size=attended.shape[-2:], mode="bilinear", align_corners=False)
        mask = mask.to(dtype=attended.dtype).clamp(0.0, 1.0)
        denominator = mask.sum(dim=(2, 3)).clamp_min(torch.finfo(attended.dtype).eps)
        guided_pool = (attended * mask).sum(dim=(2, 3)) / denominator
        mix = self.mask_guided_mix
        return (1.0 - mix) * global_pool + mix * guided_pool

    def forward(self, x: list[torch.Tensor]):
        if len(x) != 3:
            raise ContractError(f"Run12 head expected [P3, P4, P5], received {len(x)} maps")
        semantic = SemanticSegment.forward(self, x[:2])
        semantic_main = semantic[0] if isinstance(semantic, tuple) else semantic
        foreground = self._foreground_probability(semantic_main)
        pooled = torch.cat(
            [self._pool_scale(feature, foreground, index) for index, feature in enumerate(x)],
            dim=1,
        )
        image_logits = self.image_classifier(pooled)
        if self.training:
            if isinstance(semantic, tuple):
                return semantic[0], semantic[1], image_logits
            return semantic, None, image_logits
        return semantic, image_logits


class BinaryLesionGuidedAttentionModel(BinaryPartialMultiTaskModel):
    """Run10 model with classifier-only binary lesion-guided attention."""

    def __init__(
        self,
        cfg="yolo26s-sem.yaml",
        ch: int = 3,
        nc: int | None = 1,
        verbose: bool = True,
        num_diseases: int = 8,
        image_cls_weight: float = 0.5,
        p5_layer: int = 10,
        eca_kernel_size: int = 5,
        mask_guided_mix: float = 0.5,
        detach_mask: bool = True,
    ) -> None:
        super().__init__(
            cfg=cfg,
            ch=ch,
            nc=nc,
            verbose=False,
            num_diseases=num_diseases,
            image_cls_weight=image_cls_weight,
        )
        old_head = self.model[-1]
        if not isinstance(old_head, BinaryPartialClassificationHead):
            raise ContractError(f"Expected Run10 binary multitask head, found {type(old_head).__name__}")
        if list(old_head.f) != [16, 13]:
            raise ContractError(f"Run12 is locked to YOLO26-sem head inputs [16, 13], found {old_head.f}")
        if p5_layer < 0 or p5_layer >= len(self.model) - 1:
            raise ContractError(f"Invalid P5 layer index {p5_layer}")
        p5_module = self.model[p5_layer]
        try:
            p5_channels = int(p5_module.cv2.conv.out_channels)
        except AttributeError as exc:
            raise ContractError(f"Layer {p5_layer} is not the locked YOLO26 C2PSA P5 source") from exc

        p3_channels = old_head.classifier[0].conv.in_channels
        p4_channels = old_head.aux_head[0].conv.in_channels
        new_head = BinaryLesionGuidedAttentionHead(
            nc=1,
            ch=(p3_channels, p4_channels, p5_channels),
            num_diseases=num_diseases,
            hidden_channels=p3_channels,
            eca_kernel_size=eca_kernel_size,
            mask_guided_mix=mask_guided_mix,
            detach_mask=detach_mask,
        )
        new_head.classifier.load_state_dict(old_head.classifier.state_dict())
        if old_head.aux_head is not None:
            new_head.aux_head.load_state_dict(old_head.aux_head.state_dict())

        old_first, old_last = old_head.image_classifier[0], old_head.image_classifier[3]
        new_first, new_last = new_head.image_classifier[0], new_head.image_classifier[3]
        with torch.no_grad():
            new_first.weight.zero_()
            new_first.weight[:, : old_first.in_features].copy_(old_first.weight)
            new_first.bias.copy_(old_first.bias)
            new_last.weight.copy_(old_last.weight)
            new_last.bias.copy_(old_last.bias)

        new_head.f = [16, 13, int(p5_layer)]
        new_head.i = old_head.i
        new_head.type = f"{type(new_head).__module__}.{type(new_head).__name__}"
        new_head.np = sum(parameter.numel() for parameter in new_head.parameters())
        new_head.stride = torch.cat((old_head.stride, old_head.stride[-1:] * 2))
        self.model[-1] = new_head
        self.save = sorted(set(self.save + [int(p5_layer)]))
        self.lesion_attention_config = {
            "p5_layer": int(p5_layer),
            "eca_kernel_size": int(eca_kernel_size),
            "mask_guided_mix": float(mask_guided_mix),
            "detach_mask": bool(detach_mask),
        }
        if verbose:
            self.info()


class BinaryLesionGuidedAttentionTrainer(PartialLabelMultiTaskTrainer):
    """Run12 trainer; all non-architecture behavior is inherited from Run10."""

    def __init__(self, *args, lesion_attention: dict | None = None, **kwargs) -> None:
        self.lesion_attention = dict(lesion_attention or {})
        super().__init__(*args, **kwargs)

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        model = BinaryLesionGuidedAttentionModel(
            cfg,
            nc=1,
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            num_diseases=len(CLASS_NAMES),
            image_cls_weight=self.image_cls_weight,
            **self.lesion_attention,
        )
        if weights:
            model.load(weights)
        return model


__all__ = [
    "BinaryLesionGuidedAttentionHead",
    "BinaryLesionGuidedAttentionModel",
    "BinaryLesionGuidedAttentionTrainer",
    "IdentityInitializedECA",
    "PartialLabelSemanticDataset",
]
