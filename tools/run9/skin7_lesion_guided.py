"""Runs7: lesion-guided multi-scale dual attention for Skin7.

The semantic branch is intentionally identical to runs5.  Its foreground
probability guides P3/P4/P5 pooling only for the image-classification branch.
An ECA-style channel gate provides the second attention axis.  No loss,
sampling, data, or augmentation policy is changed in this module.
"""

from __future__ import annotations

from copy import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import SemanticSegment
from ultralytics.utils import RANK

from skin7_cbam import Skin7CBAMTrainer
from skin7_multitask import (
    DISEASE_NAMES,
    Skin7MultiTaskModel,
    Skin7MultiTaskValidator,
    Skin7SemanticClassificationHead,
)


BACKGROUND_ID = 7


class IdentityInitializedECA(nn.Module):
    """ECA-style local cross-channel gate with an identity initial state."""

    def __init__(self, kernel_size: int = 5):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(f"ECA kernel_size must be a positive odd integer, found {kernel_size}")
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        nn.init.zeros_(self.conv.weight)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        descriptor = F.adaptive_avg_pool2d(feature, 1).flatten(2).transpose(1, 2)
        gate = 2.0 * torch.sigmoid(self.conv(descriptor)).transpose(1, 2).unsqueeze(-1)
        return feature * gate


class Skin7LesionGuidedSemanticClassificationHead(Skin7SemanticClassificationHead):
    """Unchanged semantic head plus lesion-guided P3/P4/P5 classification."""

    def __init__(
        self,
        nc: int,
        ch: tuple[int, int, int],
        num_diseases: int = 7,
        hidden_channels: int = 128,
        background_id: int = BACKGROUND_ID,
        eca_kernel_size: int = 5,
        mask_guided_mix: float = 0.5,
        detach_mask: bool = True,
    ):
        if len(ch) != 3:
            raise ValueError(f"Runs7 requires P3/P4/P5 channels, found {ch}")
        if not 0.0 <= mask_guided_mix <= 1.0:
            raise ValueError(f"mask_guided_mix must be in [0, 1], found {mask_guided_mix}")
        if not 0 <= background_id < nc:
            raise ValueError(f"background_id={background_id} is invalid for nc={nc}")
        super().__init__(
            nc=nc,
            ch=(ch[0], ch[1]),
            num_diseases=num_diseases,
            hidden_channels=hidden_channels,
        )
        self.scale_channels = tuple(int(value) for value in ch)
        self.background_id = int(background_id)
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
        probability = semantic_logits.float().softmax(dim=1)
        foreground = 1.0 - probability[:, self.background_id : self.background_id + 1]
        if self.detach_mask:
            foreground = foreground.detach()
        return foreground

    def _pool_scale(self, feature: torch.Tensor, foreground: torch.Tensor, scale_index: int) -> torch.Tensor:
        attended = self.channel_attention[scale_index](feature)
        global_pool = F.adaptive_avg_pool2d(attended, 1).flatten(1)
        mask = F.interpolate(foreground, size=attended.shape[-2:], mode="bilinear", align_corners=False)
        mask = mask.to(dtype=attended.dtype).clamp_(0.0, 1.0)
        denominator = mask.sum(dim=(2, 3)).clamp_min(torch.finfo(attended.dtype).eps)
        guided_pool = (attended * mask).sum(dim=(2, 3)) / denominator
        mix = self.mask_guided_mix
        return (1.0 - mix) * global_pool + mix * guided_pool

    def forward(self, x: list[torch.Tensor]):
        if len(x) != 3:
            raise ValueError(f"Runs7 head expected [P3, P4, P5], received {len(x)} feature maps")

        # Call SemanticSegment directly so the runs5 semantic P3/P4 path is
        # unchanged and the custom three-scale classifier is evaluated once.
        semantic = SemanticSegment.forward(self, x[:2])
        semantic_main = semantic[0] if isinstance(semantic, tuple) else semantic
        foreground = self._foreground_probability(semantic_main)
        pooled = torch.cat(
            [self._pool_scale(feature, foreground, index) for index, feature in enumerate(x)], dim=1
        )
        image_logits = self.image_classifier(pooled)

        if self.training:
            if isinstance(semantic, tuple):
                return semantic[0], semantic[1], image_logits
            return semantic, None, image_logits
        return semantic, image_logits


class Skin7LesionGuidedMultiTaskModel(Skin7MultiTaskModel):
    """Runs7 model with classifier-only P3/P4/P5 lesion guidance."""

    def __init__(
        self,
        cfg="yolo26s-sem.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        num_diseases: int = 7,
        image_cls_weight: float = 0.5,
        p5_layer: int = 10,
        eca_kernel_size: int = 5,
        mask_guided_mix: float = 0.5,
        detach_mask: bool = True,
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
        if list(old_head.f) != [16, 13]:
            raise RuntimeError(f"Runs7 was preregistered for YOLO26-sem head inputs [16, 13], found {old_head.f}")
        if p5_layer < 0 or p5_layer >= len(self.model) - 1:
            raise ValueError(f"Invalid P5 layer index {p5_layer}")
        p5_module = self.model[p5_layer]
        try:
            p5_channels = int(p5_module.cv2.conv.out_channels)
        except AttributeError as exc:
            raise TypeError(f"Layer {p5_layer} is not the expected YOLO26 C2PSA P5 module") from exc

        p3_channels = old_head.classifier[0].conv.in_channels
        p4_channels = old_head.aux_head[0].conv.in_channels
        new_head = Skin7LesionGuidedSemanticClassificationHead(
            nc=old_head.nc,
            ch=(p3_channels, p4_channels, p5_channels),
            num_diseases=num_diseases,
            hidden_channels=p3_channels,
            background_id=BACKGROUND_ID,
            eca_kernel_size=eca_kernel_size,
            mask_guided_mix=mask_guided_mix,
            detach_mask=detach_mask,
        )
        new_head.classifier.load_state_dict(old_head.classifier.state_dict())
        new_head.aux_head.load_state_dict(old_head.aux_head.state_dict())

        # Preserve the runs5 P3/P4 classifier initialization.  The added P5
        # block begins at zero contribution and learns only in runs7.
        old_first, old_last = old_head.image_classifier[0], old_head.image_classifier[3]
        new_first, new_last = new_head.image_classifier[0], new_head.image_classifier[3]
        with torch.no_grad():
            new_first.weight.zero_()
            new_first.weight[:, : old_first.in_features].copy_(old_first.weight)
            new_first.bias.copy_(old_first.bias)
            new_last.weight.copy_(old_last.weight)
            new_last.bias.copy_(old_last.bias)

        new_head.f = [16, 13, p5_layer]
        new_head.i = old_head.i
        new_head.type = f"{type(new_head).__module__}.{type(new_head).__name__}"
        new_head.np = sum(parameter.numel() for parameter in new_head.parameters())
        new_head.stride = torch.cat((old_head.stride, old_head.stride[-1:] * 2))
        self.model[-1] = new_head
        self.save = sorted(set(self.save + [p5_layer]))
        self.lesion_guided_config = {
            "p5_layer": int(p5_layer),
            "eca_kernel_size": int(eca_kernel_size),
            "mask_guided_mix": float(mask_guided_mix),
            "detach_mask": bool(detach_mask),
        }
        if verbose:
            self.info()


class Skin7LesionGuidedTrainer(Skin7CBAMTrainer):
    """Runs7 trainer; CBAM inheritance is checkpoint bookkeeping only."""

    def __init__(self, *args, lesion_guided: dict | None = None, **kwargs):
        self.lesion_guided = dict(lesion_guided or {})
        super().__init__(*args, **kwargs)

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        model = Skin7LesionGuidedMultiTaskModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            num_diseases=len(DISEASE_NAMES),
            image_cls_weight=self.image_cls_weight,
            **self.lesion_guided,
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
