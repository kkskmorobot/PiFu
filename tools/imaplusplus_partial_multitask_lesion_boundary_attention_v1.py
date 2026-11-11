"""Run17 detached lesion-boundary-guided attention on the Run16 YOLO26m baseline.

The frozen Run16 BLG-MDA descriptors remain unchanged. A detached boundary map
is derived from the binary semantic probability and used to pool one additional
P3/P4/P5 descriptor block. New classifier columns are zero initialized, making
the initial logits exactly equal to Run16 while allowing boundary evidence to
be learned during training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import SemanticSegment
from ultralytics.utils import RANK

from imaplusplus_partial_multitask_lesion_attention_utility_selection_v1 import (
    UtilityEligibleBinaryLesionGuidedAttentionTrainer,
)
from imaplusplus_partial_multitask_lesion_attention_v1 import (
    BinaryLesionGuidedAttentionHead,
    BinaryLesionGuidedAttentionModel,
    IdentityInitializedECA,
    PartialLabelSemanticDataset,
)
from imaplusplus_partial_multitask_v1 import CLASS_NAMES, ContractError


class BinaryLesionBoundaryGuidedAttentionHead(BinaryLesionGuidedAttentionHead):
    """Run16 descriptors plus zero-initialized detached boundary descriptors."""

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
        if detach_mask is not True:
            raise ContractError("Run17 requires a detached semantic probability and boundary map")
        super().__init__(
            nc=nc,
            ch=ch,
            num_diseases=num_diseases,
            hidden_channels=hidden_channels,
            eca_kernel_size=eca_kernel_size,
            mask_guided_mix=mask_guided_mix,
            detach_mask=detach_mask,
        )
        old_first = self.image_classifier[0]
        expanded_first = nn.Linear(2 * old_first.in_features, old_first.out_features)
        with torch.no_grad():
            expanded_first.weight.zero_()
            expanded_first.weight[:, : old_first.in_features].copy_(old_first.weight)
            expanded_first.bias.copy_(old_first.bias)
        self.image_classifier[0] = expanded_first
        self.base_descriptor_channels = int(old_first.in_features)
        self.boundary_descriptor_channels = int(old_first.in_features)

    @staticmethod
    def _boundary_probability(foreground: torch.Tensor) -> torch.Tensor:
        if foreground.ndim != 4 or foreground.shape[1] != 1:
            raise ContractError(f"Run17 boundary source must be [N,1,H,W], found {tuple(foreground.shape)}")
        source = foreground.detach().float().clamp(0.0, 1.0)
        dx = F.pad((source[..., 1:] - source[..., :-1]).abs(), (0, 1, 0, 0))
        dy = F.pad((source[..., 1:, :] - source[..., :-1, :]).abs(), (0, 0, 0, 1))
        boundary = dx + dy
        scale = boundary.amax(dim=(2, 3), keepdim=True).clamp_min(torch.finfo(boundary.dtype).eps)
        return (boundary / scale).clamp(0.0, 1.0)

    def _pool_boundary(
        self,
        feature: torch.Tensor,
        boundary: torch.Tensor,
        scale_index: int,
    ) -> torch.Tensor:
        attended = self.channel_attention[scale_index](feature)
        mask = F.interpolate(boundary, size=attended.shape[-2:], mode="bilinear", align_corners=False)
        mask = mask.to(dtype=attended.dtype).clamp(0.0, 1.0)
        denominator = mask.sum(dim=(2, 3)).clamp_min(torch.finfo(attended.dtype).eps)
        return (attended * mask).sum(dim=(2, 3)) / denominator

    def forward(self, x: list[torch.Tensor]):
        if len(x) != 3:
            raise ContractError(f"Run17 head expected [P3, P4, P5], received {len(x)} maps")
        semantic = SemanticSegment.forward(self, x[:2])
        semantic_main = semantic[0] if isinstance(semantic, tuple) else semantic
        foreground = self._foreground_probability(semantic_main)
        boundary = self._boundary_probability(foreground)
        base_descriptors = [
            self._pool_scale(feature, foreground, index) for index, feature in enumerate(x)
        ]
        boundary_descriptors = [
            self._pool_boundary(feature, boundary, index) for index, feature in enumerate(x)
        ]
        image_logits = self.image_classifier(torch.cat(base_descriptors + boundary_descriptors, dim=1))
        if self.training:
            if isinstance(semantic, tuple):
                return semantic[0], semantic[1], image_logits
            return semantic, None, image_logits
        return semantic, image_logits


class BinaryLesionBoundaryGuidedAttentionModel(BinaryLesionGuidedAttentionModel):
    """Run16 YOLO26m model with the single boundary-descriptor factor."""

    def __init__(
        self,
        cfg="yolo26m-sem.yaml",
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
            p5_layer=p5_layer,
            eca_kernel_size=eca_kernel_size,
            mask_guided_mix=mask_guided_mix,
            detach_mask=detach_mask,
        )
        old_head = self.model[-1]
        if not isinstance(old_head, BinaryLesionGuidedAttentionHead):
            raise ContractError(f"Run17 expected the frozen Run16 attention head, found {type(old_head).__name__}")
        channels = tuple(int(value) for value in old_head.scale_channels)
        hidden_channels = int(old_head.image_classifier[0].out_features)
        new_head = BinaryLesionBoundaryGuidedAttentionHead(
            nc=1,
            ch=channels,
            num_diseases=num_diseases,
            hidden_channels=hidden_channels,
            eca_kernel_size=eca_kernel_size,
            mask_guided_mix=mask_guided_mix,
            detach_mask=detach_mask,
        )
        new_head.classifier.load_state_dict(old_head.classifier.state_dict())
        if old_head.aux_head is not None:
            new_head.aux_head.load_state_dict(old_head.aux_head.state_dict())
        new_head.channel_attention.load_state_dict(old_head.channel_attention.state_dict())

        old_first, old_last = old_head.image_classifier[0], old_head.image_classifier[3]
        new_first, new_last = new_head.image_classifier[0], new_head.image_classifier[3]
        with torch.no_grad():
            new_first.weight.zero_()
            new_first.weight[:, : old_first.in_features].copy_(old_first.weight)
            new_first.bias.copy_(old_first.bias)
            new_last.weight.copy_(old_last.weight)
            new_last.bias.copy_(old_last.bias)

        new_head.f = list(old_head.f)
        new_head.i = old_head.i
        new_head.type = f"{type(new_head).__module__}.{type(new_head).__name__}"
        new_head.np = sum(parameter.numel() for parameter in new_head.parameters())
        new_head.stride = old_head.stride.clone()
        self.model[-1] = new_head
        self.boundary_attention_config = {
            "source": "detached_binary_semantic_probability_first_difference_magnitude",
            "scales": ["P3", "P4", "P5"],
            "normalization": "per_image_max_then_normalized_weighted_average",
            "classifier_initial_contribution": "zero",
        }
        if verbose:
            self.info()


class BinaryLesionBoundaryGuidedAttentionTrainer(UtilityEligibleBinaryLesionGuidedAttentionTrainer):
    """Run16 trainer with only the Run17 boundary-attention model replacement."""

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        model = BinaryLesionBoundaryGuidedAttentionModel(
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
    "BinaryLesionBoundaryGuidedAttentionHead",
    "BinaryLesionBoundaryGuidedAttentionModel",
    "BinaryLesionBoundaryGuidedAttentionTrainer",
    "IdentityInitializedECA",
    "PartialLabelSemanticDataset",
]
