"""Run14 BLG-MDA attention with a train-prior-derived MEL-vs-NV logit margin.

Relative to Run12, architecture, initialization, data, validator, sampling,
and inference are unchanged. During training CE only, the NV competitor logit
is increased by ln(n_NV / n_MEL) for samples whose true target is MEL. This
forces a wider MEL-over-NV decision margin without reweighting MEL against all
other diseases.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils import RANK
from ultralytics.utils.loss import SemanticSegmentationLoss

from imaplusplus_partial_multitask_lesion_attention_v1 import (
    BinaryLesionGuidedAttentionHead,
    BinaryLesionGuidedAttentionModel,
    BinaryLesionGuidedAttentionTrainer,
    IdentityInitializedECA,
    PartialLabelSemanticDataset,
)
from imaplusplus_partial_multitask_v1 import CLASS_NAMES, ContractError, unpack_outputs


MEL_ID = CLASS_NAMES.index("mel")
NV_ID = CLASS_NAMES.index("nv")
RUN14_TRAIN_MEL_COUNT = 3047
RUN14_TRAIN_NV_COUNT = 5539
RUN14_MEL_NV_LOGIT_MARGIN = math.log(RUN14_TRAIN_NV_COUNT / RUN14_TRAIN_MEL_COUNT)


def apply_mel_nv_training_margin(
    image_logits: torch.Tensor,
    targets: torch.Tensor,
    margin: float = RUN14_MEL_NV_LOGIT_MARGIN,
) -> torch.Tensor:
    """Return training-only logits with the NV competitor raised for MEL targets."""
    if image_logits.ndim != 2 or image_logits.shape[1] != len(CLASS_NAMES):
        raise ContractError(f"Expected classification logits [N, 8], found {tuple(image_logits.shape)}")
    if targets.ndim != 1 or targets.shape[0] != image_logits.shape[0]:
        raise ContractError("Classification targets must be a length-N vector")
    if float(margin) != RUN14_MEL_NV_LOGIT_MARGIN:
        raise ContractError(
            f"Run14 margin drifted: {margin!r} != {RUN14_MEL_NV_LOGIT_MARGIN!r}"
        )
    adjusted = image_logits.clone()
    mel_rows = targets == MEL_ID
    if bool(mel_rows.any().item()):
        adjusted[mel_rows, NV_ID] = adjusted[mel_rows, NV_ID] + float(margin)
    return adjusted


def run14_mel_nv_margin_ce(
    image_logits: torch.Tensor,
    targets: torch.Tensor,
    margin: float = RUN14_MEL_NV_LOGIT_MARGIN,
) -> torch.Tensor:
    return F.cross_entropy(apply_mel_nv_training_margin(image_logits, targets, margin), targets)


class MELNVLogitMarginPartialLabelMultiTaskLoss(nn.Module):
    """Run12 partial-label loss with the single frozen MEL-vs-NV margin factor."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.semantic_loss = SemanticSegmentationLoss(model)
        self.image_cls_weight = float(getattr(model, "image_cls_weight", 1.0))
        if self.image_cls_weight <= 0:
            raise ContractError(f"image_cls_weight must be positive, found {self.image_cls_weight}")
        self.mel_nv_logit_margin = float(getattr(model, "mel_nv_logit_margin", float("nan")))
        if self.mel_nv_logit_margin != RUN14_MEL_NV_LOGIT_MARGIN:
            raise ContractError(
                "Run14 model margin differs from the train-prior-derived frozen value: "
                f"{self.mel_nv_logit_margin!r} != {RUN14_MEL_NV_LOGIT_MARGIN!r}"
            )

    def forward(self, preds, batch: dict[str, Any]):
        semantic_logits, auxiliary_logits, image_logits = unpack_outputs(preds)
        batch_size = image_logits.shape[0]
        has_seg = batch["has_segmentation"].to(image_logits.device).view(-1).bool()
        has_cls = batch["has_classification"].to(image_logits.device).view(-1).bool()
        targets = batch["classification_target"].to(image_logits.device).view(-1).long()
        if any(tensor.shape[0] != batch_size for tensor in (has_seg, has_cls, targets)):
            raise ContractError("Partial-label availability tensors do not match model batch size")
        if bool(((targets < 0) != (~has_cls)).any().item()):
            raise ContractError("classification_target=-1 must match has_classification=false exactly")
        if bool(((targets[has_cls] < 0) | (targets[has_cls] >= len(CLASS_NAMES))).any().item()):
            raise ContractError("Labeled classification targets must be in [0, 7]")

        masks = batch["semantic_mask"].to(image_logits.device)
        mask_has_valid = (masks != 255).flatten(1).any(1)
        if not torch.equal(mask_has_valid, has_seg):
            raise ContractError("has_segmentation does not match valid semantic-mask pixels")
        semantic_preds = (
            (semantic_logits, auxiliary_logits) if auxiliary_logits is not None else semantic_logits
        )
        semantic_total, semantic_items = self.semantic_loss(semantic_preds, batch)

        if bool(has_cls.any().item()):
            image_ce = run14_mel_nv_margin_ce(
                image_logits[has_cls], targets[has_cls], self.mel_nv_logit_margin
            )
        else:
            image_ce = image_logits.sum() * 0.0
        total = semantic_total + self.image_cls_weight * image_ce * batch_size
        items = torch.cat([semantic_items, image_ce.detach().reshape(1)])
        return total, items


class MELNVLogitMarginBinaryLesionGuidedAttentionModel(BinaryLesionGuidedAttentionModel):
    """Run12 model with a criterion-only MEL-vs-NV margin."""

    def __init__(
        self,
        cfg="yolo26s-sem.yaml",
        ch: int = 3,
        nc: int | None = 1,
        verbose: bool = True,
        num_diseases: int = 8,
        image_cls_weight: float = 0.5,
        mel_nv_logit_margin: float = RUN14_MEL_NV_LOGIT_MARGIN,
        p5_layer: int = 10,
        eca_kernel_size: int = 5,
        mask_guided_mix: float = 0.5,
        detach_mask: bool = True,
    ) -> None:
        if float(mel_nv_logit_margin) != RUN14_MEL_NV_LOGIT_MARGIN:
            raise ContractError("Run14 model margin differs from frozen train-prior value")
        super().__init__(
            cfg=cfg,
            ch=ch,
            nc=nc,
            verbose=verbose,
            num_diseases=num_diseases,
            image_cls_weight=image_cls_weight,
            p5_layer=p5_layer,
            eca_kernel_size=eca_kernel_size,
            mask_guided_mix=mask_guided_mix,
            detach_mask=detach_mask,
        )
        self.mel_nv_logit_margin = float(mel_nv_logit_margin)

    def init_criterion(self):
        return MELNVLogitMarginPartialLabelMultiTaskLoss(self)


class MELNVLogitMarginBinaryLesionGuidedAttentionTrainer(BinaryLesionGuidedAttentionTrainer):
    """Run14 Trainer; validation and inference continue to use unmodified logits."""

    def __init__(
        self,
        *args,
        mel_nv_logit_margin: float = RUN14_MEL_NV_LOGIT_MARGIN,
        **kwargs,
    ) -> None:
        if float(mel_nv_logit_margin) != RUN14_MEL_NV_LOGIT_MARGIN:
            raise ContractError("Run14 Trainer margin differs from frozen train-prior value")
        self.mel_nv_logit_margin = float(mel_nv_logit_margin)
        super().__init__(*args, **kwargs)

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        model = MELNVLogitMarginBinaryLesionGuidedAttentionModel(
            cfg,
            nc=1,
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            num_diseases=len(CLASS_NAMES),
            image_cls_weight=self.image_cls_weight,
            mel_nv_logit_margin=self.mel_nv_logit_margin,
            **self.lesion_attention,
        )
        if weights:
            model.load(weights)
        return model


__all__ = [
    "apply_mel_nv_training_margin",
    "BinaryLesionGuidedAttentionHead",
    "IdentityInitializedECA",
    "MELNVLogitMarginBinaryLesionGuidedAttentionModel",
    "MELNVLogitMarginBinaryLesionGuidedAttentionTrainer",
    "MELNVLogitMarginPartialLabelMultiTaskLoss",
    "MEL_ID",
    "NV_ID",
    "PartialLabelSemanticDataset",
    "RUN14_MEL_NV_LOGIT_MARGIN",
    "run14_mel_nv_margin_ce",
]
