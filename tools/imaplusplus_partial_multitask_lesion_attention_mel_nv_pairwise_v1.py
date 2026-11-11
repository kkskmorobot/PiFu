"""Run15 BLG-MDA attention with a direct MEL-vs-NV pairwise boundary loss.

Relative to frozen Run12, architecture, initialization, base CE, data,
sampling, augmentation, and inference are unchanged.  For true MEL training
samples only, an additive softplus term directly enforces the MEL logit above
the NV logit by the frozen train-prior margin.  Unlike Run14's eight-way CE
margin, the pairwise gradient cannot be diluted by the other six logits.
"""

from __future__ import annotations

import math
from copy import copy
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
from imaplusplus_partial_multitask_v1 import (
    CLASS_NAMES,
    ContractError,
    MEL_ID,
    NV_ID,
    PartialLabelMultiTaskValidator,
    unpack_outputs,
)


RUN15_TRAIN_MEL_COUNT = 3047
RUN15_TRAIN_NV_COUNT = 5539
RUN15_MEL_NV_PAIRWISE_MARGIN = math.log(RUN15_TRAIN_NV_COUNT / RUN15_TRAIN_MEL_COUNT)
RUN15_MEL_NV_PAIRWISE_LOSS_WEIGHT = 1.0
RUN15_MIN_BINARY_IOU = 0.70
RUN15_MIN_GROUP_MACRO_F1 = 0.50


def run15_mel_nv_pairwise_softplus(
    image_logits: torch.Tensor,
    targets: torch.Tensor,
    margin: float = RUN15_MEL_NV_PAIRWISE_MARGIN,
) -> torch.Tensor:
    """Sum true-MEL pair losses and normalize by every labeled sample."""
    if image_logits.ndim != 2 or image_logits.shape[1] != len(CLASS_NAMES):
        raise ContractError(f"Expected classification logits [N, 8], found {tuple(image_logits.shape)}")
    if targets.ndim != 1 or targets.shape[0] != image_logits.shape[0] or not targets.numel():
        raise ContractError("Classification targets must be a non-empty length-N vector")
    if float(margin) != RUN15_MEL_NV_PAIRWISE_MARGIN:
        raise ContractError(
            f"Run15 pairwise margin drifted: {margin!r} != {RUN15_MEL_NV_PAIRWISE_MARGIN!r}"
        )
    mel_rows = targets == MEL_ID
    if not bool(mel_rows.any().item()):
        return image_logits.sum() * 0.0
    boundary = image_logits[mel_rows, NV_ID] - image_logits[mel_rows, MEL_ID] + float(margin)
    return F.softplus(boundary).sum() / targets.numel()


def run15_utility_eligible(binary_iou: float, group_macro_f1: float) -> bool:
    return float(binary_iou) >= RUN15_MIN_BINARY_IOU and float(group_macro_f1) >= RUN15_MIN_GROUP_MACRO_F1


class UtilityEligiblePartialLabelMultiTaskValidator(PartialLabelMultiTaskValidator):
    """Keep training fitness unchanged while rejecting cold-start false feasibility."""

    def __init__(self, *args, selection: dict[str, float] | None = None, **kwargs) -> None:
        selection = dict(selection or {})
        self.min_binary_iou = float(selection.get("min_binary_iou", RUN15_MIN_BINARY_IOU))
        self.min_group_macro_f1 = float(selection.get("min_group_macro_f1", RUN15_MIN_GROUP_MACRO_F1))
        if self.min_binary_iou != RUN15_MIN_BINARY_IOU or self.min_group_macro_f1 != RUN15_MIN_GROUP_MACRO_F1:
            raise ContractError("Run15 utility eligibility floors drifted")
        super().__init__(*args, selection=selection, **kwargs)

    def get_stats(self):
        stats = super().get_stats()
        risk_feasible = float(stats["metrics/feasible_joint"]) == 1.0
        utility_eligible = run15_utility_eligible(
            stats["metrics/binary_lesion_iou"], stats["metrics/group_macro_f1"]
        )
        stats["metrics/risk_feasible"] = float(risk_feasible)
        stats["metrics/utility_eligible"] = float(utility_eligible)
        stats["metrics/feasible_joint"] = float(risk_feasible and utility_eligible)
        return stats


class MELNVPairwiseBoundaryPartialLabelMultiTaskLoss(nn.Module):
    """Run12 unweighted CE plus the single direct MEL-vs-NV pairwise factor."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.semantic_loss = SemanticSegmentationLoss(model)
        self.image_cls_weight = float(getattr(model, "image_cls_weight", 1.0))
        self.mel_nv_pairwise_margin = float(getattr(model, "mel_nv_pairwise_margin", float("nan")))
        self.mel_nv_pairwise_loss_weight = float(
            getattr(model, "mel_nv_pairwise_loss_weight", float("nan"))
        )
        if self.image_cls_weight <= 0:
            raise ContractError(f"image_cls_weight must be positive, found {self.image_cls_weight}")
        if self.mel_nv_pairwise_margin != RUN15_MEL_NV_PAIRWISE_MARGIN:
            raise ContractError("Run15 model pairwise margin differs from the frozen train-prior value")
        if self.mel_nv_pairwise_loss_weight != RUN15_MEL_NV_PAIRWISE_LOSS_WEIGHT:
            raise ContractError("Run15 pairwise loss weight must remain the frozen unit coefficient")

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
            labeled_logits = image_logits[has_cls]
            labeled_targets = targets[has_cls]
            image_ce = F.cross_entropy(labeled_logits, labeled_targets)
            pairwise = run15_mel_nv_pairwise_softplus(
                labeled_logits, labeled_targets, self.mel_nv_pairwise_margin
            )
        else:
            image_ce = image_logits.sum() * 0.0
            pairwise = image_logits.sum() * 0.0
        image_objective = image_ce + self.mel_nv_pairwise_loss_weight * pairwise
        total = semantic_total + self.image_cls_weight * image_objective * batch_size
        items = torch.cat([semantic_items, image_ce.detach().reshape(1), pairwise.detach().reshape(1)])
        return total, items


class MELNVPairwiseBoundaryBinaryLesionGuidedAttentionModel(BinaryLesionGuidedAttentionModel):
    """Run12 architecture with a criterion-only direct pairwise boundary loss."""

    def __init__(
        self,
        cfg="yolo26s-sem.yaml",
        ch: int = 3,
        nc: int | None = 1,
        verbose: bool = True,
        num_diseases: int = 8,
        image_cls_weight: float = 0.5,
        mel_nv_pairwise_margin: float = RUN15_MEL_NV_PAIRWISE_MARGIN,
        mel_nv_pairwise_loss_weight: float = RUN15_MEL_NV_PAIRWISE_LOSS_WEIGHT,
        p5_layer: int = 10,
        eca_kernel_size: int = 5,
        mask_guided_mix: float = 0.5,
        detach_mask: bool = True,
    ) -> None:
        if float(mel_nv_pairwise_margin) != RUN15_MEL_NV_PAIRWISE_MARGIN:
            raise ContractError("Run15 model pairwise margin differs from frozen train-prior value")
        if float(mel_nv_pairwise_loss_weight) != RUN15_MEL_NV_PAIRWISE_LOSS_WEIGHT:
            raise ContractError("Run15 model pairwise weight differs from frozen unit coefficient")
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
        self.mel_nv_pairwise_margin = float(mel_nv_pairwise_margin)
        self.mel_nv_pairwise_loss_weight = float(mel_nv_pairwise_loss_weight)

    def init_criterion(self):
        return MELNVPairwiseBoundaryPartialLabelMultiTaskLoss(self)


class MELNVPairwiseBoundaryBinaryLesionGuidedAttentionTrainer(BinaryLesionGuidedAttentionTrainer):
    """Run15 Trainer with direct pairwise loss and utility-eligible selection."""

    def __init__(
        self,
        *args,
        mel_nv_pairwise_margin: float = RUN15_MEL_NV_PAIRWISE_MARGIN,
        mel_nv_pairwise_loss_weight: float = RUN15_MEL_NV_PAIRWISE_LOSS_WEIGHT,
        **kwargs,
    ) -> None:
        if float(mel_nv_pairwise_margin) != RUN15_MEL_NV_PAIRWISE_MARGIN:
            raise ContractError("Run15 Trainer pairwise margin drifted")
        if float(mel_nv_pairwise_loss_weight) != RUN15_MEL_NV_PAIRWISE_LOSS_WEIGHT:
            raise ContractError("Run15 Trainer pairwise weight drifted")
        self.mel_nv_pairwise_margin = float(mel_nv_pairwise_margin)
        self.mel_nv_pairwise_loss_weight = float(mel_nv_pairwise_loss_weight)
        super().__init__(*args, **kwargs)

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        model = MELNVPairwiseBoundaryBinaryLesionGuidedAttentionModel(
            cfg,
            nc=1,
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            num_diseases=len(CLASS_NAMES),
            image_cls_weight=self.image_cls_weight,
            mel_nv_pairwise_margin=self.mel_nv_pairwise_margin,
            mel_nv_pairwise_loss_weight=self.mel_nv_pairwise_loss_weight,
            **self.lesion_attention,
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        self.loss_names = "bce_loss", "dice_loss", "aux_loss", "image_cls_loss", "mel_nv_pairwise_loss"
        return UtilityEligiblePartialLabelMultiTaskValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
            selection=self.selection,
        )


__all__ = [
    "BinaryLesionGuidedAttentionHead",
    "IdentityInitializedECA",
    "MELNVPairwiseBoundaryBinaryLesionGuidedAttentionModel",
    "MELNVPairwiseBoundaryBinaryLesionGuidedAttentionTrainer",
    "MELNVPairwiseBoundaryPartialLabelMultiTaskLoss",
    "PartialLabelSemanticDataset",
    "RUN15_MEL_NV_PAIRWISE_LOSS_WEIGHT",
    "RUN15_MEL_NV_PAIRWISE_MARGIN",
    "RUN15_MIN_BINARY_IOU",
    "RUN15_MIN_GROUP_MACRO_F1",
    "UtilityEligiblePartialLabelMultiTaskValidator",
    "run15_mel_nv_pairwise_softplus",
    "run15_utility_eligible",
]
