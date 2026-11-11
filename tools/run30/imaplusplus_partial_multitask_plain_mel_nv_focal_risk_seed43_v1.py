"""Run30 plain P3/P4 model with a training-only MEL-to-NV focal risk loss.

Relative to frozen Run22, the model architecture, initialization, primary
eight-class cross-entropy, data, sampling, augmentation, validation metrics,
inference and delayed early stopping are unchanged.  The only training
behavior change is a half-strength posterior focal penalty applied only to
true-MEL rows and only to their NV probability.  It emphasizes high-risk
MEL-to-NV candidates while leaving easy MEL rows and every non-MEL target
unmodified by the auxiliary term.  No parameters or inference rules are added.
"""

from __future__ import annotations

from copy import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils import RANK
from ultralytics.utils.loss import SemanticSegmentationLoss
from ultralytics.utils.torch_utils import EarlyStopping

from imaplusplus_partial_multitask_v1 import (
    CLASS_NAMES,
    BinaryPartialMultiTaskModel,
    ContractError,
    MEL_ID,
    NV_ID,
    unpack_outputs,
)
from imaplusplus_partial_multitask_utility_selection_v1 import (
    UtilityEligiblePartialLabelMultiTaskTrainer,
    UtilityEligiblePartialLabelMultiTaskValidator,
)


RUN30_MEL_NV_FOCAL_RISK_WEIGHT = 0.5
RUN30_MEL_NV_FOCAL_GAMMA = 2.0
RUN30_EARLY_STOP_START_EPOCH = 40
RUN30_PATIENCE_AFTER_START = 30


def run30_early_stop_fitness(metrics: dict[str, float], original_fitness: float) -> float:
    """Preserve Run22's utility-gated ranking without changing validation metrics."""
    if float(metrics.get("metrics/utility_eligible", 0.0)) == 1.0:
        return float(original_fitness)
    return float(metrics["metrics/joint_score"]) - 2.0


class DelayedEarlyStopping(EarlyStopping):
    """Ultralytics-compatible stopper whose patience clock starts at epoch 40."""

    def __init__(self, patience: int, start_epoch: int = RUN30_EARLY_STOP_START_EPOCH) -> None:
        super().__init__(patience=patience)
        self.start_epoch = int(start_epoch)
        self.activated = False

    def __call__(self, epoch, fitness):
        epoch = int(epoch)
        if epoch < self.start_epoch:
            self.possible_stop = False
            return False
        if not self.activated:
            self.best_fitness = 0.0
            self.best_epoch = epoch
            self.possible_stop = False
            self.activated = True
        return super().__call__(epoch, fitness)


class DelayedUtilityGatedPartialLabelMultiTaskTrainer(UtilityEligiblePartialLabelMultiTaskTrainer):
    """Frozen Run22 fitness and delayed early-stop lifecycle."""

    def validate(self):
        prior_gated_best = self.best_fitness
        metrics, original_fitness = super().validate()
        if metrics is None or original_fitness is None:
            return metrics, original_fitness
        gated_fitness = run30_early_stop_fitness(metrics, original_fitness)
        metrics["metrics/original_fitness"] = float(original_fitness)
        metrics["metrics/early_stop_fitness"] = float(gated_fitness)
        if prior_gated_best is None or gated_fitness > prior_gated_best:
            self.best_fitness = float(gated_fitness)
        else:
            self.best_fitness = float(prior_gated_best)
        return metrics, float(gated_fitness)

    def resume_training(self, ckpt):
        super().resume_training(ckpt)
        saved = dict(vars(self.stopper))
        self.stopper = DelayedEarlyStopping(
            patience=RUN30_PATIENCE_AFTER_START,
            start_epoch=RUN30_EARLY_STOP_START_EPOCH,
        )
        for name, value in saved.items():
            if name in {"best_fitness", "best_epoch", "possible_stop"}:
                setattr(self.stopper, name, value)
        if int(getattr(self, "start_epoch", 0)) >= RUN30_EARLY_STOP_START_EPOCH:
            self.stopper.activated = True


def run30_mel_nv_focal_risk(
    image_logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = RUN30_MEL_NV_FOCAL_GAMMA,
) -> torch.Tensor:
    """Mean focal -log(1-p_NV) over true-MEL rows only."""
    if image_logits.ndim != 2 or image_logits.shape[1] != len(CLASS_NAMES):
        raise ContractError(f"Expected classification logits [N, 8], found {tuple(image_logits.shape)}")
    if targets.ndim != 1 or targets.shape[0] != image_logits.shape[0] or not targets.numel():
        raise ContractError("Classification targets must be a non-empty length-N vector")
    if bool(((targets < 0) | (targets >= len(CLASS_NAMES))).any().item()):
        raise ContractError("Run30 labeled classification targets must be in [0, 7]")
    if float(gamma) != RUN30_MEL_NV_FOCAL_GAMMA:
        raise ContractError("Run30 MEL-to-NV focal gamma drifted from 2.0")
    mel_rows = targets == MEL_ID
    if not bool(mel_rows.any().item()):
        return image_logits.sum() * 0.0
    probabilities = F.softmax(image_logits[mel_rows].float(), dim=1)
    nv_probability = probabilities[:, NV_ID].clamp(max=1.0 - 1e-6)
    return (nv_probability.pow(float(gamma)) * (-torch.log1p(-nv_probability))).mean()


class MELNVFocalRiskPartialLabelMultiTaskLoss(nn.Module):
    """Frozen plain multitask loss plus Run30's single focal risk factor."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.semantic_loss = SemanticSegmentationLoss(model)
        self.image_cls_weight = float(getattr(model, "image_cls_weight", 1.0))
        self.mel_nv_focal_risk_weight = float(
            getattr(model, "mel_nv_focal_risk_weight", float("nan"))
        )
        if self.image_cls_weight <= 0:
            raise ContractError(f"image_cls_weight must be positive, found {self.image_cls_weight}")
        if self.mel_nv_focal_risk_weight != RUN30_MEL_NV_FOCAL_RISK_WEIGHT:
            raise ContractError("Run30 MEL-to-NV focal risk weight drifted from 0.5")

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
            risk_loss = run30_mel_nv_focal_risk(labeled_logits, labeled_targets)
        else:
            image_ce = image_logits.sum() * 0.0
            risk_loss = image_logits.sum() * 0.0
        image_objective = image_ce + self.mel_nv_focal_risk_weight * risk_loss
        total = semantic_total + self.image_cls_weight * image_objective * batch_size
        items = torch.cat([semantic_items, image_ce.detach().reshape(1), risk_loss.detach().reshape(1)])
        return total, items


class MELNVFocalRiskBinaryPartialMultiTaskModel(BinaryPartialMultiTaskModel):
    """Run22's unchanged plain model with a criterion-only Run30 attribute."""

    def __init__(
        self,
        cfg="yolo26s-sem.yaml",
        ch: int = 3,
        nc: int | None = 1,
        verbose: bool = True,
        num_diseases: int = 8,
        image_cls_weight: float = 0.5,
        mel_nv_focal_risk_weight: float = RUN30_MEL_NV_FOCAL_RISK_WEIGHT,
    ) -> None:
        if float(mel_nv_focal_risk_weight) != RUN30_MEL_NV_FOCAL_RISK_WEIGHT:
            raise ContractError("Run30 model MEL-to-NV focal risk weight must remain 0.5")
        super().__init__(
            cfg=cfg,
            ch=ch,
            nc=nc,
            verbose=verbose,
            num_diseases=num_diseases,
            image_cls_weight=image_cls_weight,
        )
        self.mel_nv_focal_risk_weight = float(mel_nv_focal_risk_weight)

    def init_criterion(self):
        return MELNVFocalRiskPartialLabelMultiTaskLoss(self)


class MELNVFocalRiskDelayedUtilityGatedTrainer(DelayedUtilityGatedPartialLabelMultiTaskTrainer):
    """Run22 trainer with only the preregistered Run30 focal risk loss added."""

    def __init__(
        self,
        *args,
        mel_nv_focal_risk_weight: float = RUN30_MEL_NV_FOCAL_RISK_WEIGHT,
        **kwargs,
    ) -> None:
        if float(mel_nv_focal_risk_weight) != RUN30_MEL_NV_FOCAL_RISK_WEIGHT:
            raise ContractError("Run30 Trainer MEL-to-NV focal risk weight drifted")
        self.mel_nv_focal_risk_weight = float(mel_nv_focal_risk_weight)
        super().__init__(*args, **kwargs)

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        model = MELNVFocalRiskBinaryPartialMultiTaskModel(
            cfg,
            nc=1,
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            num_diseases=len(CLASS_NAMES),
            image_cls_weight=self.image_cls_weight,
            mel_nv_focal_risk_weight=self.mel_nv_focal_risk_weight,
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        self.loss_names = (
            "bce_loss",
            "dice_loss",
            "aux_loss",
            "image_cls_loss",
            "mel_nv_focal_risk_loss",
        )
        return UtilityEligiblePartialLabelMultiTaskValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
            selection=self.selection,
        )


__all__ = [
    "DelayedEarlyStopping",
    "MELNVFocalRiskBinaryPartialMultiTaskModel",
    "MELNVFocalRiskDelayedUtilityGatedTrainer",
    "MELNVFocalRiskPartialLabelMultiTaskLoss",
    "RUN30_EARLY_STOP_START_EPOCH",
    "RUN30_PATIENCE_AFTER_START",
    "RUN30_MEL_NV_FOCAL_GAMMA",
    "RUN30_MEL_NV_FOCAL_RISK_WEIGHT",
    "run30_early_stop_fitness",
    "run30_mel_nv_focal_risk",
]
