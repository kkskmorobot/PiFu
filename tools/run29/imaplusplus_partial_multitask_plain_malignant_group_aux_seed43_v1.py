"""Run29 plain P3/P4 model with a training-only malignant-group auxiliary NLL.

Relative to frozen Run22, the model architecture, initialization, primary
eight-class cross-entropy, data, sampling, augmentation, validation metrics,
inference and delayed early stopping are unchanged.  The only training
behavior change is a half-strength hierarchical negative log-likelihood that
aggregates the existing eight logits into benign versus malignant groups.
No parameters or inference-time decision rules are added.
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
    unpack_outputs,
)
from imaplusplus_partial_multitask_utility_selection_v1 import (
    UtilityEligiblePartialLabelMultiTaskTrainer,
    UtilityEligiblePartialLabelMultiTaskValidator,
)


RUN29_MALIGNANT_CLASS_IDS = (1, 4, 7)  # bcc, mel, scc
RUN29_BENIGN_CLASS_IDS = (0, 2, 3, 5, 6)  # akiec, bkl, df, nv, vasc
RUN29_MALIGNANT_GROUP_AUX_WEIGHT = 0.5
RUN29_EARLY_STOP_START_EPOCH = 40
RUN29_PATIENCE_AFTER_START = 30


def run29_early_stop_fitness(metrics: dict[str, float], original_fitness: float) -> float:
    """Preserve Run22's utility-gated ranking without changing validation metrics."""
    if float(metrics.get("metrics/utility_eligible", 0.0)) == 1.0:
        return float(original_fitness)
    return float(metrics["metrics/joint_score"]) - 2.0


class DelayedEarlyStopping(EarlyStopping):
    """Ultralytics-compatible stopper whose patience clock starts at epoch 40."""

    def __init__(self, patience: int, start_epoch: int = RUN29_EARLY_STOP_START_EPOCH) -> None:
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
        gated_fitness = run29_early_stop_fitness(metrics, original_fitness)
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
            patience=RUN29_PATIENCE_AFTER_START,
            start_epoch=RUN29_EARLY_STOP_START_EPOCH,
        )
        for name, value in saved.items():
            if name in {"best_fitness", "best_epoch", "possible_stop"}:
                setattr(self.stopper, name, value)
        if int(getattr(self, "start_epoch", 0)) >= RUN29_EARLY_STOP_START_EPOCH:
            self.stopper.activated = True


def run29_malignant_group_nll(image_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy after exact logsumexp aggregation into benign/malignant."""
    if image_logits.ndim != 2 or image_logits.shape[1] != len(CLASS_NAMES):
        raise ContractError(f"Expected classification logits [N, 8], found {tuple(image_logits.shape)}")
    if targets.ndim != 1 or targets.shape[0] != image_logits.shape[0] or not targets.numel():
        raise ContractError("Classification targets must be a non-empty length-N vector")
    if bool(((targets < 0) | (targets >= len(CLASS_NAMES))).any().item()):
        raise ContractError("Run29 labeled classification targets must be in [0, 7]")

    benign = torch.logsumexp(image_logits[:, RUN29_BENIGN_CLASS_IDS], dim=1)
    malignant = torch.logsumexp(image_logits[:, RUN29_MALIGNANT_CLASS_IDS], dim=1)
    group_logits = torch.stack((benign, malignant), dim=1)
    malignant_target = torch.zeros_like(targets)
    for class_id in RUN29_MALIGNANT_CLASS_IDS:
        malignant_target = torch.where(targets == class_id, 1, malignant_target)
    return F.cross_entropy(group_logits, malignant_target)


class MalignantGroupAuxPartialLabelMultiTaskLoss(nn.Module):
    """Frozen plain multitask loss plus Run29's single hierarchy factor."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.semantic_loss = SemanticSegmentationLoss(model)
        self.image_cls_weight = float(getattr(model, "image_cls_weight", 1.0))
        self.malignant_group_aux_weight = float(
            getattr(model, "malignant_group_aux_weight", float("nan"))
        )
        if self.image_cls_weight <= 0:
            raise ContractError(f"image_cls_weight must be positive, found {self.image_cls_weight}")
        if self.malignant_group_aux_weight != RUN29_MALIGNANT_GROUP_AUX_WEIGHT:
            raise ContractError("Run29 malignant-group auxiliary weight drifted from 0.5")

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
            group_nll = run29_malignant_group_nll(labeled_logits, labeled_targets)
        else:
            image_ce = image_logits.sum() * 0.0
            group_nll = image_logits.sum() * 0.0
        image_objective = image_ce + self.malignant_group_aux_weight * group_nll
        total = semantic_total + self.image_cls_weight * image_objective * batch_size
        items = torch.cat([semantic_items, image_ce.detach().reshape(1), group_nll.detach().reshape(1)])
        return total, items


class MalignantGroupAuxBinaryPartialMultiTaskModel(BinaryPartialMultiTaskModel):
    """Run22's unchanged plain model with a criterion-only Run29 attribute."""

    def __init__(
        self,
        cfg="yolo26s-sem.yaml",
        ch: int = 3,
        nc: int | None = 1,
        verbose: bool = True,
        num_diseases: int = 8,
        image_cls_weight: float = 0.5,
        malignant_group_aux_weight: float = RUN29_MALIGNANT_GROUP_AUX_WEIGHT,
    ) -> None:
        if float(malignant_group_aux_weight) != RUN29_MALIGNANT_GROUP_AUX_WEIGHT:
            raise ContractError("Run29 model malignant-group auxiliary weight must remain 0.5")
        super().__init__(
            cfg=cfg,
            ch=ch,
            nc=nc,
            verbose=verbose,
            num_diseases=num_diseases,
            image_cls_weight=image_cls_weight,
        )
        self.malignant_group_aux_weight = float(malignant_group_aux_weight)

    def init_criterion(self):
        return MalignantGroupAuxPartialLabelMultiTaskLoss(self)


class MalignantGroupAuxDelayedUtilityGatedTrainer(DelayedUtilityGatedPartialLabelMultiTaskTrainer):
    """Run22 trainer with only the preregistered Run29 hierarchy loss added."""

    def __init__(
        self,
        *args,
        malignant_group_aux_weight: float = RUN29_MALIGNANT_GROUP_AUX_WEIGHT,
        **kwargs,
    ) -> None:
        if float(malignant_group_aux_weight) != RUN29_MALIGNANT_GROUP_AUX_WEIGHT:
            raise ContractError("Run29 Trainer malignant-group auxiliary weight drifted")
        self.malignant_group_aux_weight = float(malignant_group_aux_weight)
        super().__init__(*args, **kwargs)

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        model = MalignantGroupAuxBinaryPartialMultiTaskModel(
            cfg,
            nc=1,
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            num_diseases=len(CLASS_NAMES),
            image_cls_weight=self.image_cls_weight,
            malignant_group_aux_weight=self.malignant_group_aux_weight,
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
            "malignant_group_aux_loss",
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
    "MalignantGroupAuxBinaryPartialMultiTaskModel",
    "MalignantGroupAuxDelayedUtilityGatedTrainer",
    "MalignantGroupAuxPartialLabelMultiTaskLoss",
    "RUN29_EARLY_STOP_START_EPOCH",
    "RUN29_PATIENCE_AFTER_START",
    "RUN29_BENIGN_CLASS_IDS",
    "RUN29_MALIGNANT_CLASS_IDS",
    "RUN29_MALIGNANT_GROUP_AUX_WEIGHT",
    "run29_early_stop_fitness",
    "run29_malignant_group_nll",
]
