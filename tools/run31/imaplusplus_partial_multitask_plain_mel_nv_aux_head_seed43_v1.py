"""Run31 plain P3/P4 model with a training-only MEL-vs-NV auxiliary head.

Relative to frozen Run22, the semantic branch, main eight-class image head,
data, sampling, augmentation, validation metrics, inference and delayed early
stopping are unchanged. The only experimental factor is a two-logit linear
head attached to the YOLO26m main classifier's existing 256-dimensional hidden
descriptor. Its cross-entropy is evaluated only on officially labelled MEL
or NV rows and is used only during training.
"""

from __future__ import annotations

from copy import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.head import SemanticSegment
from ultralytics.utils import RANK
from ultralytics.utils.loss import SemanticSegmentationLoss
from ultralytics.utils.torch_utils import EarlyStopping

from imaplusplus_partial_multitask_v1 import (
    CLASS_NAMES,
    BinaryPartialClassificationHead,
    BinaryPartialMultiTaskModel,
    ContractError,
    MEL_ID,
    NV_ID,
)
from imaplusplus_partial_multitask_utility_selection_v1 import (
    UtilityEligiblePartialLabelMultiTaskTrainer,
    UtilityEligiblePartialLabelMultiTaskValidator,
)


RUN31_MEL_NV_AUX_WEIGHT = 0.5
RUN31_MEL_NV_AUX_CLASSES = (NV_ID, MEL_ID)
RUN31_MEL_NV_AUX_PARAMETERS = 514
RUN31_EARLY_STOP_START_EPOCH = 40
RUN31_PATIENCE_AFTER_START = 30


def run31_early_stop_fitness(metrics: dict[str, float], original_fitness: float) -> float:
    """Preserve Run22's utility-gated ranking without changing validation metrics."""
    if float(metrics.get("metrics/utility_eligible", 0.0)) == 1.0:
        return float(original_fitness)
    return float(metrics["metrics/joint_score"]) - 2.0


class DelayedEarlyStopping(EarlyStopping):
    """Ultralytics-compatible stopper whose patience clock starts at epoch 40."""

    def __init__(self, patience: int, start_epoch: int = RUN31_EARLY_STOP_START_EPOCH) -> None:
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
        gated_fitness = run31_early_stop_fitness(metrics, original_fitness)
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
            patience=RUN31_PATIENCE_AFTER_START,
            start_epoch=RUN31_EARLY_STOP_START_EPOCH,
        )
        for name, value in saved.items():
            if name in {"best_fitness", "best_epoch", "possible_stop"}:
                setattr(self.stopper, name, value)
        if int(getattr(self, "start_epoch", 0)) >= RUN31_EARLY_STOP_START_EPOCH:
            self.stopper.activated = True


class MELNVAuxClassificationHead(BinaryPartialClassificationHead):
    """Frozen main P3/P4 head plus a training-only two-class linear auxiliary."""

    def __init__(self, nc: int, ch: tuple[int, int], num_diseases: int = 8, hidden_channels: int = 128):
        super().__init__(nc=nc, ch=ch, num_diseases=num_diseases, hidden_channels=hidden_channels)
        self.mel_nv_aux_classifier = nn.Linear(hidden_channels, 2)

    def forward(self, x: list[torch.Tensor]):
        pooled = torch.cat([F.adaptive_avg_pool2d(feature, 1).flatten(1) for feature in x], dim=1)
        hidden = self.image_classifier[0](pooled)
        hidden = self.image_classifier[1](hidden)
        hidden = self.image_classifier[2](hidden)
        image_logits = self.image_classifier[3](hidden)
        semantic = SemanticSegment.forward(self, x)
        if self.training:
            mel_nv_aux_logits = self.mel_nv_aux_classifier(hidden)
            if isinstance(semantic, tuple):
                return semantic[0], semantic[1], image_logits, mel_nv_aux_logits
            return semantic, None, image_logits, mel_nv_aux_logits
        return semantic, image_logits


def unpack_run31_outputs(preds):
    """Accept Run31's four-output training path and unchanged two-output eval path."""
    if not isinstance(preds, (tuple, list)):
        raise ContractError(f"Expected multitask output tuple, found {type(preds).__name__}")
    if len(preds) == 4:
        semantic_logits, semantic_auxiliary_logits, image_logits, mel_nv_aux_logits = preds
    elif len(preds) == 2:
        semantic_logits, image_logits = preds
        semantic_auxiliary_logits = None
        mel_nv_aux_logits = None
    else:
        raise ContractError(f"Unexpected Run31 multitask output length: {len(preds)}")
    return semantic_logits, semantic_auxiliary_logits, image_logits, mel_nv_aux_logits


def run31_mel_nv_pair_rows_and_targets(targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return official MEL/NV rows and locked NV=0, MEL=1 auxiliary targets."""
    if targets.ndim != 1:
        raise ContractError("Run31 auxiliary targets must be a one-dimensional vector")
    pair_rows = (targets == NV_ID) | (targets == MEL_ID)
    pair_targets = (targets[pair_rows] == MEL_ID).long()
    return pair_rows, pair_targets


def run31_mel_nv_auxiliary_loss(
    mel_nv_aux_logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Mean two-class CE over officially labelled MEL/NV rows only."""
    if mel_nv_aux_logits.ndim != 2 or mel_nv_aux_logits.shape[1] != 2:
        raise ContractError(f"Expected MEL/NV auxiliary logits [N, 2], found {tuple(mel_nv_aux_logits.shape)}")
    if targets.ndim != 1 or targets.shape[0] != mel_nv_aux_logits.shape[0] or not targets.numel():
        raise ContractError("Run31 auxiliary targets must be a non-empty length-N vector")
    if bool(((targets < 0) | (targets >= len(CLASS_NAMES))).any().item()):
        raise ContractError("Run31 labelled classification targets must be in [0, 7]")
    pair_rows, pair_targets = run31_mel_nv_pair_rows_and_targets(targets)
    if not bool(pair_rows.any().item()):
        return mel_nv_aux_logits.sum() * 0.0
    return F.cross_entropy(mel_nv_aux_logits[pair_rows], pair_targets)


class MELNVAuxHeadPartialLabelMultiTaskLoss(nn.Module):
    """Frozen plain multitask loss plus Run31's single auxiliary-task factor."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.semantic_loss = SemanticSegmentationLoss(model)
        self.image_cls_weight = float(getattr(model, "image_cls_weight", 1.0))
        self.mel_nv_aux_weight = float(getattr(model, "mel_nv_aux_weight", float("nan")))
        if self.image_cls_weight <= 0:
            raise ContractError(f"image_cls_weight must be positive, found {self.image_cls_weight}")
        if self.mel_nv_aux_weight != RUN31_MEL_NV_AUX_WEIGHT:
            raise ContractError("Run31 MEL/NV auxiliary weight drifted from 0.5")

    def forward(self, preds, batch: dict[str, Any]):
        semantic_logits, semantic_auxiliary_logits, image_logits, mel_nv_aux_logits = unpack_run31_outputs(preds)
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
            (semantic_logits, semantic_auxiliary_logits)
            if semantic_auxiliary_logits is not None
            else semantic_logits
        )
        semantic_total, semantic_items = self.semantic_loss(semantic_preds, batch)

        if bool(has_cls.any().item()):
            labeled_logits = image_logits[has_cls]
            labeled_targets = targets[has_cls]
            image_ce = F.cross_entropy(labeled_logits, labeled_targets)
            if mel_nv_aux_logits is None:
                mel_nv_aux_ce = image_logits.sum() * 0.0
            else:
                mel_nv_aux_ce = run31_mel_nv_auxiliary_loss(mel_nv_aux_logits[has_cls], labeled_targets)
        else:
            image_ce = image_logits.sum() * 0.0
            mel_nv_aux_ce = image_logits.sum() * 0.0
        image_objective = image_ce + self.mel_nv_aux_weight * mel_nv_aux_ce
        total = semantic_total + self.image_cls_weight * image_objective * batch_size
        items = torch.cat(
            [semantic_items, image_ce.detach().reshape(1), mel_nv_aux_ce.detach().reshape(1)]
        )
        return total, items


class MELNVAuxHeadBinaryPartialMultiTaskModel(BinaryPartialMultiTaskModel):
    """Run22 model with one training-only MEL/NV auxiliary classifier."""

    def __init__(
        self,
        cfg="yolo26s-sem.yaml",
        ch: int = 3,
        nc: int | None = 1,
        verbose: bool = True,
        num_diseases: int = 8,
        image_cls_weight: float = 0.5,
        mel_nv_aux_weight: float = RUN31_MEL_NV_AUX_WEIGHT,
    ) -> None:
        if float(mel_nv_aux_weight) != RUN31_MEL_NV_AUX_WEIGHT:
            raise ContractError("Run31 model MEL/NV auxiliary weight must remain 0.5")
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
            raise ContractError(f"Expected BinaryPartialClassificationHead, found {type(old_head).__name__}")
        p3_channels = old_head.classifier[0].conv.in_channels
        p4_channels = old_head.aux_head[0].conv.in_channels
        hidden_channels = old_head.image_classifier[0].out_features
        new_head = MELNVAuxClassificationHead(
            nc=1,
            ch=(p3_channels, p4_channels),
            num_diseases=num_diseases,
            hidden_channels=hidden_channels,
        )
        new_head.classifier.load_state_dict(old_head.classifier.state_dict())
        if old_head.aux_head is not None:
            new_head.aux_head.load_state_dict(old_head.aux_head.state_dict())
        new_head.image_classifier.load_state_dict(old_head.image_classifier.state_dict())
        for attr in ("f", "i", "type"):
            setattr(new_head, attr, getattr(old_head, attr))
        new_head.np = sum(parameter.numel() for parameter in new_head.parameters())
        new_head.stride = old_head.stride
        self.model[-1] = new_head
        self.mel_nv_aux_weight = float(mel_nv_aux_weight)
        if sum(parameter.numel() for parameter in new_head.mel_nv_aux_classifier.parameters()) != RUN31_MEL_NV_AUX_PARAMETERS:
            raise ContractError("Run31 YOLO26m auxiliary classifier must contain exactly 514 parameters")
        if verbose:
            self.info()

    def init_criterion(self):
        return MELNVAuxHeadPartialLabelMultiTaskLoss(self)


class MELNVAuxHeadDelayedUtilityGatedTrainer(DelayedUtilityGatedPartialLabelMultiTaskTrainer):
    """Run22 trainer with only the preregistered Run31 auxiliary task added."""

    def __init__(self, *args, mel_nv_aux_weight: float = RUN31_MEL_NV_AUX_WEIGHT, **kwargs) -> None:
        if float(mel_nv_aux_weight) != RUN31_MEL_NV_AUX_WEIGHT:
            raise ContractError("Run31 Trainer MEL/NV auxiliary weight drifted")
        self.mel_nv_aux_weight = float(mel_nv_aux_weight)
        super().__init__(*args, **kwargs)

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        model = MELNVAuxHeadBinaryPartialMultiTaskModel(
            cfg,
            nc=1,
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            num_diseases=len(CLASS_NAMES),
            image_cls_weight=self.image_cls_weight,
            mel_nv_aux_weight=self.mel_nv_aux_weight,
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
            "mel_nv_aux_loss",
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
    "MELNVAuxClassificationHead",
    "MELNVAuxHeadBinaryPartialMultiTaskModel",
    "MELNVAuxHeadDelayedUtilityGatedTrainer",
    "MELNVAuxHeadPartialLabelMultiTaskLoss",
    "RUN31_EARLY_STOP_START_EPOCH",
    "RUN31_MEL_NV_AUX_CLASSES",
    "RUN31_MEL_NV_AUX_PARAMETERS",
    "RUN31_MEL_NV_AUX_WEIGHT",
    "RUN31_PATIENCE_AFTER_START",
    "run31_early_stop_fitness",
    "run31_mel_nv_auxiliary_loss",
    "run31_mel_nv_pair_rows_and_targets",
    "unpack_run31_outputs",
]
