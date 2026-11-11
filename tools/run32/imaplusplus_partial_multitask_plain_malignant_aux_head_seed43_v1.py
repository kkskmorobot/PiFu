"""Run32 plain P3/P4 model with a training-only malignant-vs-benign auxiliary head.

Relative to frozen Run22, the semantic branch, main eight-class image head,
data, sampling, augmentation, validation metrics, inference and delayed early
stopping are unchanged. The only experimental factor is a two-logit linear
head attached to the YOLO26m main classifier's existing 256-dimensional hidden
descriptor. Its cross-entropy is evaluated on every officially labelled
classification row and is used only during training.
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
    MALIGNANT_IDS,
)
from imaplusplus_partial_multitask_utility_selection_v1 import (
    UtilityEligiblePartialLabelMultiTaskTrainer,
    UtilityEligiblePartialLabelMultiTaskValidator,
)


RUN32_MALIGNANT_AUX_WEIGHT = 0.25
RUN32_MALIGNANT_AUX_CLASSES = ("benign", "malignant")
RUN32_MALIGNANT_AUX_PARAMETERS = 514
RUN32_EARLY_STOP_START_EPOCH = 40
RUN32_PATIENCE_AFTER_START = 30


def run32_early_stop_fitness(metrics: dict[str, float], original_fitness: float) -> float:
    """Preserve Run22's utility-gated ranking without changing validation metrics."""
    if float(metrics.get("metrics/utility_eligible", 0.0)) == 1.0:
        return float(original_fitness)
    return float(metrics["metrics/joint_score"]) - 2.0


class DelayedEarlyStopping(EarlyStopping):
    """Ultralytics-compatible stopper whose patience clock starts at epoch 40."""

    def __init__(self, patience: int, start_epoch: int = RUN32_EARLY_STOP_START_EPOCH) -> None:
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
        gated_fitness = run32_early_stop_fitness(metrics, original_fitness)
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
            patience=RUN32_PATIENCE_AFTER_START,
            start_epoch=RUN32_EARLY_STOP_START_EPOCH,
        )
        for name, value in saved.items():
            if name in {"best_fitness", "best_epoch", "possible_stop"}:
                setattr(self.stopper, name, value)
        if int(getattr(self, "start_epoch", 0)) >= RUN32_EARLY_STOP_START_EPOCH:
            self.stopper.activated = True


class MalignantAuxClassificationHead(BinaryPartialClassificationHead):
    """Frozen main P3/P4 head plus a training-only two-class linear auxiliary."""

    def __init__(self, nc: int, ch: tuple[int, int], num_diseases: int = 8, hidden_channels: int = 128):
        super().__init__(nc=nc, ch=ch, num_diseases=num_diseases, hidden_channels=hidden_channels)
        self.malignant_aux_classifier = nn.Linear(hidden_channels, 2)

    def forward(self, x: list[torch.Tensor]):
        pooled = torch.cat([F.adaptive_avg_pool2d(feature, 1).flatten(1) for feature in x], dim=1)
        hidden = self.image_classifier[0](pooled)
        hidden = self.image_classifier[1](hidden)
        hidden = self.image_classifier[2](hidden)
        image_logits = self.image_classifier[3](hidden)
        semantic = SemanticSegment.forward(self, x)
        if self.training:
            malignant_aux_logits = self.malignant_aux_classifier(hidden)
            if isinstance(semantic, tuple):
                return semantic[0], semantic[1], image_logits, malignant_aux_logits
            return semantic, None, image_logits, malignant_aux_logits
        return semantic, image_logits


def unpack_run32_outputs(preds):
    """Accept Run32's four-output training path and unchanged two-output eval path."""
    if not isinstance(preds, (tuple, list)):
        raise ContractError(f"Expected multitask output tuple, found {type(preds).__name__}")
    if len(preds) == 4:
        semantic_logits, semantic_auxiliary_logits, image_logits, malignant_aux_logits = preds
    elif len(preds) == 2:
        semantic_logits, image_logits = preds
        semantic_auxiliary_logits = None
        malignant_aux_logits = None
    else:
        raise ContractError(f"Unexpected Run32 multitask output length: {len(preds)}")
    return semantic_logits, semantic_auxiliary_logits, image_logits, malignant_aux_logits


def run32_malignant_targets(targets: torch.Tensor) -> torch.Tensor:
    """Map all official labels to locked benign=0, malignant=1 targets."""
    if targets.ndim != 1:
        raise ContractError("Run32 auxiliary targets must be a one-dimensional vector")
    if bool(((targets < 0) | (targets >= len(CLASS_NAMES))).any().item()):
        raise ContractError("Run32 labelled classification targets must be in [0, 7]")
    malignant = torch.zeros_like(targets, dtype=torch.bool)
    for class_id in MALIGNANT_IDS:
        malignant |= targets == int(class_id)
    return malignant.long()


def run32_malignant_auxiliary_loss(
    malignant_aux_logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Mean benign/malignant CE over every officially labelled row."""
    if malignant_aux_logits.ndim != 2 or malignant_aux_logits.shape[1] != 2:
        raise ContractError(
            f"Expected malignant/benign auxiliary logits [N, 2], found {tuple(malignant_aux_logits.shape)}"
        )
    if targets.ndim != 1 or targets.shape[0] != malignant_aux_logits.shape[0] or not targets.numel():
        raise ContractError("Run32 auxiliary targets must be a non-empty length-N vector")
    return F.cross_entropy(malignant_aux_logits, run32_malignant_targets(targets))


class MalignantAuxHeadPartialLabelMultiTaskLoss(nn.Module):
    """Frozen plain multitask loss plus Run32's single auxiliary-task factor."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.semantic_loss = SemanticSegmentationLoss(model)
        self.image_cls_weight = float(getattr(model, "image_cls_weight", 1.0))
        self.malignant_aux_weight = float(getattr(model, "malignant_aux_weight", float("nan")))
        if self.image_cls_weight <= 0:
            raise ContractError(f"image_cls_weight must be positive, found {self.image_cls_weight}")
        if self.malignant_aux_weight != RUN32_MALIGNANT_AUX_WEIGHT:
            raise ContractError("Run32 malignant/benign auxiliary weight drifted from 0.25")

    def forward(self, preds, batch: dict[str, Any]):
        semantic_logits, semantic_auxiliary_logits, image_logits, malignant_aux_logits = unpack_run32_outputs(preds)
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
            if malignant_aux_logits is None:
                malignant_aux_ce = image_logits.sum() * 0.0
            else:
                malignant_aux_ce = run32_malignant_auxiliary_loss(
                    malignant_aux_logits[has_cls], labeled_targets
                )
        else:
            image_ce = image_logits.sum() * 0.0
            malignant_aux_ce = image_logits.sum() * 0.0
        image_objective = image_ce + self.malignant_aux_weight * malignant_aux_ce
        total = semantic_total + self.image_cls_weight * image_objective * batch_size
        items = torch.cat(
            [semantic_items, image_ce.detach().reshape(1), malignant_aux_ce.detach().reshape(1)]
        )
        return total, items


class MalignantAuxHeadBinaryPartialMultiTaskModel(BinaryPartialMultiTaskModel):
    """Run22 model with one training-only malignant/benign auxiliary classifier."""

    def __init__(
        self,
        cfg="yolo26s-sem.yaml",
        ch: int = 3,
        nc: int | None = 1,
        verbose: bool = True,
        num_diseases: int = 8,
        image_cls_weight: float = 0.5,
        malignant_aux_weight: float = RUN32_MALIGNANT_AUX_WEIGHT,
    ) -> None:
        if float(malignant_aux_weight) != RUN32_MALIGNANT_AUX_WEIGHT:
            raise ContractError("Run32 model malignant/benign auxiliary weight must remain 0.25")
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
        new_head = MalignantAuxClassificationHead(
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
        self.malignant_aux_weight = float(malignant_aux_weight)
        if sum(parameter.numel() for parameter in new_head.malignant_aux_classifier.parameters()) != RUN32_MALIGNANT_AUX_PARAMETERS:
            raise ContractError("Run32 YOLO26m auxiliary classifier must contain exactly 514 parameters")
        if verbose:
            self.info()

    def init_criterion(self):
        return MalignantAuxHeadPartialLabelMultiTaskLoss(self)


class MalignantAuxHeadDelayedUtilityGatedTrainer(DelayedUtilityGatedPartialLabelMultiTaskTrainer):
    """Run22 trainer with only the preregistered Run32 auxiliary task added."""

    def __init__(self, *args, malignant_aux_weight: float = RUN32_MALIGNANT_AUX_WEIGHT, **kwargs) -> None:
        if float(malignant_aux_weight) != RUN32_MALIGNANT_AUX_WEIGHT:
            raise ContractError("Run32 Trainer malignant/benign auxiliary weight drifted")
        self.malignant_aux_weight = float(malignant_aux_weight)
        super().__init__(*args, **kwargs)

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        model = MalignantAuxHeadBinaryPartialMultiTaskModel(
            cfg,
            nc=1,
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            num_diseases=len(CLASS_NAMES),
            image_cls_weight=self.image_cls_weight,
            malignant_aux_weight=self.malignant_aux_weight,
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
            "malignant_aux_loss",
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
    "MalignantAuxClassificationHead",
    "MalignantAuxHeadBinaryPartialMultiTaskModel",
    "MalignantAuxHeadDelayedUtilityGatedTrainer",
    "MalignantAuxHeadPartialLabelMultiTaskLoss",
    "RUN32_EARLY_STOP_START_EPOCH",
    "RUN32_MALIGNANT_AUX_CLASSES",
    "RUN32_MALIGNANT_AUX_PARAMETERS",
    "RUN32_MALIGNANT_AUX_WEIGHT",
    "RUN32_PATIENCE_AFTER_START",
    "run32_early_stop_fitness",
    "run32_malignant_auxiliary_loss",
    "run32_malignant_targets",
    "unpack_run32_outputs",
]
