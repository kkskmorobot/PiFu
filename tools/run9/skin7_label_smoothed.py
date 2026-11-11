"""Run9: Run8 with label smoothing only on the image-classification CE.

The Run7 architecture, semantic loss, Run8 sampler, class quotas, optimizer,
augmentation and checkpoint-selection policy are inherited unchanged.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ultralytics.utils import LOGGER, RANK

from skin7_lesion_balanced import Skin7LesionBalancedTrainer
from skin7_lesion_guided import Skin7LesionGuidedMultiTaskModel
from skin7_multitask import DISEASE_NAMES, Skin7MultiTaskLoss, image_targets_from_batch


class Skin7LabelSmoothedMultiTaskLoss(Skin7MultiTaskLoss):
    """Unchanged semantic objective plus label-smoothed image CE."""

    def __init__(self, model):
        super().__init__(model)
        self.image_cls_label_smoothing = float(getattr(model, "image_cls_label_smoothing", 0.05))
        if not 0.0 < self.image_cls_label_smoothing < 1.0:
            raise ValueError(
                "Run9 image_cls_label_smoothing must be strictly between 0 and 1, "
                f"found {self.image_cls_label_smoothing}"
            )
        LOGGER.info(
            "Run9 criterion: semantic loss unchanged; "
            f"image classification CE label_smoothing={self.image_cls_label_smoothing:.2f}."
        )

    def forward(self, preds, batch):
        if len(preds) == 3:
            semantic_logits, auxiliary_logits, image_logits = preds
        elif len(preds) == 2:
            semantic_logits, image_logits = preds
            auxiliary_logits = None
        else:
            raise ValueError(f"Unexpected Run9 model output length: {len(preds)}")
        semantic_preds = (
            (semantic_logits, auxiliary_logits) if auxiliary_logits is not None else semantic_logits
        )
        semantic_total, semantic_items = self.semantic_loss(semantic_preds, batch)
        targets = image_targets_from_batch(batch, image_logits.shape[0], image_logits.device)
        image_ce = F.cross_entropy(
            image_logits,
            targets,
            label_smoothing=self.image_cls_label_smoothing,
        )
        total = semantic_total + self.image_cls_weight * image_ce * image_logits.shape[0]
        items = torch.cat([semantic_items, image_ce.detach().reshape(1)])
        return total, items


class Skin7LabelSmoothedMultiTaskModel(Skin7LesionGuidedMultiTaskModel):
    """Fixed Run7 architecture whose criterion carries the Run9 CE setting."""

    def __init__(self, *args, image_cls_label_smoothing: float = 0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_cls_label_smoothing = float(image_cls_label_smoothing)
        if not 0.0 < self.image_cls_label_smoothing < 1.0:
            raise ValueError(
                "Run9 image_cls_label_smoothing must be strictly between 0 and 1, "
                f"found {self.image_cls_label_smoothing}"
            )

    def init_criterion(self):
        return Skin7LabelSmoothedMultiTaskLoss(self)


class Skin7LabelSmoothedTrainer(Skin7LesionBalancedTrainer):
    """Run8 trainer with only image-classification label smoothing added."""

    def __init__(self, *args, classification_regularization: dict | None = None, **kwargs):
        self.classification_regularization = dict(classification_regularization or {})
        if self.classification_regularization.get("strategy") != "label_smoothing":
            raise ValueError("Run9 requires classification_regularization.strategy=label_smoothing")
        if self.classification_regularization.get("scope") != "image_classification_ce_only":
            raise ValueError("Run9 label smoothing must be scoped to image_classification_ce_only")
        self.image_cls_label_smoothing = float(self.classification_regularization.get("epsilon", 0.05))
        if self.image_cls_label_smoothing != 0.05:
            raise ValueError("Run9 is preregistered for image classification label smoothing epsilon=0.05")
        super().__init__(*args, **kwargs)

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        model = Skin7LabelSmoothedMultiTaskModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            num_diseases=len(DISEASE_NAMES),
            image_cls_weight=self.image_cls_weight,
            image_cls_label_smoothing=self.image_cls_label_smoothing,
            **self.lesion_guided,
        )
        if weights:
            model.load(weights)
        return model
