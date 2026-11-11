"""Project-local YOLO26 semantic + image-classification extension for runs5.

This module deliberately lives outside ``site-packages``.  Checkpoints pickle
these classes by module name, so keep this file in ``PiFuV2/tools`` and import
it before loading a runs5 checkpoint.
"""

from __future__ import annotations

from copy import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.engine.results import Probs
from ultralytics.models import yolo
from ultralytics.models.yolo.semantic import SemanticSegmentationPredictor, SemanticSegmentationValidator
from ultralytics.models.yolo.semantic.train import SemanticSegmentationTrainer
from ultralytics.nn.modules import SemanticSegment
from ultralytics.nn.tasks import SemanticSegmentationModel
from ultralytics.utils import RANK
from ultralytics.utils.loss import SemanticSegmentationLoss


DISEASE_NAMES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
MINORITY_IDS = (0, 1, 3)
MEL_ID = 4
NV_ID = 5


def image_targets_from_batch(batch: dict[str, Any], batch_size: int, device: torch.device) -> torch.Tensor:
    """Return one disease target per image and reject ambiguous multi-class images."""
    targets = torch.full((batch_size,), -1, dtype=torch.long, device=device)
    if "cls" in batch and "batch_idx" in batch:
        classes = batch["cls"].view(-1).long().to(device)
        batch_idx = batch["batch_idx"].view(-1).long().to(device)
        per_image = [classes[batch_idx == image_idx].unique() for image_idx in range(batch_size)]
    else:
        # SemanticDataset intentionally drops instance ``cls`` and ``batch_idx``.
        # Each Skin7 image has one lesion/disease, so recover that image target
        # from the unique non-background, non-ignore pixel label.
        masks = batch["semantic_mask"].to(device)
        per_image = []
        for image_idx in range(batch_size):
            values = masks[image_idx].unique()
            per_image.append(values[(values >= 0) & (values < len(DISEASE_NAMES))].long())
    for image_idx, image_classes in enumerate(per_image):
        if image_classes.numel() != 1:
            raise ValueError(
                f"runs5 expects exactly one disease class per image; image {image_idx} has "
                f"classes={image_classes.tolist()}"
            )
        targets[image_idx] = image_classes[0]
    return targets


class Skin7SemanticClassificationHead(SemanticSegment):
    """Standard semantic head plus a small P3/P4 global-pooling classifier."""

    def __init__(self, nc: int, ch: tuple[int, int], num_diseases: int = 7, hidden_channels: int = 128):
        super().__init__(nc=nc, ch=ch)
        self.num_diseases = num_diseases
        self.image_classifier = nn.Sequential(
            nn.Linear(sum(ch), hidden_channels),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.20),
            nn.Linear(hidden_channels, num_diseases),
        )

    def forward(self, x: list[torch.Tensor]):
        pooled = torch.cat([F.adaptive_avg_pool2d(feature, 1).flatten(1) for feature in x], dim=1)
        image_logits = self.image_classifier(pooled)
        semantic = super().forward(x)
        if self.training:
            if isinstance(semantic, tuple):
                return semantic[0], semantic[1], image_logits
            return semantic, None, image_logits
        return semantic, image_logits


class Skin7MultiTaskLoss(nn.Module):
    """Keep runs4 semantic loss unchanged and add ordinary image-level CE."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.semantic_loss = SemanticSegmentationLoss(model)
        self.image_cls_weight = float(getattr(model, "image_cls_weight", 0.5))

    def forward(self, preds, batch):
        if len(preds) == 3:
            semantic_logits, auxiliary_logits, image_logits = preds
        elif len(preds) == 2:  # validation runs the model in eval mode, where the semantic aux head is disabled
            semantic_logits, image_logits = preds
            auxiliary_logits = None
        else:
            raise ValueError(f"Unexpected runs5 model output length: {len(preds)}")
        semantic_preds = (
            (semantic_logits, auxiliary_logits) if auxiliary_logits is not None else semantic_logits
        )
        semantic_total, semantic_items = self.semantic_loss(semantic_preds, batch)
        targets = image_targets_from_batch(batch, image_logits.shape[0], image_logits.device)
        image_ce = F.cross_entropy(image_logits, targets)
        total = semantic_total + self.image_cls_weight * image_ce * image_logits.shape[0]
        items = torch.cat([semantic_items, image_ce.detach().reshape(1)])
        return total, items


class Skin7MultiTaskModel(SemanticSegmentationModel):
    """SemanticSegmentationModel whose final head also predicts image class."""

    def __init__(
        self,
        cfg="yolo26s-sem.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        num_diseases: int = 7,
        image_cls_weight: float = 0.5,
    ):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=False)
        old_head = self.model[-1]
        if not isinstance(old_head, SemanticSegment):
            raise TypeError(f"Expected SemanticSegment head, found {type(old_head).__name__}")
        p3_channels = old_head.classifier[0].conv.in_channels
        p4_channels = old_head.aux_head[0].conv.in_channels
        new_head = Skin7SemanticClassificationHead(
            nc=old_head.nc,
            ch=(p3_channels, p4_channels),
            num_diseases=num_diseases,
            hidden_channels=p3_channels,
        )
        new_head.classifier.load_state_dict(old_head.classifier.state_dict())
        if old_head.aux_head is not None:
            new_head.aux_head.load_state_dict(old_head.aux_head.state_dict())
        for attr in ("f", "i", "type"):
            setattr(new_head, attr, getattr(old_head, attr))
        new_head.np = sum(parameter.numel() for parameter in new_head.parameters())
        new_head.stride = old_head.stride
        self.model[-1] = new_head
        self.image_cls_weight = image_cls_weight
        self.num_diseases = num_diseases
        if verbose:
            self.info()

    def init_criterion(self):
        return Skin7MultiTaskLoss(self)


def _classification_stats(confusion: np.ndarray) -> dict[str, Any]:
    per_class = []
    for class_id, name in enumerate(DISEASE_NAMES):
        tp = int(confusion[class_id, class_id])
        fn = int(confusion[class_id].sum() - tp)
        fp = int(confusion[:, class_id].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append((name, precision, recall, f1))
    total = int(confusion.sum())
    return {
        "accuracy": float(np.trace(confusion) / total) if total else 0.0,
        "macro_precision": float(np.mean([row[1] for row in per_class])),
        "macro_recall": float(np.mean([row[2] for row in per_class])),
        "macro_f1": float(np.mean([row[3] for row in per_class])),
        "minority_recall": float(np.mean([per_class[idx][2] for idx in MINORITY_IDS])),
        "mel_to_nv": int(confusion[MEL_ID, NV_ID]),
        "per_class": per_class,
    }


class Skin7MultiTaskValidator(SemanticSegmentationValidator):
    """Validate both heads and expose a predeclared checkpoint fitness score."""

    def __init__(self, *args, selection: dict[str, float] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.selection = selection or {}
        self.image_confusion = np.zeros((len(DISEASE_NAMES), len(DISEASE_NAMES)), dtype=np.int64)
        self._image_logits = None

    def init_metrics(self, model):
        super().init_metrics(model)
        self.image_confusion.fill(0)

    def postprocess(self, preds):
        semantic_logits, self._image_logits = preds
        return super().postprocess(semantic_logits)

    def update_metrics(self, preds, batch):
        super().update_metrics(preds, batch)
        if self._image_logits is None:
            raise RuntimeError("The runs5 classification head did not return image logits.")
        targets = image_targets_from_batch(batch, self._image_logits.shape[0], self._image_logits.device)
        predicted = self._image_logits.argmax(1)
        for truth, pred in zip(targets.detach().cpu().tolist(), predicted.detach().cpu().tolist()):
            self.image_confusion[truth, pred] += 1

    def get_stats(self):
        stats = super().get_stats()
        cls = _classification_stats(self.image_confusion)
        foreground_iou = float(np.mean(self.metrics.per_class_iou[: len(DISEASE_NAMES)]))
        mel_support = int(self.image_confusion[MEL_ID].sum())
        mel_to_nv_rate = cls["mel_to_nv"] / mel_support if mel_support else 0.0
        score = (
            self.selection.get("macro_f1_weight", 0.50) * cls["macro_f1"]
            + self.selection.get("foreground_miou_weight", 0.25) * foreground_iou
            + self.selection.get("minority_recall_weight", 0.25) * cls["minority_recall"]
            - self.selection.get("mel_to_nv_penalty", 0.25)
            * max(0.0, mel_to_nv_rate - self.selection.get("max_mel_to_nv_rate", 18 / 167))
        )
        stats.update(
            {
                "metrics/image_accuracy": cls["accuracy"],
                "metrics/macro_precision": cls["macro_precision"],
                "metrics/macro_recall": cls["macro_recall"],
                "metrics/macro_f1": cls["macro_f1"],
                "metrics/foreground_mIoU": foreground_iou,
                "metrics/minority_recall": cls["minority_recall"],
                "metrics/mel_to_nv_rate": mel_to_nv_rate,
                "fitness": score,
            }
        )
        return stats


class Skin7MultiTaskPredictor(SemanticSegmentationPredictor):
    """Expose classifier probabilities alongside the semantic result."""

    def postprocess(self, preds, img, orig_imgs):
        semantic_logits, image_logits = preds
        results = super().postprocess(semantic_logits, img, orig_imgs)
        probabilities = image_logits.softmax(1)
        for result, probs in zip(results, probabilities):
            result.probs = Probs(probs, result.orig_shape)
        return results


class Skin7MultiTaskTrainer(SemanticSegmentationTrainer):
    """Project-local trainer used only by experiments/05."""

    def __init__(
        self,
        *args,
        image_cls_weight: float = 0.5,
        selection: dict[str, float] | None = None,
        **kwargs,
    ):
        self.image_cls_weight = image_cls_weight
        self.selection = selection or {}
        super().__init__(*args, **kwargs)

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        model = Skin7MultiTaskModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            num_diseases=len(DISEASE_NAMES),
            image_cls_weight=self.image_cls_weight,
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
