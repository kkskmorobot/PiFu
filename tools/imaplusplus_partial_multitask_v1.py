#!/usr/bin/env python3
"""Binary lesion segmentation + 8-class image classification with partial labels.

This is a project-local implementation scaffold, not a Run10 experiment.  It
keeps the YOLO26 binary semantic head, adds a plain P3/P4 global-pooling image
classifier, attaches task availability from the frozen manifest, masks losses
by task availability, and reports unified validation metrics.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from copy import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.data.dataset import DATASET_CACHE_VERSION, SemanticDataset
from ultralytics.data.utils import load_dataset_cache_file
from ultralytics.utils import LOCAL_RANK, LOGGER, TQDM
from ultralytics.engine.results import Probs
from ultralytics.models.yolo.semantic import SemanticSegmentationPredictor, SemanticSegmentationValidator
from ultralytics.models.yolo.semantic.train import SemanticSegmentationTrainer
from ultralytics.nn.modules import SemanticSegment
from ultralytics.nn.tasks import SemanticSegmentationModel
from ultralytics.utils import RANK, colorstr
from ultralytics.utils.loss import SemanticSegmentationLoss
from ultralytics.utils.torch_utils import unwrap_model

from multitask_checkpoint_contract import InterruptSafeTrainerMixin


CLASS_NAMES = ("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc", "scc")
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
MEL_ID = CLASS_TO_ID["mel"]
NV_ID = CLASS_TO_ID["nv"]
MALIGNANT_IDS = frozenset({CLASS_TO_ID["bcc"], MEL_ID, CLASS_TO_ID["scc"]})
REQUIRED_MANIFEST_COLUMNS = {
    "isic_id",
    "split",
    "dataset_role",
    "image_path",
    "mask_path",
    "has_segmentation",
    "has_classification",
    "classification_target",
    "class_name",
    "group_id",
}


class ContractError(RuntimeError):
    """Raised when data/model state does not satisfy the frozen v1 contract."""


def _as_bool(value: str, field: str, isic_id: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ContractError(f"Invalid {field}={value!r} for {isic_id}")


def load_partial_manifest(path: Path, data_root: Path) -> dict[Path, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Partial-label manifest not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ContractError(f"Partial-label manifest is empty: {path}")
    missing = REQUIRED_MANIFEST_COLUMNS - set(rows[0])
    if missing:
        raise ContractError(f"Partial-label manifest is missing columns: {sorted(missing)}")

    records: dict[Path, dict[str, Any]] = {}
    seen_ids: set[str] = set()
    for raw in rows:
        isic_id = raw["isic_id"].strip()
        if not isic_id or isic_id in seen_ids:
            raise ContractError(f"Blank or duplicate isic_id in partial manifest: {isic_id!r}")
        seen_ids.add(isic_id)
        has_segmentation = _as_bool(raw["has_segmentation"], "has_segmentation", isic_id)
        has_classification = _as_bool(raw["has_classification"], "has_classification", isic_id)
        target = int(raw["classification_target"])
        class_name = raw["class_name"].strip()
        if has_classification:
            if target not in range(len(CLASS_NAMES)) or class_name != CLASS_NAMES[target]:
                raise ContractError(
                    f"Classification mapping mismatch for {isic_id}: target={target}, class_name={class_name!r}"
                )
        elif target != -1 or class_name:
            raise ContractError(f"Unlabeled classification row must use target=-1 and blank class: {isic_id}")
        if not has_segmentation and not has_classification:
            raise ContractError(f"Sample has neither task label: {isic_id}")
        image_path = (data_root / raw["image_path"]).resolve()
        mask_path = (data_root / raw["mask_path"]).resolve()
        if image_path in records:
            raise ContractError(f"Duplicate resolved image path in partial manifest: {image_path}")
        records[image_path] = {
            **raw,
            "isic_id": isic_id,
            "has_segmentation": has_segmentation,
            "has_classification": has_classification,
            "classification_target": target,
            "image_path_resolved": image_path,
            "mask_path_resolved": mask_path,
        }
    return records


class PartialLabelSemanticDataset(SemanticDataset):
    """SemanticDataset whose task availability comes only from a frozen manifest."""

    def __init__(
        self,
        *args,
        partial_label_manifest: str | Path,
        expected_split: str,
        **kwargs,
    ) -> None:
        data = kwargs.get("data") or {}
        data_root = Path(data.get("path", "")).resolve()
        manifest_path = Path(partial_label_manifest)
        if not manifest_path.is_absolute():
            manifest_path = (data_root / manifest_path).resolve()
        self.partial_manifest_path = manifest_path
        self.expected_split = expected_split
        self.partial_records = load_partial_manifest(manifest_path, data_root)
        super().__init__(*args, **kwargs)
        active = {Path(path).resolve() for path in self.im_files}
        missing = active - set(self.partial_records)
        if missing:
            preview = ", ".join(str(path) for path in sorted(missing)[:5])
            raise ContractError(f"Manifest is missing {len(missing)} active images; first: {preview}")
        wrong_split = [
            path
            for path in active
            if self.partial_records[path]["split"].strip() != self.expected_split
        ]
        if wrong_split:
            raise ContractError(
                f"{len(wrong_split)} active images violate expected split={self.expected_split}; first={wrong_split[0]}"
            )

    def get_labels(self):
        """Load/cache image-mask pairing outside both immutable source datasets."""
        from ultralytics.data.utils import img2label_paths

        self.mask_files = img2label_paths(self.im_files, label_dir=self.data.get("masks_dir", "masks"), suffix=".png")
        data_root = Path(self.data.get("path", "")).resolve()
        cache_path = data_root / ".cache" / "partial_multitask_v1" / f"{self.expected_split}.cache"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            cache, exists = load_dataset_cache_file(cache_path), True
            assert cache["version"] == DATASET_CACHE_VERSION
            assert cache["hash"] == self._semantic_cache_hash(self.mask_files)
        except (FileNotFoundError, AssertionError, AttributeError, ModuleNotFoundError):
            cache, exists = self.cache_labels(cache_path), False
        nf, nm, nc, n = cache.pop("results")
        if exists and LOCAL_RANK in {-1, 0}:
            description = f"Scanning {cache_path}... {nf} masks, {nm} missing, {nc} corrupt"
            TQDM(None, desc=self.prefix + description, total=n, initial=n)
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))
        [cache.pop(key) for key in ("hash", "version", "msgs")]
        labels = cache["labels"]
        if not labels:
            raise ContractError(f"No valid partial-label images found in {cache_path}")
        self.im_files = [label["im_file"] for label in labels]
        self.mask_files = [label["mask_file"] for label in labels]
        return labels

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = super().__getitem__(index)
        image_path = Path(str(sample["im_file"])).resolve()
        record = self.partial_records[image_path]
        semantic_mask = sample["semantic_mask"]
        valid_semantic = bool((semantic_mask != 255).any().item())
        if record["has_segmentation"] != valid_semantic:
            raise ContractError(
                f"Manifest/mask availability mismatch for {record['isic_id']}: "
                f"manifest={record['has_segmentation']} mask_has_valid_pixels={valid_semantic}"
            )
        sample["classification_target"] = torch.tensor(record["classification_target"], dtype=torch.long)
        sample["has_classification"] = torch.tensor(record["has_classification"], dtype=torch.bool)
        sample["has_segmentation"] = torch.tensor(record["has_segmentation"], dtype=torch.bool)
        sample["isic_id"] = record["isic_id"]
        sample["group_id"] = record["group_id"].strip() or record["isic_id"]
        sample["dataset_role"] = record["dataset_role"]
        return sample

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        tensor_keys = ("classification_target", "has_classification", "has_segmentation")
        tensors = {key: [sample.pop(key) for sample in batch] for key in tensor_keys}
        result = SemanticDataset.collate_fn(batch)
        for key, values in tensors.items():
            result[key] = torch.stack(values, dim=0)
        return result


class BinaryPartialClassificationHead(SemanticSegment):
    """Unchanged binary semantic head plus a plain P3/P4 image classifier."""

    def __init__(self, nc: int, ch: tuple[int, int], num_diseases: int = 8, hidden_channels: int = 128):
        if nc != 1:
            raise ContractError(f"v1 requires binary semantic nc=1, found {nc}")
        if num_diseases != len(CLASS_NAMES):
            raise ContractError(f"v1 requires {len(CLASS_NAMES)} classification classes, found {num_diseases}")
        super().__init__(nc=nc, ch=ch)
        self.num_diseases = num_diseases
        self.classification_names = CLASS_NAMES
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


def unpack_outputs(preds):
    if not isinstance(preds, (tuple, list)):
        raise ContractError(f"Expected multitask output tuple, found {type(preds).__name__}")
    if len(preds) == 3:
        semantic_logits, auxiliary_logits, image_logits = preds
    elif len(preds) == 2:
        semantic_logits, image_logits = preds
        auxiliary_logits = None
    else:
        raise ContractError(f"Unexpected multitask output length: {len(preds)}")
    return semantic_logits, auxiliary_logits, image_logits


class PartialLabelMultiTaskLoss(nn.Module):
    """BCE+Dice on valid masks and CE on explicitly labeled images only."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.semantic_loss = SemanticSegmentationLoss(model)
        self.image_cls_weight = float(getattr(model, "image_cls_weight", 1.0))
        if not math.isfinite(self.image_cls_weight) or self.image_cls_weight < 0:
            raise ContractError(f"image_cls_weight must be finite and nonnegative, found {self.image_cls_weight}")

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
            image_ce = F.cross_entropy(image_logits[has_cls], targets[has_cls])
        else:
            image_ce = image_logits.sum() * 0.0
        total = semantic_total + self.image_cls_weight * image_ce * batch_size
        items = torch.cat([semantic_items, image_ce.detach().reshape(1)])
        return total, items


class BinaryPartialMultiTaskModel(SemanticSegmentationModel):
    """YOLO26 binary semantic model with an independent 8-class image head."""

    def __init__(
        self,
        cfg="yolo26s-sem.yaml",
        ch: int = 3,
        nc: int | None = 1,
        verbose: bool = True,
        num_diseases: int = 8,
        image_cls_weight: float = 1.0,
    ) -> None:
        if nc != 1:
            raise ContractError(f"BinaryPartialMultiTaskModel requires nc=1, found {nc}")
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=False)
        old_head = self.model[-1]
        if not isinstance(old_head, SemanticSegment):
            raise ContractError(f"Expected SemanticSegment head, found {type(old_head).__name__}")
        p3_channels = old_head.classifier[0].conv.in_channels
        p4_channels = old_head.aux_head[0].conv.in_channels
        new_head = BinaryPartialClassificationHead(
            nc=1,
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
        self.image_cls_weight = float(image_cls_weight)
        self.num_diseases = int(num_diseases)
        self.classification_names = CLASS_NAMES
        if verbose:
            self.info()

    def init_criterion(self):
        return PartialLabelMultiTaskLoss(self)


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def classification_metrics(confusion: np.ndarray) -> dict[str, Any]:
    if confusion.shape != (len(CLASS_NAMES), len(CLASS_NAMES)):
        raise ContractError(f"Expected 8x8 confusion matrix, found {confusion.shape}")
    per_class = []
    for class_id, name in enumerate(CLASS_NAMES):
        tp = int(confusion[class_id, class_id])
        fn = int(confusion[class_id].sum() - tp)
        fp = int(confusion[:, class_id].sum() - tp)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        per_class.append(
            {"class": name, "support": int(confusion[class_id].sum()), "precision": precision, "recall": recall, "f1": f1}
        )
    total = int(confusion.sum())
    correct = int(np.trace(confusion))
    malignant_support = int(sum(confusion[class_id].sum() for class_id in MALIGNANT_IDS))
    malignant_to_benign = int(
        sum(
            confusion[truth, pred]
            for truth in MALIGNANT_IDS
            for pred in range(len(CLASS_NAMES))
            if pred not in MALIGNANT_IDS
        )
    )
    return {
        "samples": total,
        "accuracy": safe_div(correct, total),
        "macro_precision": float(np.mean([row["precision"] for row in per_class])),
        "macro_recall": float(np.mean([row["recall"] for row in per_class])),
        "macro_f1": float(np.mean([row["f1"] for row in per_class])),
        "mel_to_nv": int(confusion[MEL_ID, NV_ID]),
        "mel_to_nv_rate": safe_div(int(confusion[MEL_ID, NV_ID]), int(confusion[MEL_ID].sum())),
        "nv_to_mel": int(confusion[NV_ID, MEL_ID]),
        "nv_to_mel_rate": safe_div(int(confusion[NV_ID, MEL_ID]), int(confusion[NV_ID].sum())),
        "malignant_to_benign": malignant_to_benign,
        "malignant_to_benign_rate": safe_div(malignant_to_benign, malignant_support),
        "per_class": per_class,
    }


def calibration_metrics(probabilities: np.ndarray, targets: np.ndarray, bins: int = 15) -> dict[str, float]:
    if probabilities.ndim != 2 or probabilities.shape[1] != len(CLASS_NAMES):
        raise ContractError(f"Expected probabilities [N,8], found {probabilities.shape}")
    if len(probabilities) != len(targets) or not len(targets):
        return {"nll": 0.0, "multiclass_brier": 0.0, "ece": 0.0, "ece_bins": bins}
    clipped = np.clip(probabilities.astype(np.float64), 1e-12, 1.0)
    one_hot = np.eye(len(CLASS_NAMES), dtype=np.float64)[targets]
    nll = float(-np.log(clipped[np.arange(len(targets)), targets]).mean())
    brier = float(np.square(clipped - one_hot).sum(axis=1).mean())
    confidence = clipped.max(axis=1)
    predicted = clipped.argmax(axis=1)
    correct = predicted == targets
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        selected = (confidence > lower) & (confidence <= upper) if index else (confidence >= lower) & (confidence <= upper)
        if selected.any():
            ece += float(selected.mean()) * abs(float(correct[selected].mean()) - float(confidence[selected].mean()))
    return {"nll": nll, "multiclass_brier": brier, "ece": ece, "ece_bins": bins}


class PartialLabelMultiTaskValidator(SemanticSegmentationValidator):
    """Validate binary segmentation and labeled classification subsets together."""

    def __init__(self, *args, selection: dict[str, float] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.selection = dict(selection or {})
        self.max_mel_to_nv_rate = float(self.selection.get("max_mel_to_nv_rate", 0.1077844311))
        self.image_confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
        self._image_logits = None
        self.classification_rows: list[dict[str, Any]] = []

    def init_metrics(self, model):
        super().init_metrics(model)
        self.image_confusion.fill(0)
        self.classification_rows.clear()

    def postprocess(self, preds):
        semantic_logits, _, image_logits = unpack_outputs(preds)
        self._image_logits = image_logits
        return super().postprocess(semantic_logits)

    def update_metrics(self, preds, batch):
        super().update_metrics(preds, batch)
        if self._image_logits is None:
            raise ContractError("Classification logits were not returned during validation")
        has_cls = batch["has_classification"].to(self._image_logits.device).view(-1).bool()
        targets = batch["classification_target"].to(self._image_logits.device).view(-1).long()
        probabilities = self._image_logits.softmax(1)
        predicted = probabilities.argmax(1)
        group_ids = list(batch["group_id"])
        isic_ids = list(batch["isic_id"])
        for index in torch.where(has_cls)[0].detach().cpu().tolist():
            truth = int(targets[index].item())
            pred = int(predicted[index].item())
            self.image_confusion[truth, pred] += 1
            self.classification_rows.append(
                {
                    "isic_id": str(isic_ids[index]),
                    "group_id": str(group_ids[index]),
                    "target": truth,
                    "probabilities": probabilities[index].detach().float().cpu().numpy(),
                }
            )

    def get_stats(self):
        stats = super().get_stats()
        binary_iou = float(self.metrics.miou)
        binary_dice = safe_div(2.0 * binary_iou, 1.0 + binary_iou)
        image = classification_metrics(self.image_confusion)

        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in self.classification_rows:
            grouped[row["group_id"]].append(row)
        group_confusion = np.zeros_like(self.image_confusion)
        group_probabilities: list[np.ndarray] = []
        group_targets: list[int] = []
        for group_id, rows in grouped.items():
            targets = {int(row["target"]) for row in rows}
            if len(targets) != 1:
                raise ContractError(f"Validation group {group_id} spans multiple targets: {sorted(targets)}")
            target = targets.pop()
            mean_probability = np.mean([row["probabilities"] for row in rows], axis=0)
            pred = int(mean_probability.argmax())
            group_confusion[target, pred] += 1
            group_targets.append(target)
            group_probabilities.append(mean_probability)
        group = classification_metrics(group_confusion)
        calibration = calibration_metrics(
            np.asarray(group_probabilities, dtype=np.float64),
            np.asarray(group_targets, dtype=np.int64),
        )
        joint_score = math.sqrt(max(binary_iou, 0.0) * max(group["macro_f1"], 0.0))
        feasible = bool(group["per_class"][MEL_ID]["support"]) and group["mel_to_nv_rate"] <= self.max_mel_to_nv_rate
        fitness = joint_score if feasible else joint_score - max(0.0, group["mel_to_nv_rate"] - self.max_mel_to_nv_rate)
        stats.update(
            {
                "metrics/binary_lesion_iou": binary_iou,
                "metrics/binary_lesion_dice": binary_dice,
                "metrics/image_macro_f1": image["macro_f1"],
                "metrics/image_macro_recall": image["macro_recall"],
                "metrics/group_macro_f1": group["macro_f1"],
                "metrics/group_macro_recall": group["macro_recall"],
                "metrics/group_mel_to_nv_rate": group["mel_to_nv_rate"],
                "metrics/group_malignant_to_benign_rate": group["malignant_to_benign_rate"],
                "metrics/group_nll": calibration["nll"],
                "metrics/group_brier": calibration["multiclass_brier"],
                "metrics/group_ece": calibration["ece"],
                "metrics/joint_score": joint_score,
                "metrics/feasible_joint": float(feasible),
                "fitness": fitness,
            }
        )
        return stats


class PartialLabelMultiTaskPredictor(SemanticSegmentationPredictor):
    """Expose classification probabilities alongside the binary semantic result."""

    def postprocess(self, preds, img, orig_imgs):
        semantic_logits, _, image_logits = unpack_outputs(preds)
        results = super().postprocess(semantic_logits, img, orig_imgs)
        for result, probabilities in zip(results, image_logits.softmax(1)):
            result.probs = Probs(probabilities, result.orig_shape)
        return results


class PartialLabelMultiTaskTrainer(InterruptSafeTrainerMixin, SemanticSegmentationTrainer):
    """Ultralytics trainer injection for the frozen partial-label v1 contract."""

    def __init__(
        self,
        *args,
        image_cls_weight: float = 1.0,
        selection: dict[str, float] | None = None,
        **kwargs,
    ) -> None:
        self.image_cls_weight = float(image_cls_weight)
        self.selection = dict(selection or {})
        super().__init__(*args, **kwargs)
        for name in ("mosaic", "mixup", "cutmix", "copy_paste"):
            if float(getattr(self.args, name, 0.0) or 0.0) != 0.0:
                raise ContractError(f"partial-label v1 requires {name}=0.0")
        if self.data.get("nc") != 1 or tuple(self.data.get("names", {}).values()) != ("lesion",):
            raise ContractError(f"partial-label v1 requires binary semantic names={{0: 'lesion'}}, found {self.data.get('names')}")
        if not self.data.get("partial_label_manifest"):
            raise ContractError("Dataset YAML must declare partial_label_manifest")

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        data_root = Path(self.data["path"]).resolve()
        manifest = Path(self.data["partial_label_manifest"])
        if not manifest.is_absolute():
            manifest = (data_root / manifest).resolve()
        return PartialLabelSemanticDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=self.args.rect or mode == "val",
            cache=self.args.cache or None,
            single_cls=False,
            stride=gs,
            pad=0.0,
            prefix=colorstr(f"{mode}: "),
            task="semantic",
            classes=None,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
            partial_label_manifest=manifest,
            expected_split=mode,
        )

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        model = BinaryPartialMultiTaskModel(
            cfg,
            nc=1,
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            num_diseases=len(CLASS_NAMES),
            image_cls_weight=self.image_cls_weight,
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        self.loss_names = "bce_loss", "dice_loss", "aux_loss", "image_cls_loss"
        return PartialLabelMultiTaskValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
            selection=self.selection,
        )
