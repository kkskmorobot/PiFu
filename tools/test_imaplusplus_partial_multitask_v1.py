from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from ultralytics.cfg import get_cfg

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from imaplusplus_partial_multitask_v1 import (
    CLASS_NAMES,
    ContractError,
    PartialLabelMultiTaskLoss,
    PartialLabelSemanticDataset,
    calibration_metrics,
    classification_metrics,
    load_partial_manifest,
)


class FakeSemanticHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.nc = 1


class FakeSemanticModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.model = nn.ModuleList([FakeSemanticHead()])
        self.args = SimpleNamespace(data="synthetic_binary")
        self.class_weights = None
        self.image_cls_weight = 1.0


def make_predictions(batch_size: int = 2):
    semantic = torch.randn(batch_size, 1, 4, 4, requires_grad=True)
    auxiliary = torch.randn(batch_size, 1, 4, 4, requires_grad=True)
    classification = torch.randn(batch_size, len(CLASS_NAMES), requires_grad=True)
    return semantic, auxiliary, classification


class PartialMultitaskTests(unittest.TestCase):
    def test_real_semantic_dataset_collates_partial_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images" / "train"
            mask_dir = root / "masks" / "train"
            image_dir.mkdir(parents=True)
            mask_dir.mkdir(parents=True)
            manifest = root / "manifest.csv"
            rows = []
            for index, (has_seg, has_cls, target, class_name) in enumerate(
                ((1, 0, -1, ""), (0, 1, 4, "mel"))
            ):
                isic_id = f"ISIC_fixture_{index}"
                image = np.full((48, 64, 3), 64 + index * 64, dtype=np.uint8)
                Image.fromarray(image).save(image_dir / f"{isic_id}.jpg")
                mask = np.full((48, 64), 255, dtype=np.uint8)
                if has_seg:
                    mask[:] = 0
                    mask[12:36, 16:48] = 1
                Image.fromarray(mask).save(mask_dir / f"{isic_id}.png")
                rows.append(
                    {
                        "isic_id": isic_id,
                        "split": "train",
                        "dataset_role": "fixture",
                        "image_path": f"images/train/{isic_id}.jpg",
                        "mask_path": f"masks/train/{isic_id}.png",
                        "has_segmentation": has_seg,
                        "has_classification": has_cls,
                        "classification_target": target,
                        "class_name": class_name,
                        "group_id": f"G{index}",
                    }
                )
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            hyp = get_cfg()
            for name in ("mosaic", "mixup", "cutmix", "copy_paste"):
                setattr(hyp, name, 0.0)
            dataset = PartialLabelSemanticDataset(
                img_path=str(image_dir),
                imgsz=64,
                batch_size=2,
                augment=False,
                hyp=hyp,
                rect=False,
                cache=False,
                single_cls=False,
                stride=32,
                pad=0.0,
                prefix="fixture: ",
                task="semantic",
                classes=None,
                data={"path": str(root), "masks_dir": "masks", "nc": 1, "names": {0: "lesion"}},
                fraction=1.0,
                partial_label_manifest=manifest,
                expected_split="train",
            )
            batch = PartialLabelSemanticDataset.collate_fn([dataset[0], dataset[1]])
            self.assertEqual(tuple(batch["img"].shape), (2, 3, 64, 64))
            self.assertEqual(batch["classification_target"].tolist(), [-1, 4])
            self.assertEqual(batch["has_segmentation"].tolist(), [True, False])
            self.assertEqual(batch["has_classification"].tolist(), [False, True])
            self.assertTrue(bool((batch["semantic_mask"][0] != 255).any().item()))
            self.assertTrue(bool((batch["semantic_mask"][1] == 255).all().item()))

    def test_partial_loss_masks_each_task_and_backpropagates(self) -> None:
        criterion = PartialLabelMultiTaskLoss(FakeSemanticModel())
        masks = torch.full((2, 4, 4), 255, dtype=torch.int32)
        masks[0] = 0
        masks[0, 1:3, 1:3] = 1
        batch = {
            "semantic_mask": masks,
            "has_segmentation": torch.tensor([True, False]),
            "has_classification": torch.tensor([False, True]),
            "classification_target": torch.tensor([-1, 4]),
        }
        predictions = make_predictions()
        total, items = criterion(predictions, batch)
        self.assertTrue(bool(torch.isfinite(total).item()))
        self.assertEqual(items.shape, (4,))
        total.backward()
        self.assertIsNotNone(predictions[0].grad)
        self.assertIsNotNone(predictions[2].grad)
        self.assertTrue(bool(torch.all(predictions[2].grad[0] == 0).item()))
        self.assertTrue(bool(torch.any(predictions[2].grad[1] != 0).item()))

    def test_single_task_batches_remain_finite(self) -> None:
        cases = [
            ([True, True], [False, False], [-1, -1]),
            ([False, False], [True, True], [0, 7]),
        ]
        for has_segmentation, has_classification, targets in cases:
            with self.subTest(has_segmentation=has_segmentation):
                criterion = PartialLabelMultiTaskLoss(FakeSemanticModel())
                masks = torch.full((2, 4, 4), 255, dtype=torch.int32)
                if any(has_segmentation):
                    masks[:] = 0
                    masks[:, 1:3, 1:3] = 1
                batch = {
                    "semantic_mask": masks,
                    "has_segmentation": torch.tensor(has_segmentation),
                    "has_classification": torch.tensor(has_classification),
                    "classification_target": torch.tensor(targets),
                }
                total, items = criterion(make_predictions(), batch)
                self.assertTrue(bool(torch.isfinite(total).item()))
                self.assertTrue(bool(torch.isfinite(items).all().item()))

    def test_manifest_rejects_inconsistent_class_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            manifest = tmp_path / "manifest.csv"
            fields = [
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
            ]
            with manifest.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "isic_id": "ISIC_bad",
                        "split": "train",
                        "dataset_role": "synthetic",
                        "image_path": "images/train/ISIC_bad.jpg",
                        "mask_path": "masks/train/ISIC_bad.png",
                        "has_segmentation": 1,
                        "has_classification": 1,
                        "classification_target": 4,
                        "class_name": "nv",
                        "group_id": "G1",
                    }
                )
            with self.assertRaisesRegex(ContractError, "mapping mismatch"):
                load_partial_manifest(manifest, tmp_path)

    def test_metrics_include_risk_and_calibration(self) -> None:
        confusion = np.eye(8, dtype=np.int64)
        confusion[4, 4] = 9
        confusion[4, 5] = 1
        metrics = classification_metrics(confusion)
        self.assertEqual(metrics["mel_to_nv"], 1)
        self.assertAlmostEqual(metrics["mel_to_nv_rate"], 0.1)
        probabilities = np.eye(8, dtype=np.float64)
        targets = np.arange(8, dtype=np.int64)
        calibration = calibration_metrics(probabilities, targets)
        self.assertLess(calibration["nll"], 1e-9)
        self.assertAlmostEqual(calibration["multiclass_brier"], 0.0)
        self.assertAlmostEqual(calibration["ece"], 0.0)


if __name__ == "__main__":
    unittest.main()
