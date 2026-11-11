from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from imaplusplus_partial_multitask_lesion_attention_v1 import BinaryLesionGuidedAttentionModel
from imaplusplus_partial_multitask_lesion_boundary_attention_v1 import (
    BinaryLesionBoundaryGuidedAttentionHead,
    BinaryLesionBoundaryGuidedAttentionModel,
)
from imaplusplus_partial_multitask_v1 import PartialLabelMultiTaskLoss


class Run17BoundaryAttentionTests(unittest.TestCase):
    def test_boundary_map_is_detached_and_concentrated_on_edges(self) -> None:
        foreground = torch.zeros(1, 1, 16, 16, requires_grad=True)
        with torch.no_grad():
            foreground[:, :, 4:12, 4:12] = 1.0
        boundary = BinaryLesionBoundaryGuidedAttentionHead._boundary_probability(foreground)
        self.assertFalse(boundary.requires_grad)
        self.assertEqual(tuple(boundary.shape), (1, 1, 16, 16))
        self.assertGreater(float(boundary.sum()), 0.0)
        self.assertEqual(float(boundary[:, :, 6:10, 6:10].sum()), 0.0)

    def test_zero_boundary_columns_preserve_run16_initial_outputs(self) -> None:
        torch.manual_seed(1234)
        run16 = BinaryLesionGuidedAttentionModel(cfg="yolo26m-sem.yaml", verbose=False)
        torch.manual_seed(1234)
        run17 = BinaryLesionBoundaryGuidedAttentionModel(cfg="yolo26m-sem.yaml", verbose=False)
        base_first = run16.model[-1].image_classifier[0]
        boundary_first = run17.model[-1].image_classifier[0]
        split = base_first.in_features
        self.assertTrue(torch.equal(boundary_first.weight[:, :split], base_first.weight))
        self.assertEqual(int(boundary_first.weight[:, split:].count_nonzero()), 0)
        self.assertTrue(torch.equal(boundary_first.bias, base_first.bias))

        x = torch.rand(2, 3, 64, 64)
        run16.train()
        run17.train()
        torch.manual_seed(9876)
        out16 = run16(x)
        torch.manual_seed(9876)
        out17 = run17(x)
        for left, right in zip(out16, out17):
            if left is None or right is None:
                self.assertIs(left, right)
            else:
                self.assertTrue(torch.equal(left, right))

    def test_unweighted_ce_is_inherited_and_boundary_columns_receive_gradient(self) -> None:
        torch.manual_seed(7)
        model = BinaryLesionBoundaryGuidedAttentionModel(cfg="yolo26m-sem.yaml", verbose=False).train()
        model.args = SimpleNamespace(data="synthetic_run17_unit_test")
        self.assertIsInstance(model.init_criterion(), PartialLabelMultiTaskLoss)
        outputs = model(torch.rand(2, 3, 64, 64))
        outputs[-1].sum().backward()
        first = model.model[-1].image_classifier[0]
        split = model.model[-1].base_descriptor_channels
        self.assertIsNotNone(first.weight.grad)
        self.assertGreater(float(first.weight.grad[:, split:].abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
