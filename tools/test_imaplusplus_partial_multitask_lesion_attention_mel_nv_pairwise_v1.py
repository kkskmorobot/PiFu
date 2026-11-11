from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from imaplusplus_partial_multitask_lesion_attention_mel_nv_pairwise_v1 import (
    MELNVPairwiseBoundaryBinaryLesionGuidedAttentionModel,
    MELNVPairwiseBoundaryPartialLabelMultiTaskLoss,
    RUN15_MEL_NV_PAIRWISE_LOSS_WEIGHT,
    RUN15_MEL_NV_PAIRWISE_MARGIN,
    run15_mel_nv_pairwise_softplus,
    run15_utility_eligible,
)
from imaplusplus_partial_multitask_lesion_attention_v1 import BinaryLesionGuidedAttentionModel
from imaplusplus_partial_multitask_v1 import ContractError, MEL_ID, NV_ID


class Run15MELNVPairwiseBoundaryTests(unittest.TestCase):
    def test_margin_and_weight_are_frozen(self) -> None:
        self.assertEqual(RUN15_MEL_NV_PAIRWISE_MARGIN, math.log(5539 / 3047))
        self.assertEqual(RUN15_MEL_NV_PAIRWISE_LOSS_WEIGHT, 1.0)

    def test_pairwise_loss_matches_manual_all_labeled_normalization(self) -> None:
        logits = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.4, -0.2, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, -0.1, 0.7, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.8, 0.1, 0.0, 0.0]],
            requires_grad=True,
        )
        targets = torch.tensor([MEL_ID, NV_ID, MEL_ID])
        expected = (
            F.softplus(logits[0, NV_ID] - logits[0, MEL_ID] + RUN15_MEL_NV_PAIRWISE_MARGIN)
            + F.softplus(logits[2, NV_ID] - logits[2, MEL_ID] + RUN15_MEL_NV_PAIRWISE_MARGIN)
        ) / 3
        self.assertTrue(torch.equal(run15_mel_nv_pairwise_softplus(logits, targets), expected))

    def test_pairwise_gradient_is_direct_and_only_mel_nv(self) -> None:
        logits = torch.zeros((2, 8), requires_grad=True)
        targets = torch.tensor([MEL_ID, 0])
        run15_mel_nv_pairwise_softplus(logits, targets).backward()
        self.assertLess(float(logits.grad[0, MEL_ID]), 0.0)
        self.assertGreater(float(logits.grad[0, NV_ID]), 0.0)
        untouched = logits.grad.clone()
        untouched[0, MEL_ID] = 0
        untouched[0, NV_ID] = 0
        self.assertEqual(int(untouched.count_nonzero()), 0)

    def test_non_mel_batch_has_exact_zero_pairwise_loss(self) -> None:
        logits = torch.randn(5, 8, requires_grad=True)
        targets = torch.tensor([0, 1, 2, NV_ID, 7])
        loss = run15_mel_nv_pairwise_softplus(logits, targets)
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        self.assertEqual(int(logits.grad.count_nonzero()), 0)

    def test_utility_floor_rejects_run14_cold_start(self) -> None:
        self.assertFalse(run15_utility_eligible(0.53363, 0.16923))
        self.assertTrue(run15_utility_eligible(0.70, 0.50))

    def test_model_rejects_margin_or_weight_drift(self) -> None:
        with self.assertRaisesRegex(ContractError, "pairwise margin"):
            MELNVPairwiseBoundaryBinaryLesionGuidedAttentionModel(
                verbose=False, mel_nv_pairwise_margin=RUN15_MEL_NV_PAIRWISE_MARGIN + 1e-6
            )
        with self.assertRaisesRegex(ContractError, "pairwise weight"):
            MELNVPairwiseBoundaryBinaryLesionGuidedAttentionModel(
                verbose=False, mel_nv_pairwise_loss_weight=0.5
            )

    def test_run15_initial_state_is_elementwise_identical_to_run12(self) -> None:
        torch.manual_seed(42)
        run12 = BinaryLesionGuidedAttentionModel(verbose=False)
        torch.manual_seed(42)
        run15 = MELNVPairwiseBoundaryBinaryLesionGuidedAttentionModel(verbose=False)
        self.assertEqual(run12.state_dict().keys(), run15.state_dict().keys())
        for name, value in run12.state_dict().items():
            self.assertTrue(torch.equal(value, run15.state_dict()[name]), name)
        self.assertEqual(sum(p.numel() for p in run12.parameters()), sum(p.numel() for p in run15.parameters()))

    def test_run15_criterion_and_inference_architecture_are_locked(self) -> None:
        model = MELNVPairwiseBoundaryBinaryLesionGuidedAttentionModel(verbose=False)
        model.args = SimpleNamespace(data="run15_development_data.yaml")
        criterion = model.init_criterion()
        self.assertIsInstance(criterion, MELNVPairwiseBoundaryPartialLabelMultiTaskLoss)
        self.assertEqual(criterion.mel_nv_pairwise_margin, RUN15_MEL_NV_PAIRWISE_MARGIN)
        self.assertEqual(criterion.mel_nv_pairwise_loss_weight, 1.0)
        self.assertEqual(list(model.model[-1].f), [16, 13, 10])


if __name__ == "__main__":
    unittest.main()
