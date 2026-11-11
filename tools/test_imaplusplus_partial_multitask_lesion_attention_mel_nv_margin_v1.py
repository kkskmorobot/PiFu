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

from imaplusplus_partial_multitask_lesion_attention_mel_nv_margin_v1 import (
    MELNVLogitMarginBinaryLesionGuidedAttentionModel,
    MELNVLogitMarginPartialLabelMultiTaskLoss,
    MEL_ID,
    NV_ID,
    RUN14_MEL_NV_LOGIT_MARGIN,
    apply_mel_nv_training_margin,
    run14_mel_nv_margin_ce,
)
from imaplusplus_partial_multitask_lesion_attention_v1 import BinaryLesionGuidedAttentionModel
from imaplusplus_partial_multitask_v1 import ContractError


class Run14MELNVLogitMarginTests(unittest.TestCase):
    def test_margin_is_exact_train_prior_log_ratio(self) -> None:
        self.assertEqual(RUN14_MEL_NV_LOGIT_MARGIN, math.log(5539 / 3047))

    def test_margin_changes_only_nv_logit_for_mel_targets(self) -> None:
        logits = torch.arange(24, dtype=torch.float32).reshape(3, 8)
        targets = torch.tensor([MEL_ID, NV_ID, 0])
        adjusted = apply_mel_nv_training_margin(logits, targets)
        expected = logits.clone()
        expected[0, NV_ID] += RUN14_MEL_NV_LOGIT_MARGIN
        self.assertTrue(torch.equal(adjusted, expected))
        self.assertTrue(torch.equal(logits, torch.arange(24, dtype=torch.float32).reshape(3, 8)))

    def test_margin_ce_matches_manual_reference(self) -> None:
        torch.manual_seed(9)
        logits = torch.randn(4, 8, requires_grad=True)
        targets = torch.tensor([MEL_ID, 0, NV_ID, MEL_ID])
        adjusted = logits.clone()
        adjusted[targets == MEL_ID, NV_ID] += RUN14_MEL_NV_LOGIT_MARGIN
        self.assertTrue(torch.equal(run14_mel_nv_margin_ce(logits, targets), F.cross_entropy(adjusted, targets)))

    def test_non_mel_batch_is_unmodified_ce(self) -> None:
        torch.manual_seed(10)
        logits = torch.randn(5, 8)
        targets = torch.tensor([0, 1, 2, NV_ID, 7])
        self.assertTrue(torch.equal(run14_mel_nv_margin_ce(logits, targets), F.cross_entropy(logits, targets)))

    def test_model_rejects_margin_drift(self) -> None:
        with self.assertRaisesRegex(ContractError, "frozen train-prior"):
            MELNVLogitMarginBinaryLesionGuidedAttentionModel(
                verbose=False,
                mel_nv_logit_margin=RUN14_MEL_NV_LOGIT_MARGIN + 1e-6,
            )

    def test_run14_initial_state_is_elementwise_identical_to_run12(self) -> None:
        torch.manual_seed(42)
        run12 = BinaryLesionGuidedAttentionModel(verbose=False)
        torch.manual_seed(42)
        run14 = MELNVLogitMarginBinaryLesionGuidedAttentionModel(verbose=False)
        self.assertEqual(run12.state_dict().keys(), run14.state_dict().keys())
        for name, value in run12.state_dict().items():
            self.assertTrue(torch.equal(value, run14.state_dict()[name]), name)
        self.assertEqual(sum(p.numel() for p in run12.parameters()), sum(p.numel() for p in run14.parameters()))

    def test_run14_criterion_and_inference_architecture_are_locked(self) -> None:
        model = MELNVLogitMarginBinaryLesionGuidedAttentionModel(verbose=False)
        model.args = SimpleNamespace(data="run14_development_data.yaml")
        criterion = model.init_criterion()
        self.assertIsInstance(criterion, MELNVLogitMarginPartialLabelMultiTaskLoss)
        self.assertEqual(criterion.mel_nv_logit_margin, RUN14_MEL_NV_LOGIT_MARGIN)
        self.assertEqual(list(model.model[-1].f), [16, 13, 10])


if __name__ == "__main__":
    unittest.main()
