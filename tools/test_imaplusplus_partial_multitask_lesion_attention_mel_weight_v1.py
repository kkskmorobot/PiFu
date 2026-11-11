from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F


TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from imaplusplus_partial_multitask_lesion_attention_mel_weight_v1 import (
    MelWeightedBinaryLesionGuidedAttentionModel,
    MelWeightedPartialLabelMultiTaskLoss,
    RUN11_CLASSIFICATION_CLASS_WEIGHTS,
    RUN11_MEL_CLASS_WEIGHT,
)
from imaplusplus_partial_multitask_lesion_attention_v1 import BinaryLesionGuidedAttentionModel
from imaplusplus_partial_multitask_mel_weight_v1 import run11_weighted_classification_ce
from imaplusplus_partial_multitask_v1 import ContractError


class Run13FactorialCombinationTests(unittest.TestCase):
    def test_exact_run11_weight_is_reused(self) -> None:
        self.assertEqual(RUN11_MEL_CLASS_WEIGHT, 5539 / 3047)
        self.assertEqual(
            RUN11_CLASSIFICATION_CLASS_WEIGHTS,
            (1.0, 1.0, 1.0, 1.0, 5539 / 3047, 1.0, 1.0, 1.0),
        )

    def test_model_rejects_any_weight_drift(self) -> None:
        changed = list(RUN11_CLASSIFICATION_CLASS_WEIGHTS)
        changed[4] += 1e-6
        with self.assertRaisesRegex(ContractError, "exact Run11"):
            MelWeightedBinaryLesionGuidedAttentionModel(
                verbose=False,
                classification_class_weights=tuple(changed),
            )

    def test_run13_initial_state_is_elementwise_identical_to_run12(self) -> None:
        torch.manual_seed(42)
        run12 = BinaryLesionGuidedAttentionModel(verbose=False)
        torch.manual_seed(42)
        run13 = MelWeightedBinaryLesionGuidedAttentionModel(verbose=False)
        self.assertEqual(run12.state_dict().keys(), run13.state_dict().keys())
        for name, value in run12.state_dict().items():
            self.assertTrue(torch.equal(value, run13.state_dict()[name]), name)
        self.assertEqual(sum(p.numel() for p in run12.parameters()), sum(p.numel() for p in run13.parameters()))
        self.assertEqual(list(run13.model[-1].f), [16, 13, 10])

    def test_run13_criterion_is_mel_weighted(self) -> None:
        model = MelWeightedBinaryLesionGuidedAttentionModel(verbose=False)
        model.args = SimpleNamespace(data="run13_development_data.yaml")
        criterion = model.init_criterion()
        self.assertIsInstance(criterion, MelWeightedPartialLabelMultiTaskLoss)
        self.assertEqual(criterion.classification_class_weights, RUN11_CLASSIFICATION_CLASS_WEIGHTS)

    def test_weighted_ce_matches_pytorch_reference(self) -> None:
        logits = torch.tensor(
            [[2.0, 0.0, -1.0, 0.5, -0.5, 1.2, -0.2, 0.1],
             [0.2, -0.3, 0.8, 0.1, 1.5, 0.4, -0.6, -0.8],
             [0.4, 0.1, 0.0, -0.2, 0.3, 1.8, -0.4, -0.7]],
            dtype=torch.float32,
        )
        targets = torch.tensor([0, 4, 5])
        weights = logits.new_tensor(RUN11_CLASSIFICATION_CLASS_WEIGHTS)
        actual = run11_weighted_classification_ce(logits, targets)
        expected = F.cross_entropy(logits, targets, weight=weights)
        self.assertTrue(torch.equal(actual, expected))
        self.assertFalse(torch.equal(actual, F.cross_entropy(logits, targets)))

    def test_non_mel_only_batch_matches_unweighted_ce(self) -> None:
        torch.manual_seed(7)
        logits = torch.randn(5, 8)
        targets = torch.tensor([0, 1, 2, 5, 7])
        self.assertTrue(
            torch.equal(
                run11_weighted_classification_ce(logits, targets),
                F.cross_entropy(logits, targets),
            )
        )


if __name__ == "__main__":
    unittest.main()
