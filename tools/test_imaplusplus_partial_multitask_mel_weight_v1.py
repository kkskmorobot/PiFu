from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from imaplusplus_partial_multitask_mel_weight_v1 import (  # noqa: E402
    CLASS_NAMES,
    MEL_ID,
    NV_ID,
    RUN11_CLASSIFICATION_CLASS_WEIGHTS,
    RUN11_MEL_CLASS_WEIGHT,
    RUN11_TRAIN_MEL_COUNT,
    RUN11_TRAIN_NV_COUNT,
    run11_weighted_classification_ce,
)


class Run11MelWeightedCrossEntropyTests(unittest.TestCase):
    def test_weight_is_train_nv_to_mel_ratio_and_only_mel_differs(self) -> None:
        self.assertEqual(RUN11_TRAIN_MEL_COUNT, 3047)
        self.assertEqual(RUN11_TRAIN_NV_COUNT, 5539)
        self.assertAlmostEqual(RUN11_MEL_CLASS_WEIGHT, 5539 / 3047, places=15)
        self.assertEqual(len(RUN11_CLASSIFICATION_CLASS_WEIGHTS), len(CLASS_NAMES))
        for class_id, value in enumerate(RUN11_CLASSIFICATION_CLASS_WEIGHTS):
            expected = RUN11_MEL_CLASS_WEIGHT if class_id == MEL_ID else 1.0
            self.assertEqual(value, expected)

    def test_weighted_ce_matches_locked_manual_formula(self) -> None:
        logits = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, -1.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.5, -0.5, 0.0, 0.0],
            ],
            dtype=torch.float64,
            requires_grad=True,
        )
        targets = torch.tensor([MEL_ID, NV_ID], dtype=torch.long)
        per_row = F.cross_entropy(logits, targets, reduction="none")
        expected = (RUN11_MEL_CLASS_WEIGHT * per_row[0] + per_row[1]) / (RUN11_MEL_CLASS_WEIGHT + 1.0)
        actual = run11_weighted_classification_ce(logits, targets)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_logits_contract_rejects_non_eight_class_input(self) -> None:
        with self.assertRaisesRegex(RuntimeError, r"Expected classification logits \[N, 8\]"):
            run11_weighted_classification_ce(torch.zeros(2, 7), torch.zeros(2, dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
