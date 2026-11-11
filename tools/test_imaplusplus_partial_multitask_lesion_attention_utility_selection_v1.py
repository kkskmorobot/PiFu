from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from imaplusplus_partial_multitask_lesion_attention_utility_selection_v1 import (
    RUN16_MIN_BINARY_IOU,
    RUN16_MIN_GROUP_MACRO_F1,
    UtilityEligiblePartialLabelMultiTaskValidator,
    run16_utility_eligible,
)
from imaplusplus_partial_multitask_v1 import ContractError, PartialLabelMultiTaskValidator


class Run16UtilitySelectionTests(unittest.TestCase):
    def test_frozen_utility_floors(self) -> None:
        self.assertEqual(RUN16_MIN_BINARY_IOU, 0.70)
        self.assertEqual(RUN16_MIN_GROUP_MACRO_F1, 0.50)
        self.assertTrue(run16_utility_eligible(0.70, 0.50))
        self.assertFalse(run16_utility_eligible(0.69999, 0.50))
        self.assertFalse(run16_utility_eligible(0.70, 0.49999))

    def test_validator_rejects_floor_drift(self) -> None:
        with self.assertRaisesRegex(ContractError, "utility eligibility floors"):
            UtilityEligiblePartialLabelMultiTaskValidator(
                None,
                selection={"min_binary_iou": 0.71, "min_group_macro_f1": 0.50},
            )

    def test_only_feasibility_is_tightened_and_fitness_is_unchanged(self) -> None:
        validator = object.__new__(UtilityEligiblePartialLabelMultiTaskValidator)
        validator.min_binary_iou = RUN16_MIN_BINARY_IOU
        validator.min_group_macro_f1 = RUN16_MIN_GROUP_MACRO_F1
        base = {
            "fitness": 0.31,
            "metrics/feasible_joint": 1.0,
            "metrics/binary_lesion_iou": 0.63,
            "metrics/group_macro_f1": 0.11,
        }
        with patch.object(PartialLabelMultiTaskValidator, "get_stats", return_value=dict(base)):
            stats = UtilityEligiblePartialLabelMultiTaskValidator.get_stats(validator)
        self.assertEqual(stats["fitness"], base["fitness"])
        self.assertEqual(stats["metrics/risk_feasible"], 1.0)
        self.assertEqual(stats["metrics/utility_eligible"], 0.0)
        self.assertEqual(stats["metrics/feasible_joint"], 0.0)


if __name__ == "__main__":
    unittest.main()
