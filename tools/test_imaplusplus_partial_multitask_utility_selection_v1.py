from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from imaplusplus_partial_multitask_utility_selection_v1 import (
    RUN17_MIN_BINARY_IOU,
    RUN17_MIN_GROUP_MACRO_F1,
    UtilityEligiblePartialLabelMultiTaskTrainer,
    run17_utility_eligible,
)
from imaplusplus_partial_multitask_v1 import PartialLabelMultiTaskTrainer


def test_run17_trainer_is_plain_multitask_trainer() -> None:
    assert issubclass(UtilityEligiblePartialLabelMultiTaskTrainer, PartialLabelMultiTaskTrainer)


def test_run17_utility_floors_are_frozen() -> None:
    assert RUN17_MIN_BINARY_IOU == 0.70
    assert RUN17_MIN_GROUP_MACRO_F1 == 0.50
    assert run17_utility_eligible(0.70, 0.50)
    assert not run17_utility_eligible(0.699999, 0.50)
    assert not run17_utility_eligible(0.70, 0.499999)


@pytest.mark.parametrize("binary_iou,group_macro_f1", [(0.0, 0.0), (0.8, 0.49), (0.69, 0.8)])
def test_run17_rejects_non_degenerate_floor_failures(binary_iou: float, group_macro_f1: float) -> None:
    assert not run17_utility_eligible(binary_iou, group_macro_f1)
