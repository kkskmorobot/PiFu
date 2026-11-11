"""Plain partial-label multitask trainer with Run16 utility eligibility floors.

This module changes checkpoint eligibility reporting only.  The training model,
loss, sampling, and validation fitness remain the frozen plain multitask v1
implementation.
"""

from __future__ import annotations

from copy import copy

from imaplusplus_partial_multitask_v1 import (
    ContractError,
    PartialLabelMultiTaskTrainer,
    PartialLabelMultiTaskValidator,
)


RUN17_MIN_BINARY_IOU = 0.70
RUN17_MIN_GROUP_MACRO_F1 = 0.50


def run17_utility_eligible(binary_iou: float, group_macro_f1: float, min_binary_iou: float = RUN17_MIN_BINARY_IOU, min_group_macro_f1: float = RUN17_MIN_GROUP_MACRO_F1) -> bool:
    return float(binary_iou) >= min_binary_iou and float(group_macro_f1) >= min_group_macro_f1


class UtilityEligiblePartialLabelMultiTaskValidator(PartialLabelMultiTaskValidator):
    """Reject cold-start false feasibility without changing fitness."""

    def __init__(self, *args, selection: dict[str, float] | None = None, **kwargs) -> None:
        selection = dict(selection or {})
        self.min_binary_iou = float(selection.get("min_binary_iou", RUN17_MIN_BINARY_IOU))
        self.min_group_macro_f1 = float(selection.get("min_group_macro_f1", RUN17_MIN_GROUP_MACRO_F1))
        if not (0.0 <= self.min_binary_iou <= 1.0 and 0.0 <= self.min_group_macro_f1 <= 1.0):
            raise ContractError("Utility eligibility floors must be finite values in [0, 1]")
        super().__init__(*args, selection=selection, **kwargs)

    def get_stats(self):
        stats = super().get_stats()
        original_fitness = float(stats["fitness"])
        risk_feasible = float(stats["metrics/feasible_joint"]) == 1.0
        utility_eligible = run17_utility_eligible(
            stats["metrics/binary_lesion_iou"], stats["metrics/group_macro_f1"],
            self.min_binary_iou, self.min_group_macro_f1
        )
        stats["metrics/risk_feasible"] = float(risk_feasible)
        stats["metrics/utility_eligible"] = float(utility_eligible)
        stats["metrics/feasible_joint"] = float(risk_feasible and utility_eligible)
        if float(stats["fitness"]) != original_fitness:
            raise ContractError("Run17 utility eligibility changed validation fitness")
        return stats


class UtilityEligiblePartialLabelMultiTaskTrainer(PartialLabelMultiTaskTrainer):
    """Plain baseline training behavior with corrected val-only eligibility."""

    def get_validator(self):
        self.loss_names = "bce_loss", "dice_loss", "aux_loss", "image_cls_loss"
        return UtilityEligiblePartialLabelMultiTaskValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
            selection=self.selection,
        )


__all__ = [
    "RUN17_MIN_BINARY_IOU",
    "RUN17_MIN_GROUP_MACRO_F1",
    "UtilityEligiblePartialLabelMultiTaskTrainer",
    "UtilityEligiblePartialLabelMultiTaskValidator",
    "run17_utility_eligible",
]
