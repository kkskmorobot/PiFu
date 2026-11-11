"""Run16 BLG-MDA trainer with non-degenerate val checkpoint eligibility.

Training architecture and loss are exactly the frozen Run12 implementation.
This module only carries forward the Run15 reporting/alias correction that
rejects cold-start checkpoints with weak segmentation or classification
utility. The underlying validation fitness is left unchanged.
"""

from __future__ import annotations

from copy import copy

from imaplusplus_partial_multitask_lesion_attention_v1 import (
    BinaryLesionGuidedAttentionHead,
    BinaryLesionGuidedAttentionModel,
    BinaryLesionGuidedAttentionTrainer,
    IdentityInitializedECA,
    PartialLabelSemanticDataset,
)
from imaplusplus_partial_multitask_v1 import ContractError, PartialLabelMultiTaskValidator


RUN16_MIN_BINARY_IOU = 0.70
RUN16_MIN_GROUP_MACRO_F1 = 0.50


def run16_utility_eligible(binary_iou: float, group_macro_f1: float) -> bool:
    return float(binary_iou) >= RUN16_MIN_BINARY_IOU and float(group_macro_f1) >= RUN16_MIN_GROUP_MACRO_F1


class UtilityEligiblePartialLabelMultiTaskValidator(PartialLabelMultiTaskValidator):
    """Reject cold-start false feasibility without changing fitness."""

    def __init__(self, *args, selection: dict[str, float] | None = None, **kwargs) -> None:
        selection = dict(selection or {})
        self.min_binary_iou = float(selection.get("min_binary_iou", RUN16_MIN_BINARY_IOU))
        self.min_group_macro_f1 = float(selection.get("min_group_macro_f1", RUN16_MIN_GROUP_MACRO_F1))
        if self.min_binary_iou != RUN16_MIN_BINARY_IOU or self.min_group_macro_f1 != RUN16_MIN_GROUP_MACRO_F1:
            raise ContractError("Run16 utility eligibility floors drifted")
        super().__init__(*args, selection=selection, **kwargs)

    def get_stats(self):
        stats = super().get_stats()
        original_fitness = float(stats["fitness"])
        risk_feasible = float(stats["metrics/feasible_joint"]) == 1.0
        utility_eligible = run16_utility_eligible(
            stats["metrics/binary_lesion_iou"], stats["metrics/group_macro_f1"]
        )
        stats["metrics/risk_feasible"] = float(risk_feasible)
        stats["metrics/utility_eligible"] = float(utility_eligible)
        stats["metrics/feasible_joint"] = float(risk_feasible and utility_eligible)
        if float(stats["fitness"]) != original_fitness:
            raise ContractError("Run16 utility eligibility changed validation fitness")
        return stats


class UtilityEligibleBinaryLesionGuidedAttentionTrainer(BinaryLesionGuidedAttentionTrainer):
    """Run12 training behavior with the corrected val-only eligibility rule."""

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
    "BinaryLesionGuidedAttentionHead",
    "BinaryLesionGuidedAttentionModel",
    "IdentityInitializedECA",
    "PartialLabelSemanticDataset",
    "RUN16_MIN_BINARY_IOU",
    "RUN16_MIN_GROUP_MACRO_F1",
    "UtilityEligibleBinaryLesionGuidedAttentionTrainer",
    "UtilityEligiblePartialLabelMultiTaskValidator",
    "run16_utility_eligible",
]
