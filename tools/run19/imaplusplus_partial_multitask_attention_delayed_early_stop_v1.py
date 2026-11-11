"""Run19 BLG-MDA trainer with the Run18 delayed utility-gated lifecycle.

Relative to Run18, this module changes only the image-classification head by
restoring the frozen Run16 binary-lesion-guided attention implementation.
Early stopping, validation, loss, data loading, and checkpoint behavior remain
the matched Run18 protocol.
"""

from __future__ import annotations

from ultralytics.utils.torch_utils import EarlyStopping

from imaplusplus_partial_multitask_lesion_attention_utility_selection_v1 import (
    UtilityEligibleBinaryLesionGuidedAttentionTrainer,
)


RUN19_EARLY_STOP_START_EPOCH = 40
RUN19_PATIENCE_AFTER_START = 30


def run19_early_stop_fitness(metrics: dict[str, float], original_fitness: float) -> float:
    """Use exactly the Run18 separated score domains."""
    if float(metrics.get("metrics/utility_eligible", 0.0)) == 1.0:
        return float(original_fitness)
    return float(metrics["metrics/joint_score"]) - 2.0


class DelayedEarlyStopping(EarlyStopping):
    """Ultralytics-compatible stopper whose patience clock starts at epoch 40."""

    def __init__(self, patience: int, start_epoch: int = RUN19_EARLY_STOP_START_EPOCH) -> None:
        super().__init__(patience=patience)
        self.start_epoch = int(start_epoch)
        self.activated = False

    def __call__(self, epoch, fitness):
        epoch = int(epoch)
        if epoch < self.start_epoch:
            self.possible_stop = False
            return False
        if not self.activated:
            self.best_fitness = 0.0
            self.best_epoch = epoch
            self.possible_stop = False
            self.activated = True
        return super().__call__(epoch, fitness)


class DelayedUtilityGatedBinaryLesionGuidedAttentionTrainer(
    UtilityEligibleBinaryLesionGuidedAttentionTrainer
):
    """Frozen Run16 attention architecture under the matched Run18 lifecycle."""

    def validate(self):
        prior_gated_best = self.best_fitness
        metrics, original_fitness = super().validate()
        if metrics is None or original_fitness is None:
            return metrics, original_fitness
        gated_fitness = run19_early_stop_fitness(metrics, original_fitness)
        metrics["metrics/original_fitness"] = float(original_fitness)
        metrics["metrics/early_stop_fitness"] = float(gated_fitness)
        self.best_fitness = float(
            gated_fitness if prior_gated_best is None or gated_fitness > prior_gated_best else prior_gated_best
        )
        return metrics, float(gated_fitness)

    def resume_training(self, ckpt):
        super().resume_training(ckpt)
        saved = dict(vars(self.stopper))
        self.stopper = DelayedEarlyStopping(
            patience=RUN19_PATIENCE_AFTER_START,
            start_epoch=RUN19_EARLY_STOP_START_EPOCH,
        )
        for name, value in saved.items():
            if name in {"best_fitness", "best_epoch", "possible_stop"}:
                setattr(self.stopper, name, value)
        if int(getattr(self, "start_epoch", 0)) >= RUN19_EARLY_STOP_START_EPOCH:
            self.stopper.activated = True


__all__ = [
    "DelayedEarlyStopping",
    "DelayedUtilityGatedBinaryLesionGuidedAttentionTrainer",
    "RUN19_EARLY_STOP_START_EPOCH",
    "RUN19_PATIENCE_AFTER_START",
    "run19_early_stop_fitness",
]
