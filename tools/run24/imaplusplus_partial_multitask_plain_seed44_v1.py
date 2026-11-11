"""Run24 delayed and utility-gated early stopping for the plain YOLO26m baseline.

This module intentionally changes only checkpoint/termination fitness handling.
The model, loss, data loader, validation metrics, and selection aliases are
inherited unchanged from the Run17 plain partial-label implementation.
"""

from __future__ import annotations

from ultralytics.utils.torch_utils import EarlyStopping

from imaplusplus_partial_multitask_utility_selection_v1 import (
    UtilityEligiblePartialLabelMultiTaskTrainer,
)


RUN24_EARLY_STOP_START_EPOCH = 40
RUN24_PATIENCE_AFTER_START = 30


def run24_early_stop_fitness(metrics: dict[str, float], original_fitness: float) -> float:
    """Rank utility-eligible epochs above every ineligible cold-start epoch."""
    if float(metrics.get("metrics/utility_eligible", 0.0)) == 1.0:
        return float(original_fitness)
    return float(metrics["metrics/joint_score"]) - 2.0


class DelayedEarlyStopping(EarlyStopping):
    """Ultralytics-compatible stopper whose patience clock starts at epoch 40."""

    def __init__(self, patience: int, start_epoch: int = RUN24_EARLY_STOP_START_EPOCH) -> None:
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


class DelayedUtilityGatedPartialLabelMultiTaskTrainer(UtilityEligiblePartialLabelMultiTaskTrainer):
    """Run17 plain trainer with only Run24 fitness/early-stop lifecycle changed."""

    def validate(self):
        prior_gated_best = self.best_fitness
        metrics, original_fitness = super().validate()
        if metrics is None or original_fitness is None:
            return metrics, original_fitness
        gated_fitness = run24_early_stop_fitness(metrics, original_fitness)
        metrics["metrics/original_fitness"] = float(original_fitness)
        metrics["metrics/early_stop_fitness"] = float(gated_fitness)

        # BaseTrainer.validate() updated best_fitness using the original score.
        # Recompute it in the gated score domain so best.pt follows Run24 policy.
        if prior_gated_best is None or gated_fitness > prior_gated_best:
            self.best_fitness = float(gated_fitness)
        else:
            self.best_fitness = float(prior_gated_best)
        return metrics, float(gated_fitness)

    def resume_training(self, ckpt):
        super().resume_training(ckpt)
        # _setup_train constructs the stock stopper immediately before this hook.
        saved = dict(vars(self.stopper))
        self.stopper = DelayedEarlyStopping(
            patience=RUN24_PATIENCE_AFTER_START,
            start_epoch=RUN24_EARLY_STOP_START_EPOCH,
        )
        for name, value in saved.items():
            if name in {"best_fitness", "best_epoch", "possible_stop"}:
                setattr(self.stopper, name, value)
        if int(getattr(self, "start_epoch", 0)) >= RUN24_EARLY_STOP_START_EPOCH:
            self.stopper.activated = True


__all__ = [
    "DelayedEarlyStopping",
    "DelayedUtilityGatedPartialLabelMultiTaskTrainer",
    "RUN24_EARLY_STOP_START_EPOCH",
    "RUN24_PATIENCE_AFTER_START",
    "run24_early_stop_fitness",
]
