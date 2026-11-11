"""Run20 adaptive-residual BLG-MDA under the matched Run18 lifecycle.

The sole architectural change from Run19 is replacing one fixed 0.5
lesion-guided pooling mixture with three learned, bounded per-scale gates.
Each gate starts at 0.05 and can learn within [0, 0.5].
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils import RANK
from ultralytics.utils.torch_utils import EarlyStopping

from imaplusplus_partial_multitask_lesion_attention_utility_selection_v1 import (
    UtilityEligiblePartialLabelMultiTaskValidator,
)
from imaplusplus_partial_multitask_lesion_attention_v1 import (
    BinaryLesionGuidedAttentionHead,
    BinaryLesionGuidedAttentionModel,
)
from imaplusplus_partial_multitask_v1 import CLASS_NAMES, ContractError, PartialLabelMultiTaskTrainer


RUN20_EARLY_STOP_START_EPOCH = 40
RUN20_PATIENCE_AFTER_START = 30
RUN20_INITIAL_MIX = 0.05
RUN20_MAX_MIX = 0.50


class AdaptiveResidualBinaryLesionGuidedAttentionHead(BinaryLesionGuidedAttentionHead):
    """BLG-MDA with one learnable residual mixture per P3/P4/P5 scale."""

    def __init__(
        self,
        *args,
        initial_mask_guided_mix: float = RUN20_INITIAL_MIX,
        max_mask_guided_mix: float = RUN20_MAX_MIX,
        **kwargs,
    ) -> None:
        if not 0.0 < initial_mask_guided_mix < max_mask_guided_mix <= 1.0:
            raise ContractError("Run20 requires 0 < initial_mask_guided_mix < max_mask_guided_mix <= 1")
        kwargs["mask_guided_mix"] = float(initial_mask_guided_mix)
        super().__init__(*args, **kwargs)
        self.initial_mask_guided_mix = float(initial_mask_guided_mix)
        self.max_mask_guided_mix = float(max_mask_guided_mix)
        initial_logit = math.log(initial_mask_guided_mix / (max_mask_guided_mix - initial_mask_guided_mix))
        self.mask_guided_mix_logits = nn.Parameter(
            torch.full((len(self.scale_channels),), float(initial_logit), dtype=torch.float32)
        )

    def current_mask_guided_mix(self) -> torch.Tensor:
        return self.max_mask_guided_mix * self.mask_guided_mix_logits.sigmoid()

    def _pool_scale(self, feature: torch.Tensor, foreground: torch.Tensor, scale_index: int) -> torch.Tensor:
        attended = self.channel_attention[scale_index](feature)
        global_pool = F.adaptive_avg_pool2d(attended, 1).flatten(1)
        mask = F.interpolate(foreground, size=attended.shape[-2:], mode="bilinear", align_corners=False)
        mask = mask.to(dtype=attended.dtype).clamp(0.0, 1.0)
        denominator = mask.sum(dim=(2, 3)).clamp_min(torch.finfo(attended.dtype).eps)
        guided_pool = (attended * mask).sum(dim=(2, 3)) / denominator
        mix = self.current_mask_guided_mix()[scale_index].to(dtype=attended.dtype)
        return global_pool + mix * (guided_pool - global_pool)


class AdaptiveResidualBinaryLesionGuidedAttentionModel(BinaryLesionGuidedAttentionModel):
    """Frozen Run19 model with only its pooling mixture parameterized adaptively."""

    def __init__(
        self,
        *args,
        initial_mask_guided_mix: float = RUN20_INITIAL_MIX,
        max_mask_guided_mix: float = RUN20_MAX_MIX,
        **kwargs,
    ) -> None:
        kwargs["mask_guided_mix"] = float(initial_mask_guided_mix)
        super().__init__(*args, **kwargs)
        old_head = self.model[-1]
        new_head = AdaptiveResidualBinaryLesionGuidedAttentionHead(
            nc=1,
            ch=old_head.scale_channels,
            num_diseases=old_head.image_classifier[-1].out_features,
            hidden_channels=old_head.image_classifier[0].out_features,
            eca_kernel_size=old_head.channel_attention[0].conv.kernel_size[0],
            initial_mask_guided_mix=initial_mask_guided_mix,
            max_mask_guided_mix=max_mask_guided_mix,
            detach_mask=old_head.detach_mask,
        )
        incompatible = new_head.load_state_dict(old_head.state_dict(), strict=False)
        if incompatible.unexpected_keys or incompatible.missing_keys != ["mask_guided_mix_logits"]:
            raise ContractError(
                f"Run20 head transfer mismatch: missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        new_head.f = list(old_head.f)
        new_head.i = old_head.i
        new_head.type = f"{type(new_head).__module__}.{type(new_head).__name__}"
        new_head.np = sum(parameter.numel() for parameter in new_head.parameters())
        new_head.stride = old_head.stride
        self.model[-1] = new_head
        self.lesion_attention_config.update(
            {
                "initial_mask_guided_mix": float(initial_mask_guided_mix),
                "max_mask_guided_mix": float(max_mask_guided_mix),
                "mix_parameterization": "per_scale_max_times_sigmoid",
            }
        )
        if kwargs.get("verbose", True):
            self.info()


class AdaptiveResidualUtilityTrainer(PartialLabelMultiTaskTrainer):
    """Adaptive attention model with the unchanged Run19 validator."""

    def __init__(self, *args, lesion_attention: dict | None = None, **kwargs) -> None:
        self.lesion_attention = dict(lesion_attention or {})
        super().__init__(*args, **kwargs)

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        model = AdaptiveResidualBinaryLesionGuidedAttentionModel(
            cfg,
            nc=1,
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            num_diseases=len(CLASS_NAMES),
            image_cls_weight=self.image_cls_weight,
            **self.lesion_attention,
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        from copy import copy

        self.loss_names = "bce_loss", "dice_loss", "aux_loss", "image_cls_loss"
        return UtilityEligiblePartialLabelMultiTaskValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
            selection=self.selection,
        )


def run20_early_stop_fitness(metrics: dict[str, float], original_fitness: float) -> float:
    if float(metrics.get("metrics/utility_eligible", 0.0)) == 1.0:
        return float(original_fitness)
    return float(metrics["metrics/joint_score"]) - 2.0


class DelayedEarlyStopping(EarlyStopping):
    def __init__(self, patience: int, start_epoch: int = RUN20_EARLY_STOP_START_EPOCH) -> None:
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


class DelayedUtilityGatedAdaptiveResidualAttentionTrainer(AdaptiveResidualUtilityTrainer):
    def validate(self):
        prior_gated_best = self.best_fitness
        metrics, original_fitness = super().validate()
        if metrics is None or original_fitness is None:
            return metrics, original_fitness
        gated_fitness = run20_early_stop_fitness(metrics, original_fitness)
        metrics["metrics/original_fitness"] = float(original_fitness)
        metrics["metrics/early_stop_fitness"] = float(gated_fitness)
        self.best_fitness = float(
            gated_fitness if prior_gated_best is None or gated_fitness > prior_gated_best else prior_gated_best
        )
        return metrics, float(gated_fitness)

    def resume_training(self, ckpt):
        super().resume_training(ckpt)
        saved = dict(vars(self.stopper))
        self.stopper = DelayedEarlyStopping(RUN20_PATIENCE_AFTER_START, RUN20_EARLY_STOP_START_EPOCH)
        for name, value in saved.items():
            if name in {"best_fitness", "best_epoch", "possible_stop"}:
                setattr(self.stopper, name, value)
        if int(getattr(self, "start_epoch", 0)) >= RUN20_EARLY_STOP_START_EPOCH:
            self.stopper.activated = True


__all__ = [
    "AdaptiveResidualBinaryLesionGuidedAttentionHead",
    "AdaptiveResidualBinaryLesionGuidedAttentionModel",
    "DelayedEarlyStopping",
    "DelayedUtilityGatedAdaptiveResidualAttentionTrainer",
    "RUN20_INITIAL_MIX",
    "RUN20_MAX_MIX",
    "run20_early_stop_fitness",
]
