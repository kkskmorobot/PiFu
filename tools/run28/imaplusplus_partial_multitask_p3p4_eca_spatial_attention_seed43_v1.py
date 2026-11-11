"""Run28 identity-initialized P3/P4 ECA plus spatial attention.

The sole architectural change from Run23 is adding one CBAM-style spatial gate
after each existing identity-initialized ECA gate. Each spatial gate uses the
channel mean and maximum, a 7x7 convolution, and ``2 * sigmoid`` scaling. Its
convolution starts at zero, so the complete Run28 head is exactly identical to
Run23 at initialization. The semantic branch still sees untouched P3/P4 maps.
"""

from __future__ import annotations

from copy import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils import RANK
from ultralytics.utils.torch_utils import EarlyStopping
from ultralytics.nn.modules import SemanticSegment

from imaplusplus_partial_multitask_lesion_attention_v1 import IdentityInitializedECA
from imaplusplus_partial_multitask_utility_selection_v1 import (
    UtilityEligiblePartialLabelMultiTaskValidator,
)
from imaplusplus_partial_multitask_v1 import (
    CLASS_NAMES,
    BinaryPartialClassificationHead,
    BinaryPartialMultiTaskModel,
    ContractError,
    PartialLabelMultiTaskTrainer,
)


RUN28_EARLY_STOP_START_EPOCH = 40
RUN28_PATIENCE_AFTER_START = 30
RUN28_ECA_KERNEL_SIZE = 5
RUN28_SPATIAL_KERNEL_SIZE = 7


class IdentityInitializedSpatialAttention(nn.Module):
    """CBAM-style spatial gate with an exact identity initial response."""

    def __init__(self, kernel_size: int = RUN28_SPATIAL_KERNEL_SIZE) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ContractError("Run28 spatial kernel size must be a positive odd integer")
        self.kernel_size = int(kernel_size)
        self.conv = nn.Conv2d(2, 1, self.kernel_size, padding=self.kernel_size // 2, bias=False)
        nn.init.zeros_(self.conv.weight)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        descriptor = torch.cat(
            (feature.mean(dim=1, keepdim=True), feature.amax(dim=1, keepdim=True)), dim=1
        )
        gate = 2.0 * torch.sigmoid(self.conv(descriptor))
        return feature * gate


class P3P4ECASpatialClassificationHead(BinaryPartialClassificationHead):
    """Run23 P3/P4 ECA classifier plus classifier-only spatial attention."""

    def __init__(
        self,
        nc: int,
        ch: tuple[int, int],
        num_diseases: int = 8,
        hidden_channels: int = 128,
        eca_kernel_size: int = RUN28_ECA_KERNEL_SIZE,
        spatial_kernel_size: int = RUN28_SPATIAL_KERNEL_SIZE,
    ) -> None:
        if len(ch) != 2:
            raise ContractError(f"Run28 requires exactly P3/P4 channels, found {ch}")
        super().__init__(
            nc=nc,
            ch=ch,
            num_diseases=num_diseases,
            hidden_channels=hidden_channels,
        )
        self.scale_channels = tuple(int(value) for value in ch)
        self.eca_kernel_size = int(eca_kernel_size)
        self.spatial_kernel_size = int(spatial_kernel_size)
        self.channel_attention = nn.ModuleList(
            [IdentityInitializedECA(kernel_size=self.eca_kernel_size) for _ in self.scale_channels]
        )
        self.spatial_attention = nn.ModuleList(
            [IdentityInitializedSpatialAttention(self.spatial_kernel_size) for _ in self.scale_channels]
        )

    def forward(self, x: list[torch.Tensor]):
        if len(x) != 2:
            raise ContractError(f"Run28 head expected [P3, P4], received {len(x)} maps")
        channel_attended = [module(feature) for module, feature in zip(self.channel_attention, x)]
        attended = [module(feature) for module, feature in zip(self.spatial_attention, channel_attended)]
        pooled = torch.cat([F.adaptive_avg_pool2d(feature, 1).flatten(1) for feature in attended], dim=1)
        image_logits = self.image_classifier(pooled)
        # Semantic prediction deliberately sees the original, unmodified P3/P4 maps.
        semantic = SemanticSegment.forward(self, x)
        if self.training:
            if isinstance(semantic, tuple):
                return semantic[0], semantic[1], image_logits
            return semantic, None, image_logits
        return semantic, image_logits


class P3P4ECASpatialMultiTaskModel(BinaryPartialMultiTaskModel):
    """Frozen Run23 model with only two identity-initialized spatial gates added."""

    def __init__(
        self,
        *args,
        eca_kernel_size: int = RUN28_ECA_KERNEL_SIZE,
        spatial_kernel_size: int = RUN28_SPATIAL_KERNEL_SIZE,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        old_head = self.model[-1]
        if not isinstance(old_head, BinaryPartialClassificationHead):
            raise ContractError(f"Expected Run18 plain head, found {type(old_head).__name__}")
        if list(old_head.f) != [16, 13]:
            raise ContractError(f"Run28 is locked to YOLO26-sem P3/P4 inputs [16, 13], found {old_head.f}")
        p3_channels = old_head.image_classifier[0].in_features - old_head.aux_head[0].conv.in_channels
        p4_channels = old_head.aux_head[0].conv.in_channels
        new_head = P3P4ECASpatialClassificationHead(
            nc=1,
            ch=(p3_channels, p4_channels),
            num_diseases=old_head.image_classifier[-1].out_features,
            hidden_channels=old_head.image_classifier[0].out_features,
            eca_kernel_size=eca_kernel_size,
            spatial_kernel_size=spatial_kernel_size,
        )
        incompatible = new_head.load_state_dict(old_head.state_dict(), strict=False)
        expected_missing = [
            f"channel_attention.{index}.conv.weight" for index in range(len(new_head.channel_attention))
        ] + [f"spatial_attention.{index}.conv.weight" for index in range(len(new_head.spatial_attention))]
        if incompatible.unexpected_keys or incompatible.missing_keys != expected_missing:
            raise ContractError(
                f"Run28 head transfer mismatch: missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        for attr in ("f", "i", "type"):
            setattr(new_head, attr, getattr(old_head, attr))
        new_head.type = f"{type(new_head).__module__}.{type(new_head).__name__}"
        new_head.np = sum(parameter.numel() for parameter in new_head.parameters())
        new_head.stride = old_head.stride
        self.model[-1] = new_head
        self.spatial_channel_attention_config = {
            "scales": ["P3", "P4"],
            "eca_kernel_size": int(eca_kernel_size),
            "pooling": "global_average",
            "spatial_attention": "identity_initialized_cbam_avg_max",
            "spatial_kernel_size": int(spatial_kernel_size),
            "p5_descriptor": False,
            "semantic_branch_modified": False,
        }
        if kwargs.get("verbose", True):
            self.info()


class P3P4ECASpatialUtilityTrainer(PartialLabelMultiTaskTrainer):
    """Run28 model with the unchanged Run18 utility validator."""

    def __init__(self, *args, spatial_channel_attention: dict | None = None, **kwargs) -> None:
        self.spatial_channel_attention = dict(spatial_channel_attention or {})
        super().__init__(*args, **kwargs)

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        model = P3P4ECASpatialMultiTaskModel(
            cfg,
            nc=1,
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            num_diseases=len(CLASS_NAMES),
            image_cls_weight=self.image_cls_weight,
            **self.spatial_channel_attention,
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        self.loss_names = "bce_loss", "dice_loss", "aux_loss", "image_cls_loss"
        return UtilityEligiblePartialLabelMultiTaskValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
            selection=self.selection,
        )


def run28_early_stop_fitness(metrics: dict[str, float], original_fitness: float) -> float:
    if float(metrics.get("metrics/utility_eligible", 0.0)) == 1.0:
        return float(original_fitness)
    return float(metrics["metrics/joint_score"]) - 2.0


class DelayedEarlyStopping(EarlyStopping):
    def __init__(self, patience: int, start_epoch: int = RUN28_EARLY_STOP_START_EPOCH) -> None:
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


class DelayedUtilityGatedP3P4ECASpatialTrainer(P3P4ECASpatialUtilityTrainer):
    def validate(self):
        prior_gated_best = self.best_fitness
        metrics, original_fitness = super().validate()
        if metrics is None or original_fitness is None:
            return metrics, original_fitness
        gated_fitness = run28_early_stop_fitness(metrics, original_fitness)
        metrics["metrics/original_fitness"] = float(original_fitness)
        metrics["metrics/early_stop_fitness"] = float(gated_fitness)
        self.best_fitness = float(
            gated_fitness if prior_gated_best is None or gated_fitness > prior_gated_best else prior_gated_best
        )
        return metrics, float(gated_fitness)

    def resume_training(self, ckpt):
        super().resume_training(ckpt)
        saved = dict(vars(self.stopper))
        self.stopper = DelayedEarlyStopping(RUN28_PATIENCE_AFTER_START, RUN28_EARLY_STOP_START_EPOCH)
        for name, value in saved.items():
            if name in {"best_fitness", "best_epoch", "possible_stop"}:
                setattr(self.stopper, name, value)
        if int(getattr(self, "start_epoch", 0)) >= RUN28_EARLY_STOP_START_EPOCH:
            self.stopper.activated = True


__all__ = [
    "DelayedEarlyStopping",
    "DelayedUtilityGatedP3P4ECASpatialTrainer",
    "IdentityInitializedSpatialAttention",
    "P3P4ECASpatialClassificationHead",
    "P3P4ECASpatialMultiTaskModel",
    "RUN28_EARLY_STOP_START_EPOCH",
    "RUN28_ECA_KERNEL_SIZE",
    "RUN28_PATIENCE_AFTER_START",
    "RUN28_SPATIAL_KERNEL_SIZE",
    "run28_early_stop_fitness",
]


