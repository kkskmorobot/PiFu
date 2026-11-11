"""Run33 conventional P3/P4 CBAM attention on the YOLO26m image head.

Relative to the Run22 seed43 plain baseline, the sole architectural factor is
one standard CBAM block on each P3/P4 feature used by the image-classification
head. Each block uses the conventional shared avg/max channel MLP followed by
a 7x7 avg/max spatial gate. Both gates use sigmoid scaling. The semantic branch
continues to consume the original, unmodified P3/P4 feature maps.
"""

from __future__ import annotations

from copy import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import SemanticSegment
from ultralytics.utils import RANK
from ultralytics.utils.torch_utils import EarlyStopping

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


RUN33_EARLY_STOP_START_EPOCH = 40
RUN33_PATIENCE_AFTER_START = 30
RUN33_CBAM_REDUCTION = 16
RUN33_SPATIAL_KERNEL_SIZE = 7


class StandardCBAMChannelAttention(nn.Module):
    """Standard CBAM channel attention with a shared avg/max-pool MLP."""

    def __init__(self, channels: int, reduction: int = RUN33_CBAM_REDUCTION) -> None:
        super().__init__()
        channels = int(channels)
        reduction = int(reduction)
        if channels <= 0 or reduction <= 0:
            raise ContractError("Run33 CBAM channels and reduction must be positive")
        self.channels = channels
        self.reduction = reduction
        hidden = max(channels // reduction, 1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        average = self.shared_mlp(F.adaptive_avg_pool2d(feature, 1))
        maximum = self.shared_mlp(feature.amax(dim=(2, 3), keepdim=True))
        return feature * torch.sigmoid(average + maximum)


class StandardCBAMSpatialAttention(nn.Module):
    """Standard CBAM spatial attention over channel mean/max maps."""

    def __init__(self, kernel_size: int = RUN33_SPATIAL_KERNEL_SIZE) -> None:
        super().__init__()
        kernel_size = int(kernel_size)
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ContractError("Run33 CBAM spatial kernel must be a positive odd integer")
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        descriptors = torch.cat(
            (feature.mean(dim=1, keepdim=True), feature.amax(dim=1, keepdim=True)), dim=1
        )
        return feature * torch.sigmoid(self.conv(descriptors))


class StandardCBAM(nn.Module):
    """Convolutional Block Attention Module: channel attention then spatial attention."""

    def __init__(
        self,
        channels: int,
        reduction: int = RUN33_CBAM_REDUCTION,
        spatial_kernel_size: int = RUN33_SPATIAL_KERNEL_SIZE,
    ) -> None:
        super().__init__()
        self.channel_attention = StandardCBAMChannelAttention(channels, reduction)
        self.spatial_attention = StandardCBAMSpatialAttention(spatial_kernel_size)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.spatial_attention(self.channel_attention(feature))


class P3P4StandardCBAMClassificationHead(BinaryPartialClassificationHead):
    """Run22 P3/P4 image classifier with one conventional CBAM per scale."""

    def __init__(
        self,
        nc: int,
        ch: tuple[int, int],
        num_diseases: int = 8,
        hidden_channels: int = 128,
        reduction: int = RUN33_CBAM_REDUCTION,
        spatial_kernel_size: int = RUN33_SPATIAL_KERNEL_SIZE,
    ) -> None:
        if len(ch) != 2:
            raise ContractError(f"Run33 requires exactly P3/P4 channels, found {ch}")
        super().__init__(nc=nc, ch=ch, num_diseases=num_diseases, hidden_channels=hidden_channels)
        self.scale_channels = tuple(int(value) for value in ch)
        self.cbam_reduction = int(reduction)
        self.spatial_kernel_size = int(spatial_kernel_size)
        self.cbam = nn.ModuleList(
            [
                StandardCBAM(
                    channels,
                    reduction=self.cbam_reduction,
                    spatial_kernel_size=self.spatial_kernel_size,
                )
                for channels in self.scale_channels
            ]
        )

    def forward(self, x: list[torch.Tensor]):
        if len(x) != 2:
            raise ContractError(f"Run33 head expected [P3, P4], received {len(x)} maps")
        attended = [module(feature) for module, feature in zip(self.cbam, x)]
        pooled = torch.cat(
            [F.adaptive_avg_pool2d(feature, 1).flatten(1) for feature in attended], dim=1
        )
        image_logits = self.image_classifier(pooled)
        semantic = SemanticSegment.forward(self, x)
        if self.training:
            if isinstance(semantic, tuple):
                return semantic[0], semantic[1], image_logits
            return semantic, None, image_logits
        return semantic, image_logits


class P3P4StandardCBAMMultiTaskModel(BinaryPartialMultiTaskModel):
    """Frozen Run22 model with conventional classifier-only P3/P4 CBAM blocks."""

    def __init__(
        self,
        *args,
        reduction: int = RUN33_CBAM_REDUCTION,
        spatial_kernel_size: int = RUN33_SPATIAL_KERNEL_SIZE,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        old_head = self.model[-1]
        if not isinstance(old_head, BinaryPartialClassificationHead):
            raise ContractError(f"Expected Run22 plain head, found {type(old_head).__name__}")
        if list(old_head.f) != [16, 13]:
            raise ContractError(f"Run33 is locked to YOLO26-sem P3/P4 inputs [16, 13], found {old_head.f}")
        p3_channels = old_head.image_classifier[0].in_features - old_head.aux_head[0].conv.in_channels
        p4_channels = old_head.aux_head[0].conv.in_channels
        new_head = P3P4StandardCBAMClassificationHead(
            nc=1,
            ch=(p3_channels, p4_channels),
            num_diseases=old_head.image_classifier[-1].out_features,
            hidden_channels=old_head.image_classifier[0].out_features,
            reduction=reduction,
            spatial_kernel_size=spatial_kernel_size,
        )
        incompatible = new_head.load_state_dict(old_head.state_dict(), strict=False)
        expected_missing = sorted(
            [
                f"cbam.{index}.channel_attention.shared_mlp.{layer}.weight"
                for index in range(len(new_head.cbam))
                for layer in (0, 2)
            ]
            + [f"cbam.{index}.spatial_attention.conv.weight" for index in range(len(new_head.cbam))]
        )
        if incompatible.unexpected_keys or sorted(incompatible.missing_keys) != expected_missing:
            raise ContractError(
                f"Run33 head transfer mismatch: missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        for attr in ("f", "i", "type"):
            setattr(new_head, attr, getattr(old_head, attr))
        new_head.type = f"{type(new_head).__module__}.{type(new_head).__name__}"
        new_head.np = sum(parameter.numel() for parameter in new_head.parameters())
        new_head.stride = old_head.stride
        self.model[-1] = new_head
        self.standard_cbam_config = {
            "scales": ["P3", "P4"],
            "channel_pooling": ["global_average", "global_maximum"],
            "shared_mlp_reduction": int(reduction),
            "spatial_pooling": ["channel_mean", "channel_maximum"],
            "spatial_kernel_size": int(spatial_kernel_size),
            "gate_scaling": "sigmoid",
            "pooling_after_attention": "global_average",
            "p5_descriptor": False,
            "semantic_branch_modified": False,
        }
        if kwargs.get("verbose", True):
            self.info()


class P3P4StandardCBAMUtilityTrainer(PartialLabelMultiTaskTrainer):
    """Run33 CBAM model with the unchanged Run22 utility validator."""

    def __init__(self, *args, cbam_attention: dict | None = None, **kwargs) -> None:
        self.cbam_attention = dict(cbam_attention or {})
        super().__init__(*args, **kwargs)

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        model = P3P4StandardCBAMMultiTaskModel(
            cfg,
            nc=1,
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            num_diseases=len(CLASS_NAMES),
            image_cls_weight=self.image_cls_weight,
            **self.cbam_attention,
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


def run33_early_stop_fitness(metrics: dict[str, float], original_fitness: float) -> float:
    if float(metrics.get("metrics/utility_eligible", 0.0)) == 1.0:
        return float(original_fitness)
    return float(metrics["metrics/joint_score"]) - 2.0


class DelayedEarlyStopping(EarlyStopping):
    def __init__(self, patience: int, start_epoch: int = RUN33_EARLY_STOP_START_EPOCH) -> None:
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


class DelayedUtilityGatedP3P4StandardCBAMTrainer(P3P4StandardCBAMUtilityTrainer):
    def validate(self):
        prior_gated_best = self.best_fitness
        metrics, original_fitness = super().validate()
        if metrics is None or original_fitness is None:
            return metrics, original_fitness
        gated_fitness = run33_early_stop_fitness(metrics, original_fitness)
        metrics["metrics/original_fitness"] = float(original_fitness)
        metrics["metrics/early_stop_fitness"] = float(gated_fitness)
        self.best_fitness = float(
            gated_fitness
            if prior_gated_best is None or gated_fitness > prior_gated_best
            else prior_gated_best
        )
        return metrics, float(gated_fitness)

    def resume_training(self, ckpt):
        super().resume_training(ckpt)
        saved = dict(vars(self.stopper))
        self.stopper = DelayedEarlyStopping(RUN33_PATIENCE_AFTER_START, RUN33_EARLY_STOP_START_EPOCH)
        for name, value in saved.items():
            if name in {"best_fitness", "best_epoch", "possible_stop"}:
                setattr(self.stopper, name, value)
        if int(getattr(self, "start_epoch", 0)) >= RUN33_EARLY_STOP_START_EPOCH:
            self.stopper.activated = True


__all__ = [
    "DelayedEarlyStopping",
    "DelayedUtilityGatedP3P4StandardCBAMTrainer",
    "P3P4StandardCBAMClassificationHead",
    "P3P4StandardCBAMMultiTaskModel",
    "RUN33_CBAM_REDUCTION",
    "RUN33_EARLY_STOP_START_EPOCH",
    "RUN33_PATIENCE_AFTER_START",
    "RUN33_SPATIAL_KERNEL_SIZE",
    "StandardCBAM",
    "StandardCBAMChannelAttention",
    "StandardCBAMSpatialAttention",
    "run33_early_stop_fitness",
]
