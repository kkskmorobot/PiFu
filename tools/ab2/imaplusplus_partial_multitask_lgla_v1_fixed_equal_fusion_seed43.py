"""AB2: fixed-equal-fusion ablation of the locked gate-balanced LGLA v1.

The predicted-lesion spatial prior, P3/P4 descriptors, shared projections,
classifier, optimizer, validation, and checkpoint contracts remain unchanged.
The only experimental factor is that the learned two-way fusion gate is not
used: global and lesion-local descriptors are always fused with weights
``0.5/0.5``.  The gate module remains registered and identically initialized
to preserve full-LGLA parameter parity, but it receives no gradient and the
gate-balance regularizer is disabled because there are no learned gate logits.
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
    PartialLabelMultiTaskLoss,
    PartialLabelMultiTaskTrainer,
)


AB2_EARLY_STOP_START_EPOCH = 40
AB2_PATIENCE_AFTER_START = 30
AB2_PROJECTION_DIM = 128
AB2_GATE_HIDDEN_DIM = 128
AB2_ATTENTION_EPS = 1.0e-6
AB2_MIN_ATTENTION_MASS_FRACTION = 1.0e-4
AB2_GATE_BALANCE_WEIGHT = 0.0


class LesionGuidedFixedEqualFusionClassificationHead(BinaryPartialClassificationHead):
    """P3/P4 lesion-guided classifier with fixed 0.5/0.5 fusion."""

    def __init__(
        self,
        nc: int,
        ch: tuple[int, int],
        num_diseases: int = 8,
        hidden_channels: int = 128,
        projection_dim: int = AB2_PROJECTION_DIM,
        gate_hidden_dim: int = AB2_GATE_HIDDEN_DIM,
        attention_eps: float = AB2_ATTENTION_EPS,
        min_attention_mass_fraction: float = AB2_MIN_ATTENTION_MASS_FRACTION,
    ) -> None:
        if len(ch) != 2:
            raise ContractError(f"ab2 requires exactly P3/P4 channels, found {ch}")
        if projection_dim <= 0 or gate_hidden_dim <= 0:
            raise ContractError("ab2 projection and gate hidden dimensions must be positive")
        if attention_eps <= 0.0:
            raise ContractError("ab2 attention epsilon must be positive")
        if not 0.0 <= min_attention_mass_fraction < 1.0:
            raise ContractError("ab2 minimum attention mass fraction must be in [0, 1)")
        super().__init__(
            nc=nc,
            ch=ch,
            num_diseases=num_diseases,
            hidden_channels=hidden_channels,
        )
        self.scale_channels = tuple(int(value) for value in ch)
        self.projection_dim = int(projection_dim)
        self.gate_hidden_dim = int(gate_hidden_dim)
        self.attention_eps = float(attention_eps)
        self.min_attention_mass_fraction = float(min_attention_mass_fraction)
        self.scale_projections = nn.ModuleList(
            [nn.Linear(channels, self.projection_dim) for channels in self.scale_channels]
        )
        descriptor_dim = len(self.scale_channels) * self.projection_dim
        self.global_local_gate = nn.Sequential(
            nn.Linear(2 * descriptor_dim, self.gate_hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.gate_hidden_dim, 2),
        )
        # Retain the exact full-LGLA gate module and initialization for parameter
        # parity. AB2 deliberately does not call this module in forward().
        nn.init.zeros_(self.global_local_gate[-1].weight)
        nn.init.zeros_(self.global_local_gate[-1].bias)
        self.image_classifier = nn.Sequential(
            nn.Linear(descriptor_dim, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.20),
            nn.Linear(hidden_channels, num_diseases),
        )
        self.last_gate_weights: torch.Tensor | None = None
        self.last_attention_mass_fraction: torch.Tensor | None = None
        self.last_attention_fallback: torch.Tensor | None = None
        self.fixed_fusion_weights = (0.5, 0.5)

    def _global_local_pool(
        self, feature: torch.Tensor, lesion_probability: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return stable global/local vectors plus attention mass and fallback flags."""
        attention = F.interpolate(
            lesion_probability,
            size=feature.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp_(0.0, 1.0)
        feature_fp32 = feature.float()
        global_vector = feature_fp32.mean(dim=(2, 3))
        mass = attention.sum(dim=(2, 3))
        local_vector = (feature_fp32 * attention).sum(dim=(2, 3)) / mass.clamp_min(
            self.attention_eps
        )
        spatial_area = float(feature.shape[-2] * feature.shape[-1])
        mass_fraction = mass.squeeze(1) / spatial_area
        fallback = mass_fraction <= self.min_attention_mass_fraction
        local_vector = torch.where(fallback[:, None], global_vector, local_vector)
        return global_vector, local_vector, mass_fraction, fallback

    def forward(self, x: list[torch.Tensor]):
        if len(x) != 2:
            raise ContractError(f"ab2 head expected [P3, P4], received {len(x)} maps")

        # Preserve the original semantic computation and output contract.
        semantic = SemanticSegment.forward(self, x)
        semantic_main = semantic[0] if isinstance(semantic, tuple) else semantic
        lesion_probability = torch.sigmoid(semantic_main.detach().float())

        global_parts: list[torch.Tensor] = []
        local_parts: list[torch.Tensor] = []
        mass_fractions: list[torch.Tensor] = []
        fallbacks: list[torch.Tensor] = []
        for projection, feature in zip(self.scale_projections, x):
            global_raw, local_raw, mass_fraction, fallback = self._global_local_pool(
                feature, lesion_probability
            )
            # AutoBackend converts the final EMA model to FP16 for CUDA
            # validation outside autocast. Keep pooling accumulation in FP32,
            # then match each projection's runtime weight dtype explicitly.
            projection_dtype = projection.weight.dtype
            global_parts.append(projection(global_raw.to(dtype=projection_dtype)))
            local_parts.append(projection(local_raw.to(dtype=projection_dtype)))
            mass_fractions.append(mass_fraction)
            fallbacks.append(fallback)

        global_descriptor = torch.cat(global_parts, dim=1)
        local_descriptor = torch.cat(local_parts, dim=1)
        gate_weights = torch.full(
            (global_descriptor.shape[0], 2),
            0.5,
            device=global_descriptor.device,
            dtype=global_descriptor.dtype,
        )
        fused = (
            gate_weights[:, 0:1] * global_descriptor
            + gate_weights[:, 1:2] * local_descriptor
        )
        image_logits = self.image_classifier(fused)

        # Diagnostics are detached and overwritten every forward, so they are
        # recordable by future validators without retaining an autograd graph.
        self.last_gate_weights = gate_weights.detach()
        self.last_attention_mass_fraction = torch.stack(mass_fractions, dim=1).detach()
        self.last_attention_fallback = torch.stack(fallbacks, dim=1).detach()
        if self.training:
            if isinstance(semantic, tuple):
                return semantic[0], semantic[1], image_logits
            return semantic, None, image_logits
        return semantic, image_logits


class FixedEqualFusionPartialLabelMultiTaskLoss(PartialLabelMultiTaskLoss):
    """Original partial-label loss with no gate regularizer."""

    def __init__(self, model: nn.Module):
        super().__init__(model)
        head = model.model[-1]
        if not isinstance(head, LesionGuidedFixedEqualFusionClassificationHead):
            raise ContractError(f"Expected ab2 fixed-fusion head, found {type(head).__name__}")
        self.gate_balance_weight = float(getattr(model, "gate_balance_weight", -1.0))
        if self.gate_balance_weight != AB2_GATE_BALANCE_WEIGHT:
            raise ContractError(
                f"ab2 gate balance weight changed: {self.gate_balance_weight}"
            )


class LGLAV1FixedEqualFusionMultiTaskModel(BinaryPartialMultiTaskModel):
    """Locked LGLA v1 with its adaptive gate replaced by fixed equal fusion."""

    def __init__(
        self,
        *args,
        projection_dim: int = AB2_PROJECTION_DIM,
        gate_hidden_dim: int = AB2_GATE_HIDDEN_DIM,
        attention_eps: float = AB2_ATTENTION_EPS,
        min_attention_mass_fraction: float = AB2_MIN_ATTENTION_MASS_FRACTION,
        gate_balance_weight: float = AB2_GATE_BALANCE_WEIGHT,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        old_head = self.model[-1]
        if not isinstance(old_head, BinaryPartialClassificationHead):
            raise ContractError(f"Expected Run22 plain head, found {type(old_head).__name__}")
        if list(old_head.f) != [16, 13]:
            raise ContractError(f"ab2 is locked to YOLO26-sem P3/P4 inputs [16, 13], found {old_head.f}")
        p3_channels = old_head.image_classifier[0].in_features - old_head.aux_head[0].conv.in_channels
        p4_channels = old_head.aux_head[0].conv.in_channels
        new_head = LesionGuidedFixedEqualFusionClassificationHead(
            nc=1,
            ch=(p3_channels, p4_channels),
            num_diseases=old_head.image_classifier[-1].out_features,
            hidden_channels=old_head.image_classifier[0].out_features,
            projection_dim=projection_dim,
            gate_hidden_dim=gate_hidden_dim,
            attention_eps=attention_eps,
            min_attention_mass_fraction=min_attention_mass_fraction,
        )
        # Only the semantic classifiers transfer. The new LGLA classification
        # parameters intentionally follow the same seeded initialization policy
        # as every prior classifier-head ablation.
        new_head.classifier.load_state_dict(old_head.classifier.state_dict())
        new_head.aux_head.load_state_dict(old_head.aux_head.state_dict())
        for attr in ("f", "i", "type"):
            setattr(new_head, attr, getattr(old_head, attr))
        new_head.type = f"{type(new_head).__module__}.{type(new_head).__name__}"
        new_head.np = sum(parameter.numel() for parameter in new_head.parameters())
        new_head.stride = old_head.stride
        self.model[-1] = new_head
        self.gate_balance_weight = float(gate_balance_weight)
        if self.gate_balance_weight != AB2_GATE_BALANCE_WEIGHT:
            raise ContractError(
                f"ab2 gate balance weight must be {AB2_GATE_BALANCE_WEIGHT}"
            )
        self.lgla_v1_config = {
            "scales": ["P3", "P4"],
            "attention_source": "sigmoid_main_predicted_semantic_logit",
            "attention_stop_gradient": True,
            "projection_dim": int(projection_dim),
            "gate_hidden_dim": int(gate_hidden_dim),
            "attention_eps": float(attention_eps),
            "min_attention_mass_fraction": float(min_attention_mass_fraction),
            "empty_attention_fallback": "global_average",
            "gate": "fixed_equal_half_global_local",
            "fixed_fusion_weights": [0.5, 0.5],
            "adaptive_gate_enabled": False,
            "gate_network_retained_but_inactive": True,
            "parameter_parity_with_full_lgla": True,
            "gate_balance_regularizer": "disabled_with_fixed_gate",
            "gate_balance_weight": self.gate_balance_weight,
            "p5_descriptor": False,
            "semantic_branch_modified": False,
            "ground_truth_mask_as_classifier_input": False,
        }
        if kwargs.get("verbose", True):
            self.info()

    def init_criterion(self):
        return FixedEqualFusionPartialLabelMultiTaskLoss(self)


class LGLAV1FixedEqualFusionUtilityTrainer(PartialLabelMultiTaskTrainer):
    """AB2 model with the unchanged Run22 utility validator."""

    def __init__(self, *args, lgla_v1: dict | None = None, **kwargs) -> None:
        self.lgla_v1 = dict(lgla_v1 or {})
        super().__init__(*args, **kwargs)

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        model = LGLAV1FixedEqualFusionMultiTaskModel(
            cfg,
            nc=1,
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            num_diseases=len(CLASS_NAMES),
            image_cls_weight=self.image_cls_weight,
            **self.lgla_v1,
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        self.loss_names = (
            "bce_loss",
            "dice_loss",
            "aux_loss",
            "image_cls_loss",
        )
        return UtilityEligiblePartialLabelMultiTaskValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
            selection=self.selection,
        )


def ab2_early_stop_fitness(metrics: dict[str, float], original_fitness: float) -> float:
    if float(metrics.get("metrics/utility_eligible", 0.0)) == 1.0:
        return float(original_fitness)
    return float(metrics["metrics/joint_score"]) - 2.0


class DelayedEarlyStopping(EarlyStopping):
    def __init__(self, patience: int, start_epoch: int = AB2_EARLY_STOP_START_EPOCH) -> None:
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


class DelayedUtilityGatedAB2Trainer(LGLAV1FixedEqualFusionUtilityTrainer):
    def validate(self):
        prior_gated_best = self.best_fitness
        metrics, original_fitness = super().validate()
        if metrics is None or original_fitness is None:
            return metrics, original_fitness
        gated_fitness = ab2_early_stop_fitness(metrics, original_fitness)
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
        self.stopper = DelayedEarlyStopping(AB2_PATIENCE_AFTER_START, AB2_EARLY_STOP_START_EPOCH)
        for name, value in saved.items():
            if name in {"best_fitness", "best_epoch", "possible_stop"}:
                setattr(self.stopper, name, value)
        if int(getattr(self, "start_epoch", 0)) >= AB2_EARLY_STOP_START_EPOCH:
            self.stopper.activated = True


__all__ = [
    "DelayedEarlyStopping",
    "DelayedUtilityGatedAB2Trainer",
    "AB2_ATTENTION_EPS",
    "AB2_EARLY_STOP_START_EPOCH",
    "AB2_GATE_HIDDEN_DIM",
    "AB2_GATE_BALANCE_WEIGHT",
    "AB2_MIN_ATTENTION_MASS_FRACTION",
    "AB2_PATIENCE_AFTER_START",
    "AB2_PROJECTION_DIM",
    "LGLAV1FixedEqualFusionMultiTaskModel",
    "FixedEqualFusionPartialLabelMultiTaskLoss",
    "LesionGuidedFixedEqualFusionClassificationHead",
    "ab2_early_stop_fitness",
]
