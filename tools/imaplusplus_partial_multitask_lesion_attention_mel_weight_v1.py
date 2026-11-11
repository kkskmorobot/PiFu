"""Run13 frozen combination: Run12 BLG-MDA attention plus Run11 MEL CE weight.

Relative to Run12, the only training-behavior change is the already frozen
Run11 target-class weight for MEL.  The model architecture, initialization,
partial-label loader, validator, sampler, and checkpoint contract are inherited
unchanged from Run12.
"""

from __future__ import annotations

from ultralytics.utils import RANK

from imaplusplus_partial_multitask_lesion_attention_v1 import (
    BinaryLesionGuidedAttentionHead,
    BinaryLesionGuidedAttentionModel,
    BinaryLesionGuidedAttentionTrainer,
    IdentityInitializedECA,
    PartialLabelSemanticDataset,
)
from imaplusplus_partial_multitask_mel_weight_v1 import (
    PartialLabelMultiTaskLoss as MelWeightedPartialLabelMultiTaskLoss,
    RUN11_CLASSIFICATION_CLASS_WEIGHTS,
    RUN11_MEL_CLASS_WEIGHT,
)
from imaplusplus_partial_multitask_v1 import CLASS_NAMES, ContractError


class MelWeightedBinaryLesionGuidedAttentionModel(BinaryLesionGuidedAttentionModel):
    """Run12 architecture with the exact Run11 MEL target-class CE weight."""

    def __init__(
        self,
        cfg="yolo26s-sem.yaml",
        ch: int = 3,
        nc: int | None = 1,
        verbose: bool = True,
        num_diseases: int = 8,
        image_cls_weight: float = 0.5,
        classification_class_weights: tuple[float, ...] = RUN11_CLASSIFICATION_CLASS_WEIGHTS,
        p5_layer: int = 10,
        eca_kernel_size: int = 5,
        mask_guided_mix: float = 0.5,
        detach_mask: bool = True,
    ) -> None:
        weights = tuple(float(value) for value in classification_class_weights)
        if weights != RUN11_CLASSIFICATION_CLASS_WEIGHTS:
            raise ContractError(
                "Run13 must reuse the exact Run11 classification weights: "
                f"{weights!r} != {RUN11_CLASSIFICATION_CLASS_WEIGHTS!r}"
            )
        super().__init__(
            cfg=cfg,
            ch=ch,
            nc=nc,
            verbose=verbose,
            num_diseases=num_diseases,
            image_cls_weight=image_cls_weight,
            p5_layer=p5_layer,
            eca_kernel_size=eca_kernel_size,
            mask_guided_mix=mask_guided_mix,
            detach_mask=detach_mask,
        )
        self.classification_class_weights = weights

    def init_criterion(self):
        return MelWeightedPartialLabelMultiTaskLoss(self)


class MelWeightedBinaryLesionGuidedAttentionTrainer(BinaryLesionGuidedAttentionTrainer):
    """Run13 Trainer; only the classification criterion differs from Run12."""

    def __init__(
        self,
        *args,
        classification_class_weights: tuple[float, ...] = RUN11_CLASSIFICATION_CLASS_WEIGHTS,
        **kwargs,
    ) -> None:
        weights = tuple(float(value) for value in classification_class_weights)
        if weights != RUN11_CLASSIFICATION_CLASS_WEIGHTS:
            raise ContractError("Run13 Trainer classification weights differ from frozen Run11")
        self.classification_class_weights = weights
        super().__init__(*args, **kwargs)

    def get_model(self, cfg=None, weights=None, verbose: bool = True):
        model = MelWeightedBinaryLesionGuidedAttentionModel(
            cfg,
            nc=1,
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            num_diseases=len(CLASS_NAMES),
            image_cls_weight=self.image_cls_weight,
            classification_class_weights=self.classification_class_weights,
            **self.lesion_attention,
        )
        if weights:
            model.load(weights)
        return model


__all__ = [
    "BinaryLesionGuidedAttentionHead",
    "IdentityInitializedECA",
    "MelWeightedBinaryLesionGuidedAttentionModel",
    "MelWeightedBinaryLesionGuidedAttentionTrainer",
    "MelWeightedPartialLabelMultiTaskLoss",
    "PartialLabelSemanticDataset",
    "RUN11_CLASSIFICATION_CLASS_WEIGHTS",
    "RUN11_MEL_CLASS_WEIGHT",
]
