from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = PROJECT_ROOT / "tools"
RUN33_DIR = TOOLS_DIR / "run33"
for path in (TOOLS_DIR, RUN33_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from imaplusplus_partial_multitask_p3p4_standard_cbam_seed43_v1 import (
    DelayedEarlyStopping,
    P3P4StandardCBAMClassificationHead,
    StandardCBAM,
    StandardCBAMChannelAttention,
    StandardCBAMSpatialAttention,
    run33_early_stop_fitness,
)


def test_standard_cbam_head_uses_only_p3_p4():
    head = P3P4StandardCBAMClassificationHead(
        nc=1, ch=(16, 24), hidden_channels=16, reduction=16, spatial_kernel_size=7
    )
    assert len(head.cbam) == 2
    assert head.scale_channels == (16, 24)
    assert all(isinstance(module, StandardCBAM) for module in head.cbam)
    assert not hasattr(head, "mask_guided_mix")
    assert not hasattr(head, "detach_mask")
    assert not hasattr(head, "p5_layer")


def test_standard_cbam_channel_and_spatial_math_is_finite_and_not_identity_locked():
    torch.manual_seed(33)
    feature = torch.randn(2, 16, 11, 13)
    channel = StandardCBAMChannelAttention(16, reduction=16)
    spatial = StandardCBAMSpatialAttention(kernel_size=7)
    output = spatial(channel(feature))
    assert output.shape == feature.shape
    assert torch.isfinite(output).all()
    assert not torch.equal(output, feature)
    assert sum(parameter.numel() for parameter in channel.parameters()) == 32
    assert sum(parameter.numel() for parameter in spatial.parameters()) == 98


def test_standard_cbam_validates_parameters():
    for constructor in (
        lambda: StandardCBAMChannelAttention(0, reduction=16),
        lambda: StandardCBAMChannelAttention(16, reduction=0),
        lambda: StandardCBAMSpatialAttention(kernel_size=5),
    ):
        try:
            constructor()
        except Exception as error:
            assert "Run33" in str(error)
        else:
            raise AssertionError("Run33 accepted an invalid standard CBAM parameter")


def test_attention_parameters_receive_classification_gradient():
    torch.manual_seed(33)
    head = P3P4StandardCBAMClassificationHead(
        nc=1, ch=(16, 24), hidden_channels=16, reduction=16, spatial_kernel_size=7
    )
    head.train()
    features = [
        torch.randn(2, 16, 16, 16, requires_grad=True),
        torch.randn(2, 24, 8, 8, requires_grad=True),
    ]
    image_logits = head(features)[-1]
    image_logits.square().mean().backward()
    for cbam in head.cbam:
        for parameter in cbam.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            assert float(parameter.grad.abs().sum()) > 0.0


def test_ineligible_fitness_is_below_eligible_fitness():
    cold = run33_early_stop_fitness(
        {"metrics/utility_eligible": 0.0, "metrics/joint_score": 0.45}, 0.31
    )
    mature = run33_early_stop_fitness(
        {"metrics/utility_eligible": 1.0, "metrics/joint_score": 0.60}, 0.12
    )
    assert cold == -1.55
    assert mature == 0.12
    assert mature > cold


def test_patience_clock_cannot_start_before_epoch_40():
    stopper = DelayedEarlyStopping(patience=30, start_epoch=40)
    for epoch in range(1, 40):
        assert stopper(epoch, 0.9 if epoch == 6 else 0.1) is False
        assert stopper.possible_stop is False
    assert stopper.activated is False
    assert stopper(40, 0.2) is False
    assert stopper.activated is True
    for epoch in range(41, 70):
        assert stopper(epoch, 0.1) is False
    assert stopper.possible_stop is True
    assert stopper(70, 0.1) is True
