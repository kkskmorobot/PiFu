from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = PROJECT_ROOT / "tools"
RUN26_DIR = TOOLS_DIR / "run26"
for path in (TOOLS_DIR, RUN26_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from imaplusplus_partial_multitask_p3p4_eca_spatial_attention_seed44_v1 import (
    DelayedEarlyStopping,
    IdentityInitializedSpatialAttention,
    P3P4ECASpatialClassificationHead,
    run26_early_stop_fitness,
)


def test_eca_spatial_head_is_identity_initialized_and_uses_only_p3_p4():
    head = P3P4ECASpatialClassificationHead(
        nc=1, ch=(16, 24), hidden_channels=16, eca_kernel_size=5, spatial_kernel_size=7
    )
    assert len(head.channel_attention) == 2
    assert len(head.spatial_attention) == 2
    assert head.scale_channels == (16, 24)
    assert not hasattr(head, "mask_guided_mix")
    assert not hasattr(head, "detach_mask")
    features = [torch.randn(2, 16, 16, 16), torch.randn(2, 24, 8, 8)]
    for channel, spatial, feature in zip(head.channel_attention, head.spatial_attention, features):
        assert torch.count_nonzero(channel.conv.weight).item() == 0
        assert torch.count_nonzero(spatial.conv.weight).item() == 0
        assert torch.equal(spatial(channel(feature)), feature)


def test_spatial_gate_validates_kernel_and_is_exact_identity():
    feature = torch.randn(2, 7, 11, 13)
    module = IdentityInitializedSpatialAttention(kernel_size=7)
    assert torch.equal(module(feature), feature)
    try:
        IdentityInitializedSpatialAttention(kernel_size=4)
    except Exception as error:
        assert "positive odd" in str(error)
    else:
        raise AssertionError("Run26 accepted an even spatial kernel")


def test_attention_parameters_receive_classification_gradient():
    head = P3P4ECASpatialClassificationHead(
        nc=1, ch=(16, 24), hidden_channels=16, eca_kernel_size=5, spatial_kernel_size=7
    )
    head.train()
    features = [
        torch.randn(2, 16, 16, 16, requires_grad=True),
        torch.randn(2, 24, 8, 8, requires_grad=True),
    ]
    image_logits = head(features)[-1]
    image_logits.square().mean().backward()
    for module in head.channel_attention:
        assert module.conv.weight.grad is not None
        assert torch.isfinite(module.conv.weight.grad).all()
        assert float(module.conv.weight.grad.abs().sum()) > 0.0
    for module in head.spatial_attention:
        assert module.conv.weight.grad is not None
        assert torch.isfinite(module.conv.weight.grad).all()
        assert float(module.conv.weight.grad.abs().sum()) > 0.0


def test_ineligible_fitness_is_below_eligible_fitness():
    cold = run26_early_stop_fitness(
        {"metrics/utility_eligible": 0.0, "metrics/joint_score": 0.45}, 0.31
    )
    mature = run26_early_stop_fitness(
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


