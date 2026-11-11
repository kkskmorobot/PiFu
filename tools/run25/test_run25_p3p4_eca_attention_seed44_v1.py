from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = PROJECT_ROOT / "tools"
RUN25_DIR = TOOLS_DIR / "run25"
for path in (TOOLS_DIR, RUN25_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from imaplusplus_partial_multitask_p3p4_eca_attention_seed44_v1 import (
    DelayedEarlyStopping,
    P3P4ECAClassificationHead,
    run25_early_stop_fitness,
)


def test_eca_head_is_identity_initialized_and_uses_only_p3_p4():
    head = P3P4ECAClassificationHead(nc=1, ch=(16, 24), hidden_channels=16, eca_kernel_size=5)
    assert len(head.channel_attention) == 2
    assert head.scale_channels == (16, 24)
    assert not hasattr(head, "mask_guided_mix")
    assert not hasattr(head, "detach_mask")
    for module in head.channel_attention:
        assert torch.count_nonzero(module.conv.weight).item() == 0
    features = [torch.randn(2, 16, 16, 16), torch.randn(2, 24, 8, 8)]
    for module, feature in zip(head.channel_attention, features):
        assert torch.equal(module(feature), feature)


def test_eca_parameters_receive_classification_gradient():
    head = P3P4ECAClassificationHead(nc=1, ch=(16, 24), hidden_channels=16, eca_kernel_size=5)
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


def test_ineligible_fitness_is_below_eligible_fitness():
    cold = run25_early_stop_fitness(
        {"metrics/utility_eligible": 0.0, "metrics/joint_score": 0.45}, 0.31
    )
    mature = run25_early_stop_fitness(
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

