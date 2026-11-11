from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = PROJECT_ROOT / "tools"
RUN20_DIR = TOOLS_DIR / "run20"
for path in (TOOLS_DIR, RUN20_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from imaplusplus_partial_multitask_adaptive_residual_attention_v1 import (
    AdaptiveResidualBinaryLesionGuidedAttentionHead,
    DelayedEarlyStopping,
    RUN20_INITIAL_MIX,
    RUN20_MAX_MIX,
    run20_early_stop_fitness,
)


def make_head():
    return AdaptiveResidualBinaryLesionGuidedAttentionHead(
        nc=1,
        ch=(16, 32, 64),
        num_diseases=8,
        hidden_channels=16,
        eca_kernel_size=5,
        initial_mask_guided_mix=RUN20_INITIAL_MIX,
        max_mask_guided_mix=RUN20_MAX_MIX,
        detach_mask=True,
    )


def test_three_scale_mix_is_learnable_bounded_and_initialized_at_005():
    head = make_head()
    mix = head.current_mask_guided_mix()
    assert isinstance(head.mask_guided_mix_logits, torch.nn.Parameter)
    assert tuple(mix.shape) == (3,)
    assert torch.allclose(mix, torch.full((3,), 0.05), atol=1e-7)
    assert bool(((mix > 0.0) & (mix < 0.5)).all())


def test_mix_parameter_receives_gradient_from_residual_pooling():
    head = make_head()
    feature = torch.randn(2, 16, 8, 8, requires_grad=True)
    foreground = torch.rand(2, 1, 16, 16)
    loss = head._pool_scale(feature, foreground, 0).square().mean()
    loss.backward()
    assert head.mask_guided_mix_logits.grad is not None
    assert torch.isfinite(head.mask_guided_mix_logits.grad).all()
    assert float(head.mask_guided_mix_logits.grad[0].abs()) > 0.0


def test_ineligible_fitness_is_strictly_below_eligible_domain():
    cold = run20_early_stop_fitness(
        {"metrics/utility_eligible": 0.0, "metrics/joint_score": 0.45}, 0.31
    )
    mature = run20_early_stop_fitness(
        {"metrics/utility_eligible": 1.0, "metrics/joint_score": 0.60}, 0.12
    )
    assert cold == -1.55
    assert mature == 0.12
    assert mature > cold


def test_patience_clock_cannot_start_before_epoch_40():
    stopper = DelayedEarlyStopping(patience=30, start_epoch=40)
    for epoch in range(1, 40):
        assert stopper(epoch, 0.9 if epoch == 6 else 0.1) is False
    assert stopper.activated is False
    assert stopper(40, 0.2) is False
    for epoch in range(41, 70):
        assert stopper(epoch, 0.1) is False
    assert stopper(70, 0.1) is True
