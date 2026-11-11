from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = PROJECT_ROOT / "tools"
LG3_DIR = TOOLS_DIR / "lg3"
for path in (TOOLS_DIR, LG3_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from imaplusplus_partial_multitask_lgla_v1_gate_balance_seed44 import (
    DelayedEarlyStopping,
    LesionGuidedGlobalLocalClassificationHead,
    classified_batch_mean_gate_logit_balance,
    lg3_early_stop_fitness,
)


def make_head():
    torch.manual_seed(44)
    return LesionGuidedGlobalLocalClassificationHead(
        nc=1,
        ch=(16, 24),
        hidden_channels=16,
        projection_dim=8,
        gate_hidden_dim=8,
        attention_eps=1.0e-6,
        min_attention_mass_fraction=1.0e-4,
    )


def test_lgla_shapes_gate_and_diagnostics_are_finite():
    head = make_head().train()
    features = [torch.randn(3, 16, 16, 16), torch.randn(3, 24, 8, 8)]
    semantic, auxiliary, image_logits = head(features)
    assert semantic.shape == (3, 1, 16, 16)
    assert auxiliary.shape == (3, 1, 8, 8)
    assert image_logits.shape == (3, 8)
    assert head.last_gate_weights.shape == (3, 2)
    assert torch.isfinite(head.last_gate_weights).all()
    assert torch.allclose(head.last_gate_weights.sum(1), torch.ones(3))
    assert torch.equal(head.last_gate_weights, torch.full((3, 2), 0.5))
    assert head.last_attention_mass_fraction.shape == (3, 2)
    assert head.last_attention_fallback.shape == (3, 2)
    assert not hasattr(head, "p5_layer")


def test_empty_attention_falls_back_to_global_without_nan():
    head = make_head()
    feature = torch.randn(2, 16, 7, 9)
    zero_attention = torch.zeros(2, 1, 14, 18)
    global_vector, local_vector, mass_fraction, fallback = head._global_local_pool(
        feature, zero_attention
    )
    assert torch.equal(local_vector, global_vector)
    assert torch.equal(mass_fraction, torch.zeros(2))
    assert fallback.tolist() == [True, True]
    assert torch.isfinite(local_vector).all()


def test_classification_gradient_cannot_reach_semantic_classifier_through_attention():
    head = make_head().train()
    features = [
        torch.randn(2, 16, 16, 16, requires_grad=True),
        torch.randn(2, 24, 8, 8, requires_grad=True),
    ]
    image_logits = head(features)[-1]
    image_logits.square().mean().backward()
    assert all(parameter.grad is None for parameter in head.classifier.parameters())
    assert all(parameter.grad is None for parameter in head.aux_head.parameters())
    assert all(feature.grad is not None and torch.isfinite(feature.grad).all() for feature in features)
    assert all(
        projection.weight.grad is not None and torch.isfinite(projection.weight.grad).all()
        for projection in head.scale_projections
    )
    assert head.global_local_gate[-1].weight.grad is not None
    assert torch.isfinite(head.global_local_gate[-1].weight.grad).all()


def test_local_pooling_uses_predicted_probability_weights():
    head = make_head()
    feature = torch.zeros(1, 16, 2, 2)
    feature[:, :, 0, 0] = 4.0
    attention = torch.zeros(1, 1, 2, 2)
    attention[:, :, 0, 0] = 1.0
    global_vector, local_vector, mass_fraction, fallback = head._global_local_pool(
        feature, attention
    )
    assert torch.equal(global_vector, torch.ones_like(global_vector))
    assert torch.equal(local_vector, torch.full_like(local_vector, 4.0))
    assert torch.equal(mass_fraction, torch.tensor([0.25]))
    assert fallback.tolist() == [False]


def test_gate_balance_uses_only_classified_samples_and_allows_sample_variation():
    logits = torch.tensor(
        [
            [0.0, 1.0],   # classified, local-minus-global = +1
            [99.0, -99.0],  # unlabeled and must not affect the regularizer
            [0.0, -1.0],  # classified, local-minus-global = -1
        ],
        requires_grad=True,
    )
    has_cls = torch.tensor([True, False, True])
    balanced = classified_batch_mean_gate_logit_balance(logits, has_cls)
    assert torch.equal(balanced, torch.tensor(0.0))
    collapsed = classified_batch_mean_gate_logit_balance(
        torch.tensor([[1.0, -1.0], [2.0, 0.0]]), torch.tensor([True, True])
    )
    assert torch.equal(collapsed, torch.tensor(4.0))


def test_ineligible_fitness_is_below_eligible_fitness():
    cold = lg3_early_stop_fitness(
        {"metrics/utility_eligible": 0.0, "metrics/joint_score": 0.45}, 0.31
    )
    mature = lg3_early_stop_fitness(
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
