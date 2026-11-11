from pathlib import Path
import sys

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = PROJECT_ROOT / "tools"
RUN30_DIR = TOOLS_DIR / "run30"
for path in (TOOLS_DIR, RUN30_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from imaplusplus_partial_multitask_plain_mel_nv_focal_risk_seed43_v1 import (
    DelayedEarlyStopping,
    RUN30_MEL_NV_FOCAL_GAMMA,
    RUN30_MEL_NV_FOCAL_RISK_WEIGHT,
    run30_early_stop_fitness,
    run30_mel_nv_focal_risk,
)


def test_focal_risk_penalizes_high_nv_probability_for_true_mel():
    targets = torch.tensor([4])
    low_risk = torch.zeros(1, 8)
    low_risk[0, 4] = 4.0
    high_risk = torch.zeros(1, 8)
    high_risk[0, 5] = 4.0
    assert run30_mel_nv_focal_risk(low_risk, targets) < run30_mel_nv_focal_risk(high_risk, targets)


def test_non_mel_rows_have_zero_auxiliary_gradient():
    logits = torch.randn(3, 8, requires_grad=True)
    targets = torch.tensor([1, 5, 7])
    loss = run30_mel_nv_focal_risk(logits, targets)
    loss.backward()
    assert loss.item() == 0.0
    assert torch.equal(logits.grad, torch.zeros_like(logits.grad))


def test_only_true_mel_rows_determine_the_mean():
    mel_logits = torch.tensor([[0.1, -0.2, 0.3, 0.0, 0.4, 1.2, -0.1, 0.2]])
    mel_only = run30_mel_nv_focal_risk(mel_logits, torch.tensor([4]))
    mixed = torch.cat((mel_logits, torch.full((2, 8), 50.0)), dim=0)
    mixed_loss = run30_mel_nv_focal_risk(mixed, torch.tensor([4, 5, 1]))
    assert torch.allclose(mel_only, mixed_loss, atol=1e-7)


def test_focal_formula_matches_preregistered_expression():
    logits = torch.tensor([[0.4, -0.2, 0.8, -0.6, 1.1, 0.2, -0.4, 0.3]])
    p_nv = logits.softmax(dim=1)[:, 5]
    expected = (p_nv.pow(2.0) * (-torch.log1p(-p_nv))).mean()
    actual = run30_mel_nv_focal_risk(logits, torch.tensor([4]))
    assert torch.allclose(actual, expected, atol=1e-7)


def test_ineligible_fitness_and_delayed_patience_are_frozen():
    cold = run30_early_stop_fitness(
        {"metrics/utility_eligible": 0.0, "metrics/joint_score": 0.45}, 0.31
    )
    mature = run30_early_stop_fitness(
        {"metrics/utility_eligible": 1.0, "metrics/joint_score": 0.60}, 0.12
    )
    assert cold == -1.55
    assert mature == 0.12
    stopper = DelayedEarlyStopping(patience=30, start_epoch=40)
    for epoch in range(1, 40):
        assert stopper(epoch, 0.9 if epoch == 6 else 0.1) is False
    assert stopper(40, 0.2) is False
    for epoch in range(41, 70):
        assert stopper(epoch, 0.1) is False
    assert stopper(70, 0.1) is True


def test_run30_config_has_exactly_preregistered_risk_factor():
    config = yaml.safe_load((RUN30_DIR / "run30_preregistered_config_v1.yaml").read_text(encoding="utf-8"))
    assert config["ultralytics"]["seed"] == 43
    assert config["attention_ablation"]["enabled"] is False
    risk = config["risk_loss"]
    assert risk["true_class_id"] == 4 and risk["risk_class_id"] == 5
    assert risk["gamma"] == RUN30_MEL_NV_FOCAL_GAMMA
    assert risk["weight_inside_classification_objective"] == RUN30_MEL_NV_FOCAL_RISK_WEIGHT
    assert risk["non_mel_auxiliary_gradient"] == "zero"
    assert risk["new_parameters"] == 0
    assert risk["inference_rule"] == "unchanged_eight_class_argmax"
