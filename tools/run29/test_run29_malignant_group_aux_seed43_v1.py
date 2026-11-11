from pathlib import Path
import sys

import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = PROJECT_ROOT / "tools"
RUN29_DIR = TOOLS_DIR / "run29"
for path in (TOOLS_DIR, RUN29_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from imaplusplus_partial_multitask_plain_malignant_group_aux_seed43_v1 import (
    DelayedEarlyStopping,
    RUN29_BENIGN_CLASS_IDS,
    RUN29_MALIGNANT_CLASS_IDS,
    RUN29_MALIGNANT_GROUP_AUX_WEIGHT,
    run29_early_stop_fitness,
    run29_malignant_group_nll,
)


def test_group_partition_is_complete_and_disjoint():
    assert set(RUN29_MALIGNANT_CLASS_IDS) == {1, 4, 7}
    assert set(RUN29_BENIGN_CLASS_IDS) == {0, 2, 3, 5, 6}
    assert set(RUN29_MALIGNANT_CLASS_IDS).isdisjoint(RUN29_BENIGN_CLASS_IDS)
    assert set(RUN29_MALIGNANT_CLASS_IDS) | set(RUN29_BENIGN_CLASS_IDS) == set(range(8))


def test_group_nll_rewards_probability_mass_in_the_correct_group():
    targets = torch.tensor([4, 5])
    wrong = torch.zeros(2, 8)
    wrong[0, 5] = 4.0
    wrong[1, 4] = 4.0
    right = torch.zeros(2, 8)
    right[0, 4] = 4.0
    right[1, 5] = 4.0
    assert run29_malignant_group_nll(right, targets) < run29_malignant_group_nll(wrong, targets)


def test_group_nll_equals_nll_of_summed_eight_class_probabilities():
    logits = torch.tensor([[0.4, -0.2, 0.8, -0.6, 1.1, 0.2, -0.4, 0.3]])
    target = torch.tensor([4])
    probs = logits.softmax(dim=1)
    expected = -torch.log(probs[:, RUN29_MALIGNANT_CLASS_IDS].sum(dim=1)).mean()
    assert torch.allclose(run29_malignant_group_nll(logits, target), expected, atol=1e-7)


def test_ineligible_fitness_and_delayed_patience_are_frozen():
    cold = run29_early_stop_fitness(
        {"metrics/utility_eligible": 0.0, "metrics/joint_score": 0.45}, 0.31
    )
    mature = run29_early_stop_fitness(
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


def test_run29_config_has_exactly_preregistered_risk_factor():
    config = yaml.safe_load((RUN29_DIR / "run29_preregistered_config_v1.yaml").read_text(encoding="utf-8"))
    assert config["ultralytics"]["seed"] == 43
    assert config["multitask"]["image_head"] == "plain_p3_p4_global_average_pool_mlp_dropout020"
    assert config["attention_ablation"]["enabled"] is False
    assert config["risk_loss"]["enabled"] is True
    assert config["risk_loss"]["weight_inside_classification_objective"] == RUN29_MALIGNANT_GROUP_AUX_WEIGHT
    assert config["risk_loss"]["new_parameters"] == 0
    assert config["risk_loss"]["inference_rule"] == "unchanged_eight_class_argmax"
