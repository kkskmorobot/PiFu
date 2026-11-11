from pathlib import Path

import pytest
import torch
import yaml

from imaplusplus_partial_multitask_plain_mel_nv_aux_head_seed43_v1 import (
    DelayedEarlyStopping,
    MELNVAuxClassificationHead,
    RUN31_MEL_NV_AUX_CLASSES,
    RUN31_MEL_NV_AUX_PARAMETERS,
    RUN31_MEL_NV_AUX_WEIGHT,
    run31_early_stop_fitness,
    run31_mel_nv_auxiliary_loss,
    run31_mel_nv_pair_rows_and_targets,
)


RUN31_DIR = Path(__file__).resolve().parent


def test_pair_mapping_is_locked_to_nv_zero_and_mel_one():
    targets = torch.tensor([4, 5, 1, 4, 7, 5])
    rows, pair_targets = run31_mel_nv_pair_rows_and_targets(targets)
    assert RUN31_MEL_NV_AUX_CLASSES == (5, 4)
    assert rows.tolist() == [True, True, False, True, False, True]
    assert pair_targets.tolist() == [1, 0, 1, 0]


def test_auxiliary_ce_ignores_non_pair_rows_and_gradients():
    logits = torch.tensor(
        [[0.0, 1.0], [1.0, 0.0], [8.0, -8.0], [0.0, 0.0]],
        requires_grad=True,
    )
    targets = torch.tensor([4, 5, 1, 7])
    loss = run31_mel_nv_auxiliary_loss(logits, targets)
    expected = torch.nn.functional.cross_entropy(logits[:2], torch.tensor([1, 0]))
    assert torch.allclose(loss, expected)
    loss.backward()
    assert torch.count_nonzero(logits.grad[2:]).item() == 0


def test_auxiliary_loss_is_zero_when_batch_has_no_mel_or_nv():
    logits = torch.randn(3, 2, requires_grad=True)
    loss = run31_mel_nv_auxiliary_loss(logits, torch.tensor([0, 1, 7]))
    assert loss.item() == 0.0
    loss.backward()
    assert torch.count_nonzero(logits.grad).item() == 0


def test_yolo26m_auxiliary_head_adds_exactly_514_parameters_and_eval_omits_it():
    head = MELNVAuxClassificationHead(nc=1, ch=(8, 16), hidden_channels=256)
    assert sum(parameter.numel() for parameter in head.mel_nv_aux_classifier.parameters()) == 514
    assert RUN31_MEL_NV_AUX_PARAMETERS == 514
    features = [torch.randn(2, 8, 16, 16), torch.randn(2, 16, 8, 8)]
    head.train()
    training_output = head(features)
    assert len(training_output) == 4
    assert training_output[2].shape == (2, 8)
    assert training_output[3].shape == (2, 2)
    head.eval()
    with torch.no_grad():
        evaluation_output = head(features)
    assert len(evaluation_output) == 2
    assert evaluation_output[1].shape == (2, 8)


def test_delayed_early_stop_and_utility_gate_remain_frozen():
    stopper = DelayedEarlyStopping(patience=30, start_epoch=40)
    for epoch in range(40):
        assert stopper(epoch, 0.6) is False
    assert stopper.activated is False
    assert stopper(40, 0.6) is False
    assert stopper.activated is True
    eligible = {"metrics/utility_eligible": 1.0, "metrics/joint_score": 0.7}
    ineligible = {"metrics/utility_eligible": 0.0, "metrics/joint_score": 0.7}
    assert run31_early_stop_fitness(eligible, 0.65) == pytest.approx(0.65)
    assert run31_early_stop_fitness(ineligible, 0.65) == pytest.approx(-1.3)


def test_run31_config_has_exactly_preregistered_auxiliary_factor():
    config = yaml.safe_load((RUN31_DIR / "run31_preregistered_config_v1.yaml").read_text(encoding="utf-8"))
    assert config["contract"]["single_preregistered_factor"] == (
        "add_training_only_shared_hidden_mel_vs_nv_auxiliary_head_weight_0_5_to_run22"
    )
    auxiliary = config["risk_auxiliary_head"]
    assert auxiliary["class_ids"] == [5, 4]
    assert auxiliary["classes"] == ["nv", "mel"]
    assert auxiliary["weight_inside_classification_objective"] == RUN31_MEL_NV_AUX_WEIGHT
    assert auxiliary["application_scope"] == "training_only"
    assert auxiliary["new_parameters"] == RUN31_MEL_NV_AUX_PARAMETERS
    assert auxiliary["inference_rule"] == "unchanged_eight_class_argmax"
    assert config["attention_ablation"]["enabled"] is False
