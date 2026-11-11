from pathlib import Path

import pytest
import torch
import yaml

from imaplusplus_partial_multitask_plain_malignant_aux_head_seed43_v1 import (
    DelayedEarlyStopping,
    MalignantAuxClassificationHead,
    RUN32_MALIGNANT_AUX_CLASSES,
    RUN32_MALIGNANT_AUX_PARAMETERS,
    RUN32_MALIGNANT_AUX_WEIGHT,
    run32_early_stop_fitness,
    run32_malignant_auxiliary_loss,
    run32_malignant_targets,
)


RUN32_DIR = Path(__file__).resolve().parent


def test_all_eight_classes_map_to_locked_benign_and_malignant_groups():
    targets = torch.arange(8)
    assert RUN32_MALIGNANT_AUX_CLASSES == ("benign", "malignant")
    assert run32_malignant_targets(targets).tolist() == [0, 1, 0, 0, 1, 0, 0, 1]


def test_auxiliary_ce_uses_every_officially_labeled_row():
    logits = torch.tensor(
        [[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 2.0]],
        requires_grad=True,
    )
    targets = torch.tensor([0, 1, 5, 7])
    loss = run32_malignant_auxiliary_loss(logits, targets)
    expected = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 0, 1]))
    assert torch.allclose(loss, expected)
    loss.backward()
    assert torch.count_nonzero(logits.grad).item() == logits.numel()


def test_auxiliary_loss_rejects_missing_or_invalid_official_targets():
    with pytest.raises(Exception):
        run32_malignant_auxiliary_loss(torch.randn(2, 2), torch.tensor([1, -1]))
    with pytest.raises(Exception):
        run32_malignant_auxiliary_loss(torch.randn(2, 2), torch.tensor([1]))


def test_yolo26m_auxiliary_head_adds_exactly_514_parameters_and_eval_omits_it():
    head = MalignantAuxClassificationHead(nc=1, ch=(8, 16), hidden_channels=256)
    assert sum(parameter.numel() for parameter in head.malignant_aux_classifier.parameters()) == 514
    assert RUN32_MALIGNANT_AUX_PARAMETERS == 514
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
    assert run32_early_stop_fitness(eligible, 0.65) == pytest.approx(0.65)
    assert run32_early_stop_fitness(ineligible, 0.65) == pytest.approx(-1.3)


def test_run32_config_has_exactly_preregistered_auxiliary_factor():
    config = yaml.safe_load((RUN32_DIR / "run32_preregistered_config_v1.yaml").read_text(encoding="utf-8"))
    assert config["contract"]["single_preregistered_factor"] == (
        "add_training_only_shared_hidden_malignant_vs_benign_auxiliary_head_weight_0_25_to_run22"
    )
    auxiliary = config["risk_auxiliary_head"]
    assert auxiliary["benign_class_ids"] == [0, 2, 3, 5, 6]
    assert auxiliary["malignant_class_ids"] == [1, 4, 7]
    assert auxiliary["classes"] == ["benign", "malignant"]
    assert auxiliary["weight_inside_classification_objective"] == RUN32_MALIGNANT_AUX_WEIGHT
    assert auxiliary["normalization"] == "all_officially_labeled_classification_rows"
    assert auxiliary["application_scope"] == "training_only"
    assert auxiliary["new_parameters"] == RUN32_MALIGNANT_AUX_PARAMETERS
    assert auxiliary["inference_rule"] == "unchanged_eight_class_argmax"
    assert config["attention_ablation"]["enabled"] is False
