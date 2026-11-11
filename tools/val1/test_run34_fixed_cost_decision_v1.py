from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import yaml


RUN34_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("run34_entry", RUN34_DIR / "evaluate_run34_from_pycharm.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def config():
    return yaml.safe_load((RUN34_DIR / "run34_preregistered_config_v1.yaml").read_text(encoding="utf-8"))


def test_cost_matrix_is_exactly_preregistered():
    matrix = MODULE.build_cost_matrix(config())
    assert matrix.shape == (8, 8)
    assert np.array_equal(np.diag(matrix), np.zeros(8))
    assert matrix[MODULE.MEL_ID, MODULE.NV_ID] == 8.0
    for truth in MODULE.MALIGNANT_IDS:
        for prediction in range(8):
            if prediction not in MODULE.MALIGNANT_IDS and not (
                truth == MODULE.MEL_ID and prediction == MODULE.NV_ID
            ):
                assert matrix[truth, prediction] == 4.0


def test_cost_decision_uses_posterior_expected_cost():
    matrix = MODULE.build_cost_matrix(config())
    probabilities = np.asarray([[0.0, 0.25, 0.0, 0.0, 0.0, 0.75, 0.0, 0.0]], dtype=np.float64)
    assert probabilities.argmax(axis=1).tolist() == [MODULE.NV_ID]
    assert MODULE.cost_predictions(probabilities, matrix).tolist() == [1]


def test_unambiguous_correct_class_remains_selected():
    matrix = MODULE.build_cost_matrix(config())
    probabilities = np.eye(8, dtype=np.float64)
    assert MODULE.cost_predictions(probabilities, matrix).tolist() == list(range(8))


def test_entry_has_no_training_or_test_switch():
    source = (RUN34_DIR / "evaluate_run34_from_pycharm.py").read_text(encoding="utf-8")
    assert "trainer.train(" not in source
    assert ".backward(" not in source
    assert "--allow-test" not in source
    assert "--split" not in source


def test_shared_tools_directory_is_importable_from_run34_entry():
    assert str(MODULE.TOOLS_DIR) in sys.path


def test_formal_evaluation_restores_lossless_runtime_ema():
    source = (RUN34_DIR / "evaluate_run34_from_pycharm.py").read_text(encoding="utf-8")
    assert 'runtime_state.pt' in source
    assert 'get("training_state", {}).get("ema")' in source
    assert "model.load_state_dict(exact_state, strict=True)" in source


def test_formal_evaluation_uses_original_unfused_training_validation_path():
    source = (RUN34_DIR / "evaluate_run34_from_pycharm.py").read_text(encoding="utf-8")
    assert "flat = validator(frozen_trainer)" in source
    assert "flat = validator(model=model)" not in source
    assert "training_path_unfused_fp32_ema_cuda_autocast" in source


def test_legacy_pycharm_entry_delegates_to_canonical_evaluator():
    source = (RUN34_DIR / "run_run34_from_pycharm.py").read_text(encoding="utf-8")
    assert "from evaluate_run34_from_pycharm import main" in source


def test_acceptance_gate_is_frozen_three_seed_gate():
    gate = config()["acceptance_gate"]
    assert gate["min_mean_delta_mel_to_nv"] == -0.10
    assert gate["min_mean_delta_malignant_to_benign"] == -0.05
    assert gate["min_mean_delta_group_macro_f1"] == -0.03
    assert gate["min_mean_delta_joint"] == -0.02
    assert gate["min_each_seed_delta_group_macro_f1"] == -0.06
