#!/usr/bin/env python3
"""Static and metric-contract tests for Test2; never load test/external images."""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("test2_entry", HERE / "test.py")
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class TestContract(unittest.TestCase):
    def test_config_and_frozen_inputs(self) -> None:
        config = module.load_config()
        self.assertEqual(len(config["checkpoints"]), 3)
        self.assertEqual([row["seed"] for row in config["checkpoints"]], [42, 43, 44])
        self.assertEqual(config["evaluation"]["classification_bootstrap_strategy"], "stratified_by_true_class")
        inputs = module.validate_inputs(config)
        self.assertEqual(inputs["internal"]["counts"]["classification"], 56)
        self.assertEqual(inputs["external"]["counts"]["classification"], 0)
        self.assertFalse(inputs["pad_ufes_20_present"])

    def test_restricted_metrics_and_bootstrap_are_deterministic(self) -> None:
        targets = np.asarray([module.MEL_ID] * 16 + [module.NV_ID] * 40, dtype=np.int64)
        probabilities = np.full((56, len(module.CLASS_NAMES)), 0.01, dtype=np.float64)
        probabilities[np.arange(56), targets] = 0.93
        metrics = module.restricted_metrics(targets, probabilities)
        for name in ("accuracy", "mel_recall", "nv_recall", "balanced_recall"):
            self.assertEqual(metrics[name], 1.0)
        self.assertEqual(metrics["mel_to_nv_rate"], 0.0)
        self.assertEqual(metrics["nv_to_mel_rate"], 0.0)
        first = module.bootstrap_ci(targets, probabilities, seed=20260911, repetitions=100)
        second = module.bootstrap_ci(targets, probabilities, seed=20260911, repetitions=100)
        self.assertEqual(first, second)
        self.assertTrue(all(0.0 <= low <= high <= 1.0 for low, high in first.values()))

    def test_entry_contains_no_training_or_backward_call(self) -> None:
        tree = ast.parse((HERE / "test.py").read_text(encoding="utf-8"))
        forbidden = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in {"train", "backward", "step", "zero_grad"}:
                    forbidden.append((node.func.attr, node.lineno))
        self.assertEqual(forbidden, [])

    def test_release_gate_records_current_authorization(self) -> None:
        module.validate_release()
        output = module.resolve(module.load_config()["evaluation"]["output"])
        self.assertFalse(output.exists())

    def test_generated_input_report_is_non_inferential(self) -> None:
        report = json.loads((HERE / "inputs.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "PASS")
        self.assertFalse(report["model_inference_performed"])
        self.assertFalse(report["pad_ufes_20_used"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
