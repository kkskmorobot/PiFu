#!/usr/bin/env python3
"""Reproduce Run18 validation with the lossless bundle EMA; never trains or accesses test."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from evaluate_run34_from_pycharm import load_config, validate_one_checkpoint


def main() -> None:
    config = load_config()
    temporary = Path(tempfile.mkdtemp(prefix="run34_exact_ema_reproduction_"))
    try:
        result = validate_one_checkpoint(config["checkpoints"][0], config, temporary)
        deltas = result["argmax_reproduction_delta"]
        if any(abs(float(value)) > 5e-4 for value in deltas.values()):
            raise RuntimeError(f"Run18 exact-EMA reproduction failed: {deltas}")
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "run": result["run"],
                    "seed": result["seed"],
                    "argmax_reproduction_delta": deltas,
                    "weight_source": result["weight_source"],
                    "training_performed": False,
                    "test_metrics_used": False,
                    "external_metrics_used": False,
                },
                indent=2,
            )
        )
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


if __name__ == "__main__":
    main()
