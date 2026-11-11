from pathlib import Path
import sys

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = PROJECT_ROOT / "tools"
RUN22_DIR = TOOLS_DIR / "run22"
for path in (TOOLS_DIR, RUN22_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from imaplusplus_partial_multitask_plain_seed43_v1 import (
    DelayedEarlyStopping,
    run22_early_stop_fitness,
)


def test_ineligible_fitness_is_below_eligible_fitness():
    cold = run22_early_stop_fitness(
        {"metrics/utility_eligible": 0.0, "metrics/joint_score": 0.45}, 0.31
    )
    mature = run22_early_stop_fitness(
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
    assert stopper.best_epoch == 40
    for epoch in range(41, 70):
        assert stopper(epoch, 0.1) is False
    assert stopper.possible_stop is True
    assert stopper(70, 0.1) is True


def test_improvement_after_start_resets_patience():
    stopper = DelayedEarlyStopping(patience=30, start_epoch=40)
    assert stopper(40, 0.2) is False
    assert stopper(55, 0.3) is False
    assert stopper.best_epoch == 55
    assert stopper(84, 0.2) is False
    assert stopper(85, 0.2) is True


def test_run22_changes_only_formal_seed_to_43():
    config = yaml.safe_load((RUN22_DIR / "run22_preregistered_config_v1.yaml").read_text(encoding="utf-8"))
    assert config["contract"]["single_preregistered_factor"] == "change_only_random_seed_from_42_to_43"
    assert config["ultralytics"]["seed"] == 43
    assert config["ultralytics"]["batch"] == 16
    assert config["ultralytics"]["imgsz"] == 640
    assert config["multitask"]["image_head"] == "plain_p3_p4_global_average_pool_mlp_dropout020"
    assert config["attention_ablation"]["enabled"] is False
