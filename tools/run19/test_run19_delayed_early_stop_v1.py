from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = PROJECT_ROOT / "tools"
RUN19_DIR = TOOLS_DIR / "run19"
for path in (TOOLS_DIR, RUN19_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from imaplusplus_partial_multitask_attention_delayed_early_stop_v1 import (
    DelayedEarlyStopping,
    DelayedUtilityGatedBinaryLesionGuidedAttentionTrainer,
    run19_early_stop_fitness,
)
from imaplusplus_partial_multitask_lesion_attention_utility_selection_v1 import (
    UtilityEligibleBinaryLesionGuidedAttentionTrainer,
)


def test_run19_inherits_frozen_run16_attention_trainer():
    assert issubclass(
        DelayedUtilityGatedBinaryLesionGuidedAttentionTrainer,
        UtilityEligibleBinaryLesionGuidedAttentionTrainer,
    )


def test_ineligible_fitness_is_strictly_below_eligible_domain():
    cold = run19_early_stop_fitness(
        {"metrics/utility_eligible": 0.0, "metrics/joint_score": 0.45}, 0.31
    )
    mature = run19_early_stop_fitness(
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


def test_activated_stopper_state_round_trip():
    original = DelayedEarlyStopping(patience=30, start_epoch=40)
    assert original(40, 0.2) is False
    assert original(55, 0.3) is False
    saved = dict(vars(original))
    restored = DelayedEarlyStopping(patience=30, start_epoch=40)
    for name, value in saved.items():
        setattr(restored, name, value)
    assert vars(restored) == saved
    assert restored(84, 0.2) is False
    assert restored(85, 0.2) is True
