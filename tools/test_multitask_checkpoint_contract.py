from __future__ import annotations

import sys
import tempfile
import unittest
import random
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from multitask_checkpoint_contract import (
    BUNDLE_VERSION,
    CheckpointBundleManager,
    CheckpointContractError,
    capture_runtime_state,
    capture_rng_state,
    restore_runtime_state,
)


def write_trainer_checkpoint(path: Path, epoch: int) -> None:
    torch.save(
        {
            "epoch": epoch,
            "best_fitness": 0.5,
            "model": None,
            "ema": nn.Linear(2, 2),
            "optimizer": {"state": {0: {"step": 1}}, "param_groups": [{"params": [0]}]},
            "scaler": {"scale": 1.0},
            "train_args": {"epochs": 10},
        },
        path,
    )


def runtime_state(epoch: int, fingerprint: str) -> dict:
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    return {
        "bundle_version": BUNDLE_VERSION,
        "fingerprint": fingerprint,
        "scheduler": {"last_epoch": epoch},
        "stopper": {"best_epoch": epoch},
        "selection_state": {},
        "training_state": {
            "model": model.state_dict(),
            "ema": model.state_dict(),
            "ema_updates": epoch + 1,
            "optimizer": optimizer.state_dict(),
            "scaler": {},
        },
        "rng": capture_rng_state(),
        "data_progress": {"completed_epoch": epoch, "next_epoch": epoch + 1, "sampler": {}},
        "interruption": {},
    }


class CheckpointContractTests(unittest.TestCase):
    def test_capture_restore_is_lossless_for_live_training_state(self) -> None:
        class ResettingLoader:
            def __init__(self) -> None:
                self.sampler = None
                self.generator = torch.Generator().manual_seed(123)

            def reset(self) -> None:
                torch.empty((), dtype=torch.int64).random_(generator=self.generator)
                random.random()

        model = nn.Linear(2, 2)
        ema = nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0 - 0.1 * epoch)
        model(torch.ones(2, 2)).sum().backward()
        optimizer.step()
        trainer = SimpleNamespace(
            model=model,
            ema=SimpleNamespace(ema=ema, updates=7),
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            stopper=SimpleNamespace(best_epoch=3, best_fitness=0.7),
            epoch=3,
            train_loader=ResettingLoader(),
        )
        fingerprint = "e" * 64
        state = capture_runtime_state(trainer, fingerprint=fingerprint, selection_state={"best": {"epoch": 3}})
        expected_model = {key: value.clone() for key, value in model.state_dict().items()}
        expected_optimizer = state["training_state"]["optimizer"]
        expected_loader_generator = state["data_progress"]["sampler"]["loader_generator_state"]
        expected_python_rng = state["rng"]["python"]

        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(10)
            for parameter in ema.parameters():
                parameter.zero_()
        optimizer.param_groups[0]["lr"] = 9.0
        trainer.ema.updates = 0
        restore_runtime_state(trainer, state, fingerprint)

        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, expected_model[key]))
        actual_optimizer = optimizer.state_dict()
        self.assertEqual(actual_optimizer["param_groups"], expected_optimizer["param_groups"])
        self.assertEqual(set(actual_optimizer["state"]), set(expected_optimizer["state"]))
        for parameter_id, expected_state in expected_optimizer["state"].items():
            actual_state = actual_optimizer["state"][parameter_id]
            self.assertEqual(set(actual_state), set(expected_state))
            for key, expected_value in expected_state.items():
                actual_value = actual_state[key]
                if isinstance(expected_value, torch.Tensor):
                    self.assertTrue(torch.equal(actual_value, expected_value))
                else:
                    self.assertEqual(actual_value, expected_value)
        self.assertEqual(trainer.ema.updates, 7)
        self.assertTrue(torch.equal(trainer.train_loader.generator.get_state(), expected_loader_generator))
        self.assertEqual(random.getstate(), expected_python_rng)

    def test_publish_validate_auto_and_incomplete_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            fingerprint = "a" * 64
            manager = CheckpointBundleManager(tmp_path / "bundles", fingerprint=fingerprint, keep_last=2)
            checkpoint = tmp_path / "last.pt"
            write_trainer_checkpoint(checkpoint, epoch=0)
            bundle = manager.publish(checkpoint, runtime_state(0, fingerprint))
            self.assertTrue((bundle.path / "_SUCCESS").is_file())
            self.assertEqual(manager.resolve("auto").epoch, 0)

            incomplete = manager.root / "epoch_0001"
            incomplete.mkdir()
            (incomplete / "trainer.pt").write_bytes(b"partial")
            with self.assertRaisesRegex(CheckpointContractError, "Incomplete"):
                manager.validate(incomplete)
            self.assertEqual(manager.resolve("auto").epoch, 0)

    def test_rotation_preserves_aliased_best(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            fingerprint = "b" * 64
            manager = CheckpointBundleManager(tmp_path / "bundles", fingerprint=fingerprint, keep_last=1)
            checkpoint = tmp_path / "last.pt"
            write_trainer_checkpoint(checkpoint, epoch=0)
            first = manager.publish(checkpoint, runtime_state(0, fingerprint))
            manager.set_alias("best_joint_score", first)

            write_trainer_checkpoint(checkpoint, epoch=1)
            second = manager.publish(checkpoint, runtime_state(1, fingerprint))
            self.assertTrue(first.path.is_dir())
            self.assertTrue(second.path.is_dir())
            self.assertEqual(manager.resolve("auto").epoch, 1)

    def test_explicit_resume_rejects_fingerprint_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            source = CheckpointBundleManager(tmp_path / "bundles", fingerprint="c" * 64)
            checkpoint = tmp_path / "last.pt"
            write_trainer_checkpoint(checkpoint, epoch=2)
            bundle = source.publish(checkpoint, runtime_state(2, "c" * 64))
            incompatible = CheckpointBundleManager(source.root, fingerprint="d" * 64)
            with self.assertRaisesRegex(CheckpointContractError, "fingerprint mismatch"):
                incompatible.resolve(bundle.path)


if __name__ == "__main__":
    unittest.main()
