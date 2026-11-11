#!/usr/bin/env python3
"""Atomic, interrupt-safe checkpoint bundles for PiFuV2 multitask training."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import random
import shutil
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


BUNDLE_VERSION = "pifuv2_multitask_checkpoint_v2"
REQUIRED_TRAINER_KEYS = {"epoch", "best_fitness", "ema", "optimizer", "scaler", "train_args"}
REQUIRED_RUNTIME_KEYS = {
    "bundle_version",
    "fingerprint",
    "scheduler",
    "stopper",
    "selection_state",
    "rng",
    "data_progress",
    "training_state",
}
REQUIRED_TRAINING_STATE_KEYS = {"model", "ema", "ema_updates", "optimizer", "scaler"}


class CheckpointContractError(RuntimeError):
    """Raised when a checkpoint is incomplete or incompatible."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_fingerprint(config: dict[str, Any], files: list[Path]) -> str:
    """Hash behavior-changing configuration and immutable contract files."""
    digest = hashlib.sha256()
    digest.update(json.dumps(config, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"))
    for path in sorted((Path(path).resolve() for path in files), key=lambda value: str(value).casefold()):
        if not path.is_file():
            raise FileNotFoundError(f"Fingerprint input not found: {path}")
        encoded = str(path).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    # Windows rejects fsync on a read-only CRT descriptor (EBADF), even when
    # the underlying file exists and is readable.  rb+ preserves contents and
    # gives fsync a writable descriptor without modifying payload bytes.
    with path.open("rb+") as handle:
        os.fsync(handle.fileno())


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    _fsync_file(temporary)
    os.replace(temporary, path)


def _torch_load(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def validate_trainer_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file() or not zipfile_is_torch_archive(path):
        raise CheckpointContractError(f"Trainer checkpoint is missing or not a torch ZIP archive: {path}")
    checkpoint = _torch_load(path)
    if not isinstance(checkpoint, dict):
        raise CheckpointContractError(f"Trainer checkpoint payload is not a mapping: {path}")
    missing = REQUIRED_TRAINER_KEYS - set(checkpoint)
    if missing:
        raise CheckpointContractError(f"Trainer checkpoint is missing keys {sorted(missing)}: {path}")
    epoch = checkpoint.get("epoch")
    if not isinstance(epoch, int) or epoch < 0:
        raise CheckpointContractError(f"Trainer checkpoint has invalid epoch={epoch!r}: {path}")
    if not checkpoint.get("optimizer"):
        raise CheckpointContractError(f"Trainer checkpoint optimizer is absent or stripped: {path}")
    if checkpoint.get("ema") is None:
        raise CheckpointContractError(f"Trainer checkpoint EMA is absent: {path}")
    return checkpoint


def zipfile_is_torch_archive(path: Path) -> bool:
    import zipfile

    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.namelist()
            return any(name.endswith("data.pkl") for name in members) and any("/data/" in name for name in members)
    except zipfile.BadZipFile:
        return False


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_states = state.get("torch_cuda") or []
    if cuda_states:
        if not torch.cuda.is_available():
            raise CheckpointContractError("Checkpoint contains CUDA RNG state but CUDA is unavailable")
        if len(cuda_states) != torch.cuda.device_count():
            raise CheckpointContractError(
                f"CUDA RNG device count mismatch: checkpoint={len(cuda_states)} runtime={torch.cuda.device_count()}"
            )
        torch.cuda.set_rng_state_all(cuda_states)


def _sampler_state(trainer) -> dict[str, Any]:
    loader = getattr(trainer, "train_loader", None)
    sampler = getattr(loader, "sampler", None)
    result: dict[str, Any] = {"sampler_type": type(sampler).__name__ if sampler is not None else ""}
    if sampler is not None and hasattr(sampler, "state_dict"):
        result["sampler_state_dict"] = sampler.state_dict()
    elif sampler is not None and hasattr(sampler, "epoch"):
        result["sampler_epoch"] = int(sampler.epoch)
    generator = getattr(loader, "generator", None)
    if generator is not None:
        result["loader_generator_state"] = generator.get_state()
    return result


def _to_cpu_clone(value: Any) -> Any:
    """Detach tensors so runtime_state is a lossless, device-neutral snapshot."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _to_cpu_clone(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_to_cpu_clone(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu_clone(child) for child in value)
    return copy.deepcopy(value)


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def capture_training_state(trainer) -> dict[str, Any]:
    """Capture FP32 live state because Ultralytics last.pt quantizes resume tensors."""
    model = getattr(trainer, "model", None)
    ema_wrapper = getattr(trainer, "ema", None)
    ema_model = getattr(ema_wrapper, "ema", None)
    optimizer = getattr(trainer, "optimizer", None)
    scaler = getattr(trainer, "scaler", None)
    if model is None or ema_model is None or optimizer is None or scaler is None:
        raise CheckpointContractError("Trainer model/EMA/optimizer/scaler are not initialized")
    return {
        "model": _to_cpu_clone(_unwrap_model(model).state_dict()),
        "ema": _to_cpu_clone(_unwrap_model(ema_model).state_dict()),
        "ema_updates": int(getattr(ema_wrapper, "updates", 0)),
        "optimizer": _to_cpu_clone(optimizer.state_dict()),
        "scaler": _to_cpu_clone(scaler.state_dict()),
    }


def capture_runtime_state(
    trainer,
    *,
    fingerprint: str,
    selection_state: dict[str, Any],
    interruption: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stopper = getattr(trainer, "stopper", None)
    scheduler = getattr(trainer, "scheduler", None)
    if stopper is None or scheduler is None:
        raise CheckpointContractError("Trainer stopper/scheduler are not initialized")
    return {
        "bundle_version": BUNDLE_VERSION,
        "fingerprint": fingerprint,
        "scheduler": scheduler.state_dict(),
        "stopper": dict(vars(stopper)),
        "selection_state": selection_state,
        "training_state": capture_training_state(trainer),
        "rng": capture_rng_state(),
        "data_progress": {
            "completed_epoch": int(trainer.epoch),
            "next_epoch": int(trainer.epoch) + 1,
            "sampler": _sampler_state(trainer),
        },
        "interruption": dict(interruption or {}),
        "versions": {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        },
        "saved_local_time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }


def restore_runtime_state(trainer, state: dict[str, Any], expected_fingerprint: str) -> None:
    missing = REQUIRED_RUNTIME_KEYS - set(state)
    if missing:
        raise CheckpointContractError(f"Runtime state is missing keys: {sorted(missing)}")
    if state["bundle_version"] != BUNDLE_VERSION:
        raise CheckpointContractError(
            f"Checkpoint bundle version mismatch: {state['bundle_version']} != {BUNDLE_VERSION}"
        )
    if state["fingerprint"] != expected_fingerprint:
        raise CheckpointContractError(
            f"Checkpoint fingerprint mismatch: {state['fingerprint']} != {expected_fingerprint}"
        )
    training_state = state["training_state"]
    missing_training = REQUIRED_TRAINING_STATE_KEYS - set(training_state)
    if missing_training:
        raise CheckpointContractError(f"Training state is missing keys: {sorted(missing_training)}")
    _unwrap_model(trainer.model).load_state_dict(training_state["model"], strict=True)
    _unwrap_model(trainer.ema.ema).load_state_dict(training_state["ema"], strict=True)
    trainer.ema.updates = int(training_state["ema_updates"])
    trainer.optimizer.load_state_dict(training_state["optimizer"])
    trainer.scaler.load_state_dict(training_state["scaler"])
    trainer.scheduler.load_state_dict(state["scheduler"])
    for name, value in state["stopper"].items():
        setattr(trainer.stopper, name, value)
    sampler_state = state["data_progress"].get("sampler", {})
    loader = getattr(trainer, "train_loader", None)
    sampler = getattr(loader, "sampler", None)
    if sampler_state.get("sampler_type") and type(sampler).__name__ != sampler_state["sampler_type"]:
        raise CheckpointContractError(
            f"Sampler type mismatch: {type(sampler).__name__} != {sampler_state['sampler_type']}"
        )
    if "sampler_state_dict" in sampler_state:
        if not hasattr(sampler, "load_state_dict"):
            raise CheckpointContractError("Saved sampler state_dict cannot be restored by current sampler")
        sampler.load_state_dict(sampler_state["sampler_state_dict"])
    elif "sampler_epoch" in sampler_state and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(int(sampler_state["sampler_epoch"]))
    generator = getattr(loader, "generator", None)
    saved_generator_state = sampler_state.get("loader_generator_state")
    if saved_generator_state is not None and generator is None:
        raise CheckpointContractError("Saved loader generator state has no runtime target")
    if loader is not None and hasattr(loader, "reset"):
        # InfiniteDataLoader creates an iterator during setup. Rebuild it first:
        # iterator construction consumes loader.generator for a worker base seed.
        # Restoring the saved generator before reset would therefore advance the
        # next-epoch sample permutation compared with uninterrupted training.
        loader.reset()
    if saved_generator_state is not None:
        generator.set_state(saved_generator_state)
    # Restore global RNG last so iterator reconstruction cannot perturb the
    # exact safe-epoch snapshot.
    restore_rng_state(state["rng"])


@dataclass(frozen=True)
class ValidatedBundle:
    path: Path
    epoch: int
    trainer_checkpoint: Path
    runtime_state: Path
    fingerprint: str


class CheckpointBundleManager:
    """Publish, validate, discover, alias, and rotate complete bundles."""

    def __init__(self, root: Path, *, fingerprint: str, keep_last: int = 3) -> None:
        self.root = Path(root).resolve()
        self.fingerprint = fingerprint
        self.keep_last = int(keep_last)
        if not fingerprint or len(fingerprint) != 64:
            raise ValueError("Checkpoint fingerprint must be a SHA-256 hex string")
        if self.keep_last < 1:
            raise ValueError("keep_last must be at least 1")
        self.alias_dir = self.root / "aliases"

    def publish(self, trainer_checkpoint: Path, runtime_state: dict[str, Any]) -> ValidatedBundle:
        trainer_checkpoint = Path(trainer_checkpoint).resolve()
        checkpoint = validate_trainer_checkpoint(trainer_checkpoint)
        epoch = int(checkpoint["epoch"])
        if runtime_state.get("fingerprint") != self.fingerprint:
            raise CheckpointContractError("Runtime fingerprint differs from bundle manager fingerprint")
        if runtime_state.get("data_progress", {}).get("completed_epoch") != epoch:
            raise CheckpointContractError("Trainer checkpoint epoch and runtime completed_epoch differ")
        self.root.mkdir(parents=True, exist_ok=True)
        final = self.root / f"epoch_{epoch:04d}"
        temporary = self.root / f".epoch_{epoch:04d}.tmp-{os.getpid()}"
        if final.exists():
            existing = self.validate(final)
            if existing.fingerprint == self.fingerprint:
                return existing
            raise CheckpointContractError(f"Refusing to overwrite incompatible complete bundle: {final}")
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=False)
        trainer_copy = temporary / "trainer.pt"
        runtime_path = temporary / "runtime_state.pt"
        shutil.copy2(trainer_checkpoint, trainer_copy)
        torch.save(runtime_state, runtime_path)
        _fsync_file(trainer_copy)
        _fsync_file(runtime_path)
        manifest = {
            "bundle_version": BUNDLE_VERSION,
            "epoch": epoch,
            "fingerprint": self.fingerprint,
            "files": {
                "trainer.pt": {"bytes": trainer_copy.stat().st_size, "sha256": sha256_file(trainer_copy)},
                "runtime_state.pt": {"bytes": runtime_path.stat().st_size, "sha256": sha256_file(runtime_path)},
            },
        }
        _atomic_json(temporary / "manifest.json", manifest)
        success = temporary / "_SUCCESS"
        success.write_text(json.dumps({"epoch": epoch, "fingerprint": self.fingerprint}), encoding="utf-8")
        _fsync_file(success)
        os.replace(temporary, final)
        _atomic_json(self.root / "latest.json", {"bundle": final.name, "epoch": epoch})
        validated = self.validate(final)
        self.rotate()
        return validated

    def validate(self, path: Path) -> ValidatedBundle:
        path = Path(path).resolve()
        success = path / "_SUCCESS"
        manifest_path = path / "manifest.json"
        trainer_path = path / "trainer.pt"
        runtime_path = path / "runtime_state.pt"
        if not path.is_dir() or not success.is_file() or not manifest_path.is_file():
            raise CheckpointContractError(f"Incomplete checkpoint bundle: {path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("bundle_version") != BUNDLE_VERSION:
            raise CheckpointContractError(f"Unsupported checkpoint bundle version: {path}")
        for name, file_path in (("trainer.pt", trainer_path), ("runtime_state.pt", runtime_path)):
            expected = manifest.get("files", {}).get(name, {})
            if not file_path.is_file() or file_path.stat().st_size != expected.get("bytes"):
                raise CheckpointContractError(f"Missing or wrong-sized {name}: {path}")
            if sha256_file(file_path) != expected.get("sha256"):
                raise CheckpointContractError(f"SHA-256 mismatch for {name}: {path}")
        checkpoint = validate_trainer_checkpoint(trainer_path)
        runtime = _torch_load(runtime_path)
        missing = REQUIRED_RUNTIME_KEYS - set(runtime)
        if missing:
            raise CheckpointContractError(f"Runtime state is missing keys {sorted(missing)}: {path}")
        missing_training = REQUIRED_TRAINING_STATE_KEYS - set(runtime["training_state"])
        if missing_training:
            raise CheckpointContractError(f"Training state is missing keys {sorted(missing_training)}: {path}")
        epoch = int(checkpoint["epoch"])
        if epoch != int(manifest["epoch"]) or runtime["data_progress"]["completed_epoch"] != epoch:
            raise CheckpointContractError(f"Epoch disagreement inside bundle: {path}")
        if runtime["fingerprint"] != manifest["fingerprint"]:
            raise CheckpointContractError(f"Fingerprint disagreement inside bundle: {path}")
        return ValidatedBundle(path, epoch, trainer_path, runtime_path, runtime["fingerprint"])

    def resolve(self, value: str | Path = "auto") -> ValidatedBundle:
        if str(value).lower() != "auto":
            candidate = Path(value).resolve()
            if candidate.is_file() and candidate.name == "trainer.pt":
                candidate = candidate.parent
            bundle = self.validate(candidate)
            if bundle.fingerprint != self.fingerprint:
                raise CheckpointContractError("Explicit resume bundle fingerprint mismatch")
            return bundle
        candidates: list[ValidatedBundle] = []
        if self.root.is_dir():
            for path in self.root.glob("epoch_[0-9][0-9][0-9][0-9]"):
                try:
                    bundle = self.validate(path)
                except CheckpointContractError:
                    continue
                if bundle.fingerprint == self.fingerprint:
                    candidates.append(bundle)
        if not candidates:
            raise CheckpointContractError(f"No complete compatible checkpoint bundle under {self.root}")
        return max(candidates, key=lambda bundle: bundle.epoch)

    def set_alias(self, name: str, bundle: ValidatedBundle) -> None:
        if not name.replace("_", "").isalnum():
            raise ValueError(f"Invalid checkpoint alias: {name!r}")
        validated = self.validate(bundle.path)
        _atomic_json(self.alias_dir / f"{name}.json", {"bundle": validated.path.name, "epoch": validated.epoch})

    def _pinned_bundle_names(self) -> set[str]:
        result: set[str] = set()
        if self.alias_dir.is_dir():
            for path in self.alias_dir.glob("*.json"):
                try:
                    result.add(json.loads(path.read_text(encoding="utf-8"))["bundle"])
                except (OSError, KeyError, json.JSONDecodeError):
                    continue
        return result

    def rotate(self) -> None:
        complete: list[ValidatedBundle] = []
        for path in self.root.glob("epoch_[0-9][0-9][0-9][0-9]"):
            try:
                complete.append(self.validate(path))
            except CheckpointContractError:
                continue
        complete.sort(key=lambda bundle: bundle.epoch, reverse=True)
        pinned = self._pinned_bundle_names()
        for bundle in complete[self.keep_last :]:
            if bundle.path.name not in pinned:
                shutil.rmtree(bundle.path)


class SignalLatch:
    """Record SIGINT/SIGTERM without interrupting an optimizer step."""

    def __init__(self) -> None:
        self.signum: int | None = None
        self.requested_at: float | None = None

    @property
    def requested(self) -> bool:
        return self.signum is not None

    def handler(self, signum, _frame) -> None:
        if self.signum is None:
            self.signum = int(signum)
            self.requested_at = time.time()

    def metadata(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "signal": signal.Signals(self.signum).name if self.signum is not None else "",
            "requested_unix_time": self.requested_at,
            "save_boundary": "completed_epoch",
        }


@contextlib.contextmanager
def installed_signal_latch(latch: SignalLatch):
    previous = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, latch.handler)
    try:
        yield latch
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


class InterruptSafeTrainerMixin:
    """Add atomic bundles, full control-state restore, and epoch-boundary signals."""

    def __init__(self, *args, checkpoint_contract: dict[str, Any] | None = None, **kwargs):
        contract = dict(checkpoint_contract or {})
        self.checkpoint_contract_enabled = bool(contract.get("enabled", False))
        self.checkpoint_fingerprint = str(contract.get("fingerprint", ""))
        self.checkpoint_keep_last = int(contract.get("keep_last", 3))
        self.resume_bundle_path = Path(contract["resume_bundle"]).resolve() if contract.get("resume_bundle") else None
        self.selection_state: dict[str, dict[str, Any]] = dict(contract.get("selection_state") or {})
        self._signal_latch = SignalLatch()
        self._checkpoint_callback_added = False
        self._last_bundled_epoch = -1
        super().__init__(*args, **kwargs)
        if self.checkpoint_contract_enabled and not bool(getattr(self.args, "save", False)):
            raise CheckpointContractError("Checkpoint contract requires save=True")
        root = Path(contract.get("root") or (Path(self.save_dir) / "checkpoint_bundles"))
        self.bundle_manager = (
            CheckpointBundleManager(root, fingerprint=self.checkpoint_fingerprint, keep_last=self.checkpoint_keep_last)
            if self.checkpoint_contract_enabled
            else None
        )

    def _selection_updates(self) -> list[str]:
        if not getattr(self, "metrics", None):
            return []
        candidates = {
            "best_binary_iou": ("metrics/binary_lesion_iou", False),
            "best_macro_f1": ("metrics/group_macro_f1", False),
            "best_joint_score": ("metrics/joint_score", False),
            "best_feasible_joint": ("metrics/joint_score", True),
        }
        updated = []
        feasible = float(self.metrics.get("metrics/feasible_joint", 0.0)) == 1.0
        for alias, (metric_name, requires_feasible) in candidates.items():
            if metric_name not in self.metrics or (requires_feasible and not feasible):
                continue
            value = float(self.metrics[metric_name])
            previous = self.selection_state.get(alias)
            if previous is None or value > float(previous["value"]):
                self.selection_state[alias] = {"value": value, "epoch": int(self.epoch), "metric": metric_name}
                updated.append(alias)
        return updated

    def _safe_epoch_boundary(self, _trainer) -> None:
        if self._signal_latch.requested:
            self.stop = True

    def _seed_epoch_boundary_rng(self, _trainer) -> None:
        """Make each safe epoch independently reproducible across process restarts."""
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "-1")))
        base_seed = int(getattr(self.args, "seed", 0)) + 1 + max(rank, 0)
        epoch_seed = (base_seed + int(self.epoch) * 1_000_003) % (2**32)
        random.seed(epoch_seed)
        np.random.seed(epoch_seed)
        torch.manual_seed(epoch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(epoch_seed)
        loader = getattr(self, "train_loader", None)
        sampler = getattr(loader, "sampler", None)
        generator = getattr(sampler, "generator", None)
        if generator is None:
            generator = getattr(loader, "generator", None)
        if generator is not None:
            generator.manual_seed((epoch_seed + 0x5DEECE66D) % (2**63 - 1))
        if rank < 0 and loader is not None and hasattr(loader, "reset"):
            # PyTorch RandomSampler lazily consumes an extra zero-length
            # randperm when an iterator is exhausted. A continued
            # InfiniteDataLoader and a freshly resumed iterator would
            # otherwise start the next epoch from different generator offsets.
            # Rebuilding on every single-process epoch makes both paths equal.
            loader.reset()
            random.seed(epoch_seed)
            np.random.seed(epoch_seed)
            torch.manual_seed(epoch_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(epoch_seed)

    def _publish_safe_epoch_bundle(self, _trainer) -> None:
        """Publish after save/model callbacks, at the actual next-epoch boundary."""
        if not self.checkpoint_contract_enabled:
            return
        if self._last_bundled_epoch == int(self.epoch):
            return  # final_eval emits on_fit_epoch_end again after stripping last.pt
        checkpoint = _torch_load(Path(self.last))
        checkpoint_epoch = int(checkpoint.get("epoch", -1))
        if checkpoint_epoch == -1 and not checkpoint.get("optimizer") and self._last_bundled_epoch >= 0:
            return  # final_eval callback after Ultralytics strips last.pt
        if checkpoint_epoch != int(self.epoch):
            raise CheckpointContractError(
                f"Trainer epoch/checkpoint mismatch at safe boundary: trainer={self.epoch}, checkpoint={checkpoint_epoch}"
            )
        updated_aliases = self._selection_updates()
        runtime = capture_runtime_state(
            self,
            fingerprint=self.checkpoint_fingerprint,
            selection_state=self.selection_state,
            interruption=self._signal_latch.metadata(),
        )
        bundle = self.bundle_manager.publish(Path(self.last), runtime)
        for alias in updated_aliases:
            self.bundle_manager.set_alias(alias, bundle)
        self.bundle_manager.rotate()
        self._last_bundled_epoch = int(self.epoch)

    def train(self, *args, **kwargs):
        if self.checkpoint_contract_enabled and not self._checkpoint_callback_added:
            # Appended last so integration callbacks cannot perturb the epoch's
            # reproducible RNG streams before the first batch is drawn.
            self.add_callback("on_train_epoch_start", self._seed_epoch_boundary_rng)
            self.add_callback("on_train_epoch_end", self._safe_epoch_boundary)
            # BaseTrainer saves last.pt before on_fit_epoch_end. Appending this
            # callback captures state after all existing fit-epoch callbacks,
            # immediately before the loop either breaks or starts the next epoch.
            self.add_callback("on_fit_epoch_end", self._publish_safe_epoch_bundle)
            self._checkpoint_callback_added = True
        if not self.checkpoint_contract_enabled:
            return super().train(*args, **kwargs)
        try:
            with installed_signal_latch(self._signal_latch):
                return super().train(*args, **kwargs)
        except BaseException:
            emergency = self.bundle_manager.root / f"emergency_incomplete_{int(time.time())}.json"
            _atomic_json(
                emergency,
                {
                    "status": "INCOMPLETE_NOT_AUTO_RESUMABLE",
                    "reason": "exception_before_safe_epoch_bundle",
                    "epoch": int(getattr(self, "epoch", -1)),
                    "signal": self._signal_latch.metadata(),
                },
            )
            raise

    def save_model(self):
        return super().save_model()

    def resume_training(self, ckpt):
        super().resume_training(ckpt)
        if not self.checkpoint_contract_enabled or self.resume_bundle_path is None:
            return
        bundle = self.bundle_manager.resolve(self.resume_bundle_path)
        runtime = _torch_load(bundle.runtime_state)
        restore_runtime_state(self, runtime, self.checkpoint_fingerprint)
        self.selection_state = dict(runtime["selection_state"])
