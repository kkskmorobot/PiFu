"""Run8 lesion-balanced sampling on the fixed Run7 model and loss.

Each class keeps exactly its existing per-epoch image count.  Within a class,
lesions have equal total sampling probability; after selecting a lesion, one
of its available image variants is selected uniformly.  This removes the
implicit up-weighting caused by lesions with many augmented or duplicate
images without changing the established class exposure.
"""

from __future__ import annotations

import csv
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import Sampler

from ultralytics.data.build import InfiniteDataLoader, seed_worker
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.torch_utils import torch_distributed_zero_first

from skin7_lesion_guided import Skin7LesionGuidedTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class LesionBalancedSampler(Sampler[int]):
    """Keep class quotas fixed, then balance lesions within each class."""

    def __init__(
        self,
        lesion_to_indices: dict[str, list[int]],
        *,
        num_samples: int,
        seed: int,
        lesion_to_class: dict[str, str] | None = None,
    ):
        if not lesion_to_indices:
            raise ValueError("LesionBalancedSampler received no lesion groups.")
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, found {num_samples}")
        self.lesion_ids = tuple(sorted(lesion_to_indices))
        self.groups = tuple(tuple(lesion_to_indices[key]) for key in self.lesion_ids)
        if any(not group for group in self.groups):
            raise ValueError("Every lesion group must contain at least one dataset index.")
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.epoch = 0
        self.lesion_to_class = dict(lesion_to_class or {})
        if set(self.lesion_ids) != set(self.lesion_to_class):
            raise ValueError("lesion_to_class must identify every active lesion exactly once.")
        class_to_positions: dict[str, list[int]] = defaultdict(list)
        class_draw_counts: Counter[str] = Counter()
        for position, (lesion_id, group) in enumerate(zip(self.lesion_ids, self.groups)):
            disease = self.lesion_to_class[lesion_id]
            class_to_positions[disease].append(position)
            class_draw_counts[disease] += len(group)
        if sum(class_draw_counts.values()) != self.num_samples:
            raise ValueError(
                "Run8 fixed class quotas require num_samples to equal the active dataset size; "
                f"found {self.num_samples} samples and {sum(class_draw_counts.values())} active images."
            )
        self.class_to_positions = {key: tuple(value) for key, value in sorted(class_to_positions.items())}
        self.class_draw_counts = dict(sorted(class_draw_counts.items()))

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """Set the next logical epoch, including after checkpoint resume."""
        if epoch < 0:
            raise ValueError(f"epoch must be non-negative, found {epoch}")
        self.epoch = int(epoch)

    def sample_epoch(self, epoch: int) -> list[int]:
        """Materialize one deterministic epoch without changing sampler state."""
        generator = torch.Generator()
        generator.manual_seed(self.seed + int(epoch))
        result: list[int] = []
        for disease, lesion_positions in self.class_to_positions.items():
            choices = torch.randint(
                len(lesion_positions),
                (self.class_draw_counts[disease],),
                generator=generator,
            ).tolist()
            for choice in choices:
                group = self.groups[lesion_positions[choice]]
                image_position = int(torch.randint(len(group), (1,), generator=generator).item())
                result.append(group[image_position])
        permutation = torch.randperm(len(result), generator=generator).tolist()
        return [result[index] for index in permutation]

    def __iter__(self) -> Iterator[int]:
        logical_epoch = self.epoch
        indices = self.sample_epoch(logical_epoch)
        self.epoch = logical_epoch + 1
        return iter(indices)


def _read_metadata(metadata_path: Path) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Run8 train metadata not found: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"image_id", "lesion_id", "dx"}
    if not rows or required - set(rows[0]):
        raise ValueError(f"Run8 metadata must contain {sorted(required)}: {metadata_path}")

    image_rows: dict[str, dict[str, str]] = {}
    lesion_to_class: dict[str, str] = {}
    for row in rows:
        image_id, lesion_id, disease = (row[key].strip() for key in ("image_id", "lesion_id", "dx"))
        if not image_id or not lesion_id or not disease:
            raise ValueError(f"Blank image_id, lesion_id, or dx in metadata row: {row}")
        if image_id in image_rows:
            raise ValueError(f"Duplicate image_id in Run8 metadata: {image_id}")
        previous = lesion_to_class.setdefault(lesion_id, disease)
        if previous != disease:
            raise ValueError(f"Lesion {lesion_id} has inconsistent classes: {previous} and {disease}")
        image_rows[image_id] = row
    return image_rows, lesion_to_class


def build_lesion_groups(dataset, metadata_path: Path) -> tuple[dict[str, list[int]], dict[str, str], dict]:
    """Map the active Ultralytics train dataset indices to metadata lesions."""
    image_rows, all_lesion_classes = _read_metadata(metadata_path)
    image_files = tuple(getattr(dataset, "im_files", ()))
    if len(image_files) != len(dataset):
        raise RuntimeError(
            f"Run8 requires one im_files entry per dataset item; found {len(image_files)} files for {len(dataset)} items."
        )

    groups: dict[str, list[int]] = defaultdict(list)
    active_lesion_classes: dict[str, str] = {}
    active_image_classes: Counter[str] = Counter()
    missing: list[str] = []
    for index, image_path in enumerate(image_files):
        image_id = Path(image_path).stem
        row = image_rows.get(image_id)
        if row is None:
            missing.append(image_id)
            continue
        lesion_id, disease = row["lesion_id"].strip(), row["dx"].strip()
        groups[lesion_id].append(index)
        active_lesion_classes[lesion_id] = disease
        active_image_classes[disease] += 1
    if missing:
        preview = ", ".join(missing[:10])
        raise RuntimeError(f"Run8 metadata is missing {len(missing)} active train images; first: {preview}")

    lesion_classes = Counter(active_lesion_classes.values())
    group_sizes = [len(indices) for indices in groups.values()]
    audit = {
        "images": len(image_files),
        "lesions": len(groups),
        "multi_image_lesions": sum(size > 1 for size in group_sizes),
        "max_images_per_lesion": max(group_sizes),
        "images_by_class": dict(sorted(active_image_classes.items())),
        "lesions_by_class": dict(sorted(lesion_classes.items())),
        "unused_metadata_lesions": len(all_lesion_classes) - len(groups),
    }
    return dict(groups), active_lesion_classes, audit


def build_loader_with_sampler(dataset, *, batch: int, workers: int, sampler: Sampler[int], drop_last: bool):
    """Ultralytics 8.4.101 loader construction with a project sampler injected."""
    dataset_len = len(dataset)
    batch = min(batch, dataset_len)
    samples = len(sampler)
    drop_last = drop_last and bool(batch) and dataset_len % batch != 0
    batches = (samples // batch if drop_last else math.ceil(samples / batch)) if batch else 0
    device_count = torch.cuda.device_count()
    cpu_count = os.cpu_count() or 1
    num_workers = min(cpu_count // max(device_count, 1), workers, 0 if batches <= 1 else batches)
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return InfiniteDataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=False,
        num_workers=num_workers,
        sampler=sampler,
        prefetch_factor=4 if num_workers > 0 else None,
        pin_memory=device_count > 0,
        collate_fn=getattr(dataset, "collate_fn", None),
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=drop_last,
    )


class Skin7LesionBalancedTrainer(Skin7LesionGuidedTrainer):
    """Run7 trainer with only the train sampling policy replaced."""

    def __init__(self, *args, lesion_sampling: dict | None = None, **kwargs):
        self.lesion_sampling = dict(lesion_sampling or {})
        if not self.lesion_sampling.get("enabled", False):
            raise ValueError("Run8 requires lesion_sampling.enabled=true")
        metadata = Path(self.lesion_sampling.get("metadata", ""))
        self.lesion_metadata = metadata if metadata.is_absolute() else (PROJECT_ROOT / metadata).resolve()
        self.sampler_audit: dict = {}
        super().__init__(*args, **kwargs)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        if mode != "train":
            return super().get_dataloader(dataset_path, batch_size=batch_size, rank=rank, mode=mode)
        if rank != -1:
            raise RuntimeError("Run8 lesion-balanced sampler is preregistered for single-GPU training only.")
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        groups, lesion_classes, self.sampler_audit = build_lesion_groups(dataset, self.lesion_metadata)
        sampler = LesionBalancedSampler(
            groups,
            num_samples=len(dataset),
            seed=int(self.args.seed),
            lesion_to_class=lesion_classes,
        )
        LOGGER.info(
            "Run8 lesion-balanced sampler: "
            f"{self.sampler_audit['images']} draws/epoch from {self.sampler_audit['lesions']} lesions; "
            f"{self.sampler_audit['multi_image_lesions']} lesions have multiple active images; "
            f"max={self.sampler_audit['max_images_per_lesion']}."
        )
        LOGGER.info(f"Run8 active image counts by class: {self.sampler_audit['images_by_class']}")
        LOGGER.info(f"Run8 equal-lesion target counts by class: {self.sampler_audit['lesions_by_class']}")
        return build_loader_with_sampler(
            dataset,
            batch=batch_size,
            workers=self.args.workers,
            sampler=sampler,
            drop_last=self.args.compile,
        )

    def _setup_train(self):
        super()._setup_train()
        sampler = getattr(self.train_loader, "sampler", None)
        if not isinstance(sampler, LesionBalancedSampler):
            raise RuntimeError(f"Run8 train loader lost LesionBalancedSampler: {type(sampler).__name__}")
        # InfiniteDataLoader creates its iterator before resume_training sets
        # start_epoch. Reset here so a resumed run starts at seed+start_epoch,
        # not at epoch zero or from prefetched indices.
        sampler.set_epoch(self.start_epoch)
        self.train_loader.reset()
        LOGGER.info(f"Run8 sampler aligned to logical epoch {self.start_epoch} (seed={sampler.seed}).")
