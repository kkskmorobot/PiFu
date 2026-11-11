#!/usr/bin/env python3
"""Read-only train/val loader contract smoke for the combined partial data."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from imaplusplus_partial_multitask_v1 import PartialLabelSemanticDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "train": {"total": 23_462, "segmentation": 10_483, "classification": 13_715, "joint": 736},
    "val": {"total": 1_744, "segmentation": 292, "classification": 1_488, "joint": 36},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_yaml = args.data.resolve() if args.data.is_absolute() else (PROJECT_ROOT / args.data).resolve()
    if not data_yaml.is_file():
        raise FileNotFoundError(data_yaml)
    from ultralytics.cfg import get_cfg
    from ultralytics.data.utils import check_det_dataset

    data = check_det_dataset(str(data_yaml))
    manifest = Path(data["partial_label_manifest"])
    if not manifest.is_absolute():
        manifest = (Path(data["path"]) / manifest).resolve()
    cfg = get_cfg(overrides={"task": "semantic", "mode": "val", "imgsz": args.imgsz, "batch": 3})
    for name in ("mosaic", "mixup", "cutmix", "copy_paste"):
        setattr(cfg, name, 0.0)
    reports = {}
    for split in ("train", "val"):
        dataset = PartialLabelSemanticDataset(
            img_path=data[split],
            imgsz=args.imgsz,
            batch_size=3,
            augment=False,
            hyp=cfg,
            rect=False,
            cache=False,
            single_cls=False,
            stride=32,
            pad=0.0,
            prefix=f"{split}: ",
            task="semantic",
            classes=None,
            data=data,
            fraction=1.0,
            partial_label_manifest=manifest,
            expected_split=split,
        )
        records = [dataset.partial_records[Path(path).resolve()] for path in dataset.im_files]
        counts = Counter()
        indices = {}
        for index, record in enumerate(records):
            has_seg = record["has_segmentation"]
            has_cls = record["has_classification"]
            counts["segmentation"] += int(has_seg)
            counts["classification"] += int(has_cls)
            counts["joint"] += int(has_seg and has_cls)
            key = "joint" if has_seg and has_cls else "segmentation_only" if has_seg else "classification_only"
            indices.setdefault(key, index)
        actual = {"total": len(dataset), **{key: counts[key] for key in ("segmentation", "classification", "joint")}}
        if actual != EXPECTED[split]:
            raise RuntimeError(f"Unexpected {split} availability counts: {actual} != {EXPECTED[split]}")
        if set(indices) != {"segmentation_only", "classification_only", "joint"}:
            raise RuntimeError(f"{split} is missing a partial-label role: {sorted(indices)}")
        batch = PartialLabelSemanticDataset.collate_fn([dataset[indices[key]] for key in sorted(indices)])
        if tuple(batch["img"].shape) != (3, 3, args.imgsz, args.imgsz):
            raise RuntimeError(f"Unexpected {split} image batch shape: {tuple(batch['img'].shape)}")
        role_pairs = list(zip(batch["has_segmentation"].tolist(), batch["has_classification"].tolist()))
        if set(role_pairs) != {(True, False), (False, True), (True, True)}:
            raise RuntimeError(f"Unexpected {split} availability batch: {role_pairs}")
        reports[split] = {**actual, "sampled_roles": role_pairs}
    print(
        json.dumps(
            {
                "status": "PASS",
                "scope": "combined train/val loader only; test not exposed or read",
                "data": str(data_yaml),
                "cache_root": str(Path(data["path"]) / ".cache" / "partial_multitask_v1"),
                "splits": reports,
                "test_metrics_used": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
