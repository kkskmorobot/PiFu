"""Portable reproduction entry points; historical files are read-only inputs.

Examples: python reproduction/run.py check
          python reproduction/run.py train --run lgx --smoke
          python reproduction/run.py validation --run lgx
          python reproduction/run.py evaluate --suite test2 --run lgx
"""
from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import sys
import textwrap
from collections import Counter
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import bootstrap

ROOT = bootstrap.PROJECT_ROOT
RUNS = ("run18", "run22", "run24", "run21", "run23", "run25", "run26", "run27", "run28", "run33", "lg1", "lg2", "lgx", "lg3", "ab1", "ab2")
DATA = None


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def json_save(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def imported(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def resolve_source(relative):
    path = Path(relative)
    if path.parts and path.parts[0] == "Data":
        return DATA.joinpath(*path.parts[1:])
    return ROOT / path


def runner_and_config(run, strict=True):
    module = imported(ROOT / "tools" / run / f"train_{run}_from_pycharm.py", f"repro_original_{run}")
    if strict:
        config = module.load_locked_config()  # Audit only: original config SHA/hyperparameters.
    else:
        import yaml
        config = yaml.safe_load(module.CONFIG_PATH.read_text(encoding="utf-8-sig"))
    return module, config


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def check_manifest(path, full_hash=False):
    rows = read_rows(path)
    missing = []
    counts = Counter()
    for row in rows:
        counts[row["split"]] += 1
        for field in ("image_path", "mask_path"):
            candidate = DATA / row[field]
            if not candidate.is_file():
                missing.append(str(candidate))
        if full_hash and row.get("image_sha256"):
            candidate = DATA / row["image_path"]
            if candidate.is_file() and sha256(candidate) != row["image_sha256"]:
                raise RuntimeError(f"Image content differs from manifest: {candidate}")
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} manifest assets: {missing[:5]}")
    return {"path": str(path), "sha256": sha256(path), "rows": len(rows), "splits": dict(counts)}


def verify_training_inputs(run, full_hash=False, strict=True):
    runner, config = runner_and_config(run, strict=strict)
    checks = {}
    for relative, expected in runner.PROTECTED_SHA256.items():
        path = resolve_source(relative)
        if not path.is_file():
            raise FileNotFoundError(f"Required training asset missing: {relative}")
        actual = sha256(path)
        if strict and actual != expected:
            raise RuntimeError(f"Original protected input missing/changed: {relative}")
        checks[relative] = {"expected": expected, "actual": actual, "matches_original": actual == expected}
    if hasattr(runner, "SOURCE_MULTITASK_MANIFEST"):
        runner.SOURCE_MULTITASK_MANIFEST = DATA / "processed_isic2019_classification_incremental_v1/multitask_manifest.csv"
    manifest = ROOT / "tools" / run / f"{run}_development_manifest.csv"
    counts = dict(Counter(r["split"] for r in read_rows(manifest)))
    if strict:
        expected, counts = runner.development_manifest_payload()
        if manifest.read_bytes() != expected:
            raise RuntimeError(f"Development manifest projection changed: {manifest}")
    report = check_manifest(manifest, full_hash)
    report.update(run=run, original_config_sha256=sha256(runner.CONFIG_PATH), task_counts=counts, protected_hashes=checks)
    return runner, config, report


def frozen_index():
    index = json.loads((ROOT / "reproduction/frozen_models.json").read_text(encoding="utf-8-sig"))
    return index["models"]


def verify_frozen(run, spec=None):
    record = frozen_index()[run]
    path = ROOT / record["weights"]
    if not path.is_file() or sha256(path) != record["sha256"]:
        raise RuntimeError(f"Frozen evaluation model missing/changed: {path}")
    if spec:
        for index_key, spec_key in (("source_fingerprint", "fingerprint"), ("source_trainer_sha256", "trainer_sha256"), ("source_runtime_sha256", "runtime_sha256")):
            if record[index_key] != spec[spec_key]:
                raise RuntimeError(f"Frozen source identity mismatch: {run}/{index_key}")
        if record["source_bundle"].replace("\\", "/") != spec["bundle"].replace("\\", "/"):
            raise RuntimeError(f"Frozen bundle identity mismatch: {run}")
    return record


def load_frozen(run):
    import torch
    # Import the original definition before loading its trusted local checkpoint.
    for source in (ROOT / "tools" / run).glob("imaplusplus*.py"):
        importlib.import_module(source.stem)
    record = verify_frozen(run)
    checkpoint = torch.load(ROOT / record["weights"], map_location="cpu", weights_only=False)
    model = checkpoint.get("model") if isinstance(checkpoint, dict) else checkpoint
    if model is None:
        raise RuntimeError(f"Missing model object: {run}")
    model.float().cpu().eval()
    if record.get("state_tensor_count") and len(model.state_dict()) != record["state_tensor_count"]:
        raise RuntimeError(f"Frozen tensor count mismatch: {run}")
    return model, {"source": str(ROOT / record["weights"]), "sha256": record["sha256"], "source_bundle": record["source_bundle"], "state": "exact_fp32_ema_export", "original_runtime_sha256": record["source_runtime_sha256"]}


def user_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def verify_bundle(value):
    """Validate a newly trained complete bundle against its own recorded hashes."""
    from multitask_checkpoint_contract import BUNDLE_VERSION
    path = user_path(value)
    alias = None
    if path.is_file() and path.suffix.lower() == ".json":
        alias = path
        pointer = json.loads(path.read_text(encoding="utf-8-sig"))
        relative = Path(pointer["bundle"])
        path = relative if relative.is_absolute() else (path.parent.parent if path.parent.name == "aliases" else path.parent) / relative
        path = path.resolve()
    manifest_path, success_path = path / "manifest.json", path / "_SUCCESS"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    success = json.loads(success_path.read_text(encoding="utf-8-sig"))
    if manifest.get("bundle_version") != BUNDLE_VERSION:
        raise RuntimeError(f"Unsupported checkpoint bundle version: {path}")
    for key in ("epoch", "fingerprint"):
        if success.get(key) != manifest.get(key):
            raise RuntimeError(f"Incomplete checkpoint: manifest/_SUCCESS {key} disagreement: {path}")
    if alias is not None and int(pointer["epoch"]) != int(manifest["epoch"]):
        raise RuntimeError(f"Alias points to the wrong epoch: {alias}")
    for name in ("trainer.pt", "runtime_state.pt"):
        candidate = path / name
        expected = manifest["files"][name]
        if candidate.stat().st_size != int(expected["bytes"]) or sha256(candidate) != expected["sha256"]:
            raise RuntimeError(f"Checkpoint content differs from its own manifest: {candidate}")
    return {"kind": "new_training_bundle", "bundle": str(path), "alias": str(alias) if alias else None, "epoch": int(manifest["epoch"]), "fingerprint": manifest["fingerprint"], "manifest_sha256": sha256(manifest_path), "files": manifest["files"], "state": "exact_fp32_training_ema"}


def source_record(args):
    if args.bundle:
        return verify_bundle(args.bundle)
    if args.weights:
        path = user_path(args.weights)
        if not path.is_file():
            raise FileNotFoundError(path)
        return {"kind": "deployment_checkpoint", "weights": str(path), "sha256": sha256(path), "state": "deployment_weights_not_guaranteed_lossless_training_ema", "precision_note": "May contain FP16-rounded weights. Converting to FP32 cannot recover discarded precision."}
    return {"kind": "frozen_paper_export", **verify_frozen(args.run)}


def load_requested(args, record):
    import torch
    if record["kind"] == "frozen_paper_export":
        return load_frozen(args.run)
    for source in (ROOT / "tools" / args.run).glob("imaplusplus*.py"):
        importlib.import_module(source.stem)
    if record["kind"] == "new_training_bundle":
        # Original loader reconstructs the architecture then strictly restores lossless EMA.
        loader = imported(ROOT / "tools/test2/test.py", "repro_bundle_loader")
        model, _ = loader.load_exact_ema(Path(record["bundle"]))
        return model, record
    checkpoint = torch.load(record["weights"], map_location="cpu", weights_only=False)
    model = checkpoint
    if isinstance(checkpoint, dict):
        model = checkpoint.get("ema")
        if model is None:
            model = checkpoint.get("model")
    if model is None or not hasattr(model, "state_dict"):
        raise RuntimeError("--weights requires a complete PyTorch/Ultralytics model checkpoint, not a bare state_dict")
    source = {**record, "stored_floating_dtypes": sorted({str(t.dtype) for t in model.state_dict().values() if t.is_floating_point()})}
    return model.float().cpu().eval(), source


def output_dir(label):
    parent = ROOT / "outputs"
    parent.mkdir(exist_ok=True)
    path = parent / f"{label}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    path.mkdir(exist_ok=False)
    return path


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def smoke_rows(rows, count):
    # Include all available classes, then segmentation-only examples, then fill in source order.
    selected, seen = [], set()
    for code in range(8):
        candidate = next((r for r in rows if r["classification_target"] == str(code)), None)
        if candidate and len(selected) < count:
            selected.append(candidate)
            seen.add(candidate["isic_id"])
    for row in sorted(rows, key=lambda r: r["has_segmentation"] != "1"):
        if len(selected) >= count:
            break
        if row["isic_id"] not in seen:
            selected.append(row)
            seen.add(row["isic_id"])
    return selected


def portable_data(manifest, destination, splits, limit=0):
    import yaml
    rows = read_rows(manifest)
    selected = []
    image_lists = {}
    for split in splits:
        source_rows = [r for r in rows if r["split"] == split]
        active = smoke_rows(source_rows, limit) if limit else source_rows
        if not active:
            raise RuntimeError(f"No rows for {split}: {manifest}")
        selected += active
        image_list = destination / f"{split}_images.txt"
        image_list.write_text("\n".join((DATA / r["image_path"]).resolve().as_posix() for r in active) + "\n", encoding="utf-8")
        image_lists[split] = str(image_list.resolve())
    generated_manifest = destination / "manifest.csv"
    write_csv(generated_manifest, selected)
    payload = {
        "path": str(DATA), "train": image_lists.get("train"),
        "val": image_lists.get("val", image_lists.get("external")), "test": image_lists.get("test"),
        "masks_dir": "masks", "nc": 1, "names": {0: "lesion"},
        "partial_label_manifest": str(generated_manifest.resolve()),
        "classification_taxonomy": str((ROOT / "tools/classification_taxonomy_v1.yaml").resolve()),
    }
    data_yaml = destination / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return data_yaml, generated_manifest, selected


def trainer_factory(runner, config, ultra, checkpoint_contract):
    """Reuse the exact original trainer call and its keyword expressions, without old release paths."""
    tree = ast.parse(Path(runner.__file__).read_text(encoding="utf-8-sig"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "train_formal")
    call = next(node.value for node in ast.walk(function) if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "trainer" for t in node.targets))
    class_name = call.func.id
    imported_class = next((node.module, alias.name) for node in ast.walk(function) if isinstance(node, ast.ImportFrom) for alias in node.names if (alias.asname or alias.name) == class_name)
    trainer_class = getattr(importlib.import_module(imported_class[0]), imported_class[1])
    kwargs = {}
    namespace = {"config": config, "ultra": ultra, "checkpoint_contract": checkpoint_contract, "float": float, "int": int}
    for keyword in call.keywords:
        kwargs[keyword.arg] = eval(compile(ast.Expression(keyword.value), runner.__file__, "eval"), {"__builtins__": {}}, namespace)
    return trainer_class, kwargs


def original_loss_names(runner, config):
    trainer_class, _ = trainer_factory(runner, config, dict(config["ultralytics"]), {})
    tree = ast.parse(textwrap.dedent(inspect.getsource(trainer_class.get_validator)))
    return next(ast.literal_eval(node.value) for node in ast.walk(tree) if isinstance(node, ast.Assign) and any(isinstance(target, ast.Attribute) and target.attr == "loss_names" for target in node.targets))


def configure_early_stopping(trainer_class, config, run, start_override=None, patience_override=None):
    """Honor editable YAML values while preserving the original stopper algorithm."""
    module = importlib.import_module(trainer_class.__module__)
    section = config.get("early_stopping", {})
    changes = {}
    for field, suffix, override in (("start_epoch", "EARLY_STOP_START_EPOCH", start_override), ("patience_after_start", "PATIENCE_AFTER_START", patience_override)):
        name = f"{run.upper()}_{suffix}"
        original = getattr(module, name)
        requested = override if override is not None else section.get(field, original)
        value = int(requested)
        if value != requested or value < 0:
            raise ValueError(f"early_stopping.{field} must be a nonnegative integer")
        setattr(module, name, value)
        changes[field] = {"original": original, "effective": value}
    return changes


def train(args):
    import torch
    runner, config, evidence = verify_training_inputs(args.run, strict=False)
    out = output_dir(f"{'SMOKE_NOT_THESIS_' if args.smoke else ''}train_{args.run}")
    inputs = out / "inputs"
    inputs.mkdir()
    yaml_path, manifest, rows = portable_data(runner.DEVELOPMENT_MANIFEST, inputs, ("train", "val"), 32 if args.smoke else 0)
    ultra = {**config["ultralytics"], **config["augmentations"]}
    ultra.update(model=str(ROOT / config["ultralytics"]["model"]), data=str(yaml_path), project=str(out), name="training", exist_ok=False)
    overrides = {key: getattr(args, key) for key in ("epochs", "batch", "workers", "device", "seed", "imgsz", "lr0", "patience") if getattr(args, key) is not None}
    ultra.update(overrides)
    if args.smoke:
        ultra.update(epochs=1, batch=2, workers=0, device=0 if torch.cuda.is_available() else "cpu", amp=torch.cuda.is_available(), plots=False)
    elif not torch.cuda.is_available() and args.device is None:
        ultra.update(device="cpu", amp=False)
        overrides.update(device="cpu", amp=False)
    fingerprint = hashlib.sha256(json.dumps({"input_evidence": evidence, "overrides": ultra, "early_stop_start_epoch_override": args.early_stop_start_epoch}, sort_keys=True).encode()).hexdigest()
    contract = {"enabled": True, "fingerprint": fingerprint, "root": out / "training/checkpoint_bundles", "keep_last": int(config["checkpoint"]["keep_last"]), "resume_bundle": None}
    trainer_class, kwargs = trainer_factory(runner, config, ultra, contract)
    early_stopping = configure_early_stopping(trainer_class, config, args.run, args.early_stop_start_epoch, args.patience)
    json_save(out / "launch.json", {"mode": "SMOKE_NOT_THESIS" if args.smoke else "FULL_RETRAINING", "run": args.run, "inputs": evidence, "trainer": f"{trainer_class.__module__}.{trainer_class.__name__}", "overrides": ultra, "early_stopping": early_stopping, "user_or_hardware_overrides": overrides, "config_matches_original": sha256(runner.CONFIG_PATH) == runner.LOCKED_CONFIG_SHA256, "portable_manifest_sha256": sha256(manifest), "fingerprint": fingerprint, "row_count": len(rows), "prepared_only": args.prepare})
    if not args.prepare:
        os.chdir(ROOT / "tools")  # Original local AMP check model lives here.
        instance = trainer_class(**kwargs)
        instance.train()
        json_save(out / "complete.json", {"status": "COMPLETE", "mode": "SMOKE_NOT_THESIS" if args.smoke else "FULL_RETRAINING", "epoch": int(instance.epoch)})
    print(json.dumps({"status": "PREPARED" if args.prepare else "COMPLETE", "output": str(out)}, ensure_ascii=False))


def test_config(suite, full_hash=False):
    import yaml
    source = ROOT / "tools" / suite
    original = imported(source / "test.py", f"repro_{suite}_original")
    config = original.load_config()
    for section, split, role in (("internal_test", "test", "imaplusplus_binary"), ("external", "external", "isic2018_task1_incremental")):
        spec = config[section]
        for field, hash_field in (("yaml", "yaml_sha256"), ("manifest", "manifest_sha256"), ("source_manifest", "source_manifest_sha256")):
            if sha256(resolve_source(spec[field])) != spec[hash_field]:
                raise RuntimeError(f"Original test input changed: {spec[field]}")
        original.validate_manifest(ROOT / spec["manifest"], spec["manifest_sha256"], split, role, int(spec["total_samples"]), int(spec["classification_samples"]))
        check_manifest(ROOT / spec["manifest"], full_hash)
    if sha256(ROOT / config["taxonomy"]["path"]) != config["taxonomy"]["sha256"]:
        raise RuntimeError("Taxonomy differs from frozen test")
    return original, config


def evaluate(args):
    import torch
    original, config = test_config(args.suite)
    custom = bool(args.bundle or args.weights)
    if custom:
        if not args.run:
            raise ValueError("--bundle/--weights requires --run to identify the model architecture")
        runner, training_config = runner_and_config(args.run, strict=False)
        provided_source = source_record(args)
        specs = [{"run": args.run, "seed": training_config["ultralytics"]["seed"], "bundle": provided_source.get("bundle", str(Path(provided_source["weights"]).parent) if "weights" in provided_source else "")}]
        loss_count = len(original_loss_names(runner, training_config))
        if loss_count not in (4, 5):
            raise RuntimeError(f"Unsupported original loss count: {loss_count}")
        engine_suite = "test1" if loss_count == 4 else "test2"
        original = imported(ROOT / "tools" / engine_suite / "test.py", f"repro_custom_{engine_suite}_engine")
    else:
        specs = [s for s in config["checkpoints"] if not args.run or normalize_run(s["run"]) == args.run]
        if not specs:
            raise ValueError(f"{args.run} is not a frozen candidate of {args.suite}")
    mode = ("NEW_TRAINING_EVALUATION" if args.bundle else "DEPLOYMENT_WEIGHTS_EVALUATION") if custom else "SMOKE_NOT_THESIS" if args.limit else "FROZEN_TEST_REPRODUCTION"
    out = output_dir(f"{'SMOKE_NOT_THESIS_' if args.limit else ''}{'new_' if custom else ''}{args.suite}")
    portable = copy.deepcopy(config)
    if args.limit:
        portable["evaluation"]["workers"] = 0
    for section, split in (("internal_test", "test"), ("external", "external")):
        folder = out / section
        folder.mkdir()
        data_yaml, manifest, rows = portable_data(ROOT / config[section]["manifest"], folder, (split,), args.limit)
        portable[section].update(yaml=str(data_yaml), manifest=str(manifest), segmentation_samples=len(rows))
    evidence = {args.run: provided_source} if custom else {normalize_run(s["run"]): verify_frozen(normalize_run(s["run"]), s) for s in specs}
    json_save(out / "launch.json", {"mode": mode, "smoke_not_thesis": bool(args.limit), "suite": args.suite, "original_config_sha256": sha256(ROOT / "tools" / args.suite / f"{args.suite}.yaml"), "models": evidence, "original_config": config, "generated_config": portable, "prepared_only": args.prepare})
    if args.prepare:
        print(json.dumps({"status": "PREPARED", "output": str(out)}))
        return
    if args.limit:
        original.restricted_metrics = lambda targets, probabilities: {"scope": "SMOKE_NOT_THESIS", "support": {"total": len(targets)}, "accuracy": float((probabilities.argmax(1) == targets).mean()) if len(targets) else None}
        original.bootstrap_ci = lambda *a, **k: {}
    per_seed = []
    for spec in specs:
        run = normalize_run(spec["run"])
        # Only change the transport/loading hook; the original evaluation math is retained.
        original.load_exact_ema = (lambda _bundle: load_requested(args, provided_source)) if custom else (lambda _bundle, chosen=run: load_frozen(chosen))
        result = {"run": spec["run"], "seed": spec["seed"]}
        for section in ("internal_test", "external"):
            folder = out / run / section
            folder.mkdir(parents=True)
            result[section] = original.evaluate_dataset(spec, portable, section, folder)
        per_seed.append(result)
    report = {"status": "COMPLETE", "mode": mode, "smoke_not_thesis": bool(args.limit), "per_seed": per_seed}
    if not args.limit and len(per_seed) == 3:
        report["aggregate"] = original.aggregate_seed_metrics(per_seed)
    json_save(out / "result.json", report)
    print(json.dumps({"status": "COMPLETE", "output": str(out)}))


def validation(args):
    import numpy as np
    import torch
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_dataloader
    from ultralytics.data.utils import check_det_dataset
    from imaplusplus_partial_multitask_v1 import PartialLabelMultiTaskValidator, PartialLabelSemanticDataset
    original = importlib.import_module("evaluate_imaplusplus_multitask_v1")
    runner, config, evidence = verify_training_inputs(args.run, strict=False)
    record = source_record(args)
    out = output_dir(f"{'SMOKE_NOT_THESIS_' if args.limit else ''}validation_{args.run}")
    inputs = out / "inputs"
    inputs.mkdir()
    data_yaml, manifest, rows = portable_data(runner.DEVELOPMENT_MANIFEST, inputs, ("val",), args.limit)
    mode = "NEW_TRAINING_EVALUATION" if args.bundle else "DEPLOYMENT_WEIGHTS_EVALUATION" if args.weights else "SMOKE_NOT_THESIS" if args.limit else "FROZEN_VALIDATION_REPRODUCTION"
    json_save(out / "launch.json", {"mode": mode, "smoke_not_thesis": bool(args.limit), "input_evidence": evidence, "rows": len(rows), "weights": record, "prepared_only": args.prepare})
    if args.prepare:
        print(json.dumps({"status": "PREPARED", "output": str(out)}))
        return
    model, source = load_requested(args, record)
    data = check_det_dataset(str(data_yaml), split="val")
    model_path = Path(record["bundle"]) / "trainer.pt" if args.bundle else Path(record["weights"]) if args.weights else ROOT / record["weights"]
    cfg = get_cfg(overrides={"task": "semantic", "mode": "val", "data": str(data_yaml), "model": str(model_path), "split": "val", "imgsz": 640, "batch": 16, "workers": 0 if args.limit else 8, "device": "0" if torch.cuda.is_available() else "cpu", "plots": False, "save_json": False, "verbose": False})
    dataset = PartialLabelSemanticDataset(img_path=data["val"], imgsz=640, batch_size=16, augment=False, hyp=cfg, rect=True, cache=False, single_cls=False, stride=max(int(model.stride.max()), 32), pad=0.0, prefix="Portable validation: ", task="semantic", classes=None, data=data, fraction=1.0, partial_label_manifest=manifest, expected_split="val")
    loader = build_dataloader(dataset, batch=16, workers=cfg.workers, shuffle=False, rank=-1, drop_last=False)
    validator = PartialLabelMultiTaskValidator(dataloader=loader, save_dir=out, args=cfg, selection={k: config["selection"][k] for k in ("max_mel_to_nv_rate", "min_binary_iou", "min_group_macro_f1")})
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    loss_names = original_loss_names(runner, config)
    frozen = SimpleNamespace(device=device, data=data, amp=device.type == "cuda", ema=SimpleNamespace(ema=model), model=model, args=SimpleNamespace(compile=False), loss_items=torch.zeros(len(loss_names), device=device), stopper=SimpleNamespace(possible_stop=False), epoch=0, epochs=1, world_size=1, label_loss_items=lambda _loss, prefix="val": {})
    metrics = validator(frozen)
    group_rows = original.aggregate_groups(validator.classification_rows)
    group_metrics = original.classification_metrics(original.confusion_from_rows(group_rows, "true_id", "pred_id"))
    probability = np.asarray([r["probabilities"] for r in group_rows])
    targets = np.asarray([r["true_id"] for r in group_rows])
    calibration = original.calibration_metrics(probability, targets)
    serializable = [{k: v for k, v in row.items() if k != "probabilities"} | {"probabilities": row["probabilities"].tolist()} for row in group_rows]
    json_save(out / "result.json", {"status": "COMPLETE", "mode": mode, "smoke_not_thesis": bool(args.limit), "run": args.run, "weight_source": source, "metrics": {k: float(v) for k, v in metrics.items()}, "group_classification": group_metrics, "group_calibration": calibration, "group_predictions": serializable})
    print(json.dumps({"status": "COMPLETE", "output": str(out)}))


def normalize_run(value):
    value = str(value).lower()
    return "run" + value if value.isdigit() else value


def main(argv=None):
    global DATA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="Read-only check of original configs, data paths and frozen weights")
    check.add_argument("--full-data-hash", action="store_true")
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--run", type=normalize_run, choices=RUNS, required=True)
    train_parser.add_argument("--smoke", action="store_true", help="One epoch on small train/val subsets; NOT thesis results")
    train_parser.add_argument("--prepare", action="store_true", help="Validate/build an execution plan without training")
    train_parser.add_argument("--epochs", type=int)
    train_parser.add_argument("--batch", type=int)
    train_parser.add_argument("--device")
    train_parser.add_argument("--workers", type=int)
    train_parser.add_argument("--seed", type=int)
    train_parser.add_argument("--imgsz", type=int)
    train_parser.add_argument("--lr0", type=float)
    train_parser.add_argument("--patience", type=int, help="Early-stop patience after activation; 0 disables early stopping")
    train_parser.add_argument("--early-stop-start-epoch", type=int, help="First epoch at which early stopping is active")
    val_parser = commands.add_parser("validation")
    val_parser.add_argument("--run", type=normalize_run, choices=RUNS, required=True)
    val_parser.add_argument("--limit", type=int, default=0)
    val_parser.add_argument("--prepare", action="store_true")
    val_source = val_parser.add_mutually_exclusive_group()
    val_source.add_argument("--bundle", type=Path, help="New training epoch bundle directory or alias JSON; load exact FP32 EMA")
    val_source.add_argument("--weights", type=Path, help="Deployment .pt; not guaranteed to preserve lossless training EMA")
    eval_parser = commands.add_parser("evaluate")
    eval_parser.add_argument("--suite", choices=("test1", "test2"), required=True)
    eval_parser.add_argument("--run", type=normalize_run, choices=RUNS)
    eval_parser.add_argument("--limit", type=int, default=0, help="Partial inference smoke only, not a frozen result")
    eval_parser.add_argument("--prepare", action="store_true")
    eval_source = eval_parser.add_mutually_exclusive_group()
    eval_source.add_argument("--bundle", type=Path, help="New training epoch bundle directory or alias JSON; requires --run")
    eval_source.add_argument("--weights", type=Path, help="Deployment .pt; requires --run, not guaranteed lossless EMA")
    args = parser.parse_args(argv)
    if hasattr(args, "limit") and args.limit < 0:
        parser.error("--limit cannot be negative")
    DATA = bootstrap.configure(args.data_root, require_data=True)
    if args.command == "check":
        reports = []
        for run in RUNS:
            print(f"Checking {run}...", flush=True)
            _, _, report = verify_training_inputs(run, args.full_data_hash and run == RUNS[0])
            report["frozen"] = verify_frozen(run)
            reports.append(report)
        for suite in ("test1", "test2"):
            _, config = test_config(suite, args.full_data_hash)
            for spec in config["checkpoints"]:
                verify_frozen(normalize_run(spec["run"]), spec)
        print(json.dumps({"status": "PASS", "runs": len(reports), "data_root": str(DATA), "full_image_hash_check": args.full_data_hash}, ensure_ascii=False))
    elif args.command == "train":
        train(args)
    elif args.command == "validation":
        validation(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
