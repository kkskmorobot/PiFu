from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from ultralytics.utils.torch_utils import get_flops, get_flops_with_torch_profiler


PROJECT = Path(r"E:\App\DaiMa\JetBrains\Project\PiFuV2")
APP_PATH = PROJECT / "QianDuan" / "app.py"
OUT = Path(r"C:\Users\ROBOT\Documents\皮肤识别\unified_performance_results.json")
MODELS = [
    ("run22", "Plain seed43 representative"),
    ("lgx", "LGLA v1 gate-balance seed43 representative"),
]
IMGSZ = 640
DIRECT_WARMUP = 20
DIRECT_REPEATS = 100
PIPELINE_WARMUP = 5
PIPELINE_REPEATS = 30


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stats(values: list[float]) -> dict:
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, max(0, int(np.ceil(0.95 * len(ordered))) - 1))]
    return {
        "n": len(values),
        "mean_ms": round(statistics.fmean(values), 4),
        "median_ms": round(statistics.median(values), 4),
        "p95_ms": round(p95, 4),
        "min_ms": round(min(values), 4),
        "max_ms": round(max(values), 4),
        "throughput_fps_from_mean": round(1000.0 / statistics.fmean(values), 4),
    }


spec = importlib.util.spec_from_file_location("pifuv2_frontend_app", APP_PATH)
app = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = app
spec.loader.exec_module(app)

sample_path = sorted((PROJECT / "Data" / "processed_skin7_v3" / "images" / "test").glob("*"))[0]
with Image.open(sample_path) as source:
    sample_image = source.convert("RGB").copy()

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this unified benchmark")

torch.backends.cudnn.benchmark = True
torch.set_grad_enabled(False)
results = []

for model_id, description in MODELS:
    metadata = app.find_model(model_id)
    wrapper = app.load_model(metadata)
    model = wrapper.model.float().eval()

    parameters_total = sum(p.numel() for p in model.parameters())
    parameters_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Ultralytics reports FLOPs as two operations per multiply-accumulate.
    model.cpu()
    thop_gflops_raw = float(get_flops(model, imgsz=IMGSZ))
    profiler_gflops_raw = float(get_flops_with_torch_profiler(model, imgsz=IMGSZ))
    thop_gflops = round(thop_gflops_raw, 6) if thop_gflops_raw > 0 else None
    profiler_gflops = round(profiler_gflops_raw, 6) if profiler_gflops_raw > 0 else None

    model.cuda()
    x = torch.randn(1, 3, IMGSZ, IMGSZ, device="cuda", dtype=torch.float32)
    with torch.inference_mode():
        for _ in range(DIRECT_WARMUP):
            model(x)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        direct_times = []
        for _ in range(DIRECT_REPEATS):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(x)
            end.record()
            end.synchronize()
            direct_times.append(float(start.elapsed_time(end)))
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()

    predictor = app.partial_label_predictor_class()
    predict_kwargs = {
        "source": sample_image,
        "imgsz": IMGSZ,
        "device": 0,
        "verbose": False,
        "conf": 0.50,
        "predictor": predictor,
    }
    for _ in range(PIPELINE_WARMUP):
        wrapper.predict(**predict_kwargs)
    torch.cuda.synchronize()
    pipeline_wall = []
    reported_preprocess = []
    reported_inference = []
    reported_postprocess = []
    for _ in range(PIPELINE_REPEATS):
        started = time.perf_counter()
        prediction = wrapper.predict(**predict_kwargs)[0]
        torch.cuda.synchronize()
        pipeline_wall.append((time.perf_counter() - started) * 1000.0)
        speed = prediction.speed or {}
        reported_preprocess.append(float(speed.get("preprocess", 0.0)))
        reported_inference.append(float(speed.get("inference", 0.0)))
        reported_postprocess.append(float(speed.get("postprocess", 0.0)))

    results.append(
        {
            "model_id": model_id,
            "description": description,
            "experiment": metadata["experiment"],
            "weights_path": str(metadata["weights"]),
            "weights_sha256": sha256(metadata["weights"]),
            "weights_bytes": metadata["weights"].stat().st_size,
            "parameters_total": parameters_total,
            "parameters_trainable": parameters_trainable,
            "flops": {
                "input": [1, 3, IMGSZ, IMGSZ],
                "ultralytics_thop_gflops": thop_gflops,
                "torch_profiler_gflops": profiler_gflops,
                "interpretation": "forward pass; 2 operations per multiply-accumulate; unsupported operations may be omitted by profiler",
                "missing_reason": None if thop_gflops is not None else "Ultralytics THOP profiling returned 0 or failed",
            },
            "direct_forward_fp32_cuda": {
                "warmup": DIRECT_WARMUP,
                "repeats": DIRECT_REPEATS,
                **stats(direct_times),
                "peak_allocated_mib": round(peak_allocated / 1024**2, 3),
                "peak_reserved_mib": round(peak_reserved / 1024**2, 3),
            },
            "ultralytics_pipeline_fp32_cuda": {
                "warmup": PIPELINE_WARMUP,
                "repeats": PIPELINE_REPEATS,
                "source": str(sample_path),
                "source_size": list(sample_image.size),
                "wall_clock": stats(pipeline_wall),
                "reported_preprocess": stats(reported_preprocess),
                "reported_inference": stats(reported_inference),
                "reported_postprocess": stats(reported_postprocess),
            },
        }
    )

    del prediction, predictor, x, model, wrapper
    app.CURRENT_WRAPPER = None
    app.CURRENT_MODEL_ID = None
    gc.collect()
    torch.cuda.empty_cache()

plain, lgla = results
comparison = {
    "parameter_delta": lgla["parameters_total"] - plain["parameters_total"],
    "parameter_delta_percent": round((lgla["parameters_total"] / plain["parameters_total"] - 1.0) * 100.0, 4),
    "thop_gflops_delta": None if plain["flops"]["ultralytics_thop_gflops"] is None or lgla["flops"]["ultralytics_thop_gflops"] is None else round(lgla["flops"]["ultralytics_thop_gflops"] - plain["flops"]["ultralytics_thop_gflops"], 6),
    "thop_gflops_delta_percent": None if plain["flops"]["ultralytics_thop_gflops"] is None or lgla["flops"]["ultralytics_thop_gflops"] is None else round((lgla["flops"]["ultralytics_thop_gflops"] / plain["flops"]["ultralytics_thop_gflops"] - 1.0) * 100.0, 4),
    "direct_forward_mean_ms_delta": round(lgla["direct_forward_fp32_cuda"]["mean_ms"] - plain["direct_forward_fp32_cuda"]["mean_ms"], 4),
    "direct_forward_mean_ms_delta_percent": round((lgla["direct_forward_fp32_cuda"]["mean_ms"] / plain["direct_forward_fp32_cuda"]["mean_ms"] - 1.0) * 100.0, 4),
    "pipeline_mean_ms_delta": round(lgla["ultralytics_pipeline_fp32_cuda"]["wall_clock"]["mean_ms"] - plain["ultralytics_pipeline_fp32_cuda"]["wall_clock"]["mean_ms"], 4),
    "pipeline_mean_ms_delta_percent": round((lgla["ultralytics_pipeline_fp32_cuda"]["wall_clock"]["mean_ms"] / plain["ultralytics_pipeline_fp32_cuda"]["wall_clock"]["mean_ms"] - 1.0) * 100.0, 4),
}

report = {
    "title": "Plain 与 LGLA v1 统一性能及 FLOPs 基准",
    "tested_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    "protocol": {
        "device": torch.cuda.get_device_name(0),
        "cuda_runtime": torch.version.cuda,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "os": platform.platform(),
        "input": [1, 3, IMGSZ, IMGSZ],
        "precision": "FP32",
        "batch": 1,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "timing_boundary_direct": "CUDA event around model(x), excludes preprocessing/postprocessing and host transfer",
        "timing_boundary_pipeline": "wall clock around wrapper.predict on the same in-memory PIL image, includes Ultralytics preprocessing/postprocessing and synchronization",
        "warning": "Single-machine engineering benchmark, not a clinical or cross-hardware result. Do not compare with screenshot request_total_ms or batch duration as if they were the same timing boundary.",
    },
    "models": results,
    "comparison_lgla_minus_plain": comparison,
}
OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"comparison": comparison, "models": [{"id": item["model_id"], "params": item["parameters_total"], "gflops": item["flops"]["ultralytics_thop_gflops"], "direct_ms": item["direct_forward_fp32_cuda"]["mean_ms"], "pipeline_ms": item["ultralytics_pipeline_fp32_cuda"]["wall_clock"]["mean_ms"]} for item in results]}, ensure_ascii=False, indent=2))
