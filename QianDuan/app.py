"""PiFuV2 本地皮肤病灶模型展示与推理服务。"""

from __future__ import annotations

import argparse
import base64
import csv
import gc
import importlib
import io
import json
import math
import random
import re
import sys
import threading
import time
import uuid
import webbrowser
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError


FRONTEND_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = FRONTEND_ROOT.parent
TOOLS_ROOT = PROJECT_ROOT / "tools"
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
INDEX_PATH = FRONTEND_ROOT / "index.html"
RUNTIME_ROOT = FRONTEND_ROOT / "runtime"
ANNOTATIONS_PATH = RUNTIME_ROOT / "annotations.jsonl"
SAMPLE_IMAGES_ROOT = PROJECT_ROOT / "Data" / "processed_skin7_v3" / "images"
SAMPLE_DATA_ROOT = SAMPLE_IMAGES_ROOT.parent
MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_BATCH_COUNT = 100

CLASS_INFO = {
    "akiec": {"zh": "日光性角化 / 鲍恩病", "en": "Actinic Keratosis", "risk": "medium"},
    "bcc": {"zh": "基底细胞癌", "en": "Basal Cell Carcinoma", "risk": "high"},
    "bkl": {"zh": "良性角化样病变", "en": "Benign Keratosis", "risk": "low"},
    "df": {"zh": "皮肤纤维瘤", "en": "Dermatofibroma", "risk": "low"},
    "mel": {"zh": "黑色素瘤", "en": "Melanoma", "risk": "high"},
    "nv": {"zh": "黑色素细胞痣", "en": "Melanocytic Nevus", "risk": "low"},
    "vasc": {"zh": "血管性病变", "en": "Vascular Lesion", "risk": "low"},
    "scc": {"zh": "鳞状细胞癌", "en": "Squamous Cell Carcinoma", "risk": "high"},
}

MODEL_LOCK = threading.RLock()
SAMPLE_LOCK = threading.Lock()
BATCH_LOCK = threading.RLock()
CURRENT_MODEL_ID: str | None = None
CURRENT_WRAPPER: Any = None
LAST_SAMPLE: dict[str, Path | None] = {"test": None, "val": None}
BATCH_JOBS: dict[str, dict[str, Any]] = {}


def _prepend_import_path(path: Path) -> None:
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


_prepend_import_path(TOOLS_ROOT)


def model_family(run_number: int | None) -> tuple[str, int]:
    if run_number is None:
        return "YOLO26 二值分割 + 八分类（LGLA）", 8
    if run_number <= 3:
        return "YOLO11 实例分割", 7
    if run_number == 4:
        return "YOLO26 七类语义分割", 7
    if run_number <= 9:
        return "YOLO26 七类多任务", 7
    return "YOLO26 二值分割 + 八分类", 8


def discover_models() -> list[dict[str, Any]]:
    """扫描全部正式 results/weights/best.pt，兼容数字与字母实验编号。"""
    models: list[dict[str, Any]] = []
    if not EXPERIMENTS_ROOT.is_dir():
        return models
    for experiment in EXPERIMENTS_ROOT.iterdir():
        if not experiment.is_dir():
            continue
        match = re.match(r"^([A-Za-z0-9]+)_", experiment.name)
        if not match:
            continue
        series = match.group(1).lower()
        run_number = int(series) if series.isdigit() else None
        candidates = [
            path
            for path in experiment.glob("**/weights/best.pt")
            if "results" in {part.lower() for part in path.parts}
            and "smoke" not in {part.lower() for part in path.parts}
        ]
        if not candidates:
            continue
        weights = sorted(candidates, key=lambda path: (len(path.parts), str(path)))[0]
        readable = experiment.name.split("_", 1)[1].replace("_", " ")
        family, class_count = model_family(run_number)
        model_id = f"run{run_number}" if run_number is not None else series
        display_id = f"Run{run_number:02d}" if run_number is not None else series.upper()
        models.append(
            {
                "id": model_id,
                "run": run_number if run_number is not None else series,
                "run_number": run_number,
                "series": series,
                "tool_dir": f"run{run_number}" if run_number is not None else series,
                "label": f"{display_id} · {readable} · best.pt",
                "experiment": experiment.name,
                "weights": weights,
                "family": family,
                "class_count": class_count,
                "trained_at": weights.stat().st_mtime,
            }
        )
    return sorted(models, key=lambda item: item["trained_at"], reverse=True)


def public_model(model: dict[str, Any]) -> dict[str, Any]:
    return {key: model[key] for key in ("id", "run", "label", "experiment", "family", "class_count")}


def find_model(model_id: str | None) -> dict[str, Any]:
    models = discover_models()
    if not models:
        raise RuntimeError("experiments 中未找到正式 best.pt 模型")
    if not model_id:
        return models[0]
    for model in models:
        if model["id"] == model_id.lower():
            return model
    raise ValueError(f"未知模型：{model_id}")


def import_model_contract(model: dict[str, Any]) -> None:
    """在 PyTorch 反序列化前导入对应 Run 的自定义模型类。"""
    run_number = model["run_number"]
    if run_number is None:
        run_root = TOOLS_ROOT / model["tool_dir"]
        if not run_root.is_dir():
            raise FileNotFoundError(f"缺少模型结构目录：{run_root}")
        _prepend_import_path(run_root)
        sources = sorted(run_root.glob("imaplusplus*.py"))
        if not sources:
            raise FileNotFoundError(f"{model['series'].upper()} 缺少 imaplusplus 模型定义")
        for source in sources:
            importlib.import_module(source.stem)
        return
    if run_number <= 4:
        return
    if 5 <= run_number <= 9:
        run_root = TOOLS_ROOT / f"run{run_number}"
        _prepend_import_path(run_root)
        module_name = {
            5: "skin7_multitask",
            6: "skin7_cbam",
            7: "skin7_lesion_guided",
            8: "skin7_lesion_balanced",
            9: "skin7_label_smoothed",
        }[run_number]
        importlib.import_module("skin7_multitask")
        importlib.import_module(module_name)
        return
    if 10 <= run_number <= 17:
        module_name = {
            10: "imaplusplus_partial_multitask_v1",
            11: "imaplusplus_partial_multitask_mel_weight_v1",
            12: "imaplusplus_partial_multitask_lesion_attention_v1",
            13: "imaplusplus_partial_multitask_lesion_attention_mel_weight_v1",
            14: "imaplusplus_partial_multitask_lesion_attention_mel_nv_margin_v1",
            15: "imaplusplus_partial_multitask_lesion_attention_mel_nv_pairwise_v1",
            16: "imaplusplus_partial_multitask_lesion_attention_v1",
            17: "imaplusplus_partial_multitask_v1",
        }[run_number]
        importlib.import_module(module_name)
        return
    run_root = TOOLS_ROOT / f"run{run_number}"
    if not run_root.is_dir():
        raise FileNotFoundError(f"缺少模型结构目录：{run_root}")
    _prepend_import_path(run_root)
    sources = sorted(run_root.glob("imaplusplus*.py"))
    if not sources:
        raise FileNotFoundError(f"Run{run_number} 缺少 imaplusplus 模型定义")
    for source in sources:
        importlib.import_module(source.stem)


def load_model(model: dict[str, Any]):
    global CURRENT_MODEL_ID, CURRENT_WRAPPER
    with MODEL_LOCK:
        if CURRENT_MODEL_ID == model["id"] and CURRENT_WRAPPER is not None:
            return CURRENT_WRAPPER
        import_model_contract(model)
        from ultralytics import YOLO

        if CURRENT_WRAPPER is not None:
            del CURRENT_WRAPPER
            CURRENT_WRAPPER = None
            CURRENT_MODEL_ID = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        task = "segment" if model["run_number"] is not None and model["run_number"] <= 3 else "semantic"
        CURRENT_WRAPPER = YOLO(str(model["weights"]), task=task)
        CURRENT_MODEL_ID = model["id"]
        return CURRENT_WRAPPER


def partial_label_predictor_class():
    from imaplusplus_partial_multitask_v1 import PartialLabelMultiTaskPredictor, unpack_outputs
    from ultralytics.engine.results import Probs
    from ultralytics.models.yolo.semantic import SemanticSegmentationPredictor

    class ThresholdedPartialLabelPredictor(PartialLabelMultiTaskPredictor):
        def postprocess(self, preds, img, orig_imgs):
            semantic_logits, _, image_logits = unpack_outputs(preds)
            threshold = float(self.args.conf if self.args.conf is not None else 0.5)
            threshold = min(0.999, max(0.001, threshold))
            semantic_logits = semantic_logits - math.log(threshold / (1.0 - threshold))
            results = SemanticSegmentationPredictor.postprocess(self, semantic_logits, img, orig_imgs)
            for result, probabilities in zip(results, image_logits.softmax(1)):
                result.probs = Probs(probabilities, result.orig_shape)
            return results

    return ThresholdedPartialLabelPredictor


def skin7_predictor_class():
    from skin7_multitask import Skin7MultiTaskPredictor

    return Skin7MultiTaskPredictor


def png_base64(rgba: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(rgba).save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def build_visualization(mask: np.ndarray) -> tuple[dict[str, str], dict[str, Any]]:
    height, width = mask.shape
    positive = mask.astype(bool)
    area = int(positive.sum())
    fill = np.zeros((height, width, 4), dtype=np.uint8)
    fill[positive] = (244, 63, 94, 255)
    binary = positive.astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_pixels = np.zeros_like(binary)
    line_width = max(2, round(max(height, width) / 500))
    if contours:
        cv2.drawContours(contour_pixels, contours, -1, 255, thickness=line_width, lineType=cv2.LINE_AA)
    contour_rgba = np.zeros((height, width, 4), dtype=np.uint8)
    contour_rgba[contour_pixels > 0] = (190, 18, 60, 255)
    bbox = None
    major_axis = 0.0
    irregularity = 0.0
    if area and contours:
        ys, xs = np.where(positive)
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        bbox = {
            "x": 100.0 * x0 / width,
            "y": 100.0 * y0 / height,
            "width": 100.0 * (x1 - x0 + 1) / width,
            "height": 100.0 * (y1 - y0 + 1) / height,
        }
        largest = max(contours, key=cv2.contourArea)
        major_axis = float(max(cv2.minAreaRect(largest)[1]))
        contour_area = max(float(cv2.contourArea(largest)), 1.0)
        perimeter = float(cv2.arcLength(largest, True))
        irregularity = max(0.0, perimeter * perimeter / (4.0 * math.pi * contour_area) - 1.0)
    shape_text = "(平滑)" if irregularity < 0.15 else "(中等)" if irregularity < 0.35 else "(不规则)"
    return (
        {"mask_png": png_base64(fill), "contour_png": png_base64(contour_rgba)},
        {
            "area_px": area,
            "ratio_percent": 100.0 * area / max(height * width, 1),
            "major_axis_px": major_axis,
            "irregularity": irregularity,
            "shape_text": shape_text,
            "bbox_percent": bbox,
            "image_width": width,
            "image_height": height,
        },
    )


def diagnosis_payload(codes: list[str], probabilities: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    order = np.argsort(probabilities)[::-1]
    ranked: list[dict[str, Any]] = []
    for index in order:
        code = codes[int(index)]
        info = CLASS_INFO[code]
        ranked.append(
            {
                "code": code,
                "name_zh": info["zh"],
                "name_en": info["en"],
                "percent": float(probabilities[index] * 100.0),
                "malignant": info["risk"] == "high",
            }
        )
    if not ranked or float(np.max(probabilities, initial=0.0)) <= 0.0:
        return (
            {
                "code": "none",
                "name_zh": "未检出明确病灶",
                "name_en": "No Confident Detection",
                "confidence_percent": 0.0,
                "risk_level": "low",
                "risk_text": "未检出明确目标",
                "suggestion": "当前阈值下未检出明确病灶，可调整阈值后重新分析",
                "disclaimer": "仅供科研与毕业论文演示，不构成临床诊断。",
            },
            ranked,
        )
    primary = ranked[0]
    risk = CLASS_INFO[primary["code"]]["risk"]
    if risk == "high":
        risk_text = "模型提示高风险"
        suggestion = "模型提示恶性类别置信度较高，请由皮肤科医师结合病理检查复核"
    elif risk == "medium":
        risk_text = "模型提示中度风险"
        suggestion = "模型提示需关注的角化性病变，请由专业医师进一步复核"
    else:
        risk_text = "模型提示低风险"
        suggestion = "模型倾向良性类别，仍需结合临床表现与医生判断"
    return (
        {
            "code": primary["code"],
            "name_zh": primary["name_zh"],
            "name_en": primary["name_en"],
            "confidence_percent": primary["percent"],
            "risk_level": risk,
            "risk_text": risk_text,
            "suggestion": suggestion,
            "disclaimer": "仅供科研与毕业论文演示，不构成临床诊断。",
        },
        ranked,
    )


def class_codes_for_model(wrapper: Any, run_number: int | None) -> list[str]:
    if run_number is not None and run_number <= 9:
        return list(CLASS_INFO)[:7]
    names = tuple(getattr(wrapper.model, "classification_names", ()))
    return list(names) if names else list(CLASS_INFO)


def instance_result(result: Any, codes: list[str]) -> tuple[np.ndarray, np.ndarray, str]:
    height, width = result.orig_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    scores = np.zeros(len(codes), dtype=np.float32)
    if result.boxes is not None:
        classes = result.boxes.cls.detach().long().cpu().tolist()
        confidences = result.boxes.conf.detach().float().cpu().tolist()
        for class_id, confidence in zip(classes, confidences):
            if class_id in range(len(scores)):
                scores[class_id] = max(scores[class_id], float(confidence))
    if result.masks is not None:
        for polygon in result.masks.xy:
            points = np.rint(np.asarray(polygon)).astype(np.int32)
            if len(points) >= 3:
                cv2.fillPoly(mask, [points], 1)
    return mask, scores, "实例分割检测置信度"


def semantic_result(result: Any, codes: list[str], has_classifier: bool) -> tuple[np.ndarray, np.ndarray, str]:
    if result.semantic_mask is None:
        raise RuntimeError("模型结果缺少语义分割掩膜")
    class_map = result.semantic_mask.data.detach().byte().cpu().numpy()
    if has_classifier:
        if result.probs is None:
            raise RuntimeError("多任务模型结果缺少分类概率")
        scores = result.probs.data.detach().float().cpu().numpy()
        mask = class_map > 0 if len(codes) == 8 else class_map < len(codes)
        return mask.astype(np.uint8), scores, "图像分类概率"
    mask = class_map < len(codes)
    counts = np.asarray([(class_map == index).sum() for index in range(len(codes))], dtype=np.float64)
    scores = counts / counts.sum() if counts.sum() else counts
    return mask.astype(np.uint8), scores.astype(np.float32), "前景像素类别占比"


def infer_image(image_bytes: bytes, model_id: str | None, threshold: float, filename: str) -> dict[str, Any]:
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.load()
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("上传内容不是可读取的图片") from error
    if image.width < 16 or image.height < 16:
        raise ValueError("图片尺寸过小，至少需要 16 × 16 像素")
    if image.width > 12000 or image.height > 12000:
        raise ValueError("图片尺寸过大，单边不能超过 12000 像素")
    model = find_model(model_id)
    started = time.perf_counter()
    with MODEL_LOCK:
        wrapper = load_model(model)
        common = {
            "source": image,
            "imgsz": 640,
            "device": 0 if torch.cuda.is_available() else "cpu",
            "verbose": False,
            "conf": threshold,
        }
        run_number = model["run_number"]
        if run_number is not None and run_number <= 4:
            results = wrapper.predict(**common)
        elif run_number is not None and run_number <= 9:
            results = wrapper.predict(predictor=skin7_predictor_class(), **common)
        else:
            results = wrapper.predict(predictor=partial_label_predictor_class(), **common)
    if not results:
        raise RuntimeError("模型没有返回推理结果")
    result = results[0]
    codes = class_codes_for_model(wrapper, model["run_number"])
    if model["run_number"] is not None and model["run_number"] <= 3:
        mask, probabilities, probability_mode = instance_result(result, codes)
    else:
        has_classifier = model["run_number"] is None or model["run_number"] >= 5
        mask, probabilities, probability_mode = semantic_result(result, codes, has_classifier=has_classifier)
    visualization, segmentation = build_visualization(mask)
    diagnosis, ranked = diagnosis_payload(codes, probabilities)
    speed = result.speed or {}
    return {
        "case_id": f"#DERMA-{datetime.now():%Y}-{uuid.uuid4().hex[:6].upper()}",
        "filename": Path(filename or "uploaded_image").name,
        "model": public_model(model),
        "threshold": threshold,
        "probability_mode": probability_mode,
        "diagnosis": diagnosis,
        "probabilities": ranked,
        "segmentation": segmentation,
        "visualization": visualization,
        "timing": {
            "preprocess_ms": float(speed.get("preprocess", 0.0)),
            "inference_ms": float(speed.get("inference", 0.0)),
            "postprocess_ms": float(speed.get("postprocess", 0.0)),
            "request_total_ms": (time.perf_counter() - started) * 1000.0,
        },
    }


def random_sample(split: str) -> Path:
    if split not in LAST_SAMPLE:
        raise ValueError("样本集只支持 test 或 val")
    sample_root = SAMPLE_IMAGES_ROOT / split
    if not sample_root.is_dir():
        raise FileNotFoundError(f"未找到样本目录：{sample_root}")
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    samples = [path for path in sample_root.iterdir() if path.is_file() and path.suffix.lower() in extensions]
    if not samples:
        raise FileNotFoundError(f"{split} 目录中没有可读取图片")
    with SAMPLE_LOCK:
        previous = LAST_SAMPLE[split]
        choices = [path for path in samples if path != previous] or samples
        selected = random.choice(choices)
        LAST_SAMPLE[split] = selected
    return selected


def sample_images(split: str) -> list[Path]:
    sample_root = SAMPLE_IMAGES_ROOT / split
    if not sample_root.is_dir():
        raise FileNotFoundError(f"未找到样本目录：{sample_root}")
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return [path for path in sample_root.iterdir() if path.is_file() and path.suffix.lower() in extensions]


def select_batch_samples(count: int = MAX_BATCH_COUNT) -> list[tuple[str, Path, str]]:
    """按真实病种分层抽样，同时尽量保持 test 与 val 各占一半。"""
    if count < 1 or count > MAX_BATCH_COUNT:
        raise ValueError(f"批量预测数量必须位于 1 到 {MAX_BATCH_COUNT} 之间")
    class_codes = list(CLASS_INFO)[:7]
    pools: dict[str, dict[str, list[Path]]] = {}
    for split in ("test", "val"):
        images_by_stem = {path.stem: path for path in sample_images(split)}
        metadata_path = SAMPLE_DATA_ROOT / "metadata" / f"{split}.csv"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"缺少用于平衡抽样的标签清单：{metadata_path}")
        split_pools = {code: [] for code in class_codes}
        with metadata_path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                code = (row.get("dx") or "").strip().lower()
                image = images_by_stem.get((row.get("image_id") or "").strip())
                if code in split_pools and image is not None:
                    split_pools[code].append(image)
        pools[split] = split_pools

    available_codes = [code for code in class_codes if pools["test"][code] or pools["val"][code]]
    if sum(len(pools[split][code]) for split in pools for code in available_codes) < count:
        raise ValueError("test 与 val 中可用图片总数不足")

    base, remainder = divmod(count, len(available_codes))
    extra_codes = set(random.sample(available_codes, remainder))
    quotas = {code: base + int(code in extra_codes) for code in available_codes}
    deficit = 0
    for code in available_codes:
        available = len(pools["test"][code]) + len(pools["val"][code])
        if quotas[code] > available:
            deficit += quotas[code] - available
            quotas[code] = available
    while deficit:
        candidates = [
            code for code in available_codes
            if quotas[code] < len(pools["test"][code]) + len(pools["val"][code])
        ]
        if not candidates:
            raise ValueError("无法完成病种平衡抽样")
        minimum = min(quotas[code] for code in candidates)
        code = random.choice([item for item in candidates if quotas[item] == minimum])
        quotas[code] += 1
        deficit -= 1

    target_test = count // 2
    allocation = {
        code: {
            "test": min(quotas[code] // 2, len(pools["test"][code])),
            "val": min(quotas[code] // 2, len(pools["val"][code])),
        }
        for code in available_codes
    }
    remaining_by_code = {
        code: quotas[code] - allocation[code]["test"] - allocation[code]["val"]
        for code in available_codes
    }
    current_test = sum(values["test"] for values in allocation.values())
    for code in random.sample(available_codes, len(available_codes)):
        while remaining_by_code[code]:
            test_capacity = len(pools["test"][code]) - allocation[code]["test"]
            val_capacity = len(pools["val"][code]) - allocation[code]["val"]
            prefer_test = current_test < target_test
            if prefer_test and test_capacity:
                split = "test"
            elif val_capacity:
                split = "val"
            elif test_capacity:
                split = "test"
            else:
                raise ValueError(f"{code} 类样本不足，无法完成平衡抽样")
            allocation[code][split] += 1
            current_test += int(split == "test")
            remaining_by_code[code] -= 1

    selected: list[tuple[str, Path, str]] = []
    for code in available_codes:
        for split in ("test", "val"):
            amount = allocation[code][split]
            selected.extend((split, path, code) for path in random.sample(pools[split][code], amount))
    random.shuffle(selected)
    return selected


def public_batch_job(job: dict[str, Any]) -> dict[str, Any]:
    payload = {
        key: job[key]
        for key in (
            "id",
            "status",
            "model",
            "threshold",
            "total",
            "completed",
            "failed",
            "created_at",
            "finished_at",
            "duration_ms",
            "error",
            "class_distribution",
            "items",
        )
    }
    payload["items"] = list(job["items"])
    return payload


def run_batch_job(batch_id: str, selected: list[tuple[str, Path, str]]) -> None:
    with BATCH_LOCK:
        job = BATCH_JOBS[batch_id]
        job["status"] = "running"
        job["started_perf"] = time.perf_counter()
    try:
        for index, (split, path, ground_truth) in enumerate(selected, start=1):
            item_id = f"{index:03d}"
            try:
                result = infer_image(path.read_bytes(), job["model_id"], job["threshold"], path.name)
                summary = {
                    "id": item_id,
                    "filename": path.name,
                    "split": split,
                    "ground_truth": ground_truth,
                    "ground_truth_zh": CLASS_INFO[ground_truth]["zh"],
                    "status": "completed",
                    "diagnosis": result["diagnosis"],
                    "segmentation_ratio": result["segmentation"]["ratio_percent"],
                    "inference_ms": result["timing"]["inference_ms"],
                }
                detail = {"path": path, "result": result, "summary": summary}
            except Exception as error:
                summary = {
                    "id": item_id,
                    "filename": path.name,
                    "split": split,
                    "ground_truth": ground_truth,
                    "ground_truth_zh": CLASS_INFO[ground_truth]["zh"],
                    "status": "failed",
                    "error": str(error),
                }
                detail = {"path": path, "result": None, "summary": summary}
            with BATCH_LOCK:
                current = BATCH_JOBS[batch_id]
                current["items"].append(summary)
                current["details"][item_id] = detail
                current["completed"] = index
                if summary["status"] == "failed":
                    current["failed"] += 1
        with BATCH_LOCK:
            current = BATCH_JOBS[batch_id]
            current["status"] = "completed"
            current["finished_at"] = datetime.now().isoformat(timespec="seconds")
            current["duration_ms"] = (time.perf_counter() - current["started_perf"]) * 1000.0
    except Exception as error:
        with BATCH_LOCK:
            current = BATCH_JOBS[batch_id]
            current["status"] = "failed"
            current["error"] = str(error)
            current["finished_at"] = datetime.now().isoformat(timespec="seconds")
            current["duration_ms"] = (time.perf_counter() - current["started_perf"]) * 1000.0


def start_batch_job(model_id: str | None, threshold: float) -> dict[str, Any]:
    model = find_model(model_id)
    selected = select_batch_samples(MAX_BATCH_COUNT)
    distribution = {
        code: sum(ground_truth == code for _, _, ground_truth in selected)
        for code in list(CLASS_INFO)[:7]
    }
    batch_id = uuid.uuid4().hex[:12]
    job = {
        "id": batch_id,
        "status": "queued",
        "model_id": model["id"],
        "model": public_model(model),
        "threshold": threshold,
        "total": len(selected),
        "completed": 0,
        "failed": 0,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "finished_at": None,
        "duration_ms": None,
        "error": None,
        "class_distribution": [
            {"code": code, "name_zh": CLASS_INFO[code]["zh"], "count": amount}
            for code, amount in distribution.items()
        ],
        "items": [],
        "details": {},
    }
    with BATCH_LOCK:
        finished_ids = [key for key, value in BATCH_JOBS.items() if value["status"] in {"completed", "failed"}]
        for old_id in finished_ids[:-1]:
            BATCH_JOBS.pop(old_id, None)
        BATCH_JOBS[batch_id] = job
    threading.Thread(target=run_batch_job, args=(batch_id, selected), daemon=True, name=f"batch-{batch_id}").start()
    return public_batch_job(job)


def get_batch_job(batch_id: str) -> dict[str, Any]:
    with BATCH_LOCK:
        job = BATCH_JOBS.get(batch_id)
        if job is None:
            raise FileNotFoundError("批量预测任务不存在或已过期")
        return public_batch_job(job)


def get_batch_detail(batch_id: str, item_id: str) -> dict[str, Any]:
    with BATCH_LOCK:
        job = BATCH_JOBS.get(batch_id)
        detail = job["details"].get(item_id) if job else None
    if detail is None:
        raise FileNotFoundError("该批量预测条目尚未完成或不存在")
    if detail["result"] is None:
        return {"item": detail["summary"], "result": None, "source_image": None}
    path: Path = detail["path"]
    source_image = {
        "mime": image_content_type(path),
        "base64": base64.b64encode(path.read_bytes()).decode("ascii"),
    }
    return {"item": detail["summary"], "result": detail["result"], "source_image": source_image}


def image_content_type(path: Path) -> str:
    return {".png": "image/png", ".webp": "image/webp", ".bmp": "image/bmp"}.get(path.suffix.lower(), "image/jpeg")


class DermaSegHandler(BaseHTTPRequestHandler):
    server_version = "DermaSeg/2.0"

    def log_message(self, format_string: str, *args: Any) -> None:
        print(f"[{datetime.now():%H:%M:%S}] {self.address_string()} {format_string % args}")

    def send_bytes(self, payload: bytes, content_type: str, status: int = HTTPStatus.OK, extra_headers=None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(payload)

    def send_json(self, data: dict[str, Any], status: int = HTTPStatus.OK) -> None:
        payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_bytes(payload, "application/json; charset=utf-8", status)

    def send_error_json(self, error: Exception, status: int = HTTPStatus.BAD_REQUEST) -> None:
        print(f"[error] {type(error).__name__}: {error}")
        self.send_json({"error": str(error)}, status)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path in {"/", "/index.html"}:
                self.send_bytes(INDEX_PATH.read_bytes(), "text/html; charset=utf-8")
                return
            if parsed.path == "/api/models":
                models = discover_models()
                self.send_json({"models": [public_model(model) for model in models], "default_model": models[0]["id"] if models else None})
                return
            if parsed.path == "/api/health":
                models = discover_models()
                device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
                sample_counts = {
                    split: len(list((SAMPLE_IMAGES_ROOT / split).glob("*"))) if (SAMPLE_IMAGES_ROOT / split).is_dir() else 0
                    for split in ("test", "val")
                }
                self.send_json({"status": "ok", "device": device, "model_count": len(models), "loaded_model": CURRENT_MODEL_ID, "sample_counts": sample_counts})
                return
            sample_match = re.fullmatch(r"/api/sample/(test|val)", parsed.path)
            if sample_match:
                split = sample_match.group(1)
                sample = random_sample(split)
                self.send_bytes(sample.read_bytes(), image_content_type(sample), extra_headers={"X-Filename": quote(sample.name), "X-Split": split})
                return
            batch_detail_match = re.fullmatch(r"/api/batch/([a-f0-9]{12})/(\d{3})", parsed.path)
            if batch_detail_match:
                self.send_json(get_batch_detail(batch_detail_match.group(1), batch_detail_match.group(2)))
                return
            batch_match = re.fullmatch(r"/api/batch/([a-f0-9]{12})", parsed.path)
            if batch_match:
                self.send_json(get_batch_job(batch_match.group(1)))
                return
            self.send_json({"error": "接口不存在"}, HTTPStatus.NOT_FOUND)
        except FileNotFoundError as error:
            self.send_error_json(error, HTTPStatus.NOT_FOUND)
        except Exception as error:
            self.send_error_json(error, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if parsed.path == "/api/batch/start":
                query = parse_qs(parsed.query)
                threshold = float(query.get("threshold", ["0.5"])[0])
                if not 0.01 <= threshold <= 0.99:
                    raise ValueError("分割阈值必须位于 0.01 到 0.99 之间")
                model_id = query.get("model", [None])[0]
                self.send_json(start_batch_job(model_id, threshold), HTTPStatus.ACCEPTED)
                return
            if parsed.path == "/api/infer":
                if content_length <= 0:
                    raise ValueError("请先上传图片")
                if content_length > MAX_IMAGE_BYTES:
                    raise ValueError("图片不能超过 25 MB")
                body = self.rfile.read(content_length)
                query = parse_qs(parsed.query)
                threshold = float(query.get("threshold", ["0.5"])[0])
                if not 0.01 <= threshold <= 0.99:
                    raise ValueError("分割阈值必须位于 0.01 到 0.99 之间")
                model_id = query.get("model", [None])[0]
                filename = unquote(query.get("filename", ["uploaded_image"])[0])
                self.send_json(infer_image(body, model_id, threshold, filename))
                return
            if parsed.path == "/api/annotations":
                if content_length <= 0 or content_length > 1024 * 1024:
                    raise ValueError("复核记录大小无效")
                record = json.loads(self.rfile.read(content_length).decode("utf-8"))
                if not record.get("case_id") or not record.get("model"):
                    raise ValueError("复核记录缺少病例或模型信息")
                annotation_id = f"ANN-{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4].upper()}"
                record["annotation_id"] = annotation_id
                RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
                with ANNOTATIONS_PATH.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                self.send_json({"saved": True, "annotation_id": annotation_id}, HTTPStatus.CREATED)
                return
            self.send_json({"error": "接口不存在"}, HTTPStatus.NOT_FOUND)
        except (ValueError, json.JSONDecodeError) as error:
            self.send_error_json(error, HTTPStatus.BAD_REQUEST)
        except Exception as error:
            self.send_error_json(error, HTTPStatus.INTERNAL_SERVER_ERROR)


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="启动 DermaSeg 本地模型演示系统")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not INDEX_PATH.is_file():
        raise FileNotFoundError(INDEX_PATH)
    if not TOOLS_ROOT.is_dir() or not EXPERIMENTS_ROOT.is_dir():
        raise FileNotFoundError("QianDuan 必须位于 PiFuV2 项目根目录下")
    models = discover_models()
    if not models:
        raise RuntimeError("未发现可用于网页推理的正式 best.pt")
    server = ReusableThreadingHTTPServer((args.host, args.port), DermaSegHandler)
    url = f"http://{args.host}:{args.port}"
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"DermaSeg 已启动：{url}")
    print(f"设备：{device}；可用模型：{len(models)}；默认模型：{models[0]['id']}")
    print(f"随机样本目录：{SAMPLE_IMAGES_ROOT}")
    print("按 Ctrl+C 停止服务。")
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.3)
    except KeyboardInterrupt:
        print("\nDermaSeg 已停止。")
    finally:
        server.server_close()


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reproduction"))
    from front import main as portable_main
    portable_main()
