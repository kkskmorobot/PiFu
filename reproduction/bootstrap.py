"""Portable import paths shared by the reproduction entrypoints (standard library only)."""

from __future__ import annotations

import os
import filecmp
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = PROJECT_ROOT / "tools"
VENDOR_ROOT = PROJECT_ROOT / "vendor"
DATA_ROOT = Path(os.environ.get("PIFU_DATA_ROOT", str(PROJECT_ROOT.parent / "PiFuData"))).expanduser().resolve()


def configure(data_root: str | Path | None = None, require_data: bool = False) -> Path:
    """Prefer the preserved framework and model definitions; return the external data root.

    Explicit ``data_root`` wins over ``PIFU_DATA_ROOT``. Otherwise PiFuData is expected
    beside the project directory, so moving both directories together needs no editing.
    Call this before importing torch, ultralytics, or custom checkpoint classes.
    """
    global DATA_ROOT
    configured = data_root or os.environ.get("PIFU_DATA_ROOT") or PROJECT_ROOT.parent / "PiFuData"
    DATA_ROOT = Path(configured).expanduser().resolve()
    if require_data and not DATA_ROOT.is_dir():
        raise FileNotFoundError(
            f"Dataset directory missing: {DATA_ROOT}. Place PiFuData beside PiFu "
            "or set PIFU_DATA_ROOT to its location."
        )
    if not (VENDOR_ROOT / "ultralytics" / "__init__.py").is_file():
        raise FileNotFoundError(f"Preserved Ultralytics source is missing: {VENDOR_ROOT / 'ultralytics'}")

    model_directories = (
        sorted((p for p in TOOLS_ROOT.iterdir() if p.is_dir() and not p.name.startswith(".")), key=lambda p: p.name)
        if TOOLS_ROOT.is_dir() else []
    )
    preferred = [VENDOR_ROOT, TOOLS_ROOT, *model_directories, PROJECT_ROOT]
    preferred_strings = [str(path) for path in preferred if path.is_dir()]
    normalized = {os.path.normcase(path) for path in preferred_strings}
    sys.path[:] = preferred_strings + [path for path in sys.path if os.path.normcase(path) not in normalized]
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PIFU_DATA_ROOT"] = str(DATA_ROOT)
    for variable, directory in (
        ("YOLO_CONFIG_DIR", "ultralytics"),
        ("MPLCONFIGDIR", "matplotlib"),
        ("TORCH_HOME", "torch"),
    ):
        os.environ.setdefault(variable, str(PROJECT_ROOT / "runtime" / directory))
        # Ultralytics tests writability at import time and otherwise silently falls
        # back to the user's profile. Create each selected cache directory first.
        Path(os.environ[variable]).expanduser().mkdir(parents=True, exist_ok=True)
    font_source = VENDOR_ROOT / "font" / "DejaVuSans.ttf"
    license_source = VENDOR_ROOT / "font" / "LICENSE_DEJAVU"
    if not font_source.is_file() or not license_source.is_file():
        raise FileNotFoundError(f"Bundled plotting font/license missing in {font_source.parent}")
    # This Ultralytics version appends its own 'Ultralytics' subdirectory to
    # YOLO_CONFIG_DIR. check_font searches there before trying a network download.
    font_cache = Path(os.environ["YOLO_CONFIG_DIR"]).expanduser() / "Ultralytics"
    font_cache.mkdir(parents=True, exist_ok=True)
    for source, name in (
        (font_source, "DejaVuSans.ttf"),
        (font_source, "Arial.ttf"),
        (license_source, "LICENSE_DEJAVU"),
    ):
        destination = font_cache / name
        if not destination.is_file() or not filecmp.cmp(source, destination, shallow=False):
            shutil.copyfile(source, destination)
    (font_cache / "FONT_NOTE.txt").write_text(
        "Arial.ttf is a compatibility alias of the bundled DejaVuSans.ttf, not Microsoft Arial.\n"
        "This local font is used only for generated plot labels and prevents an automatic font download.\n"
        "Model weights, input images, preprocessing, and numerical evaluation are unchanged.\n"
        "The original filename and the accompanying LICENSE_DEJAVU are retained.\n",
        encoding="utf-8",
    )
    os.environ.setdefault("YOLO_OFFLINE", "true")
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    return DATA_ROOT
