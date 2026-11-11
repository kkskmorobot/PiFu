"""Verify the shipped project/data manifests using the Python standard library only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path, PurePosixPath, PureWindowsPath


sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", dest="mode", action="store_const", const="quick", help="Hash all project files and data metadata; check image existence/size without hashing image content (default).")
    mode.add_argument("--full", dest="mode", action="store_const", const="full", help="Hash every project and data payload file.")
    parser.set_defaults(mode="quick")
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("PIFU_DATA_ROOT", str(PROJECT_ROOT.parent / "PiFuData"))))
    return parser.parse_args()


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_records(manifest: Path) -> list[dict]:
    payload = json.loads(manifest.read_text(encoding="utf-8-sig"))
    records = payload.get("files") if isinstance(payload, dict) else payload
    if not isinstance(records, list) or not records:
        raise ValueError(f"Manifest must contain a nonempty 'files' array: {manifest}")
    return records


def verify_manifest(root: Path, manifest_name: str, skip_image_hashes: bool) -> dict:
    root = root.expanduser().resolve()
    manifest = root / manifest_name
    records = load_records(manifest)
    seen: set[str] = set()
    checked = hashed = skipped = total_bytes = 0
    failures: list[str] = []
    started = last_progress = time.monotonic()
    for record in records:
        checked += 1
        try:
            if not isinstance(record, dict):
                raise ValueError("record is not an object")
            raw_path = record.get("path")
            if not isinstance(raw_path, str) or not raw_path:
                raise ValueError("missing relative path")
            portable = PurePosixPath(raw_path.replace("\\", "/"))
            if portable.is_absolute() or PureWindowsPath(raw_path).drive or ".." in portable.parts:
                raise ValueError(f"unsafe path: {raw_path}")
            path_key = portable.as_posix().casefold()
            if path_key in seen:
                raise ValueError(f"duplicate path: {raw_path}")
            seen.add(path_key)
            path = root.joinpath(*portable.parts)
            if not path.resolve().is_relative_to(root):
                raise ValueError(f"path escapes package directory: {raw_path}")
            expected_bytes = record.get("bytes")
            expected_hash = record.get("sha256")
            if not isinstance(expected_bytes, int) or isinstance(expected_bytes, bool) or expected_bytes < 0:
                raise ValueError(f"invalid byte count: {raw_path}")
            if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_hash):
                raise ValueError(f"invalid SHA-256: {raw_path}")
            if not path.is_file():
                raise FileNotFoundError(f"missing: {raw_path}")
            size = path.stat().st_size
            if size != expected_bytes:
                raise ValueError(f"size differs: {raw_path}: {size} != {expected_bytes}")
            total_bytes += size
            if skip_image_hashes and path.suffix.lower() in IMAGE_SUFFIXES:
                skipped += 1
            else:
                if digest_file(path) != expected_hash.lower():
                    raise ValueError(f"SHA-256 differs: {raw_path}")
                hashed += 1
        except (OSError, ValueError, TypeError) as error:
            failures.append(f"entry {checked}: {error}")
        now = time.monotonic()
        if now - last_progress >= 10:
            print(f"{manifest_name}: {checked:,}/{len(records):,} checked, {len(failures)} failures", flush=True)
            last_progress = now

    return {
        "manifest": str(manifest),
        "status": "PASS" if not failures else "FAIL",
        "files_checked": checked,
        "files_sha256_verified": hashed,
        "image_hashes_skipped": skipped,
        "checked_bytes": total_bytes,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "failures": failures,
    }


def main() -> int:
    args = parse_args()
    results = []
    for root, manifest_name, skip in (
        (PROJECT_ROOT, "PACKAGE_FILES.json", False),
        (args.data_root, "DATA_FILES.json", args.mode == "quick"),
    ):
        print(f"Checking {root / manifest_name} ({args.mode})", flush=True)
        try:
            results.append(verify_manifest(root, manifest_name, skip))
        except (OSError, ValueError, TypeError) as error:
            results.append({"manifest": str(root / manifest_name), "status": "FAIL", "failures": [str(error)]})
    passed = all(result["status"] == "PASS" for result in results)
    print(json.dumps({"status": "PASS" if passed else "FAIL", "mode": args.mode, "manifests": results}, ensure_ascii=False, indent=2))
    if args.mode == "quick":
        print("Quick mode checks image existence and size. Use --full for image SHA-256 verification.")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
