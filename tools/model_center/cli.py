#!/usr/bin/env python3
"""Operational CLI for provider capability checks and weight manifests."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import shutil
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .contracts import sha256_file, write_json
    from .registry import environment_report, provider_names, write_environment_report
except ImportError:  # Allow ``python tools/model_center/cli.py ...`` for operators.
    from model_center.contracts import sha256_file, write_json
    from model_center.registry import environment_report, provider_names, write_environment_report


def _token_present() -> bool:
    if os.environ.get("MODELSCOPE_API_TOKEN"):
        return True
    candidates = (
        Path.home() / ".modelscope" / "credentials" / "git_token",
        Path.home() / ".modelscope" / "credentials" / "session",
        Path.home() / ".modelscope" / "credentials" / "user",
    )
    return any(path.is_file() and path.stat().st_size > 0 for path in candidates)


def _file_manifest(root: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sizeBytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return files


def _weight_fingerprint(files: list[dict[str, Any]]) -> str:
    payload = [
        [str(item["path"]), int(item["sizeBytes"]), str(item["sha256"])]
        for item in files
    ]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()


def _modelscope_inventory(model_id: str, revision: str) -> list[dict[str, Any]]:
    encoded_id = "/".join(urllib.parse.quote(part, safe="") for part in model_id.split("/"))
    query = urllib.parse.urlencode({"Revision": revision, "Root": "", "Recursive": "true"})
    url = f"https://modelscope.cn/api/v1/models/{encoded_id}/repo/files?{query}"
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = json.load(response)
    data = payload.get("Data", payload) if isinstance(payload, dict) else payload
    values = data.get("Files", data) if isinstance(data, dict) else data
    if not isinstance(values, list):
        raise RuntimeError("ModelScope inventory response does not contain a file list")
    return [item for item in values if isinstance(item, dict) and item.get("Type") == "blob"]


def _validate_snapshot(
    root: Path,
    inventory: list[dict[str, Any]],
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    missing: list[str] = []
    mismatched: list[str] = []
    omitted_empty: list[dict[str, Any]] = []
    for item in inventory:
        relative = str(item.get("Path") or item.get("Name") or "")
        expected_size = int(item.get("Size") or 0)
        target = root / relative
        if expected_size == 0:
            if not target.is_file():
                omitted_empty.append({"path": relative, "reason": "remote zero-byte placeholder"})
            elif target.stat().st_size != 0:
                mismatched.append(f"{relative}: expected 0, got {target.stat().st_size}")
            else:
                omitted_empty.append({"path": relative, "reason": "remote zero-byte placeholder"})
            continue
        if not target.is_file():
            missing.append(relative)
            continue
        actual_size = target.stat().st_size
        if actual_size != expected_size:
            mismatched.append(f"{relative}: expected {expected_size}, got {actual_size}")
    return missing, mismatched, omitted_empty


def _download_modelscope(args: argparse.Namespace) -> int:
    if not _token_present():
        raise SystemExit("ModelScope credential not found; refusing an unauthenticated weight download")
    target = args.target_dir.resolve()
    cache = args.cache_dir.resolve()
    if args.dry_run:
        print(json.dumps({"source": "modelscope", "modelId": args.model_id, "revision": args.revision, "target": str(target), "tokenPresent": True}, ensure_ascii=False, indent=2))
        return 0
    cache.mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    inventory = _modelscope_inventory(args.model_id, args.revision)
    missing, mismatched, omitted_empty = _validate_snapshot(target, inventory)
    if missing or mismatched or not target.is_dir():
        try:
            from modelscope import snapshot_download  # type: ignore[import-not-found]
        except ImportError as exc:
            raise SystemExit("ModelScope SDK is not installed in this runtime; install it with --no-deps before retrying") from exc
        signature = inspect.signature(snapshot_download)
        kwargs: dict[str, Any] = {}
        for name, value in (
            ("revision", args.revision),
            ("cache_dir", str(cache)),
            ("local_dir", str(target)),
            ("max_workers", args.max_workers),
        ):
            if name in signature.parameters and value is not None:
                kwargs[name] = value
        # Do not pass token explicitly.  The SDK reads the existing ModelScope
        # credential store; this keeps the token out of argv and process metadata.
        resolved = snapshot_download(args.model_id, **kwargs)
        resolved_path = Path(str(resolved)).resolve()
        if resolved_path != target and not target.exists():
            shutil.copytree(resolved_path, target)
        if not target.is_dir():
            raise RuntimeError(f"ModelScope did not materialize a directory: {target}")
        inventory = _modelscope_inventory(args.model_id, args.revision)
        missing, mismatched, omitted_empty = _validate_snapshot(target, inventory)
        if missing or mismatched:
            raise RuntimeError(
                "ModelScope snapshot is incomplete: "
                + "; ".join([*(f"missing {item}" for item in missing), *mismatched])
            )
    else:
        print("MODELSCOPE_CACHE_REUSED", target, flush=True)
    for item in omitted_empty:
        placeholder = target / str(item["path"])
        placeholder.parent.mkdir(parents=True, exist_ok=True)
        placeholder.touch(exist_ok=True)
    files = _file_manifest(target)
    manifest = {
        "schemaVersion": 1,
        "source": "modelscope",
        "modelId": args.model_id,
        "revision": args.revision,
        "downloadedAtUtc": datetime.now(timezone.utc).isoformat(),
        "target": str(target),
        "fileCount": len(files),
        "remoteBlobCount": len(inventory),
        "totalBytes": sum(item["sizeBytes"] for item in files),
        "weightFingerprint": _weight_fingerprint(files),
        "files": files,
        "pipelineConfigPath": str((target / "checkpoints" / "pipeline.yaml").resolve())
        if (target / "checkpoints" / "pipeline.yaml").is_file()
        else None,
        "tokenRecorded": False,
        "omittedZeroByteRemoteFiles": omitted_empty,
        "licenseAcknowledgement": args.license_acknowledgement,
    }
    write_json(args.manifest.resolve(), manifest)
    print("MODELSCOPE_DOWNLOAD_READY", target, len(files), flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="InsertAny3D model center operations")
    sub = parser.add_subparsers(dest="command", required=True)
    report = sub.add_parser("env-report")
    report.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    report.add_argument("--provider", action="append", choices=provider_names())
    report.add_argument("--output", type=Path, required=True)
    download = sub.add_parser("download-modelscope")
    download.add_argument("--model-id", default="facebook/sam-3d-objects")
    download.add_argument("--revision", required=True)
    download.add_argument("--cache-dir", type=Path, default=Path("/opt/data/private/ljn/.cache/modelscope"))
    download.add_argument("--target-dir", type=Path, required=True)
    download.add_argument("--manifest", type=Path, required=True)
    download.add_argument("--max-workers", type=int, default=1)
    download.add_argument("--license-acknowledgement", default="SAM License reviewed by project owner")
    download.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "env-report":
        names = args.provider or list(provider_names())
        value = write_environment_report(names, args.project_root.resolve(), args.output)
        print("MODEL_CENTER_ENV_REPORT_READY", args.output, len(value["providers"]), flush=True)
        return 0
    return _download_modelscope(args)


if __name__ == "__main__":
    raise SystemExit(main())
