#!/usr/bin/env python3
"""Build a self-contained InsertAny3D debugging bundle on the server.

The normal pipeline keeps useful intermediate files in numbered stage
directories, but those directories are awkward to inspect after a remote run.
This tool creates a curated, task-local bundle while also retaining a complete
copy of the original task directory under ``99_raw_pipeline``.

In addition to copying the existing RGB GIM results, it renders the raw depth
files to stable PNG previews, runs GIM on those previews, and creates one 2x2
grid per view:

    top row    depth scene | depth generated
    bottom row RGB scene   | RGB generated

The top row uses the same RGB correspondences after depth/pose filtering.  A
separate depth-image GIM can still be retained as a labelled diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GIM_TOOL = Path(__file__).with_name("run_gim_match.py")
DEFAULT_GIM_PYTHON = PROJECT_ROOT / "third_party" / "gim" / ".venv" / "bin" / "python"
VIEW_ORDER = ("left", "center", "right")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="整理 InsertAny3D 任务的完整调试证据包")
    parser.add_argument("--run-root", required=True, type=Path, help="已完成批处理的远端 run 根目录")
    parser.add_argument("--output-root", required=True, type=Path, help="调试包输出目录；必须位于 run-root 外部")
    parser.add_argument("--task", dest="tasks", action="append", help="只处理指定 task；可重复")
    parser.add_argument("--gim-python", type=Path, default=DEFAULT_GIM_PYTHON)
    parser.add_argument("--gim-model", default="gim_roma", choices=("gim_dkm", "gim_roma", "gim_loftr", "gim_lightglue"))
    parser.add_argument("--cuda-device", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--grid-cell-size", type=int, default=640)
    parser.add_argument("--max-grid-matches", type=int, default=100)
    parser.add_argument("--rerun-rgb", action="store_true", help="即使已有 RGB GIM 结果也重新计算")
    parser.add_argument("--reuse-existing-depth", action="store_true", help="复用输出目录中已有的深度 matches.json，不重新占用 GPU")
    parser.add_argument("--skip-depth-gim", action="store_true", help="只生成深度预览，不运行深度 GIM")
    parser.add_argument("--no-raw", action="store_true", help="不复制 99_raw_pipeline；默认会保留完整原始任务目录")
    return parser.parse_args()


def safe_task_id(value: str) -> str:
    task_id = str(value).strip()
    if not task_id or task_id in {".", ".."} or Path(task_id).name != task_id or "/" in task_id or "\\" in task_id:
        raise ValueError(f"不安全的 task id: {value!r}")
    return task_id


def read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def first_existing_file(*values: Any) -> Path | None:
    """Return the first recorded path that is currently readable."""
    for value in values:
        if value is None or str(value).strip() == "":
            continue
        path = Path(str(value))
        if path.is_file():
            return path
    return None


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def copy_file(source: Path, target: Path) -> bool:
    if not source.is_file():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return True


def copy_tree(source: Path, target: Path) -> bool:
    if not source.is_dir():
        return False
    shutil.copytree(source, target, dirs_exist_ok=True)
    return True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_pose_provenance(task_source: Path) -> dict[str, Any]:
    pose = read_json(task_source / "05_pose" / "pose.json", {}) or {}
    expected = pose.get("provenance", {}).get("sourceFileDigests", {})
    checks = []
    for view, files in expected.items():
        for role, record in files.items():
            path = Path(str(record.get("path", ""))) if isinstance(record, dict) else Path("")
            exists = path.is_file()
            actual = sha256_file(path) if exists else None
            wanted = record.get("sha256") if isinstance(record, dict) else None
            checks.append(
                {
                    "view": view,
                    "role": role,
                    "path": str(path),
                    "exists": exists,
                    "expectedSha256": wanted,
                    "actualSha256": actual,
                    "matches": bool(exists and wanted and actual == wanted),
                }
            )
    return {
        "poseStatus": pose.get("status"),
        "runId": pose.get("provenance", {}).get("runId"),
        "candidateId": pose.get("provenance", {}).get("candidateId"),
        "hasRecordedDigests": bool(expected),
        "allMatched": bool(checks) and all(item["matches"] for item in checks),
        "checks": checks,
    }


def infer_view(*values: str) -> str | None:
    text = " ".join(values).lower()
    for view in VIEW_ORDER:
        if view in text:
            return view
    return None


def image_shape(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"无法读取图片: {path}")
    height, width = image.shape[:2]
    return int(width), int(height)


def load_depth(raw_path: Path, width: int, height: int) -> np.ndarray:
    values = np.fromfile(raw_path, dtype="<f4")
    expected = width * height
    if values.size != expected:
        # Keep the diagnostic tool useful for older runs whose camera metadata
        # was not copied: infer a square image only when that is unambiguous.
        side = int(round(math.sqrt(values.size)))
        if side * side == values.size:
            width = height = side
            expected = values.size
        else:
            raise ValueError(f"深度尺寸不匹配 {raw_path}: {values.size} floats，期望 {expected}")
    return values[:expected].reshape(height, width)


def write_depth_previews(raw_path: Path, color_path: Path, gray_path: Path, stats_path: Path, width: int, height: int) -> dict[str, Any]:
    depth = load_depth(raw_path, width, height)
    valid = depth[np.isfinite(depth) & (depth > 0)]
    if valid.size == 0:
        lo, hi = 0.0, 1.0
        normalized = np.zeros(depth.shape, dtype=np.float32)
    else:
        lo = float(np.percentile(valid, 1.0))
        hi = float(np.percentile(valid, 99.0))
        if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
            lo = float(valid.min())
            hi = float(valid.max())
        if hi <= lo:
            hi = lo + 1.0
        normalized = np.clip((depth - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
        normalized[~np.isfinite(depth) | (depth <= 0)] = 0.0
    gray = np.round(normalized * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    color[gray == 0] = 0
    color_path.parent.mkdir(parents=True, exist_ok=True)
    gray_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(color_path), color)
    # GIM accepts ordinary RGB/gray image files.  Keep this input grayscale so
    # the matcher sees depth edges rather than a synthetic color gradient.
    cv2.imwrite(str(gray_path), gray)
    metadata = {
        "raw": str(raw_path),
        "width": int(depth.shape[1]),
        "height": int(depth.shape[0]),
        "valid_ratio": float(valid.size / depth.size) if depth.size else 0.0,
        "percentile_1": lo,
        "percentile_99": hi,
        "min_valid": float(valid.min()) if valid.size else None,
        "max_valid": float(valid.max()) if valid.size else None,
        "visualization": "linear percentile 1..99; invalid values black; turbo preview + grayscale GIM input",
    }
    write_json(stats_path, metadata)
    return metadata


def run_command(command: list[str], log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(shlex.quote(str(item)) for item in command)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"COMMAND: {printable}\n")
        log.write(f"CUDA_VISIBLE_DEVICES: {env.get('CUDA_VISIBLE_DEVICES', '')}\n\n")
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
        return process.wait()


def choose_matches(source_matches: dict[str, Any], target_root: Path) -> dict[str, Any]:
    """Copy match JSON while replacing paths with bundle-local paths."""
    result = dict(source_matches)
    result["source_image0"] = source_matches.get("image0")
    result["source_image1"] = source_matches.get("image1")
    result["image0"] = "scene.png"
    result["image1"] = "generated.png"
    result["match_image"] = "match.png"
    result["warp_image"] = "warp.png"
    result["bundle_relative_paths"] = True
    write_json(target_root / "matches.json", result)
    return result


def select_match_indices(matches: dict[str, Any], limit: int) -> np.ndarray:
    points = np.asarray(matches.get("mkpts0", []), dtype=np.float32)
    if points.ndim != 2 or points.shape[1:] != (2,):
        return np.empty((0,), dtype=np.int64)
    count = len(points)
    inliers = np.asarray(matches.get("inliers", np.ones(count)), dtype=bool).reshape(-1)
    if len(inliers) != count:
        inliers = np.ones(count, dtype=bool)
    confidence = np.asarray(matches.get("confidence", np.ones(count)), dtype=np.float32).reshape(-1)
    if len(confidence) != count:
        confidence = np.ones(count, dtype=np.float32)
    candidates = np.flatnonzero(inliers)
    if len(candidates) == 0:
        candidates = np.arange(count)
    order = candidates[np.argsort(confidence[candidates])[::-1]]
    return order[: max(0, limit)]


def _fit_image(image: np.ndarray, size: int) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = image[:, :, :3]
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def _draw_row(canvas: np.ndarray, left: np.ndarray, right: np.ndarray, matches: dict[str, Any], y: int, size: int, limit: int) -> None:
    left_fit = _fit_image(left, size)
    right_fit = _fit_image(right, size)
    canvas[y : y + size, :size] = left_fit
    canvas[y : y + size, size:] = right_fit
    indices = select_match_indices(matches, limit)
    points0 = np.asarray(matches.get("mkpts0", []), dtype=np.float32)
    points1 = np.asarray(matches.get("mkpts1", []), dtype=np.float32)
    if points0.ndim != 2 or points1.shape != points0.shape:
        return
    h0, w0 = left.shape[:2]
    h1, w1 = right.shape[:2]
    for index in indices:
        x0 = int(round(float(points0[index, 0]) * size / max(1, w0)))
        y0 = y + int(round(float(points0[index, 1]) * size / max(1, h0)))
        x1 = size + int(round(float(points1[index, 0]) * size / max(1, w1)))
        y1 = y + int(round(float(points1[index, 1]) * size / max(1, h1)))
        cv2.line(canvas, (x0, y0), (x1, y1), (60, 220, 80), 1, cv2.LINE_AA)
        cv2.circle(canvas, (x0, y0), 3, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (x1, y1), 3, (0, 255, 255), -1, cv2.LINE_AA)


def _draw_filtered_row(canvas: np.ndarray, left: np.ndarray, right: np.ndarray, matches: dict[str, Any], y: int, size: int, limit: int) -> None:
    left_fit = _fit_image(left, size)
    right_fit = _fit_image(right, size)
    canvas[y : y + size, :size] = left_fit
    canvas[y : y + size, size:] = right_fit
    points0 = np.asarray(matches.get("mkpts0", []), dtype=np.float32)
    points1 = np.asarray(matches.get("mkpts1", []), dtype=np.float32)
    accepted = np.asarray(matches.get("inliers", []), dtype=bool).reshape(-1)
    if points0.ndim != 2 or points1.shape != points0.shape or len(accepted) != len(points0):
        return
    indices = np.linspace(0, len(points0) - 1, min(len(points0), max(0, limit)), dtype=np.int64) if len(points0) else []
    h0, w0 = left.shape[:2]
    h1, w1 = right.shape[:2]
    for index in indices:
        x0 = int(round(float(points0[index, 0]) * size / max(1, w0)))
        y0 = y + int(round(float(points0[index, 1]) * size / max(1, h0)))
        x1 = size + int(round(float(points1[index, 0]) * size / max(1, w1)))
        y1 = y + int(round(float(points1[index, 1]) * size / max(1, h1)))
        color = (60, 220, 80) if accepted[index] else (50, 70, 230)
        cv2.line(canvas, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)
        cv2.circle(canvas, (x0, y0), 3, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (x1, y1), 3, color, -1, cv2.LINE_AA)


def make_grid(
    output: Path,
    depth_scene: Path,
    depth_generated: Path,
    depth_matches: dict[str, Any],
    rgb_scene: Path,
    rgb_generated: Path,
    rgb_matches: dict[str, Any],
    pose_matches: dict[str, Any] | None,
    cell_size: int,
    max_matches: int,
) -> None:
    depth_left = cv2.imread(str(depth_scene), cv2.IMREAD_COLOR)
    depth_right = cv2.imread(str(depth_generated), cv2.IMREAD_COLOR)
    rgb_left = cv2.imread(str(rgb_scene), cv2.IMREAD_COLOR)
    rgb_right = cv2.imread(str(rgb_generated), cv2.IMREAD_COLOR)
    if any(image is None for image in (depth_left, depth_right, rgb_left, rgb_right)):
        raise ValueError("GIM 2x2 网格输入图片读取失败")
    canvas = np.zeros((cell_size * 2, cell_size * 2, 3), dtype=np.uint8)
    if pose_matches:
        _draw_filtered_row(canvas, depth_left, depth_right, pose_matches, 0, cell_size, max_matches)
    else:
        _draw_row(canvas, depth_left, depth_right, depth_matches, 0, cell_size, max_matches)
    _draw_row(canvas, rgb_left, rgb_right, rgb_matches, cell_size, cell_size, max_matches)
    # Labels are deliberately ASCII so OpenCV's built-in font is portable.
    labels = (("DEPTH-FILTERED / SCENE", 8, 22), ("DEPTH-FILTERED / GENERATED", cell_size + 8, 22),
              ("RGB / SCENE", 8, cell_size + 22), ("RGB / GENERATED", cell_size + 8, cell_size + 22))
    overlay = canvas.copy()
    for label, x, y in labels:
        cv2.rectangle(overlay, (x - 4, y - 18), (x + 190, y + 5), (0, 0, 0), -1)
        cv2.putText(overlay, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    canvas = cv2.addWeighted(overlay, 0.62, canvas, 0.38, 0)
    cv2.line(canvas, (cell_size, 0), (cell_size, cell_size * 2), (255, 255, 255), 2)
    cv2.line(canvas, (0, cell_size), (cell_size * 2, cell_size), (255, 255, 255), 2)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas)


def run_or_copy_rgb(
    source_pair: Path,
    scene_rgb: Path,
    generated_rgb: Path,
    target_rgb: Path,
    args: argparse.Namespace,
    env: dict[str, str],
) -> tuple[dict[str, Any], str]:
    target_rgb.mkdir(parents=True, exist_ok=True)
    copy_file(scene_rgb, target_rgb / "scene.png")
    copy_file(generated_rgb, target_rgb / "generated.png")
    source_matches_path = source_pair / "matches.json"
    if source_matches_path.is_file() and not args.rerun_rgb:
        source_matches = read_json(source_matches_path, {}) or {}
        copy_file(source_pair / "match.png", target_rgb / "match.png")
        copy_file(source_pair / "warp.png", target_rgb / "warp.png")
        write_json(target_rgb / "matches.source.json", source_matches)
        matches = choose_matches(source_matches, target_rgb)
        return matches, "copied"
    command = [str(args.gim_python), str(GIM_TOOL), "--image0", str(target_rgb / "scene.png"), "--image1", str(target_rgb / "generated.png"), "--output-dir", str(target_rgb), "--model", args.gim_model]
    return_code = run_command(command, target_rgb / "gim.log", env)
    if return_code != 0 or not (target_rgb / "matches.json").is_file():
        return {}, f"failed:{return_code}"
    matches = read_json(target_rgb / "matches.json", {}) or {}
    # Preserve the absolute source references separately and normalize the
    # public JSON for a bundle that can move between machines.
    write_json(target_rgb / "matches.source.json", matches)
    matches = choose_matches(matches, target_rgb)
    return matches, "rerun"


def process_pair(
    task_source: Path,
    task_bundle: Path,
    view: str,
    pair_index: int,
    args: argparse.Namespace,
    env: dict[str, str],
    pose_diagnostic: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fallback_pair = task_source / "04_gim" / f"pair_{pair_index:02d}"
    recorded_files = pose_diagnostic.get("files", {}) if pose_diagnostic else {}
    if not isinstance(recorded_files, dict):
        recorded_files = {}
    recorded_matches = first_existing_file(recorded_files.get("matches"))
    source_pair = recorded_matches.parent if recorded_matches else fallback_pair
    source_matches = read_json(source_pair / "matches.json", {}) or {}
    scene_rgb = first_existing_file(
        recorded_files.get("scene_image"), source_matches.get("image0"),
        task_source / "step1" / view / "image.png",
    )
    scene_raw = first_existing_file(
        recorded_files.get("unity_depth"),
        scene_rgb.with_name("image.raw") if scene_rgb else None,
        task_source / "step1" / view / "image.raw",
    )
    scene_camera = first_existing_file(
        recorded_files.get("unity_camera"),
        scene_rgb.with_name("image.camera.json") if scene_rgb else None,
        task_source / "step1" / view / "image.camera.json",
    )
    generated_rgb = first_existing_file(
        recorded_files.get("generated_image"), source_matches.get("image1"),
        task_source / "03_rendered_3dgs" / "source" / "images" / f"{view}.png",
    )
    generated_raw = first_existing_file(
        recorded_files.get("generated_depth"),
        task_source / "03_rendered_3dgs" / "source" / "depths" / "absdepth" / f"{view}.raw",
    )
    pair_bundle = task_bundle / "03_gim" / f"{view}_pair"
    rgb_dir = pair_bundle / "rgb"
    depth_dir = pair_bundle / "depth"
    pair_bundle.mkdir(parents=True, exist_ok=True)
    required = (scene_rgb, generated_rgb, scene_raw, generated_raw)
    if any(path is None for path in required):
        raise FileNotFoundError(f"{view} GIM 输入不完整: {scene_rgb}, {generated_rgb}, {scene_raw}, {generated_raw}")
    assert scene_rgb is not None and generated_rgb is not None and scene_raw is not None and generated_raw is not None

    # A replay may intentionally consume Unity captures or a TRELLIS asset
    # from an earlier run.  Preserve those exact external inputs in the
    # curated bundle instead of assuming task-local step1 files exist.
    scene_view_dir = task_bundle / "01_edit" / "scene_views" / view
    copy_file(scene_rgb, scene_view_dir / "image.png")
    copy_file(scene_raw, scene_view_dir / "image.raw")
    if scene_camera:
        copy_file(scene_camera, scene_view_dir / "image.camera.json")
    (pair_bundle / "ERROR.txt").unlink(missing_ok=True)

    rgb_matches, rgb_status = run_or_copy_rgb(source_pair, scene_rgb, generated_rgb, rgb_dir, args, env)
    scene_width, scene_height = image_shape(scene_rgb)
    generated_width, generated_height = image_shape(generated_rgb)
    scene_depth_preview = depth_dir / "scene_depth.png"
    generated_depth_preview = depth_dir / "generated_depth.png"
    scene_depth_input = depth_dir / "scene_depth_input.png"
    generated_depth_input = depth_dir / "generated_depth_input.png"
    scene_stats = write_depth_previews(scene_raw, scene_depth_preview, scene_depth_input, depth_dir / "scene_depth_stats.json", scene_width, scene_height)
    generated_stats = write_depth_previews(generated_raw, generated_depth_preview, generated_depth_input, depth_dir / "generated_depth_stats.json", generated_width, generated_height)
    copy_file(scene_raw, depth_dir / "scene.raw")
    copy_file(generated_raw, depth_dir / "generated.raw")

    depth_matches: dict[str, Any] = {}
    depth_status = "skipped"
    if not args.skip_depth_gim and args.reuse_existing_depth and (depth_dir / "matches.json").is_file():
        depth_matches = read_json(depth_dir / "matches.json", {}) or {}
        depth_status = "reused"
    elif not args.skip_depth_gim:
        command = [str(args.gim_python), str(GIM_TOOL), "--image0", str(scene_depth_input), "--image1", str(generated_depth_input), "--output-dir", str(depth_dir), "--model", args.gim_model]
        return_code = run_command(command, depth_dir / "gim.log", env)
        if return_code == 0 and (depth_dir / "matches.json").is_file():
            depth_matches = read_json(depth_dir / "matches.json", {}) or {}
            write_json(depth_dir / "matches.source.json", depth_matches)
            depth_matches = choose_matches(depth_matches, depth_dir)
            depth_status = "ready"
        else:
            depth_status = f"failed:{return_code}"
    if not depth_matches:
        depth_matches = {"mkpts0": [], "mkpts1": [], "inliers": [], "confidence": [], "match_count": 0, "inlier_count": 0}

    pose_matches = None
    if pose_diagnostic:
        pose_matches = {
            "mkpts0": pose_diagnostic.get("depthValidScenePixels", []),
            "mkpts1": pose_diagnostic.get("depthValidGeneratedPixels", []),
            "inliers": pose_diagnostic.get("jointInliers", []),
        }
        write_json(pair_bundle / "pose_filtered_matches.json", pose_matches)
    grid_path = pair_bundle / "match_2x2.png"
    make_grid(
        grid_path, scene_depth_preview, generated_depth_preview, depth_matches,
        scene_rgb, generated_rgb, rgb_matches, pose_matches,
        args.grid_cell_size, args.max_grid_matches,
    )
    pair_manifest = {
        "schemaVersion": 1,
        "view": view,
        "pairIndex": pair_index,
        "rgbStatus": rgb_status,
        "depthStatus": depth_status,
        "rgb": {
            "matchCount": int(rgb_matches.get("match_count", 0)),
            "inlierCount": int(rgb_matches.get("inlier_count", 0)),
            "inlierRatio": float(rgb_matches.get("inlier_ratio", 0.0)),
            "directory": "rgb",
        },
        "depth": {
            "matchCount": int(depth_matches.get("match_count", 0)),
            "inlierCount": int(depth_matches.get("inlier_count", 0)),
            "inlierRatio": float(depth_matches.get("inlier_ratio", 0.0)),
            "directory": "depth",
            "sceneStats": scene_stats,
            "generatedStats": generated_stats,
        },
        "grid": "match_2x2.png",
        "topRowSource": "pose_depth_valid_correspondences" if pose_matches else "legacy_depth_gim_fallback",
        "source": {
            "sceneRgb": str(scene_rgb),
            "generatedRgb": str(generated_rgb),
            "sceneDepth": str(scene_raw),
            "sceneCamera": str(scene_camera) if scene_camera else None,
            "generatedDepth": str(generated_raw),
            "originalGimPair": str(source_pair),
        },
    }
    write_json(pair_bundle / "pair_manifest.json", pair_manifest)
    return pair_manifest


def write_readme(task_bundle: Path) -> None:
    task_bundle.joinpath("README.md").write_text(
        "# InsertAny3D 调试包\n\n"
        "本目录由 `build_debug_bundle.py` 自动生成。`99_raw_pipeline` 保留原始任务目录，前面的目录是便于人工检查的整理版。\n\n"
        "## 目录\n\n"
        "- `00_batch_metadata/`：批次 job、batch manifest 和调度日志。\n"
        "- `01_edit/`：编辑后的中心图，以及步骤 1 的 Unity 三视图。\n"
        "- `02_trellis/`：TRELLIS 输入、sample.ply/sample.glb、Gaussian 渲染 RGB/深度/相机。\n"
        "- `03_gim/<view>_pair/`：每个视角的 RGB 匹配、深度有效点、最终接受/拒绝点和 `match_2x2.png`。\n"
        "- `03_gim/multiview_summary.png/json`：单视角拟合、交叉验证、留一验证和联合位姿门禁。\n"
        "- `04_sags/`：SAGS 输入标注（mask、annotated、points）和所有结果 PLY/预览。\n"
        "- `05_pose/`：姿态估计及候选姿态。\n"
        "- `06_logs/`：阶段日志、pipeline manifest，以及当前 run 的唯一 evidence。\n"
        "- `99_raw_pipeline/`：未裁剪的原始任务输出，可能包含历史 run。\n\n"
        "## 2x2 图\n\n"
        "每个 `03_gim/<view>_pair/match_2x2.png` 的上排在深度图上显示同一批 RGB 对应点经过深度和位姿过滤后的状态，下排显示 RGB 几何匹配。绿色为最终接受，红色为拒绝。单独的 depth/matches.json 仍只是深度图 GIM 诊断。\n",
        encoding="utf-8",
    )


def process_task(task_source: Path, task_bundle: Path, args: argparse.Namespace, env: dict[str, str]) -> dict[str, Any]:
    task_id = safe_task_id(task_source.name)
    if task_bundle.exists():
        if task_bundle.is_symlink() or not task_bundle.is_dir():
            raise RuntimeError(f"拒绝覆盖非普通任务目录: {task_bundle}")
        shutil.rmtree(task_bundle)
    task_bundle.mkdir(parents=True, exist_ok=True)
    pipeline_manifest = read_json(task_source / "manifest.json", {}) or {}
    input_image = first_existing_file(
        pipeline_manifest.get("input_image"), task_source / "edited" / "center.png",
    )
    if input_image:
        copy_file(input_image, task_bundle / "01_edit" / "center.png")
        for name in ("edit_manifest.json", "prompt.txt", "response.json", "center.camera.json", "center.txt"):
            copy_file(input_image.parent / name, task_bundle / "01_edit" / "edited_metadata" / name)
    copy_tree(task_source / "edited", task_bundle / "01_edit" / "edited_metadata")
    copy_tree(task_source / "step1", task_bundle / "01_edit" / "scene_views")
    copy_tree(task_source / "02_trellis", task_bundle / "02_trellis" / "input_and_asset")
    sample_ply = first_existing_file(
        pipeline_manifest.get("sample_ply"), task_source / "02_trellis" / "sample.ply",
    )
    if sample_ply:
        copy_file(sample_ply, task_bundle / "02_trellis" / "input_and_asset" / "sample.ply")
        for name in ("sample.glb", "manifest.json", "processed_00.png"):
            copy_file(sample_ply.parent / name, task_bundle / "02_trellis" / "input_and_asset" / name)
    copy_tree(task_source / "03_rendered_3dgs", task_bundle / "02_trellis" / "gaussian_render")
    copy_tree(task_source / "01_segmentation", task_bundle / "04_sags" / "input_annotation")
    copy_tree(task_source / "06_sags", task_bundle / "04_sags" / "results")
    copy_tree(task_source / "05_pose", task_bundle / "05_pose")
    selected_evidence = first_existing_file(
        pipeline_manifest.get("evidence"),
        task_source / "evidence" / str(pipeline_manifest.get("run_id", "")) / "manifest.json",
    )
    selected_evidence_rel = None
    if selected_evidence:
        selected_evidence_rel = Path("06_logs/evidence") / selected_evidence.parent.name / "manifest.json"
        copy_tree(
            selected_evidence.parent,
            task_bundle / "06_logs" / "evidence" / selected_evidence.parent.name,
        )
    copy_file(task_source / "provenance.json", task_bundle / "06_logs" / "provenance.json")
    copy_tree(task_source / "logs", task_bundle / "06_logs")
    for name in ("manifest.json", "task_manifest.json", "prompts.json"):
        copy_file(task_source / name, task_bundle / "06_logs" / name)
    unity_manifest = first_existing_file(pipeline_manifest.get("unity_manifest"))
    if unity_manifest:
        copy_file(unity_manifest, task_bundle / "06_logs" / "task_manifest.json")
    # The generated composite is useful when comparing SAGS' mask against the
    # actual Gaussian render, so include an explicit copy in its input folder.
    copy_file(task_source / "03_rendered_3dgs" / "source" / "images" / "center.png", task_bundle / "04_sags" / "input_annotation" / "generated_center.png")
    for name in ("multiview_summary.json", "multiview_summary.png", "left_validation.png", "center_validation.png", "right_validation.png"):
        copy_file(task_source / "04_gim" / name, task_bundle / "03_gim" / name)
    multiview = read_json(task_source / "04_gim" / "multiview_summary.json", {}) or {}
    diagnostics_by_view = {
        str(item.get("name")): item
        for item in multiview.get("views", [])
        if isinstance(item, dict) and item.get("name")
    }

    # Prefer the view encoded in each existing RGB matches.json.  The default
    # pipeline writes pair_00/01/02 in left/center/right order, but explicit
    # --gim-pair jobs are allowed to use a different order.
    pair_indices: dict[str, int] = {}
    for matches_path in sorted((task_source / "04_gim").glob("pair_*/matches.json")):
        source_matches = read_json(matches_path, {}) or {}
        view = infer_view(str(source_matches.get("image0", "")), str(source_matches.get("image1", "")))
        if view:
            try:
                pair_indices[view] = int(matches_path.parent.name.rsplit("_", 1)[1])
            except (ValueError, IndexError):
                pass

    pair_manifests = []
    for index, view in enumerate(VIEW_ORDER):
        pair_index = pair_indices.get(view, index)
        try:
            pair_manifests.append(process_pair(task_source, task_bundle, view, pair_index, args, env, diagnostics_by_view.get(view)))
        except Exception as exc:
            pair_manifests.append({"view": view, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            pair_dir = task_bundle / "03_gim" / f"{view}_pair"
            pair_dir.mkdir(parents=True, exist_ok=True)
            (pair_dir / "ERROR.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")

    if not args.no_raw:
        copy_tree(task_source, task_bundle / "99_raw_pipeline")
    write_readme(task_bundle)
    provenance_check = verify_pose_provenance(task_source)
    write_json(task_bundle / "06_logs" / "provenance_check.json", provenance_check)
    bundle_manifest = {
        "schemaVersion": 1,
        "taskId": task_id,
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "sourceRun": str(task_source),
        "sourcePipelineStatus": pipeline_manifest.get("status"),
        "gimModel": args.gim_model,
        "cudaDevice": args.cuda_device,
        "depthGimDiagnosticOnly": True,
        "selectedEvidence": str(selected_evidence_rel) if selected_evidence_rel else None,
        "multiviewSummary": "03_gim/multiview_summary.png" if (task_bundle / "03_gim" / "multiview_summary.png").is_file() else None,
        "provenanceCheck": provenance_check,
        "pairCount": len(pair_manifests),
        "pairs": pair_manifests,
        "rawPipelineIncluded": not args.no_raw,
    }
    write_json(task_bundle / "bundle_manifest.json", bundle_manifest)
    return bundle_manifest


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    output_root = args.output_root.resolve()
    if not run_root.is_dir():
        raise SystemExit(f"run-root 不存在: {run_root}")
    if not args.gim_python.is_file():
        raise SystemExit(f"GIM Python 不存在: {args.gim_python}")
    if args.grid_cell_size < 128 or args.max_grid_matches < 0:
        raise SystemExit("grid 参数无效")
    try:
        common = os.path.commonpath((str(run_root), str(output_root)))
        if common == str(run_root):
            raise SystemExit("output-root 必须位于 run-root 外部，避免把调试包递归复制进自身")
    except ValueError:
        pass
    if args.tasks:
        task_ids = [safe_task_id(item) for item in args.tasks]
    else:
        task_ids = sorted(path.name for path in run_root.iterdir() if path.is_dir() and path.name.startswith("Task_"))
    if not task_ids:
        raise SystemExit(f"run-root 下没有 Task_* 目录: {run_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    env["PYTHONUNBUFFERED"] = "1"
    results = []
    for task_id in task_ids:
        source = run_root / task_id
        if not source.is_dir():
            results.append({"taskId": task_id, "status": "missing"})
            continue
        print(f"[debug-bundle] {task_id}", flush=True)
        result = process_task(source, output_root / task_id, args, env)
        results.append(result)
    for name in ("batch_manifest.json", "batch.console.log", "insert_jobs.json"):
        copy_file(run_root / name, output_root / "00_batch_metadata" / name)
    root_manifest = {
        "schemaVersion": 1,
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "runRoot": str(run_root),
        "outputRoot": str(output_root),
        "gimModel": args.gim_model,
        "tasks": results,
    }
    write_json(output_root / "bundle_manifest.json", root_manifest)
    print("DEBUG_BUNDLE_READY", output_root, len(results), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
