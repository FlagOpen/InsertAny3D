#!/usr/bin/env python3
"""Executable bridge for image segmentation, TRELLIS, GIM and 3D pose.

Each invocation owns one task directory.  Pose estimation is enabled when the
Unity scene images are accompanied by aligned depth and camera metadata lists.
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
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = PROJECT_ROOT / "tools"
DEFAULT_TRELLIS_PYTHON = PROJECT_ROOT / "third_party" / "TRELLIS" / ".venv" / "bin" / "python"
DEFAULT_GIM_PYTHON = PROJECT_ROOT / "third_party" / "gim" / ".venv" / "bin" / "python"
POSE_RENDER_TOOL = TOOLS_ROOT / "render_trellis_views.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="串联分割、TRELLIS、3DGS 渲染和 GIM")
    parser.add_argument("--input-image", required=True, type=Path, help="编辑后的中心视角图片")
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--output-dir", type=Path, help="单个任务的独立输出目录（兼容入口）")
    output.add_argument("--run-root", type=Path, help="场景/批次输出根；与 --task-id 组合后自动建立任务目录")
    parser.add_argument("--task-id", help="稳定任务 ID；使用 --run-root 时必填")
    prompt = parser.add_mutually_exclusive_group()
    prompt.add_argument("--prompt", help="检测短语；提供后自动执行分割")
    prompt.add_argument("--task-prompt", help="任务描述；由 auto_segment 确定性改写")
    parser.add_argument("--input-ply", type=Path, help="已有 TRELLIS PLY；提供后跳过生成")
    parser.add_argument("--gim-pair", nargs=2, action="append", metavar=("IMAGE0", "IMAGE1"), help="显式 GIM 图片对，可重复")
    parser.add_argument("--scene-image", action="append", type=Path, help="场景图片；未给 --gim-pair 时依次与生成视图配对")
    parser.add_argument("--scene-depth", action="append", type=Path, help="与 --scene-image 同序的 Unity image.raw；提供后自动求 pose")
    parser.add_argument("--scene-camera", action="append", type=Path, help="与 --scene-image 同序的 Unity image.camera.json")
    parser.add_argument("--scene-mask", action="append", type=Path, help="可选：与 --scene-image 同序的锚点允许区域 mask")
    parser.add_argument("--generated-mask", action="append", type=Path, help="可选：与生成视图同序的锚点允许区域 mask")
    parser.add_argument("--unity-manifest", type=Path, help="Unity task_manifest.json；用于锚点 ROI、真实相机重渲染和证据链")
    parser.add_argument("--trellis-model", default=os.environ.get("TRELLIS_MODEL", "microsoft/TRELLIS-image-large"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sparse-steps", type=int)
    parser.add_argument("--slat-steps", type=int)
    parser.add_argument("--sparse-cfg", type=float)
    parser.add_argument("--slat-cfg", type=float)
    parser.add_argument(
        "--seg-engine",
        choices=("auto", "langsam", "legacy"),
        default="legacy",
        help="首批默认 legacy；auto 仅用于开发期 LangSAM 失败回退",
    )
    parser.add_argument(
        "--trellis-input",
        choices=("composite", "cutout"),
        default="composite",
        help="composite 先重建锚点+插入物体组合图；cutout 保留旧的只重建插入物体入口",
    )
    parser.add_argument(
        "--trellis-mask-prompt",
        dest="trellis_mask_prompts",
        action="append",
        default=[],
        help="组合路线中用于 TRELLIS 输入抠图的英文检测词；可重复并取并集",
    )
    parser.add_argument("--render-resolution", type=int, default=1024)
    parser.add_argument("--render-radius", type=float, default=1.5)
    parser.add_argument("--render-fov", type=float, default=53.1301023542)
    parser.add_argument("--render-mode", choices=("sphere", "anchor"), default="anchor", help="sphere 为环绕视图，anchor 为 yaw/pitch/左右三视图")
    parser.add_argument("--render-yaw-degrees", type=float, default=0.0)
    parser.add_argument("--render-pitch-degrees", type=float, default=12.0)
    parser.add_argument("--render-distance", type=float, default=1.5)
    parser.add_argument("--render-side-angle-degrees", type=float, default=24.0)
    parser.add_argument("--render-yaw-offsets", default=None)
    parser.add_argument("--render-view-names", default="left,center,right")
    parser.add_argument("--render-latitudes", default="10,20,30")
    parser.add_argument("--render-views-per-latitude", type=int, default=30)
    parser.add_argument("--gim-model", default="gim_roma", choices=("gim_dkm", "gim_roma", "gim_loftr", "gim_lightglue"))
    parser.add_argument("--coarse-pose-view-names", default="center", help="相机精化粗位姿使用的生成视角名，逗号分隔")
    parser.add_argument("--pose-view-names", default="all", help="参与联合 pose 的生成视角名，逗号分隔；all 表示全部")
    parser.add_argument("--pose-primary-view-name", help="可选主视图；要求其独立位姿至少得到另一视图正向佐证")
    parser.add_argument("--pose-generated-axis", choices=("identity", "legacy-flip-z"), default="legacy-flip-z")
    parser.add_argument("--pose-ransac-threshold", type=float, default=0.1)
    parser.add_argument("--pose-ransac-iterations", type=int, default=2000)
    parser.add_argument("--pose-min-inliers", type=int, default=6)
    parser.add_argument("--pose-max-matches-per-view", type=int, default=1000)
    parser.add_argument("--pose-max-depth-relative-spread", type=float, default=0.1)
    parser.add_argument("--pose-min-view-inliers", type=int, default=6)
    parser.add_argument("--pose-min-view-inlier-ratio", type=float, default=0.01)
    parser.add_argument("--pose-min-cross-view-inliers", type=int, default=3)
    parser.add_argument("--pose-min-cross-view-ratio", type=float, default=0.005)
    parser.add_argument("--pose-spatial-grid-size", type=int, default=8)
    parser.add_argument("--gim-anchor-roi-radius", type=float, default=256.0, help="Unity 锚点投影周围的圆形 ROI 半径，像素")
    parser.add_argument(
        "--gim-aligned-max-displacement",
        type=float,
        default=0.0,
        help="按 Unity 外参重渲染后允许的最大 GIM 像素位移；0 表示关闭",
    )
    parser.add_argument("--disable-camera-refinement", action="store_true", help="关闭粗位姿后按 Unity 外参重渲染")
    parser.add_argument("--trellis-python", type=Path, default=DEFAULT_TRELLIS_PYTHON)
    parser.add_argument("--gim-python", type=Path, default=DEFAULT_GIM_PYTHON)
    parser.add_argument("--cuda-device", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--skip-segmentation", action="store_true")
    parser.add_argument("--skip-trellis", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--skip-gim", action="store_true")
    parser.add_argument("--skip-pose", action="store_true")
    parser.add_argument("--run-sags", action="store_true", help="组合物体渲染后，用分割点驱动 SAGS 输出插入物体 PLY")
    parser.add_argument("--sags-python", type=Path, default=DEFAULT_TRELLIS_PYTHON)
    parser.add_argument("--sags-view-name", default="center")
    parser.add_argument("--sags-output-ply", type=Path)
    parser.add_argument("--sags-points-per-mask", type=int, default=4)
    parser.add_argument("--sags-force-seed-radius", type=int, default=2)
    parser.add_argument("--sags-no-force-seed", action="store_true")
    parser.add_argument("--sags-points-json", type=Path, help="已有 SAGS points.json；省略时使用本流程的 auto_segment 输出")
    parser.add_argument("--sags-mask", type=Path, help="已有完整 SAGS mask.png；省略时使用本流程的 auto_segment 输出")
    parser.add_argument("--sags-mask-id", type=int, default=-1)
    parser.add_argument("--sags-threshold", type=float, default=0.5)
    parser.add_argument("--sags-min-votes", type=int, default=2)
    parser.add_argument("--sags-visibility-depth-tolerance", type=float, default=0.02)
    parser.add_argument("--sags-gd-interval", type=int, default=-1)
    parser.add_argument("--run-id", help="本次流水线运行 ID；默认自动生成")
    parser.add_argument("--candidate-id", help="本次候选 ID；默认由输入和关键配置生成")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层必须是对象: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "sizeBytes": path.stat().st_size,
    }


def _xyz(value: dict[str, Any], name: str) -> tuple[float, float, float]:
    try:
        result = tuple(float(value[axis]) for axis in "xyz")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} 缺少 xyz") from exc
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} 包含无效数值")
    return result


def _quaternion_rotate_inverse(value: dict[str, Any], vector: tuple[float, float, float]) -> tuple[float, float, float]:
    try:
        x, y, z, w = (float(value[key]) for key in ("x", "y", "z", "w"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("rotationXyzw 格式错误") from exc
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        raise ValueError("rotationXyzw 是零四元数")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    # R(q)^T * vector, written explicitly to keep the orchestrator dependency-free.
    vx, vy, vz = vector
    return (
        (1 - 2 * (y * y + z * z)) * vx + 2 * (x * y + z * w) * vy + 2 * (x * z - y * w) * vz,
        2 * (x * y - z * w) * vx + (1 - 2 * (x * x + z * z)) * vy + 2 * (y * z + x * w) * vz,
        2 * (x * z + y * w) * vx + 2 * (y * z - x * w) * vy + (1 - 2 * (x * x + y * y)) * vz,
    )


def _project_anchor(unity_manifest: Path, camera_path: Path) -> tuple[float, float]:
    manifest = _read_json(unity_manifest)
    camera = _read_json(camera_path)
    anchor = _xyz(manifest["anchorPosition"], "anchorPosition")
    pose = camera["cameraToWorld"]
    center = _xyz(pose["position"], "cameraToWorld.position")
    local = _quaternion_rotate_inverse(
        pose["rotationXyzw"],
        (anchor[0] - center[0], anchor[1] - center[1], anchor[2] - center[2]),
    )
    if local[2] <= 0:
        raise ValueError(f"锚点位于相机后方: {camera_path}")
    intr = camera["intrinsics"]
    return (
        float(intr["fx"]) * local[0] / local[2] + float(intr["cx"]) - 0.5,
        float(intr["cy"]) - float(intr["fy"]) * local[1] / local[2] - 0.5,
    )


def _new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def _candidate_id(args: argparse.Namespace) -> str:
    payload = {
        "inputSha256": _sha256_file(args.input_image),
        "inputPlySha256": _sha256_file(args.input_ply) if args.input_ply and args.input_ply.is_file() else None,
        "seed": args.seed,
        "model": args.trellis_model,
        "trellisInput": args.trellis_input,
        "trellisMaskPrompts": args.trellis_mask_prompts,
        "render": {
            "resolution": args.render_resolution,
            "fov": args.render_fov,
            "yaw": args.render_yaw_degrees,
            "pitch": args.render_pitch_degrees,
            "distance": args.render_distance,
            "side": args.render_side_angle_degrees,
        },
        "pose": {
            "generatedAxis": args.pose_generated_axis,
            "cameraRefinement": not args.disable_camera_refinement,
            "coarseViewNames": args.coarse_pose_view_names,
            "viewNames": args.pose_view_names,
            "primaryViewName": args.pose_primary_view_name,
            "alignedMaxDisplacement": args.gim_aligned_max_displacement,
        },
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return f"candidate-{digest}"


def _run_stage(name: str, command: list[str], log_path: Path, env: dict[str, str], manifest: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    manifest["stages"].setdefault(name, {})
    manifest["stages"][name].update({"status": "running", "command": command, "log": str(log_path)})
    _json_dump(Path(manifest["manifest_path"]), manifest)
    printable = " ".join(shlex.quote(str(part)) for part in command)
    print(f"[{name}] {printable}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"COMMAND: {printable}\n")
        log.write(f"CUDA_VISIBLE_DEVICES: {env.get('CUDA_VISIBLE_DEVICES', '')}\n\n")
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        assert process.stdout is not None
        for line in process.stdout:
            print(f"[{name}] {line}", end="", flush=True)
            log.write(line)
        return_code = process.wait()
    manifest["stages"][name].update({
        "status": "ready" if return_code == 0 else "failed",
        "return_code": return_code,
        "duration_seconds": round(time.time() - started, 3),
    })
    _json_dump(Path(manifest["manifest_path"]), manifest)
    if return_code != 0:
        raise RuntimeError(f"阶段 {name} 失败，详见 {log_path}")


def _stage_env(cuda_device: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_device
    env["PYTHONUNBUFFERED"] = "1"
    # The server keeps the downloaded tokenizer/model cache outside /root.
    # Using it by default makes the existing GroundingDINO/SAM fallback
    # offline-reproducible; callers can override either variable explicitly.
    cache = Path("/opt/data/private/ljn/.cache/huggingface")
    if cache.is_dir():
        env.setdefault("HF_HOME", str(cache))
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    return env


def _sorted_rendered_images(render_dir: Path) -> list[Path]:
    images = sorted((render_dir / "source" / "images").glob("*.png"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    if not images:
        raise FileNotFoundError(f"没有找到 TRELLIS 渲染图片: {render_dir}")
    return images


def _first_rendered_image(render_dir: Path, index: int = 0) -> Path:
    images = _sorted_rendered_images(render_dir)
    return images[index % len(images)]


def _anchor_rendered_image(render_dir: Path, scene_image: Path, index: int, scene_count: int, names: list[str]) -> Path:
    """Choose a named left/center/right render for a corresponding scene view."""
    image_dir = render_dir / "source" / "images"
    stem = scene_image.stem.lower()
    if scene_count == 1 and "center" in names:
        preferred = "center"
    else:
        preferred = None
        for name in names:
            if name.lower() in stem:
                preferred = name
                break
        if preferred is None and index < len(names):
            preferred = names[index]
    if preferred:
        candidate = image_dir / f"{preferred}.png"
        if candidate.is_file():
            return candidate
    return _first_rendered_image(render_dir, index)


def _render_asset(
    args: argparse.Namespace,
    sample_ply: Path,
    render_dir: Path,
    stage_name: str,
    logs_dir: Path,
    env: dict[str, str],
    manifest: dict[str, Any],
    coarse_pose: Path | None = None,
) -> None:
    if args.render_mode == "anchor":
        command = [
            str(args.trellis_python), str(POSE_RENDER_TOOL),
            "--input-ply", str(sample_ply),
            "--output-dir", str(render_dir),
            "--resolution", str(args.render_resolution),
            "--fov", str(args.render_fov),
            "--yaw-degrees", str(args.render_yaw_degrees),
            "--pitch-degrees", str(args.render_pitch_degrees),
            "--distance", str(args.render_distance),
            "--side-angle-degrees", str(args.render_side_angle_degrees),
            "--view-names", str(args.render_view_names),
        ]
        if args.render_yaw_offsets is not None:
            command += ["--yaw-offsets", str(args.render_yaw_offsets)]
        if coarse_pose is not None:
            if not args.scene_camera or not args.unity_manifest:
                raise ValueError("按 Unity 外参渲染需要 scene-camera 和 unity-manifest")
            command += ["--coarse-pose", str(coarse_pose), "--unity-manifest", str(args.unity_manifest)]
            for camera in args.scene_camera:
                command += ["--unity-camera", str(camera)]
    else:
        if coarse_pose is not None:
            raise ValueError("sphere 模式不支持 Unity 外参重渲染")
        command = [
            str(args.trellis_python), str(TOOLS_ROOT / "render_trellis_3dgs.py"),
            "--input-ply", str(sample_ply), "--output-dir", str(render_dir),
            "--resolution", str(args.render_resolution), "--radius", str(args.render_radius),
            "--fov", str(args.render_fov), "--latitudes", args.render_latitudes,
            "--views-per-latitude", str(args.render_views_per_latitude),
        ]
    _run_stage(stage_name, command, logs_dir / f"{stage_name}.log", env, manifest)


def _pair_records(args: argparse.Namespace, render_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if args.gim_pair:
        for left, right in args.gim_pair:
            records.append({"image0": Path(left), "image1": Path(right)})
        return records
    scene_images = args.scene_image or [args.input_image]
    names = [name.strip() for name in args.render_view_names.split(",") if name.strip()]
    for index, scene_image in enumerate(scene_images):
        generated_image = (
            _anchor_rendered_image(render_dir, scene_image, index, len(scene_images), names)
            if args.render_mode == "anchor"
            else _first_rendered_image(render_dir, index)
        )
        record: dict[str, Any] = {"image0": scene_image, "image1": generated_image}
        if args.scene_depth:
            record.update(
                {
                    "scene_depth": args.scene_depth[index],
                    "scene_camera": args.scene_camera[index],
                    "generated_depth": render_dir / "source" / "depths" / "absdepth" / f"{generated_image.stem}.raw",
                }
            )
        if args.scene_mask:
            record["scene_mask"] = args.scene_mask[index]
        if args.generated_mask:
            record["generated_mask"] = args.generated_mask[index]
        records.append(record)
    return records


def _run_gim_pairs(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    gim_dir: Path,
    stage_prefix: str,
    logs_dir: Path,
    env: dict[str, str],
    manifest: dict[str, Any],
    aligned_cameras: bool,
    skip_execution: bool = False,
) -> None:
    results = []
    for index, record in enumerate(records):
        image0, image1 = record["image0"], record["image1"]
        pair_dir = gim_dir / f"pair_{index:02d}"
        record["pair_dir"] = pair_dir
        if skip_execution:
            continue
        if not image0.is_file() or not image1.is_file():
            raise FileNotFoundError(f"GIM 输入不存在: {image0}, {image1}")
        command = [
            str(args.gim_python), str(TOOLS_ROOT / "run_gim_match.py"),
            "--image0", str(image0), "--image1", str(image1),
            "--output-dir", str(pair_dir), "--model", args.gim_model,
            "--seed", str(args.seed), "--auto-mask1-nonblack", "--allow-empty",
        ]
        if args.unity_manifest and record.get("scene_camera"):
            anchor_x, anchor_y = _project_anchor(args.unity_manifest, record["scene_camera"])
            roi = [str(anchor_x), str(anchor_y), str(args.gim_anchor_roi_radius)]
            command += ["--roi0", *roi]
            if aligned_cameras:
                command += ["--roi1", *roi]
            record["anchor_roi"] = {"cx": anchor_x, "cy": anchor_y, "radius": args.gim_anchor_roi_radius}
        if record.get("scene_mask"):
            command += ["--mask0", str(record["scene_mask"])]
        if record.get("generated_mask"):
            command += ["--mask1", str(record["generated_mask"])]
        if aligned_cameras and args.gim_aligned_max_displacement > 0:
            command += ["--max-aligned-displacement", str(args.gim_aligned_max_displacement)]
        _run_stage(
            f"{stage_prefix}_pair_{index:02d}", command,
            logs_dir / f"{stage_prefix}_pair_{index:02d}.log", env, manifest,
        )
        results.append({"image0": str(image0), "image1": str(image1), "output_dir": str(pair_dir), "anchorRoi": record.get("anchor_roi")})
    manifest["stages"][stage_prefix] = {"status": "skipped" if skip_execution else "ready", "pairs": results}
    _json_dump(Path(manifest["manifest_path"]), manifest)


def _run_pose_fit(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    render_dir: Path,
    output: Path,
    diagnostics_dir: Path,
    stage_name: str,
    logs_dir: Path,
    env: dict[str, str],
    manifest: dict[str, Any],
    requested_names: set[str] | None,
    no_quality_gate: bool = False,
) -> dict[str, Any]:
    selected = [
        record for record in records
        if requested_names is None or record["image1"].stem.lower() in requested_names
    ]
    if not selected:
        raise ValueError(f"{stage_name} 没有匹配的生成视角")
    command = [
        str(args.gim_python), str(TOOLS_ROOT / "estimate_similarity_pose.py"),
        "--generated-cameras", str(render_dir / "source" / "sparse" / "0" / "cameras.txt"),
        "--generated-images", str(render_dir / "source" / "sparse" / "0" / "images.txt"),
        "--output", str(output), "--diagnostics-dir", str(diagnostics_dir),
        "--generated-axis", args.pose_generated_axis,
        "--ransac-threshold", str(args.pose_ransac_threshold),
        "--ransac-iterations", str(args.pose_ransac_iterations),
        "--min-inliers", str(args.pose_min_inliers),
        "--min-view-inliers", str(args.pose_min_view_inliers),
        "--min-view-inlier-ratio", str(args.pose_min_view_inlier_ratio),
        "--min-cross-view-inliers", str(args.pose_min_cross_view_inliers),
        "--min-cross-view-ratio", str(args.pose_min_cross_view_ratio),
        "--max-matches-per-view", str(args.pose_max_matches_per_view),
        "--max-depth-relative-spread", str(args.pose_max_depth_relative_spread),
        "--spatial-grid-size", str(args.pose_spatial_grid_size),
        "--seed", str(args.seed), "--run-id", args.run_id,
        "--candidate-id", args.candidate_id, "--exit-zero-on-rejected",
    ]
    if no_quality_gate:
        command += ["--no-quality-gate", "--allow-single-view"]
    elif args.pose_primary_view_name:
        command += ["--primary-view-name", args.pose_primary_view_name]
    for record in selected:
        matches = record["pair_dir"] / "matches.json"
        required = (matches, record["scene_depth"], record["scene_camera"], record["generated_depth"])
        if not all(Path(path).is_file() for path in required):
            missing = [str(path) for path in required if not Path(path).is_file()]
            raise FileNotFoundError(f"{stage_name} 输入不存在: {missing}")
        command += ["--view", *(str(path) for path in required)]
    _run_stage(stage_name, command, logs_dir / f"{stage_name}.log", env, manifest)
    value = _read_json(output)
    if value.get("status") != "ready":
        manifest["stages"][stage_name]["status"] = "rejected"
        manifest["stages"][stage_name]["rejection_reasons"] = value.get("validation", {}).get("rejectionReasons", [])
        _json_dump(Path(manifest["manifest_path"]), manifest)
    return value


def _write_evidence(args: argparse.Namespace, manifest: dict[str, Any]) -> Path:
    evidence_dir = args.output_dir / "evidence" / args.run_id
    records_dir = evidence_dir / "records"
    paths: set[Path] = {args.input_image.resolve(), Path(manifest["manifest_path"]).resolve()}
    for optional in (args.unity_manifest, args.sags_points_json, args.sags_mask, args.input_ply):
        if optional and optional.is_file():
            paths.add(optional.resolve())
    for key in ("sample_ply", "trellis_input_path", "pose", "sags_ply"):
        recorded = manifest.get(key)
        if recorded and Path(recorded).is_file():
            paths.add(Path(recorded).resolve())
    for values in (args.scene_image, args.scene_depth, args.scene_camera, args.scene_mask, args.generated_mask):
        for path in values or []:
            if path.is_file():
                paths.add(path.resolve())
    patterns = (
        "00_trellis_input/mask.png", "00_trellis_input/manifest.json",
        "01_segmentation/mask.png", "01_segmentation/points.json", "01_segmentation/manifest.json",
        "02_trellis/sample.ply", "02_trellis/manifest.json",
        "03_rendered_3dgs/views.json", "03_rendered_3dgs/source/sparse/0/*.txt",
        "03_rendered_3dgs/source/images/*.png", "03_rendered_3dgs/source/depths/absdepth/*.raw",
        "03_rendered_3dgs_initial/views.json", "03_rendered_3dgs_initial/source/sparse/0/*.txt",
        "04_gim/pair_*/matches.json", "04_gim/multiview_summary.*",
        "04_gim_initial/pair_*/matches.json", "04_gim_initial/multiview_summary.*",
        "05_pose/*.json", "06_sags/*.json", "06_sags/*.ply",
        "06_sags/diagnostics/**/*.png", "06_sags/diagnostics/**/*.json",
    )
    for pattern in patterns:
        paths.update(path.resolve() for path in args.output_dir.glob(pattern) if path.is_file())
    artifacts: list[dict[str, Any]] = []
    for path in sorted(paths, key=str):
        item = _artifact(path)
        try:
            relative = path.relative_to(args.output_dir.resolve())
            item["taskRelativePath"] = relative.as_posix()
            if path.suffix.lower() in {".json", ".txt"} and path.stat().st_size <= 20 * 1024 * 1024:
                target = records_dir / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                item["evidenceCopy"] = str(target.resolve())
        except ValueError:
            item["taskRelativePath"] = None
        artifacts.append(item)
    evidence = {
        "schemaVersion": 2,
        "runId": args.run_id,
        "candidateId": args.candidate_id,
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "taskId": args.task_id,
        "pipelineStatus": manifest.get("status"),
        "selectedPose": manifest.get("pose"),
        "selectedSagsPly": manifest.get("sags_ply"),
        "artifactCount": len(artifacts),
        "artifacts": artifacts,
    }
    evidence_path = evidence_dir / "manifest.json"
    _json_dump(evidence_path, evidence)
    _json_dump(args.output_dir / "provenance.json", {**evidence, "artifacts": artifacts})
    return evidence_path


def main() -> int:
    args = parse_args()
    if args.run_root:
        if not args.task_id:
            raise SystemExit("--run-root 必须同时提供 --task-id")
        task_id = args.task_id.strip()
        if not task_id or task_id in (".", "..") or Path(task_id).name != task_id or any(char in task_id for char in ("/", "\\")):
            raise SystemExit(f"task-id 不是安全的目录名: {args.task_id!r}")
        args.task_id = task_id
        args.output_dir = args.run_root / task_id
    elif args.task_id:
        task_id = args.task_id.strip()
        if not task_id or task_id in (".", "..") or Path(task_id).name != task_id or any(char in task_id for char in ("/", "\\")):
            raise SystemExit(f"task-id 不是安全的目录名: {args.task_id!r}")
        args.task_id = task_id
    else:
        args.task_id = args.output_dir.name
    if not args.input_image.is_file():
        raise SystemExit(f"输入图片不存在: {args.input_image}")
    if args.unity_manifest and not args.unity_manifest.is_file():
        raise SystemExit(f"Unity task manifest 不存在: {args.unity_manifest}")
    if args.input_ply and not args.input_ply.is_file():
        raise SystemExit(f"输入 PLY 不存在: {args.input_ply}")
    if not args.trellis_python.is_file():
        raise SystemExit(f"TRELLIS Python 不存在: {args.trellis_python}")
    if not args.gim_python.is_file():
        raise SystemExit(f"GIM Python 不存在: {args.gim_python}")
    if args.run_sags and not args.sags_python.is_file():
        raise SystemExit(f"SAGS Python 不存在: {args.sags_python}")
    if (
        args.pose_ransac_threshold <= 0 or args.pose_ransac_iterations < 1
        or args.pose_min_inliers < 3 or args.pose_max_matches_per_view < 0
        or args.pose_max_depth_relative_spread < 0 or args.pose_spatial_grid_size < 1
        or args.gim_anchor_roi_radius <= 0 or args.gim_aligned_max_displacement < 0
    ):
        raise SystemExit("pose threshold/iterations/min-inliers/max-matches 参数无效")
    if bool(args.scene_depth) != bool(args.scene_camera):
        raise SystemExit("--scene-depth 与 --scene-camera 必须同时提供")
    pose_requested = bool(args.scene_depth) and not args.skip_pose
    if pose_requested:
        if args.gim_pair:
            raise SystemExit("自动 pose 目前要求使用 --scene-image/--scene-depth/--scene-camera，不能与 --gim-pair 混用")
        scene_count = len(args.scene_image or [args.input_image])
        if len(args.scene_depth) != scene_count or len(args.scene_camera) != scene_count:
            raise SystemExit("--scene-image、--scene-depth、--scene-camera 数量必须一致")
    scene_count = len(args.scene_image or [args.input_image])
    if args.scene_mask and len(args.scene_mask) != scene_count:
        raise SystemExit("--scene-mask 必须与 scene-image 数量一致")
    if args.generated_mask and len(args.generated_mask) != scene_count:
        raise SystemExit("--generated-mask 必须与 scene-image 数量一致")
    if args.trellis_input == "composite" and args.run_sags and args.render_mode != "anchor":
        raise SystemExit("组合物体运行 SAGS 需要 --render-mode anchor，以便 points.json 与生成 center.png 对齐")
    if args.trellis_mask_prompts and args.trellis_input != "composite":
        raise SystemExit("--trellis-mask-prompt 只适用于 --trellis-input composite")
    if args.sags_force_seed_radius < 0:
        raise SystemExit("--sags-force-seed-radius 不能为负")
    if args.sags_points_per_mask < 1:
        raise SystemExit("--sags-points-per-mask 必须大于 0")
    if args.run_sags and (args.sags_min_votes < 1 or not 0 <= args.sags_threshold <= 1):
        raise SystemExit("SAGS threshold/min-votes 参数无效")

    args.run_id = args.run_id or _new_run_id()
    args.candidate_id = args.candidate_id or _candidate_id(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = args.output_dir / "logs"
    manifest: dict[str, Any] = {
        "manifest_path": str(args.output_dir / "manifest.json"),
        "schemaVersion": 2,
        "run_id": args.run_id,
        "candidate_id": args.candidate_id,
        "task_id": args.task_id,
        "input_image": str(args.input_image.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "cuda_device": args.cuda_device,
        "trellis_input": args.trellis_input,
        "trellis_mask_prompts": args.trellis_mask_prompts,
        "task_prompt": args.task_prompt,
        "object_prompt": args.prompt,
        "run_sags": args.run_sags,
        "unity_manifest": str(args.unity_manifest.resolve()) if args.unity_manifest else None,
        "segmentation_config": {
            "requested_engine": args.seg_engine,
            "prompt": args.prompt,
            "task_prompt": args.task_prompt,
        },
        "sags_config": {
            "view_name": args.sags_view_name,
            "points_per_mask": args.sags_points_per_mask,
            "force_seed_radius": args.sags_force_seed_radius,
            "force_seed": not args.sags_no_force_seed,
            "points_json": str(args.sags_points_json.resolve()) if args.sags_points_json else None,
            "mask": str(args.sags_mask.resolve()) if args.sags_mask else None,
            "mask_id": args.sags_mask_id,
            "threshold": args.sags_threshold,
            "min_votes": args.sags_min_votes,
            "visibility_depth_tolerance": args.sags_visibility_depth_tolerance,
            "gd_interval": args.sags_gd_interval,
        },
        "render_config": {
            "mode": args.render_mode,
            "resolution": args.render_resolution,
            "fov_degrees": args.render_fov,
            "yaw_degrees": args.render_yaw_degrees,
            "pitch_degrees": args.render_pitch_degrees,
            "distance": args.render_distance,
            "side_angle_degrees": args.render_side_angle_degrees,
            "radius": args.render_radius,
            "yaw_offsets": args.render_yaw_offsets if args.render_yaw_offsets is not None else "generated from side_angle_degrees",
            "view_names": args.render_view_names,
            "latitudes": args.render_latitudes,
            "views_per_latitude": args.render_views_per_latitude,
        },
        "pose_config": {
            "requested": pose_requested,
            "coarse_view_names": args.coarse_pose_view_names,
            "view_names": args.pose_view_names,
            "primary_view_name": args.pose_primary_view_name,
            "generated_axis": args.pose_generated_axis,
            "ransac_threshold": args.pose_ransac_threshold,
            "ransac_iterations": args.pose_ransac_iterations,
            "min_inliers": args.pose_min_inliers,
            "max_matches_per_view": args.pose_max_matches_per_view,
            "max_depth_relative_spread": args.pose_max_depth_relative_spread,
            "min_view_inliers": args.pose_min_view_inliers,
            "min_view_inlier_ratio": args.pose_min_view_inlier_ratio,
            "min_cross_view_inliers": args.pose_min_cross_view_inliers,
            "min_cross_view_ratio": args.pose_min_cross_view_ratio,
            "spatial_grid_size": args.pose_spatial_grid_size,
            "camera_refinement": not args.disable_camera_refinement,
            "gim_anchor_roi_radius": args.gim_anchor_roi_radius,
            "gim_aligned_max_displacement": args.gim_aligned_max_displacement,
        },
        "stages": {},
        "warnings": [],
    }
    _json_dump(Path(manifest["manifest_path"]), manifest)
    env = _stage_env(args.cuda_device)

    # An explicit union mask keeps TRELLIS focused on the anchor + inserted
    # object and avoids rembg selecting a small overlapping fragment.
    composite_input = args.input_image
    trellis_mask_dir = args.output_dir / "00_trellis_input"
    if args.trellis_mask_prompts and not args.input_ply:
        command = [
            str(args.trellis_python),
            str(TOOLS_ROOT / "auto_segment.py"),
            "--input", str(args.input_image),
            "--output-dir", str(trellis_mask_dir),
            "--engine", args.seg_engine,
        ]
        for prompt_value in args.trellis_mask_prompts:
            command += ["--prompt", prompt_value]
        _run_stage("trellis_input_segmentation", command, logs_dir / "trellis_input_segmentation.log", env, manifest)
        composite_input = trellis_mask_dir / "cutout.png"
        if not composite_input.is_file():
            raise SystemExit(f"TRELLIS 组合蒙版输出不存在: {composite_input}")
    else:
        manifest["stages"]["trellis_input_segmentation"] = {
            "status": "skipped",
            "reason": "no mask prompts or --input-ply",
        }
        _json_dump(Path(manifest["manifest_path"]), manifest)

    # Downstream segmentation is deliberately deferred until the generated
    # center view exists, so SAGS points use the generated image coordinates.
    segmentation_dir = args.output_dir / "01_segmentation"
    cutout = None
    trellis_input_path = composite_input if args.trellis_input == "composite" else None
    if args.trellis_input == "cutout" and (args.skip_segmentation or not (args.prompt or args.task_prompt)):
        cutout = args.input_image
    segmentation_deferred = args.trellis_input == "composite" and not args.skip_segmentation and (args.prompt or args.task_prompt)
    if args.trellis_input == "cutout" and not args.skip_segmentation and (args.prompt or args.task_prompt):
        command = [str(args.trellis_python), str(TOOLS_ROOT / "auto_segment.py"), "--input", str(args.input_image), "--output-dir", str(segmentation_dir), "--engine", args.seg_engine, "--points-per-mask", str(args.sags_points_per_mask)]
        if args.prompt:
            command += ["--prompt", args.prompt]
        else:
            command += ["--task-prompt", args.task_prompt]
        try:
            _run_stage("segmentation", command, logs_dir / "segmentation.log", env, manifest)
            cutout = segmentation_dir / "cutout.png"
        except RuntimeError:
            if not args.continue_on_error:
                raise
            manifest["warnings"].append("segmentation failed; continuing with original image")
    elif segmentation_deferred:
        manifest["stages"]["segmentation"] = {
            "status": "deferred",
            "reason": "composite route segments the generated center render",
        }
        _json_dump(Path(manifest["manifest_path"]), manifest)
    else:
        manifest["stages"]["segmentation"] = {
            "status": "skipped",
            "reason": "no prompt, --skip-segmentation, or no cutout mode",
        }
        _json_dump(Path(manifest["manifest_path"]), manifest)

    # Stage 2: image -> Gaussian/mesh.
    trellis_dir = args.output_dir / "02_trellis"
    if args.input_ply:
        sample_ply = args.input_ply.resolve()
        manifest["stages"]["trellis"] = {"status": "skipped", "reason": "--input-ply", "sample_ply": str(sample_ply)}
        _json_dump(Path(manifest["manifest_path"]), manifest)
    elif args.skip_trellis:
        raise SystemExit("--skip-trellis 需要同时提供 --input-ply")
    else:
        trellis_input = composite_input if args.trellis_input == "composite" else cutout
        if trellis_input is None or not trellis_input.is_file():
            raise SystemExit(f"TRELLIS 输入图片不存在: {trellis_input}")
        trellis_input_path = trellis_input
        command = [str(args.trellis_python), str(TOOLS_ROOT / "generate_trellis_asset.py"), "--input-image", str(trellis_input), "--output-dir", str(trellis_dir), "--model", args.trellis_model, "--seed", str(args.seed)]
        for flag, value in (("--sparse-steps", args.sparse_steps), ("--slat-steps", args.slat_steps), ("--sparse-cfg", args.sparse_cfg), ("--slat-cfg", args.slat_cfg)):
            if value is not None:
                command += [flag, str(value)]
        try:
            _run_stage("trellis", command, logs_dir / "trellis.log", env, manifest)
        except RuntimeError:
            if not args.continue_on_error:
                raise
        sample_ply = trellis_dir / "sample.ply"

    manifest["trellis_input_path"] = str(trellis_input_path.resolve()) if trellis_input_path and trellis_input_path.exists() else None

    # Stage 3: canonical render -> center-view coarse pose -> exact Unity
    # camera render.  Existing/diagnostic runs can disable the refinement and
    # retain the legacy single render directory.
    render_dir = args.output_dir / "03_rendered_3dgs"
    refinement_enabled = bool(
        pose_requested and args.unity_manifest and args.render_mode == "anchor"
        and not args.disable_camera_refinement and not args.skip_render and not args.skip_gim
    )
    manifest["camera_refinement"] = {"enabled": refinement_enabled}
    if args.skip_render:
        manifest["stages"]["render"] = {"status": "skipped"}
        _json_dump(Path(manifest["manifest_path"]), manifest)
    else:
        if not sample_ply.is_file():
            raise SystemExit(f"渲染阶段找不到 sample.ply: {sample_ply}")
        if refinement_enabled:
            initial_render_dir = args.output_dir / "03_rendered_3dgs_initial"
            _render_asset(args, sample_ply, initial_render_dir, "render_initial", logs_dir, env, manifest)
            coarse_records = _pair_records(args, initial_render_dir)
            _run_gim_pairs(
                args, coarse_records, args.output_dir / "04_gim_initial", "gim_initial",
                logs_dir, env, manifest, aligned_cameras=False,
            )
            coarse_names = {
                name.strip().lower()
                for name in args.coarse_pose_view_names.split(",")
                if name.strip()
            }
            if not coarse_names:
                raise ValueError("coarse-pose-view-names 不能为空")
            if "all" in coarse_names:
                coarse_names = None
            coarse_pose = args.output_dir / "05_pose" / "coarse_pose.json"
            coarse_value = _run_pose_fit(
                args, coarse_records, initial_render_dir, coarse_pose,
                args.output_dir / "04_gim_initial", "pose_coarse",
                logs_dir, env, manifest, coarse_names, no_quality_gate=True,
            )
            if coarse_value.get("status") != "ready":
                raise RuntimeError("粗位姿失败，不能转换 Unity 相机")
            _render_asset(args, sample_ply, render_dir, "render_aligned", logs_dir, env, manifest, coarse_pose)
            manifest["camera_refinement"].update(
                {"initialRender": str(initial_render_dir), "coarsePose": str(coarse_pose), "alignedRender": str(render_dir)}
            )
        else:
            _render_asset(args, sample_ply, render_dir, "render", logs_dir, env, manifest)

    # Final GIM and pose always refer to the same final render directory.
    pair_records = _pair_records(args, render_dir)
    _run_gim_pairs(
        args, pair_records, args.output_dir / "04_gim", "gim",
        logs_dir, env, manifest, aligned_cameras=refinement_enabled,
        skip_execution=args.skip_gim,
    )
    pose_value = None
    if pose_requested:
        requested = {name.strip().lower() for name in args.pose_view_names.split(",") if name.strip()}
        requested_names = None if "all" in requested else requested
        pose_output = args.output_dir / "05_pose" / "pose.json"
        pose_value = _run_pose_fit(
            args, pair_records, render_dir, pose_output, args.output_dir / "04_gim", "pose",
            logs_dir, env, manifest, requested_names,
        )
        manifest["pose"] = str(pose_output)
    else:
        manifest["stages"].setdefault("pose", {"status": "skipped", "reason": "no Unity depth/camera metadata or --skip-pose"})

    # Segment the final aligned center render so mask.png, points.json and the
    # SAGS camera set have exactly the same pixels and camera metadata.
    if segmentation_deferred:
        if args.render_mode != "anchor":
            raise SystemExit("composite 路线的 auto_segment 需要 anchor 渲染得到 center.png")
        generated_center = render_dir / "source" / "images" / "center.png"
        if not generated_center.is_file():
            raise SystemExit(f"组合物体 center 渲染不存在，无法执行分割: {generated_center}")
        command = [
            str(args.trellis_python), str(TOOLS_ROOT / "auto_segment.py"),
            "--input", str(generated_center), "--output-dir", str(segmentation_dir),
            "--engine", args.seg_engine, "--points-per-mask", str(args.sags_points_per_mask),
        ]
        command += ["--prompt", args.prompt] if args.prompt else ["--task-prompt", args.task_prompt]
        _run_stage("segmentation", command, logs_dir / "segmentation.log", env, manifest)
        cutout = segmentation_dir / "cutout.png"

    if args.run_sags:
        points_json = args.sags_points_json or segmentation_dir / "points.json"
        mask_path = args.sags_mask or segmentation_dir / "mask.png"
        if not points_json.is_file() or not mask_path.is_file():
            raise SystemExit(f"SAGS 完整标注不存在: {points_json}, {mask_path}")
        model_dir = render_dir / "model"
        if not model_dir.is_dir():
            raise SystemExit(f"SAGS model 目录不存在: {model_dir}")
        sags_output = args.sags_output_ply or args.output_dir / "06_sags" / "inserted_object.ply"
        command = [
            str(args.sags_python), str(TOOLS_ROOT / "run_sags_text.py"),
            "--model-dir", str(model_dir), "--points-json", str(points_json),
            "--mask", str(mask_path), "--output-ply", str(sags_output),
            "--view-name", args.sags_view_name,
            "--mask-id", str(args.sags_mask_id), "--threshold", str(args.sags_threshold),
            "--min-votes", str(args.sags_min_votes),
            "--visibility-depth-tolerance", str(args.sags_visibility_depth_tolerance),
            "--gd-interval", str(args.sags_gd_interval),
            "--force-seed-radius", str(args.sags_force_seed_radius),
            "--diagnostics-dir", str(args.output_dir / "06_sags" / "diagnostics"),
        ]
        if args.sags_no_force_seed:
            command.append("--no-force-seed")
        _run_stage("sags", command, logs_dir / "sags.log", env, manifest)
        manifest["sags_ply"] = str(sags_output) if sags_output.is_file() else None

    manifest["sample_ply"] = str(sample_ply) if sample_ply.exists() else None
    manifest["cutout"] = str(cutout) if cutout and cutout.exists() else None
    failed_stages = [name for name, value in manifest["stages"].items() if value.get("status") == "failed"]
    rejected_stages = [name for name, value in manifest["stages"].items() if value.get("status") == "rejected"]
    manifest["status"] = "failed" if failed_stages else "rejected" if rejected_stages else "ready"
    manifest["failed_stages"] = failed_stages
    manifest["rejected_stages"] = rejected_stages
    manifest["evidence"] = str((args.output_dir / "evidence" / args.run_id / "manifest.json").resolve())
    _json_dump(Path(manifest["manifest_path"]), manifest)
    _write_evidence(args, manifest)
    _json_dump(Path(manifest["manifest_path"]), manifest)
    if failed_stages:
        print("INSERT_PIPELINE_FAILED", ",".join(failed_stages), args.output_dir, flush=True)
        return 1
    if rejected_stages:
        print("INSERT_PIPELINE_REJECTED", ",".join(rejected_stages), args.output_dir, flush=True)
        return 2
    print("INSERT_PIPELINE_READY", args.output_dir, flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit("已取消")
