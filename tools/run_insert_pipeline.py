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
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from model_center.contracts import CoordinateContract, GenerationRequest, RenderRequest
from model_center.config import ProviderProfile, resolve_profile
from model_center.registry import get_provider, provider_environment_report, provider_names
from model_center.renderers.provider_render import build_render_command
from model_center.segmentation.manager import MaskManager, MaskManagerConfig
from model_center.transforms.gaussian_ply import transform_gaussian_ply


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = PROJECT_ROOT / "tools"
DEFAULT_TRELLIS_PYTHON = PROJECT_ROOT / "third_party" / "TRELLIS" / ".venv" / "bin" / "python"
DEFAULT_GIM_PYTHON = PROJECT_ROOT / "third_party" / "gim" / ".venv" / "bin" / "python"

_ANCHOR_PROMPT_ALIASES = {
    "粉色的猪": "pig",
    "小猪": "pig",
    "猪": "pig",
    "南瓜": "pumpkin",
    "拖拉机": "tractor",
    "农用车": "tractor",
    "引擎盖": "tractor",
    "汽车": "car",
    "车辆": "vehicle",
    "牛": "cow",
    "奶牛": "cow",
    "人": "person",
}


def _infer_anchor_mask_prompt(args: argparse.Namespace) -> str:
    candidates = []
    if args.anchor_mask_prompt:
        candidates.append(str(args.anchor_mask_prompt).strip())
    candidates.extend(str(value).strip() for value in reversed(args.trellis_mask_prompts or []) if str(value).strip())
    for candidate in candidates:
        lowered = candidate.lower()
        for source, target in _ANCHOR_PROMPT_ALIASES.items():
            if source in candidate:
                return target
        if all(ord(char) < 128 for char in candidate) and len(candidate) <= 80:
            return candidate
    return ""


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
    parser.add_argument(
        "--model-provider",
        choices=provider_names(),
        default=os.environ.get("INSERTANY3D_MODEL_PROVIDER", "trellis"),
        help="3D 生成 provider；所有 provider 最终都写标准 Gaussian PLY",
    )
    parser.add_argument("--model-profile", default="default", help="provider profile 名称，写入 manifest")
    parser.add_argument("--provider-options-json", default="{}", help="provider-specific JSON options")
    parser.add_argument("--model-input-mask", type=Path, help="provider 生成前的单物体 mask；SAM3D 必需")
    parser.add_argument("--model-mask-prompt", action="append", default=[], help="生成前 mask 的英文 prompt，可重复")
    parser.add_argument("--model-dir", type=Path, help="SAM3D ModelScope materialized model directory")
    parser.add_argument("--model-config-path", type=Path, help="SAM3D pipeline.yaml override")
    parser.add_argument("--hunyuan-python", type=Path, default=PROJECT_ROOT / "third_party" / "Hunyuan3D-2" / ".venv" / "bin" / "python")
    parser.add_argument("--hunyuan-model-path", help="Hunyuan local path or model id")
    parser.add_argument("--hunyuan-shape-subfolder", default="hunyuan3d-dit-v2-0")
    parser.add_argument("--hunyuan-texture", action="store_true", help="run optional Hunyuan paint stage")
    parser.add_argument("--mesh-to-gaussian-density", type=float, default=32.0)
    parser.add_argument("--mesh-to-gaussian-thickness", type=float, default=0.002)
    parser.add_argument("--mesh-to-gaussian-max-points", type=int, default=250000)
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
    parser.add_argument("--anchor-mask-prompt", help="用于三视图位姿筛选的简短英文锚点类别")
    parser.add_argument("--anchor-mask-box-threshold", type=float, default=0.25)
    parser.add_argument("--anchor-mask-dilation", type=int, default=16)
    parser.add_argument("--skip-anchor-masking", action="store_true", help="兼容/测试入口；最终位姿不执行语义锚点筛选")
    parser.add_argument("--render-resolution", type=int, default=1024)
    parser.add_argument("--render-radius", type=float, default=1.5)
    parser.add_argument("--render-fov", type=float, default=53.1301023542)
    parser.add_argument("--render-mode", choices=("sphere", "anchor"), default="anchor", help="sphere 为环绕视图，anchor 为 yaw/pitch/左右三视图")
    parser.add_argument("--render-yaw-degrees", type=float, default=0.0)
    parser.add_argument("--render-pitch-degrees", type=float, default=12.0)
    parser.add_argument("--render-distance", type=float, default=1.5)
    parser.add_argument("--render-near", type=float, help="override provider coordinate-contract near plane")
    parser.add_argument("--render-far", type=float, help="override provider coordinate-contract far plane")
    parser.add_argument("--render-side-angle-degrees", type=float, default=24.0)
    parser.add_argument("--render-yaw-offsets", default=None)
    parser.add_argument("--render-view-names", default="left,center,right")
    parser.add_argument("--render-latitudes", default="10,20,30")
    parser.add_argument("--render-views-per-latitude", type=int, default=30)
    parser.add_argument("--gim-model", default="gim_roma", choices=("gim_dkm", "gim_roma", "gim_loftr", "gim_lightglue"))
    parser.add_argument("--coarse-pose-view-names", default="center", help="相机精化粗位姿使用的生成视角名，逗号分隔")
    parser.add_argument("--pose-view-names", default="all", help="参与联合 pose 的生成视角名，逗号分隔；all 表示全部")
    parser.add_argument("--pose-primary-view-name", help=argparse.SUPPRESS)
    parser.add_argument("--pose-generated-axis", choices=("identity", "legacy-flip-z"), default="legacy-flip-z")
    parser.add_argument("--pose-ransac-threshold", type=float, default=0.1)
    parser.add_argument("--pose-ransac-iterations", type=int, default=3000)
    parser.add_argument("--pose-min-inliers", type=int, default=6)
    parser.add_argument("--pose-max-matches-per-view", type=int, default=0)
    parser.add_argument("--pose-max-depth-relative-spread", type=float, default=0.1)
    parser.add_argument("--pose-min-view-inliers", type=int, default=6)
    parser.add_argument("--pose-min-view-inlier-ratio", type=float, default=0.01)
    parser.add_argument("--pose-cross-view-neighbors", type=int, default=16)
    parser.add_argument("--pose-cross-view-min-support", type=int, default=2)
    parser.add_argument("--pose-cross-view-fallback-support", type=int, default=1)
    parser.add_argument("--pose-min-consistent-points", type=int, default=30)
    parser.add_argument("--pose-min-consistent-view-points", type=int, default=6)
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
    parser.add_argument(
        "--sags-view-mode",
        choices=("ring6", "legacy"),
        default="ring6",
        help="SAGS 视角策略；ring6 默认六视角独立标注，legacy 保留中心点投影流程",
    )
    parser.add_argument(
        "--sags-yaw-offsets",
        default="0,60,120,180,240,300",
        help="ring6 相对中心 yaw 偏移，逗号分隔",
    )
    parser.add_argument(
        "--sags-view-names",
        default="center,ring_060,ring_120,ring_180,ring_240,ring_300",
        help="ring6 视角名称，必须与 yaw 偏移数量相同",
    )
    parser.add_argument("--sags-output-ply", type=Path)
    parser.add_argument("--sags-points-per-mask", type=int, default=4)
    parser.add_argument("--sags-force-seed-radius", type=int, default=2)
    parser.add_argument("--sags-no-force-seed", action="store_true")
    parser.add_argument("--sags-points-json", type=Path, help="已有 SAGS points.json；省略时使用本流程的 auto_segment 输出")
    parser.add_argument("--sags-mask", type=Path, help="已有完整 SAGS mask.png；省略时使用本流程的 auto_segment 输出")
    parser.add_argument("--sags-mask-id", type=int, default=-1)
    parser.add_argument("--sags-threshold", type=float, default=0.5)
    parser.add_argument("--sags-min-votes", type=int, default=2)
    parser.add_argument(
        "--sags-independent-min-prior-coverage",
        type=float,
        default=0.25,
        help="ring6 非源视角标注覆盖中心 Gaussian 几何先验的最低比例；0 关闭门控",
    )
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


def _cached_huggingface_revision(model_id: str) -> str | None:
    """Read only the public refs/main marker; never inspect credentials."""
    if "/" not in str(model_id):
        return None
    cache_roots = []
    if os.environ.get("HF_HOME"):
        cache_roots.append(Path(os.environ["HF_HOME"]))
    cache_roots.append(Path("/opt/data/private/ljn/.cache/huggingface"))
    cache_name = "models--" + str(model_id).replace("/", "--")
    for root in cache_roots:
        ref = root / "hub" / cache_name / "refs" / "main"
        if ref.is_file():
            value = ref.read_text(encoding="utf-8").strip()
            if value:
                return value
    return None


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
        "provider": args.model_provider,
        "profile": args.model_profile,
        "providerOptions": _provider_options(args),
        "weights": _provider_weight_descriptor(args),
        "inputMaskSha256": _sha256_file(args.model_input_mask) if args.model_input_mask and args.model_input_mask.is_file() else None,
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
            "policy": "point_consistency_joint_fit",
            "anchorMaskPrompt": _infer_anchor_mask_prompt(args),
            "anchorMaskDilation": args.anchor_mask_dilation,
            "crossViewNeighbors": args.pose_cross_view_neighbors,
            "crossViewMinSupport": args.pose_cross_view_min_support,
            "crossViewFallbackSupport": args.pose_cross_view_fallback_support,
            "alignedMaxDisplacement": args.gim_aligned_max_displacement,
        },
        "sags": {
            "viewMode": args.sags_view_mode,
            "yawOffsets": args.sags_yaw_offsets,
            "viewNames": args.sags_view_names,
            "threshold": args.sags_threshold,
            "minVotes": args.sags_min_votes,
            "independentMinPriorCoverage": args.sags_independent_min_prior_coverage,
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
    stage_record = {
        "status": "ready" if return_code == 0 else "failed",
        "return_code": return_code,
        "duration_seconds": round(time.time() - started, 3),
    }
    if return_code != 0:
        try:
            tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]
        except OSError:
            tail = []
        stage_record["error"] = "\n".join(tail) or f"stage process exited with code {return_code}"
    manifest["stages"][name].update(stage_record)
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


def _provider_profile(args: argparse.Namespace) -> ProviderProfile:
    try:
        value = json.loads(str(args.provider_options_json or "{}"))
    except json.JSONDecodeError as exc:
        raise ValueError("--provider-options-json 必须是合法 JSON 对象") from exc
    if not isinstance(value, dict):
        raise ValueError("--provider-options-json 顶层必须是对象")
    return resolve_profile(args.model_provider, args.model_profile, value)


def _provider_options(args: argparse.Namespace) -> dict[str, Any]:
    profile = _provider_profile(args)
    options = profile.merged_options()
    if args.model_provider == "trellis":
        for key in ("sparse_steps", "slat_steps", "sparse_cfg", "slat_cfg"):
            value_from_args = getattr(args, key, None)
            if value_from_args is not None:
                options.setdefault(key, value_from_args)
    if args.model_provider == "sam3d" and args.model_dir:
        options.setdefault("model_dir", str(args.model_dir.resolve()))
    if args.model_provider == "sam3d" and args.model_config_path:
        options.setdefault("config_path", str(args.model_config_path.resolve()))
    if args.model_provider == "hunyuan":
        if args.hunyuan_model_path:
            options.setdefault("model_path", args.hunyuan_model_path)
        if args.hunyuan_shape_subfolder:
            options.setdefault("shape_subfolder", args.hunyuan_shape_subfolder)
        if args.hunyuan_texture:
            options.setdefault("texture", True)
    return options


def _provider_weight_descriptor(args: argparse.Namespace) -> dict[str, Any]:
    if args.model_provider == "sam3d":
        model_dir = args.model_dir or PROJECT_ROOT / "third_party" / "SAM3D-Objects" / "checkpoints" / "modelscope"
        candidates = (
            model_dir.parent / "modelscope_weights_manifest.json",
            model_dir / "weights_manifest.json",
        )
        for path in candidates:
            if not path.is_file():
                continue
            try:
                value = _read_json(path)
            except (OSError, json.JSONDecodeError):
                continue
            return {
                "manifest": str(path.resolve()),
                "source": value.get("source"),
                "modelId": value.get("modelId"),
                "revision": value.get("revision"),
                "fingerprint": value.get("weightFingerprint") or _sha256_file(path),
            }
        return {"manifest": None, "modelDir": str(model_dir.resolve())}
    if args.model_provider == "hunyuan":
        model_path = args.hunyuan_model_path or os.environ.get("HUNYUAN_MODEL_PATH", "tencent/Hunyuan3D-2")
        revision = None
        path = Path(model_path)
        if path.is_dir():
            parts = path.parts
            if "snapshots" in parts and parts.index("snapshots") + 1 < len(parts):
                revision = parts[parts.index("snapshots") + 1]
            for ref in (path / "refs" / "main", path.parent / "refs" / "main"):
                if revision is None and ref.is_file():
                    revision = ref.read_text(encoding="utf-8").strip() or None
        return {
            "modelPath": model_path,
            "revision": revision,
            "shapeSubfolder": args.hunyuan_shape_subfolder,
        }
    return {
        "model": args.trellis_model,
        "revision": _cached_huggingface_revision(args.trellis_model),
    }


def _provider_instance(args: argparse.Namespace):
    provider = get_provider(args.model_provider, PROJECT_ROOT)
    runtime = args.hunyuan_python if args.model_provider == "hunyuan" else args.trellis_python
    provider.spec = replace(provider.spec, runtime_python=Path(runtime).resolve())
    return provider


def _provider_input_mask(
    args: argparse.Namespace,
    output_dir: Path,
    env: dict[str, str],
    manifest: dict[str, Any],
) -> Path | None:
    if args.model_provider == "trellis":
        return args.model_input_mask.resolve() if args.model_input_mask else None
    if args.model_input_mask:
        mask = args.model_input_mask.resolve()
        if not mask.is_file():
            raise FileNotFoundError(f"provider input mask 不存在: {mask}")
        # Validate dimensions before a model runtime allocates GPU memory.
        try:
            from PIL import Image

            with Image.open(args.input_image) as image, Image.open(mask) as mask_image:
                if image.size != mask_image.size:
                    raise ValueError(
                        f"provider input mask 尺寸 {mask_image.size} 与输入图 {image.size} 不一致"
                    )
        except ImportError:
            pass
        manifest["model_input_mask"] = str(mask)
        manifest["model_input_mask_source"] = "provided_mask"
        return mask
    prompts = list(args.model_mask_prompt or [])
    if not prompts and args.prompt:
        prompts = [args.prompt]
    if not prompts and args.task_prompt:
        prompts = [args.task_prompt]
    if not prompts:
        if args.model_provider == "sam3d":
            raise ValueError("sam3d 必须提供 --model-input-mask、--model-mask-prompt、--prompt 或 --task-prompt")
        manifest["model_input_mask_source"] = "provider_local_fallback"
        return None
    mask_dir = output_dir / "00_model_input"
    manager = MaskManager(
        TOOLS_ROOT,
        MaskManagerConfig(engine=args.seg_engine, human_confirmed=False),
    )
    mask, artifact = manager.generate(
        args.input_image,
        prompts,
        mask_dir,
        python=args.trellis_python,
    )
    manifest["model_input_mask"] = str(mask.resolve())
    manifest["model_input_mask_artifact"] = artifact.to_dict()
    return mask.resolve()


def _run_provider_generation(
    args: argparse.Namespace,
    output_dir: Path,
    env: dict[str, str],
    manifest: dict[str, Any],
    logs_dir: Path,
    input_image: Path | None = None,
) -> Path:
    provider = _provider_instance(args)
    options = _provider_options(args)
    input_mask = _provider_input_mask(args, output_dir, env, manifest)
    request = GenerationRequest(
        input_image=(input_image or args.input_image).resolve(),
        output_dir=output_dir,
        provider=args.model_provider,
        profile=args.model_profile,
        model=args.trellis_model if args.model_provider == "trellis" else None,
        seed=args.seed,
        input_mask=input_mask,
        options=options,
    )
    command = provider.generation_command(request, PROJECT_ROOT)
    # The provider command uses the provider's selected runtime.  TRELLIS's
    # existing generation script remains the implementation for that provider.
    _run_stage("model_generation", command, logs_dir / "model_generation.log", env, manifest)
    provider_manifest_path = output_dir / "provider_manifest.json"
    provider_manifest: dict[str, Any] = {}
    if provider_manifest_path.is_file():
        provider_manifest = _read_json(provider_manifest_path)
    if args.model_provider == "hunyuan":
        source_mesh = Path(str(provider_manifest.get("sourceMesh", output_dir / "source_mesh.glb")))
        if not source_mesh.is_file():
            raise FileNotFoundError(f"Hunyuan provider 没有生成 source mesh: {source_mesh}")
        sample_ply = output_dir / "sample.ply"
        converter = TOOLS_ROOT / "model_center" / "converters" / "mesh_to_gaussian.py"
        converter_command = [
            str(args.trellis_python),
            str(converter),
            "--input-mesh", str(source_mesh),
            "--output-ply", str(sample_ply),
            "--metadata", str(output_dir / "mesh_to_gaussian.json"),
            "--density", str(args.mesh_to_gaussian_density),
            "--thickness", str(args.mesh_to_gaussian_thickness),
            "--max-points", str(args.mesh_to_gaussian_max_points),
            "--seed", str(args.seed),
        ]
        _run_stage("mesh_to_gaussian", converter_command, logs_dir / "mesh_to_gaussian.log", env, manifest)
        manifest["representation"] = "surface_splats"
        manifest["source_mesh"] = str(source_mesh.resolve())
    else:
        sample_ply = Path(str(provider_manifest.get("samplePly", output_dir / "sample.ply")))
        manifest["representation"] = "native_gaussian"
    if not sample_ply.is_file():
        raise FileNotFoundError(f"provider 没有生成标准 sample.ply: {sample_ply}")
    contract_value = options.get("coordinate_contract") or provider_manifest.get("coordinateContract")
    if isinstance(contract_value, dict):
        try:
            contract = CoordinateContract.from_dict(contract_value)
        except ValueError as exc:
            raise ValueError(f"provider coordinate contract 无效: {exc}") from exc
    else:
        contract = provider.coordinate_contract(request)
    # Every generated provider crosses the same explicit Gaussian boundary.
    # Even an identity contract is materialized and hashed so a later provider
    # profile cannot accidentally skip coordinate provenance.
    transformed = sample_ply.with_name(sample_ply.stem + ".canonical.ply")
    transform_metadata = output_dir / "coordinate_transform.json"
    transform_result = transform_gaussian_ply(sample_ply, transformed, contract, transform_metadata)
    transformed.replace(sample_ply)
    manifest["coordinate_contract"] = contract.to_dict()
    manifest["coordinate_transform"] = transform_result
    manifest["provider"] = args.model_provider
    manifest["provider_profile"] = args.model_profile
    manifest["provider_manifest"] = str(provider_manifest_path.resolve()) if provider_manifest_path.is_file() else None
    manifest["coordinate_contract_status"] = provider_manifest.get(
        "coordinateContractStatus", "declared_not_provider_verified"
    )
    weight_descriptor = _provider_weight_descriptor(args)
    manifest["weight_fingerprint"] = provider_manifest.get("weightFingerprint") or weight_descriptor.get("fingerprint")
    manifest["weight_revision"] = provider_manifest.get("weightRevision") or weight_descriptor.get("revision")
    manifest["weight_identifier"] = provider_manifest.get("weightIdentifier") or weight_descriptor
    return sample_ply.resolve()


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
    contract_value = manifest.get("coordinate_contract", {})
    render_defaults = contract_value.get("renderDefaults", {}) if isinstance(contract_value, dict) else {}
    near = args.render_near if args.render_near is not None else float(render_defaults.get("near", 0.8))
    far = args.render_far if args.render_far is not None else float(render_defaults.get("far", 1.6))
    if near <= 0 or far <= near:
        raise ValueError(f"provider render near/far 无效: near={near}, far={far}")
    manifest["render_config"]["effective_near"] = near
    manifest["render_config"]["effective_far"] = far
    if args.render_mode == "anchor":
        render_request = RenderRequest(
            input_ply=sample_ply,
            output_dir=render_dir,
            mode="anchor",
            resolution=args.render_resolution,
            fov_degrees=args.render_fov,
            distance=args.render_distance,
            near=near,
            far=far,
            yaw_degrees=args.render_yaw_degrees,
            pitch_degrees=args.render_pitch_degrees,
            side_angle_degrees=args.render_side_angle_degrees,
            view_names=args.render_view_names,
        )
        command = build_render_command(render_request, PROJECT_ROOT, args.trellis_python)
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
        render_request = RenderRequest(
            input_ply=sample_ply,
            output_dir=render_dir,
            mode="sphere",
            resolution=args.render_resolution,
            fov_degrees=args.render_fov,
            radius=args.render_radius,
            near=near,
            far=far,
            latitudes=args.render_latitudes,
            views_per_latitude=args.render_views_per_latitude,
        )
        command = build_render_command(render_request, PROJECT_ROOT, args.trellis_python)
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


def _sags_view_specs(args: argparse.Namespace) -> list[tuple[str, float]]:
    names = [item.strip() for item in str(args.sags_view_names).split(",") if item.strip()]
    try:
        offsets = [float(item.strip()) for item in str(args.sags_yaw_offsets).split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("sags-yaw-offsets 必须是逗号分隔的数字") from exc
    if not names or len(names) != len(offsets):
        raise ValueError("sags-view-names 与 sags-yaw-offsets 数量必须相同且不能为空")
    if len(set(names)) != len(names):
        raise ValueError("sags-view-names 不能重复")
    if any(Path(name).name != name or name in {".", ".."} for name in names):
        raise ValueError("sags-view-names 只能包含安全文件名")
    return list(zip(names, offsets))


def _sags_source_view_name(args: argparse.Namespace, specs: list[tuple[str, float]]) -> str:
    """Choose the annotation copied to the legacy center slot and SAGS source view."""

    requested = str(args.sags_view_name or "").strip()
    names = {name for name, _ in specs}
    return requested if requested in names else specs[0][0]


def _render_sags_views(
    args: argparse.Namespace,
    sample_ply: Path,
    output_dir: Path,
    logs_dir: Path,
    env: dict[str, str],
    manifest: dict[str, Any],
) -> list[tuple[str, float]]:
    """Render SAGS views in TRELLIS canonical space, independent of Unity pose."""

    specs = _sags_view_specs(args)
    contract_value = manifest.get("coordinate_contract", {})
    render_defaults = contract_value.get("renderDefaults", {}) if isinstance(contract_value, dict) else {}
    near = args.render_near if args.render_near is not None else float(render_defaults.get("near", 0.8))
    far = args.render_far if args.render_far is not None else float(render_defaults.get("far", 1.6))
    command = [
        str(args.trellis_python), str(TOOLS_ROOT / "render_trellis_views.py"),
        "--input-ply", str(sample_ply), "--output-dir", str(output_dir),
        "--resolution", str(args.render_resolution), "--fov", str(args.render_fov),
        "--yaw-degrees", str(args.render_yaw_degrees), "--pitch-degrees", str(args.render_pitch_degrees),
        "--distance", str(args.render_distance), "--near", str(near), "--far", str(far),
        "--yaw-offsets", ",".join(str(offset) for _, offset in specs),
        "--view-names", ",".join(name for name, _ in specs),
    ]
    _run_stage("sags_render", command, logs_dir / "sags_render.log", env, manifest)
    manifest["stages"]["sags_render"].update(
        {
            "cameraMode": "canonical_ring6",
            "coordinateSpace": "trellis_canonical",
            "poseSource": "none",
            "yawOffsetsDegrees": [offset for _, offset in specs],
            "pitchDegrees": args.render_pitch_degrees,
            "distance": args.render_distance,
            "fovDegrees": args.render_fov,
        }
    )
    _json_dump(Path(manifest["manifest_path"]), manifest)
    image_dir = output_dir / "source" / "images"
    missing = [name for name, _ in specs if not (image_dir / f"{name}.png").is_file()]
    if missing or not (output_dir / "model").is_dir():
        raise RuntimeError(f"SAGS 六视角渲染输出不完整: {missing}")
    return specs


def _run_sags_view_annotations(
    args: argparse.Namespace,
    specs: list[tuple[str, float]],
    render_dir: Path,
    annotations_dir: Path,
    logs_dir: Path,
    env: dict[str, str],
    manifest: dict[str, Any],
) -> dict[str, dict[str, Path]]:
    """Run the existing auto segmenter independently for each SAGS image."""

    if not args.prompt and not args.task_prompt:
        raise ValueError("六视角 SAGS 自动标注需要 --prompt 或 --task-prompt")
    annotations: dict[str, dict[str, Path]] = {}
    for name, _ in specs:
        image = render_dir / "source" / "images" / f"{name}.png"
        output = annotations_dir / name
        command = [
            str(args.trellis_python), str(TOOLS_ROOT / "auto_segment.py"),
            "--input", str(image), "--output-dir", str(output),
            "--engine", args.seg_engine, "--points-per-mask", str(args.sags_points_per_mask),
        ]
        command += ["--prompt", args.prompt] if args.prompt else ["--task-prompt", args.task_prompt]
        _run_stage(
            f"sags_segmentation_{name}", command,
            logs_dir / f"sags_segmentation_{name}.log", env, manifest,
        )
        mask = output / "mask.png"
        points = output / "points.json"
        if not mask.is_file() or not points.is_file():
            raise RuntimeError(f"SAGS 视角 {name} 标注输出不完整: {mask}, {points}")
        annotations[name] = {"mask": mask, "points": points}
    return annotations


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


def _run_anchor_segmentation(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    output_dir: Path,
    logs_dir: Path,
    env: dict[str, str],
    manifest: dict[str, Any],
    requested_names: set[str] | None,
) -> Path | None:
    if args.skip_anchor_masking:
        manifest["stages"]["anchor_segmentation"] = {
            "status": "skipped",
            "reason": "--skip-anchor-masking",
        }
        _json_dump(Path(manifest["manifest_path"]), manifest)
        return None
    prompt = _infer_anchor_mask_prompt(args)
    if not prompt:
        raise ValueError("最终位姿需要 --anchor-mask-prompt（简短英文锚点类别）")
    args.anchor_mask_prompt = prompt
    manifest["anchor_mask_prompt"] = prompt
    if isinstance(manifest.get("pose_config"), dict):
        manifest["pose_config"]["anchor_mask_prompt"] = prompt
    selected = [
        record for record in records
        if requested_names is None or record["image1"].stem.lower() in requested_names
    ]
    command = [
        str(args.trellis_python), str(TOOLS_ROOT / "segment_anchor_views.py"),
        "--prompt", prompt,
        "--output-dir", str(output_dir),
        "--box-threshold", str(args.anchor_mask_box_threshold),
    ]
    for record in selected:
        command += [
            "--view", record["image1"].stem.lower(),
            str(record["image0"]), str(record["image1"]),
        ]
    _run_stage(
        "anchor_segmentation", command,
        logs_dir / "anchor_segmentation.log", env, manifest,
    )
    manifest["anchor_masks"] = str(output_dir.resolve())
    _json_dump(Path(manifest["manifest_path"]), manifest)
    return output_dir


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
    anchor_masks_dir: Path | None = None,
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
        "--max-matches-per-view", str(args.pose_max_matches_per_view),
        "--max-depth-relative-spread", str(args.pose_max_depth_relative_spread),
        "--spatial-grid-size", str(args.pose_spatial_grid_size),
        "--anchor-mask-dilation", str(args.anchor_mask_dilation),
        "--cross-view-neighbors", str(args.pose_cross_view_neighbors),
        "--cross-view-min-support", str(args.pose_cross_view_min_support),
        "--cross-view-fallback-support", str(args.pose_cross_view_fallback_support),
        "--min-consistent-points", str(args.pose_min_consistent_points),
        "--min-consistent-view-points", str(args.pose_min_consistent_view_points),
        "--seed", str(args.seed), "--run-id", args.run_id,
        "--candidate-id", args.candidate_id, "--exit-zero-on-rejected",
    ]
    if no_quality_gate:
        command += ["--no-quality-gate", "--allow-single-view"]
    if anchor_masks_dir is not None:
        command += ["--anchor-masks-dir", str(anchor_masks_dir)]
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
        "03_sags_views/views.json", "03_sags_views/source/sparse/0/*.txt",
        "03_sags_views/source/images/*.png", "03_sags_views/source/depths/absdepth/*.raw",
        "06_sags/annotations/**/*.png", "06_sags/annotations/**/*.json",
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


def _main_impl(args: argparse.Namespace) -> int:
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
    if args.model_provider == "hunyuan" and not args.hunyuan_python.is_file():
        raise SystemExit(f"Hunyuan Python 不存在: {args.hunyuan_python}")
    if not args.gim_python.is_file():
        raise SystemExit(f"GIM Python 不存在: {args.gim_python}")
    if args.run_sags and not args.sags_python.is_file():
        raise SystemExit(f"SAGS Python 不存在: {args.sags_python}")
    if (
        args.pose_ransac_threshold <= 0 or args.pose_ransac_iterations < 1
        or args.pose_min_inliers < 3 or args.pose_max_matches_per_view < 0
        or args.pose_max_depth_relative_spread < 0 or args.pose_spatial_grid_size < 1
        or args.gim_anchor_roi_radius <= 0 or args.gim_aligned_max_displacement < 0
        or args.anchor_mask_box_threshold <= 0 or args.anchor_mask_dilation < 0
        or args.pose_cross_view_neighbors < 1 or args.pose_cross_view_min_support < 1
        or args.pose_cross_view_fallback_support < 1
        or args.pose_min_consistent_points < 3 or args.pose_min_consistent_view_points < 1
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
    if args.run_sags and (
        args.sags_min_votes < 1
        or not 0 <= args.sags_threshold <= 1
        or not 0 <= args.sags_independent_min_prior_coverage <= 1
    ):
        raise SystemExit("SAGS threshold/min-votes/independent-prior-coverage 参数无效")
    if args.run_sags and args.sags_view_mode == "ring6":
        _sags_view_specs(args)
    if args.skip_render and not args.skip_gim and not args.gim_pair:
        raise SystemExit("--skip-render 无法提供默认生成视图；请同时使用 --skip-gim 或显式 --gim-pair")
    if args.skip_render and pose_requested:
        # ``--input-ply`` callers and synthetic tests may intentionally reuse
        # a complete render bundle.  Keep the pose input contract strict while
        # allowing the expensive renderer stage to be skipped safely.
        render_source = args.output_dir / "03_rendered_3dgs" / "source"
        required_render_files = (
            render_source / "sparse" / "0" / "cameras.txt",
            render_source / "sparse" / "0" / "images.txt",
        )
        if not all(path.is_file() for path in required_render_files):
            missing = ", ".join(str(path) for path in required_render_files if not path.is_file())
            raise SystemExit(f"--skip-render 的自动 pose 需要复用完整 render bundle，缺少: {missing}")

    args.run_id = args.run_id or _new_run_id()
    args.candidate_id = args.candidate_id or _candidate_id(args)
    weight_descriptor = _provider_weight_descriptor(args)

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
        "provider": args.model_provider,
        "provider_profile": args.model_profile,
        "provider_profile_config": _provider_profile(args).to_dict(),
        "provider_options": _provider_options(args),
        "weights": weight_descriptor,
        "weight_fingerprint": weight_descriptor.get("fingerprint"),
        "weight_revision": weight_descriptor.get("revision"),
        "weight_identifier": weight_descriptor,
        "trellis_input": args.trellis_input,
        "trellis_mask_prompts": args.trellis_mask_prompts,
        "anchor_mask_prompt": args.anchor_mask_prompt,
        "task_prompt": args.task_prompt,
        "object_prompt": args.prompt,
        "run_sags": args.run_sags,
        "unity_manifest": str(args.unity_manifest.resolve()) if args.unity_manifest else None,
        "segmentation_config": {
            "requested_engine": args.seg_engine,
            "prompt": args.prompt,
            "task_prompt": args.task_prompt,
        },
        "model_input_mask": str(args.model_input_mask.resolve()) if args.model_input_mask else None,
        "sags_config": {
            "view_name": args.sags_view_name,
            "view_mode": args.sags_view_mode,
            "yaw_offsets": args.sags_yaw_offsets,
            "view_names": args.sags_view_names,
            "points_per_mask": args.sags_points_per_mask,
            "force_seed_radius": args.sags_force_seed_radius,
            "force_seed": not args.sags_no_force_seed,
            "points_json": str(args.sags_points_json.resolve()) if args.sags_points_json else None,
            "mask": str(args.sags_mask.resolve()) if args.sags_mask else None,
            "mask_id": args.sags_mask_id,
            "threshold": args.sags_threshold,
            "min_votes": args.sags_min_votes,
            "independent_min_prior_coverage": args.sags_independent_min_prior_coverage,
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
            "near": args.render_near,
            "far": args.render_far,
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
            "policy": "point_consistency_joint_fit",
            "generated_axis": args.pose_generated_axis,
            "ransac_threshold": args.pose_ransac_threshold,
            "ransac_iterations": args.pose_ransac_iterations,
            "min_inliers": args.pose_min_inliers,
            "max_matches_per_view": args.pose_max_matches_per_view,
            "max_depth_relative_spread": args.pose_max_depth_relative_spread,
            "min_view_inliers": args.pose_min_view_inliers,
            "min_view_inlier_ratio": args.pose_min_view_inlier_ratio,
            "anchor_mask_prompt": args.anchor_mask_prompt,
            "anchor_mask_box_threshold": args.anchor_mask_box_threshold,
            "anchor_mask_dilation": args.anchor_mask_dilation,
            "cross_view_neighbors": args.pose_cross_view_neighbors,
            "cross_view_min_support": args.pose_cross_view_min_support,
            "cross_view_fallback_support": args.pose_cross_view_fallback_support,
            "min_consistent_points": args.pose_min_consistent_points,
            "min_consistent_view_points": args.pose_min_consistent_view_points,
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

    if not args.input_ply:
        provider_report = provider_environment_report(_provider_instance(args))
        report_value = provider_report.to_dict()
        manifest["provider_environment"] = report_value
        if not provider_report.available:
            manifest["status"] = "blocked"
            manifest["stages"]["provider_preflight"] = {
                "status": "blocked",
                "missing": list(provider_report.missing),
                "blocked_reasons": list(provider_report.blocked_reasons),
            }
            _json_dump(Path(manifest["manifest_path"]), manifest)
            reasons = list(provider_report.missing) + list(provider_report.blocked_reasons)
            raise SystemExit(f"provider {args.model_provider} 环境不可用: {'; '.join(reasons)}")
        manifest["stages"]["provider_preflight"] = {"status": "ready", "report": report_value}
        _json_dump(Path(manifest["manifest_path"]), manifest)

    # An explicit union mask keeps TRELLIS focused on the anchor + inserted
    # object and avoids rembg selecting a small overlapping fragment.
    composite_input = args.input_image
    trellis_mask_dir = args.output_dir / "00_trellis_input"
    if args.model_provider == "trellis" and args.trellis_mask_prompts and not args.input_ply:
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
            "reason": "provider does not use TRELLIS composite input, no mask prompts, or --input-ply",
        }
        _json_dump(Path(manifest["manifest_path"]), manifest)

    # Downstream segmentation is deliberately deferred until the generated
    # center view exists, so SAGS points use the generated image coordinates.
    segmentation_dir = args.output_dir / "01_segmentation"
    cutout = None
    trellis_input_path = composite_input if args.trellis_input == "composite" else None
    if args.model_provider != "trellis":
        # SAM3D/Hunyuan consume the managed pre-generation object mask and
        # produce a single object, so the post-render SAGS extraction stage is
        # not needed.  Keep the stage explicit for diagnostics.
        segmentation_deferred = False
        cutout = args.input_image
        manifest["stages"]["segmentation"] = {
            "status": "skipped",
            "reason": f"single-object provider {args.model_provider} uses model_input_mask",
        }
        _json_dump(Path(manifest["manifest_path"]), manifest)
    elif args.trellis_input == "cutout" and (args.skip_segmentation or not (args.prompt or args.task_prompt)):
        cutout = args.input_image
    segmentation_deferred = (
        args.model_provider == "trellis"
        and args.trellis_input == "composite"
        and not args.skip_segmentation
        and bool(args.prompt or args.task_prompt)
    )
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

    # Stage 2: image -> Gaussian/mesh.  The output directory remains named
    # ``02_trellis`` for Unity/debug compatibility; its manifest identifies the
    # actual provider and representation.
    trellis_dir = args.output_dir / "02_trellis"
    if args.input_ply:
        sample_ply = args.input_ply.resolve()
        manifest["stages"]["model_generation"] = {"status": "skipped", "reason": "--input-ply", "sample_ply": str(sample_ply)}
        provider = _provider_instance(args)
        manifest["coordinate_contract"] = provider.coordinate_contract().to_dict()
        manifest["representation"] = "external_gaussian"
        _json_dump(Path(manifest["manifest_path"]), manifest)
    elif args.skip_trellis:
        raise SystemExit("--skip-trellis 需要同时提供 --input-ply")
    else:
        if args.model_provider == "trellis":
            generation_input = composite_input if args.trellis_input == "composite" else cutout
            if generation_input is None or not generation_input.is_file():
                raise SystemExit(f"TRELLIS 输入图片不存在: {generation_input}")
        else:
            generation_input = args.input_image
        trellis_input_path = generation_input
        sample_ply = trellis_dir / "sample.ply"
        try:
            sample_ply = _run_provider_generation(
                args, trellis_dir, env, manifest, logs_dir,
                input_image=generation_input,
            )
        except RuntimeError:
            if not args.continue_on_error:
                raise
        if args.model_provider == "trellis":
            # Keep the historical stage key for consumers that inspect it.
            manifest["stages"]["trellis"] = manifest["stages"].get("model_generation", {})
        _json_dump(Path(manifest["manifest_path"]), manifest)

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
    if args.skip_render and not args.gim_pair:
        # A pose request with skip-render reuses the named images/depths from
        # the existing bundle; non-pose input-Ply checks can still avoid
        # probing for generated images altogether.
        pair_records = _pair_records(args, render_dir) if pose_requested else []
    else:
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
        anchor_masks_dir = _run_anchor_segmentation(
            args, pair_records, args.output_dir / "04_gim" / "anchor_masks",
            logs_dir, env, manifest, requested_names,
        )
        pose_output = args.output_dir / "05_pose" / "pose.json"
        pose_value = _run_pose_fit(
            args, pair_records, render_dir, pose_output, args.output_dir / "04_gim", "pose",
            logs_dir, env, manifest, requested_names, anchor_masks_dir=anchor_masks_dir,
        )
        manifest["pose"] = str(pose_output)
    else:
        manifest["stages"].setdefault("pose", {"status": "skipped", "reason": "no Unity depth/camera metadata or --skip-pose"})

    sags_render_dir: Path | None = None
    sags_annotations: dict[str, dict[str, Path]] = {}
    use_independent_sags = bool(
        args.run_sags and args.model_provider == "trellis" and args.sags_view_mode == "ring6"
        and not args.sags_points_json and not args.sags_mask
    )
    if use_independent_sags:
        sags_render_dir = args.output_dir / "03_sags_views"
        if args.skip_render:
            specs = _sags_view_specs(args)
            image_dir = sags_render_dir / "source" / "images"
            missing = [name for name, _ in specs if not (image_dir / f"{name}.png").is_file()]
            if missing or not (sags_render_dir / "model").is_dir():
                raise SystemExit(f"--skip-render 需要已有完整 SAGS 六视角目录: {sags_render_dir}; 缺少 {missing}")
            manifest["stages"]["sags_render"] = {"status": "skipped", "reason": "--skip-render", "output": str(sags_render_dir)}
        else:
            if not sample_ply.is_file():
                raise SystemExit(f"SAGS 六视角渲染找不到 sample.ply: {sample_ply}")
            specs = _render_sags_views(args, sample_ply, sags_render_dir, logs_dir, env, manifest)
        sags_annotations = _run_sags_view_annotations(
            args, specs, sags_render_dir, args.output_dir / "06_sags" / "annotations",
            logs_dir, env, manifest,
        )
        manifest["sags_render_dir"] = str(sags_render_dir.resolve())
        manifest["sags_annotations"] = {
            name: {key: str(path.resolve()) for key, path in value.items()}
            for name, value in sags_annotations.items()
        }
        manifest["sags_effective_mode"] = "ring6_independent"
        _json_dump(Path(manifest["manifest_path"]), manifest)
    elif args.run_sags and args.sags_view_mode == "ring6" and (args.sags_points_json or args.sags_mask):
        manifest["sags_effective_mode"] = "legacy_external_annotation"
    elif args.run_sags:
        manifest["sags_effective_mode"] = "legacy"

    # Segment the final aligned center render so mask.png, points.json and the
    # SAGS camera set have exactly the same pixels and camera metadata.
    if segmentation_deferred:
        if use_independent_sags:
            # center has exactly the same canonical camera as the generated
            # center view. Reuse its annotation so the new default does not
            # invoke the detector a seventh time.
            source_name = _sags_source_view_name(args, _sags_view_specs(args))
            source_annotation = sags_annotations[source_name]
            source_dir = source_annotation["mask"].parent
            segmentation_dir.mkdir(parents=True, exist_ok=True)
            for filename in ("mask.png", "cutout.png", "annotated.png", "detections.json", "points.json", "manifest.json"):
                source = source_dir / filename
                if source.is_file():
                    shutil.copy2(source, segmentation_dir / filename)
            manifest["stages"]["segmentation"] = {
                "status": "ready",
                "mode": "reused_sags_annotation",
                "source": str(source_dir.resolve()),
            }
            cutout = segmentation_dir / "cutout.png"
            _json_dump(Path(manifest["manifest_path"]), manifest)
        else:
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
        sags_output = args.sags_output_ply or args.output_dir / "06_sags" / "inserted_object.ply"
        if args.model_provider != "trellis":
            # Single-object providers already return the object Gaussian.  A
            # direct materialization keeps the Unity import contract while
            # avoiding a second semantic segmentation pass over its render.
            sags_output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sample_ply, sags_output)
            manifest["stages"]["sags"] = {
                "status": "ready",
                "mode": "provider_output",
                "provider": args.model_provider,
                "source": str(sample_ply.resolve()),
            }
            manifest["sags_effective_mode"] = "provider_output"
        else:
            points_json = args.sags_points_json or segmentation_dir / "points.json"
            mask_path = args.sags_mask or segmentation_dir / "mask.png"
            if not use_independent_sags and (not points_json.is_file() or not mask_path.is_file()):
                raise SystemExit(f"SAGS 完整标注不存在: {points_json}, {mask_path}")
            model_dir = sags_render_dir / "model" if use_independent_sags and sags_render_dir else render_dir / "model"
            if not model_dir.is_dir():
                raise SystemExit(f"SAGS model 目录不存在: {model_dir}")
            source_view_name = (
                _sags_source_view_name(args, _sags_view_specs(args))
                if use_independent_sags else args.sags_view_name
            )
            command = [
                str(args.sags_python), str(TOOLS_ROOT / "run_sags_text.py"),
                "--model-dir", str(model_dir), "--output-ply", str(sags_output),
                "--view-name", source_view_name,
                "--mask-id", str(args.sags_mask_id), "--threshold", str(args.sags_threshold),
                "--min-votes", str(args.sags_min_votes),
                "--independent-min-prior-coverage", str(args.sags_independent_min_prior_coverage),
                "--visibility-depth-tolerance", str(args.sags_visibility_depth_tolerance),
                "--gd-interval", str(args.sags_gd_interval),
                "--force-seed-radius", str(args.sags_force_seed_radius),
                "--diagnostics-dir", str(args.output_dir / "06_sags" / "diagnostics"),
            ]
            if use_independent_sags:
                command += ["--annotation-mode", "independent"]
                for name, _ in _sags_view_specs(args):
                    annotation = sags_annotations[name]
                    command += ["--view-annotation", name, str(annotation["mask"]), str(annotation["points"])]
            else:
                command += ["--points-json", str(points_json), "--mask", str(mask_path)]
            if args.sags_no_force_seed:
                command.append("--no-force-seed")
            _run_stage("sags", command, logs_dir / "sags.log", env, manifest)
        manifest["sags_ply"] = str(sags_output) if sags_output.is_file() else None
    else:
        manifest["stages"].setdefault("sags", {"status": "skipped", "reason": "--run-sags not set"})

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


def _write_fatal_manifest(args: argparse.Namespace, exception: BaseException) -> None:
    """Persist a stage-level failure even when setup/provider code raises early."""

    output_dir = args.output_dir
    if output_dir is None and args.run_root and args.task_id:
        output_dir = args.run_root / str(args.task_id)
    if output_dir is None:
        return
    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    try:
        manifest = _read_json(manifest_path) if manifest_path.is_file() else {
            "schemaVersion": 2,
            "manifest_path": str(manifest_path),
            "output_dir": str(output_dir.resolve()),
            "provider": getattr(args, "model_provider", None),
            "stages": {},
        }
        stages = manifest.setdefault("stages", {})
        active = next(
            (name for name, value in stages.items() if isinstance(value, dict) and value.get("status") == "running"),
            None,
        )
        if active is None:
            # Validation/provider checks can fail after a stage has already
            # recorded ``blocked`` or before any stage was created.  Keep the
            # most recent named stage so the batch/UI can identify the point
            # of failure instead of collapsing everything into "pipeline".
            active = next(
                (name for name, value in reversed(list(stages.items()))
                 if isinstance(value, dict) and value.get("status") in {"blocked", "running"}),
                "pipeline",
            )
        error = f"{type(exception).__name__}: {exception}"
        stages.setdefault(active, {})
        stages[active].update({
            "status": "failed",
            "error": error,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        manifest["status"] = "failed"
        manifest["failed_stages"] = [name for name, value in stages.items()
                                     if isinstance(value, dict) and value.get("status") == "failed"]
        manifest["rejected_stages"] = [name for name, value in stages.items()
                                       if isinstance(value, dict) and value.get("status") == "rejected"]
        manifest["fatal_error"] = error
        manifest["updatedAtUtc"] = datetime.now(timezone.utc).isoformat()
        output_dir.mkdir(parents=True, exist_ok=True)
        _json_dump(manifest_path, manifest)
    except Exception as write_error:
        print(f"[pipeline] unable to persist fatal manifest: {write_error}", file=sys.stderr, flush=True)


def main() -> int:
    args = parse_args()
    try:
        return _main_impl(args)
    except SystemExit as exception:
        # ``argparse`` errors happen before this function, but validation and
        # provider preflight errors happen inside _main_impl.  Persist those
        # as a task manifest before preserving the original exit code.
        _write_fatal_manifest(args, exception)
        raise
    except Exception as exception:
        _write_fatal_manifest(args, exception)
        print(f"INSERT_PIPELINE_FAILED pipeline {args.output_dir or args.run_root}", file=sys.stderr, flush=True)
        print(f"[pipeline] {type(exception).__name__}: {exception}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit("已取消")
