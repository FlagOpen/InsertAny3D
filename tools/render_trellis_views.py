#!/usr/bin/env python3
"""Render a TRELLIS Gaussian from explicit orbit cameras.

Without ``--unity-camera`` this is a canonical-space orbit renderer: every
camera looks at the canonical origin and uses the requested yaw, pitch,
distance, and FOV.  The optional Unity-camera path is reserved for pose
refinement and is kept separate from canonical SAGS ring views.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np


os.environ.setdefault("SPCONV_ALGO", "native")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRELLIS_ROOT = PROJECT_ROOT / "third_party" / "TRELLIS"
sys.path.insert(0, str(TRELLIS_ROOT))

from trellis.utils.insertany3d_render_utils import (  # noqa: E402
    _extract_3dgs_result,
    _save_result,
    load_gaussian,
)
from trellis.utils.render_utils import (  # noqa: E402
    render_frames,
    yaw_pitch_r_fov_to_extrinsics_intrinsics,
)


def _float_list(value: str, name: str) -> list[float]:
    try:
        values = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} 必须是逗号分隔的数字") from exc
    if not values:
        raise argparse.ArgumentTypeError(f"{name} 不能为空")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按显式 yaw/pitch/distance 渲染 TRELLIS Gaussian")
    parser.add_argument("--input-ply", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--yaw-degrees", type=float, default=0.0, help="中心 yaw，单位度")
    parser.add_argument("--pitch-degrees", type=float, default=12.0, help="俯仰角，单位度")
    parser.add_argument("--distance", type=float, default=1.5, help="生成物体归一化坐标中的相机距离")
    parser.add_argument("--side-angle-degrees", type=float, default=24.0, help="左右视角相对中心的偏移")
    parser.add_argument("--yaw-offsets", default=None, help="相对中心 yaw 偏移列表，例如 -24,0,24；省略时由 side-angle 生成")
    parser.add_argument("--view-names", default="left,center,right", help="与 yaw-offsets 对应的文件名列表")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--fov", type=float, default=53.1301023542)
    parser.add_argument("--near", type=float, default=0.8)
    parser.add_argument("--far", type=float, default=1.6)
    parser.add_argument("--ssaa", type=int, default=1)
    parser.add_argument("--unity-camera", action="append", type=Path, help="Unity image.camera.json；提供后按真实外参渲染，可重复")
    parser.add_argument("--coarse-pose", type=Path, help="generated_world_to_unity_world 粗位姿；显式 Unity 相机模式必填")
    parser.add_argument("--unity-manifest", type=Path, help="包含 anchorPosition 的 task_manifest.json，用于重投影校验")
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层必须是对象: {path}")
    return value


def _xyz(value: dict, name: str) -> np.ndarray:
    try:
        result = np.array([float(value[axis]) for axis in "xyz"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} 缺少 xyz") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} 包含无效数值")
    return result


def _quaternion_matrix(value: dict) -> np.ndarray:
    try:
        x, y, z, w = (float(value[key]) for key in ("x", "y", "z", "w"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("rotationXyzw 格式错误") from exc
    quaternion = np.array([x, y, z, w], dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if norm < 1e-12:
        raise ValueError("rotationXyzw 是零四元数")
    x, y, z, w = quaternion / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _pose_transform(path: Path) -> tuple[np.ndarray, np.ndarray, float, dict]:
    value = _read_json(path)
    if value.get("status") not in (None, "ready"):
        raise ValueError(f"粗位姿未通过质量门禁: {value.get('status')}")
    matrix = np.asarray(value.get("matrix4x4"), dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError("粗位姿缺少 matrix4x4")
    scale = float(value.get("uniformScale", np.cbrt(abs(np.linalg.det(matrix[:3, :3])))))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("粗位姿 scale 无效")
    rotation = matrix[:3, :3] / scale
    u, _, vt = np.linalg.svd(rotation)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        raise ValueError("粗位姿包含镜像，无法转换相机")
    return rotation, matrix[:3, 3].copy(), scale, value


def _axis_matrix(name: str) -> np.ndarray:
    if name == "identity":
        return np.eye(3, dtype=np.float64)
    if name == "legacy-flip-z":
        return np.diag([1.0, 1.0, -1.0])
    raise ValueError(f"不支持的 generatedAxis: {name!r}")


def _convert_unity_camera(
    camera: dict,
    rotation_generated_to_unity: np.ndarray,
    translation_generated_to_unity: np.ndarray,
    scale: float,
    generated_axis: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rotation_unity_to_generated = rotation_generated_to_unity.T
    axis = _axis_matrix(generated_axis)
    pose = camera["cameraToWorld"]
    center_unity = _xyz(pose["position"], "cameraToWorld.position")
    rotation_unity_camera = _quaternion_matrix(pose["rotationXyzw"])
    center_generated = axis @ rotation_unity_to_generated @ (center_unity - translation_generated_to_unity) / scale
    # Unity camera local coordinates are x-right/y-up/z-forward; COLMAP and
    # the TRELLIS renderer use x-right/y-down/z-forward.  The generated-axis
    # reflection converts Unity's left-handed world contract to the proper
    # right-handed COLMAP rotation required by the renderer.
    unity_to_cv = np.diag([1.0, -1.0, 1.0])
    rotation_w2c = unity_to_cv @ rotation_unity_camera.T @ rotation_generated_to_unity @ axis
    if np.linalg.det(rotation_w2c) < 0.999:
        raise ValueError(
            "Unity 外参转换仍是反射矩阵；相机精化需要 generatedAxis=legacy-flip-z"
        )
    extrinsic = np.eye(4, dtype=np.float32)
    extrinsic[:3, :3] = rotation_w2c.astype(np.float32)
    extrinsic[:3, 3] = (-rotation_w2c @ center_generated).astype(np.float32)
    width, height = int(camera["width"]), int(camera["height"])
    intr = camera["intrinsics"]
    intrinsic = np.array(
        [
            [float(intr["fx"]) / width, 0.0, float(intr["cx"]) / width],
            [0.0, float(intr["fy"]) / height, float(intr["cy"]) / height],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return extrinsic, intrinsic, center_generated


def _project_unity(camera: dict, point: np.ndarray) -> np.ndarray:
    pose = camera["cameraToWorld"]
    rotation = _quaternion_matrix(pose["rotationXyzw"])
    local = rotation.T @ (point - _xyz(pose["position"], "cameraToWorld.position"))
    if local[2] <= 0:
        raise ValueError("锚点位于 Unity 相机后方")
    intr = camera["intrinsics"]
    return np.array(
        [
            float(intr["fx"]) * local[0] / local[2] + float(intr["cx"]) - 0.5,
            float(intr["cy"]) - float(intr["fy"]) * local[1] / local[2] - 0.5,
        ]
    )


def _explicit_cameras(args: argparse.Namespace, names: list[str]):
    import torch

    if args.coarse_pose is None or not args.coarse_pose.is_file():
        raise ValueError("--unity-camera 模式必须提供存在的 --coarse-pose")
    if len(args.unity_camera) != len(names):
        raise ValueError("--unity-camera 与 --view-names 数量必须相同")
    rotation_ug, translation_ug, scale, pose_value = _pose_transform(args.coarse_pose)
    rotation_gu = rotation_ug.T
    generated_axis = str(pose_value.get("coordinateContract", {}).get("generatedAxis", "identity"))
    axis = _axis_matrix(generated_axis)
    extrinsics = []
    intrinsics = []
    records = []
    camera_distances = []
    anchor_unity = None
    anchor_generated = None
    if args.unity_manifest:
        manifest = _read_json(args.unity_manifest)
        anchor_unity = _xyz(manifest["anchorPosition"], "anchorPosition")
        anchor_generated = axis @ rotation_gu @ (anchor_unity - translation_ug) / scale
    for name, path in zip(names, args.unity_camera):
        camera = _read_json(path)
        width, height = int(camera["width"]), int(camera["height"])
        if width != height or width != args.resolution:
            raise ValueError(f"TRELLIS 显式相机目前要求方形且等于 resolution: {path} 是 {width}x{height}")
        extrinsic, intrinsic, center_generated = _convert_unity_camera(
            camera, rotation_ug, translation_ug, scale, generated_axis
        )
        rotation_w2c = extrinsic[:3, :3].astype(np.float64)
        translation_w2c = extrinsic[:3, 3].astype(np.float64)
        intr = camera["intrinsics"]
        reprojection_error = None
        if anchor_unity is not None and anchor_generated is not None:
            source_pixel = _project_unity(camera, anchor_unity)
            camera_point = rotation_w2c @ anchor_generated + translation_w2c
            generated_pixel = np.array(
                [
                    float(intr["fx"]) * camera_point[0] / camera_point[2] + float(intr["cx"]) - 0.5,
                    float(intr["fy"]) * camera_point[1] / camera_point[2] + float(intr["cy"]) - 0.5,
                ]
            )
            reprojection_error = float(np.linalg.norm(source_pixel - generated_pixel))
            if reprojection_error > 1e-3:
                raise ValueError(f"{name} 锚点相机转换重投影误差过大: {reprojection_error}")
            camera_distances.append(float(np.linalg.norm(center_generated - anchor_generated)))
        extrinsics.append(torch.from_numpy(extrinsic).cuda())
        intrinsics.append(torch.from_numpy(intrinsic).cuda())
        records.append(
            {
                "name": name,
                "unityCamera": str(path.resolve()),
                "generatedWorldToCamera": extrinsic.tolist(),
                "normalizedIntrinsics": intrinsic.tolist(),
                "anchorReprojectionErrorPixels": reprojection_error,
            }
        )
    metadata = {
        "mode": "unity_extrinsics_via_coarse_pose",
        "coarsePose": str(args.coarse_pose.resolve()),
        "coarsePoseStatus": pose_value.get("status"),
        "generatedToUnityScale": scale,
        "generatedAxis": generated_axis,
        "anchorUnity": anchor_unity.tolist() if anchor_unity is not None else None,
        "anchorGenerated": anchor_generated.tolist() if anchor_generated is not None else None,
        "cameraDistancesGenerated": camera_distances,
        "views": records,
    }
    return extrinsics, intrinsics, metadata


def main() -> int:
    args = parse_args()
    if not args.input_ply.is_file():
        raise SystemExit(f"输入 PLY 不存在: {args.input_ply}")
    if args.distance <= 0 or args.resolution < 1 or args.near <= 0 or args.far <= args.near:
        raise SystemExit("distance/resolution/near/far 参数无效")

    names = [item.strip() for item in args.view_names.split(",") if item.strip()]
    offsets = _float_list(args.yaw_offsets, "--yaw-offsets") if args.yaw_offsets is not None else [-args.side_angle_degrees, 0.0, args.side_angle_degrees]
    if not args.unity_camera and len(offsets) != len(names):
        raise SystemExit("--yaw-offsets 与 --view-names 的数量必须相同")
    if len(set(names)) != len(names):
        raise SystemExit("--view-names 不能重复")
    if any(not name or Path(name).name != name for name in names):
        raise SystemExit("--view-names 只能包含文件名，不能包含目录分隔符")
    if args.side_angle_degrees < 0:
        raise SystemExit("--side-angle-degrees 不能为负数")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    import torch

    gaussian = load_gaussian(str(args.input_ply))
    if args.unity_camera:
        try:
            extrinsics, intrinsics, camera_metadata = _explicit_cameras(args, names)
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"显式 Unity 相机转换失败: {exc}") from exc
        distances = camera_metadata.get("cameraDistancesGenerated", [])
        effective_near = min(args.near, max(0.01, min(distances) * 0.2)) if distances else args.near
        effective_far = max(args.far, max(distances) * 3.0) if distances else args.far
    else:
        yaws = [math.radians(args.yaw_degrees + offset) for offset in offsets]
        pitches = [math.radians(args.pitch_degrees)] * len(yaws)
        extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(
            yaws, pitches, args.distance, args.fov
        )
        camera_metadata = {
            "mode": "canonical_yaw_pitch",
            "coordinateSpace": "trellis_canonical",
            "poseSource": "none",
            "target": "origin",
        }
        effective_near, effective_far = args.near, args.far
    rendered = render_frames(
        gaussian,
        extrinsics,
        intrinsics,
        options={
            "resolution": args.resolution,
            "near": effective_near,
            "far": effective_far,
            "ssaa": args.ssaa,
            "bg_color": (0, 0, 0),
        },
    )
    if not args.unity_camera:
        empty_views = [
            name for name, image in zip(names, rendered["color"])
            if not np.any(np.asarray(image) > 0)
        ]
        if empty_views:
            raise RuntimeError(
                "canonical 渲染出现全黑视角: "
                + ", ".join(empty_views)
                + "; 请增大 distance 或调整 fov"
            )
    image_names = [f"{name}.png" for name in names]
    scene_path = args.output_dir / "source"
    model_path = args.output_dir / "model"
    _save_result(
        str(scene_path),
        rendered["color"],
        rendered["depth"],
        rendered["absdepth"],
        rendered["extr"],
        rendered["intr"],
        args.resolution,
        image_names=image_names,
    )
    _extract_3dgs_result(str(scene_path), str(model_path), str(args.input_ply), use_colmap=False)

    views = []
    for index, name in enumerate(names):
        record = {
            "index": index,
            "name": name,
            "image": f"source/images/{name}.png",
        }
        if args.unity_camera:
            record.update(camera_metadata["views"][index])
        else:
            offset = offsets[index]
            record.update(
                {
                    "yaw_degrees": args.yaw_degrees + offset,
                    "yaw_offset_degrees": offset,
                    "pitch_degrees": args.pitch_degrees,
                    "distance": args.distance,
                    "fov_degrees": args.fov,
                }
            )
        views.append(record)
    (args.output_dir / "views.json").write_text(
        json.dumps(
            {
                "mode": camera_metadata["mode"],
                "input_ply": str(args.input_ply.resolve()),
                "views": views,
                "resolution": args.resolution,
                "near": effective_near,
                "far": effective_far,
                "cameraConversion": camera_metadata,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    image_count = len(list((scene_path / "images").glob("*.png")))
    if image_count != len(names):
        raise RuntimeError(f"渲染图片数量错误: {image_count}, 期望 {len(names)}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("TRELLIS_POSE_RENDER_READY", image_count, args.output_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
