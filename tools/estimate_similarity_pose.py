#!/usr/bin/env python3
"""Estimate a generated asset's Unity pose from GIM matches and depth maps.

Each --view supplies one scene/generated correspondence set.  The scene side
uses Unity radial depth plus camera-to-world metadata; the generated side uses
TRELLIS z-depth plus COLMAP camera files.  All valid 3D correspondences are fit
together with a robust similarity transform.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class PinholeCamera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    rotation: np.ndarray
    translation: np.ndarray


@dataclass
class ViewPoints:
    name: str
    source: np.ndarray
    target: np.ndarray
    confidence: np.ndarray
    input_matches: int
    geometric_inliers: int
    depth_valid: int
    files: dict[str, str]
    file_digests: dict[str, dict[str, Any]]
    geometric_scene_pixels: np.ndarray
    geometric_generated_pixels: np.ndarray
    match_indices: np.ndarray
    scene_pixels: np.ndarray
    generated_pixels: np.ndarray


@dataclass
class SimilarityFit:
    rotation: np.ndarray
    translation: np.ndarray
    scale: float
    inliers: np.ndarray
    residuals: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GIM 对应点 + 双侧深度反投影，求 generated world 到 Unity world 的相似变换"
    )
    parser.add_argument(
        "--view",
        action="append",
        nargs=4,
        required=True,
        metavar=("MATCHES_JSON", "UNITY_DEPTH", "UNITY_CAMERA_JSON", "GENERATED_DEPTH"),
        help="可重复；matches.json 中 image0 必须是 Unity 场景图，image1 必须是生成物渲染图",
    )
    parser.add_argument("--generated-cameras", required=True, type=Path, help="TRELLIS source/sparse/0/cameras.txt")
    parser.add_argument("--generated-images", required=True, type=Path, help="TRELLIS source/sparse/0/images.txt")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--generated-axis",
        choices=("identity", "legacy-flip-z"),
        default="identity",
        help="应用到反投影 generated world 点的坐标转换；identity 对应当前导出的 point_cloud.ply",
    )
    parser.add_argument("--max-matches-per-view", type=int, default=1000)
    parser.add_argument("--min-depth", type=float, default=1e-6)
    parser.add_argument("--max-depth-relative-spread", type=float, default=0.1, help="双线性采样四邻域允许的最大相对深度跨度；0 表示关闭")
    parser.add_argument("--ransac-threshold", type=float, default=0.1, help="Unity 世界单位")
    parser.add_argument("--ransac-iterations", type=int, default=2000)
    parser.add_argument("--min-inliers", type=int, default=6)
    parser.add_argument("--min-view-inliers", type=int, default=6, help="联合位姿中每个视角至少贡献的内点数")
    parser.add_argument("--min-view-inlier-ratio", type=float, default=0.01, help="联合位姿中每个视角的最小内点比例")
    parser.add_argument("--min-cross-view-inliers", type=int, default=3, help="留一视角验证的最小支持点数")
    parser.add_argument("--min-cross-view-ratio", type=float, default=0.005, help="留一视角验证的最小支持比例")
    parser.add_argument("--spatial-grid-size", type=int, default=8, help="按场景图网格均匀选择匹配点；1 表示关闭")
    parser.add_argument("--no-quality-gate", action="store_true", help="仍生成验证统计，但不因多视角不一致拒绝输出")
    parser.add_argument("--allow-single-view", action="store_true", help="允许单视角输出 ready；粗位姿阶段使用")
    parser.add_argument(
        "--primary-view-name",
        help="主视图验证模式：联合解仍要求每个输入视图支持，且主视图独立解必须得到至少一个其他视图佐证",
    )
    parser.add_argument("--exit-zero-on-rejected", action="store_true", help="被质量门禁拒绝时仍返回 0，便于批处理保留诊断")
    parser.add_argument("--diagnostics-dir", type=Path, help="写入 multiview_summary.json/png 和逐视角图")
    parser.add_argument("--run-id", help="写入结果的数据运行 ID")
    parser.add_argument("--candidate-id", help="写入结果的候选 ID")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--without-scale", action="store_true", help="只拟合刚体变换，固定 scale=1")
    parser.add_argument("--pixel-center-offset", type=float, default=0.5)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层必须是对象: {path}")
    return value


def _file_digest(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "sizeBytes": path.stat().st_size,
    }


def _balanced_indices(
    indices: np.ndarray,
    points: np.ndarray,
    confidence: np.ndarray,
    image_size: tuple[int, int],
    limit: int,
    grid_size: int,
) -> np.ndarray:
    if limit <= 0 or len(indices) <= limit:
        return indices
    if grid_size <= 1:
        order = np.argsort(confidence[indices])[::-1]
        return indices[order[:limit]]
    width, height = image_size
    buckets: dict[tuple[int, int], list[int]] = {}
    for index in indices:
        x, y = points[index]
        cell_x = min(grid_size - 1, max(0, int(float(x) * grid_size / max(1, width))))
        cell_y = min(grid_size - 1, max(0, int(float(y) * grid_size / max(1, height))))
        buckets.setdefault((cell_y, cell_x), []).append(int(index))
    ordered = [
        sorted(values, key=lambda item: (-float(confidence[item]), item))
        for _, values in sorted(buckets.items())
    ]
    selected: list[int] = []
    level = 0
    while len(selected) < limit:
        added = False
        for values in ordered:
            if level < len(values):
                selected.append(values[level])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        level += 1
    return np.asarray(selected, dtype=np.int64)


def _number(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} 不是有限数字")
    return result


def _xyz(value: dict[str, Any], name: str) -> np.ndarray:
    return np.array([_number(value[axis], f"{name}.{axis}") for axis in "xyz"], dtype=np.float64)


def _load_unity_camera(path: Path) -> tuple[PinholeCamera, str]:
    data = _read_json(path)
    if int(data.get("schemaVersion", 0)) != 1:
        raise ValueError(f"不支持的 Unity camera schemaVersion: {path}")
    intr = data.get("intrinsics")
    pose = data.get("cameraToWorld")
    if not isinstance(intr, dict) or not isinstance(pose, dict):
        raise ValueError(f"Unity camera JSON 缺少 intrinsics/cameraToWorld: {path}")
    rotation_value = pose.get("rotationXyzw")
    if not isinstance(rotation_value, dict):
        raise ValueError(f"Unity camera JSON 缺少 rotationXyzw: {path}")
    quaternion = np.array(
        [_number(rotation_value[axis], f"rotationXyzw.{axis}") for axis in ("x", "y", "z", "w")],
        dtype=np.float64,
    )
    norm = np.linalg.norm(quaternion)
    if norm < 1e-12:
        raise ValueError(f"Unity camera quaternion 无效: {path}")
    rotation = Rotation.from_quat(quaternion / norm).as_matrix()
    camera = PinholeCamera(
        width=int(data["width"]),
        height=int(data["height"]),
        fx=_number(intr["fx"], "intrinsics.fx"),
        fy=_number(intr["fy"], "intrinsics.fy"),
        cx=_number(intr["cx"], "intrinsics.cx"),
        cy=_number(intr["cy"], "intrinsics.cy"),
        rotation=rotation,
        translation=_xyz(pose["position"], "cameraToWorld.position"),
    )
    if camera.width < 1 or camera.height < 1 or camera.fx <= 0 or camera.fy <= 0:
        raise ValueError(f"Unity camera 内参无效: {path}")
    depth_metadata = data.get("depthMetadata", {})
    depth_type = depth_metadata.get("type", "radial_distance") if isinstance(depth_metadata, dict) else "radial_distance"
    if depth_type != "radial_distance":
        raise ValueError(f"仅支持 Unity radial_distance 深度，收到 {depth_type!r}: {path}")
    return camera, str(data.get("pixelOrigin", "top_left"))


def _read_colmap_cameras(path: Path) -> dict[int, tuple[int, int, float, float, float, float]]:
    cameras: dict[int, tuple[int, int, float, float, float, float]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            values = line.split()
            camera_id = int(values[0])
            model = values[1]
            width, height = int(values[2]), int(values[3])
            params = [float(value) for value in values[4:]]
            if model == "PINHOLE" and len(params) >= 4:
                fx, fy, cx, cy = params[:4]
            elif model == "SIMPLE_PINHOLE" and len(params) >= 3:
                fx = fy = params[0]
                cx, cy = params[1:3]
            else:
                raise ValueError(f"不支持的 COLMAP camera model: {model}")
            cameras[camera_id] = (width, height, fx, fy, cx, cy)
    if not cameras:
        raise ValueError(f"没有读到 COLMAP camera: {path}")
    return cameras


def _qvec_to_matrix(qvec_wxyz: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec_wxyz
    return np.array(
        [
            [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx * qx - 2 * qy * qy],
        ],
        dtype=np.float64,
    )


def _read_colmap_images(path: Path) -> dict[str, tuple[int, np.ndarray, np.ndarray]]:
    images: dict[str, tuple[int, np.ndarray, np.ndarray]] = {}
    with path.open("r", encoding="utf-8") as handle:
        while True:
            raw = handle.readline()
            if not raw:
                break
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            values = line.split()
            if len(values) < 10:
                raise ValueError(f"COLMAP images.txt 行格式错误: {line}")
            qvec = np.array([float(value) for value in values[1:5]], dtype=np.float64)
            qvec /= np.linalg.norm(qvec)
            tvec = np.array([float(value) for value in values[5:8]], dtype=np.float64)
            camera_id = int(values[8])
            images[values[9]] = (camera_id, _qvec_to_matrix(qvec), tvec)
            handle.readline()
    if not images:
        raise ValueError(f"没有读到 COLMAP image: {path}")
    return images


def _find_generated_camera(
    depth_path: Path,
    cameras: dict[int, tuple[int, int, float, float, float, float]],
    images: dict[str, tuple[int, np.ndarray, np.ndarray]],
) -> PinholeCamera:
    stem = depth_path.stem
    match = next((value for name, value in images.items() if Path(name).stem == stem), None)
    if match is None:
        choices = ", ".join(sorted(Path(name).stem for name in images))
        raise ValueError(f"generated depth {depth_path.name} 在 images.txt 中无同名视角；可用: {choices}")
    camera_id, rotation_w2c, translation_w2c = match
    width, height, fx, fy, cx, cy = cameras[camera_id]
    return PinholeCamera(width, height, fx, fy, cx, cy, rotation_w2c, translation_w2c)


def _read_depth(path: Path, camera: PinholeCamera) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    values = np.fromfile(path, dtype="<f4")
    expected = camera.width * camera.height
    if values.size != expected:
        raise ValueError(f"深度尺寸不匹配 {path}: {values.size} floats，期望 {camera.width}x{camera.height}={expected}")
    return values.reshape(camera.height, camera.width)


def _sample_depth(
    depth: np.ndarray,
    x: float,
    y: float,
    min_depth: float,
    max_relative_spread: float,
) -> Optional[float]:
    height, width = depth.shape
    if not (0 <= x <= width - 1 and 0 <= y <= height - 1):
        return None
    x0, y0 = int(math.floor(x)), int(math.floor(y))
    x1, y1 = min(x0 + 1, width - 1), min(y0 + 1, height - 1)
    dx, dy = x - x0, y - y0
    samples = (
        (depth[y0, x0], (1 - dx) * (1 - dy)),
        (depth[y0, x1], dx * (1 - dy)),
        (depth[y1, x0], (1 - dx) * dy),
        (depth[y1, x1], dx * dy),
    )
    valid = [(float(value), weight) for value, weight in samples if math.isfinite(float(value)) and value > min_depth and weight > 0]
    if not valid:
        nearest = float(depth[int(round(y)), int(round(x))])
        return nearest if math.isfinite(nearest) and nearest > min_depth else None
    values = np.array([item[0] for item in valid], dtype=np.float64)
    weights = np.array([item[1] for item in valid], dtype=np.float64)
    if max_relative_spread > 0 and values.size > 1:
        median = float(np.median(values))
        if median <= min_depth or (float(values.max()) - float(values.min())) / median > max_relative_spread:
            return None
    return float(np.dot(values, weights) / weights.sum())


def _unity_backproject(camera: PinholeCamera, x: float, y: float, radial_depth: float, center_offset: float) -> np.ndarray:
    ray = np.array(
        [
            (x + center_offset - camera.cx) / camera.fx,
            (camera.cy - y - center_offset) / camera.fy,
            1.0,
        ],
        dtype=np.float64,
    )
    ray /= np.linalg.norm(ray)
    return camera.rotation @ (ray * radial_depth) + camera.translation


def _generated_backproject(camera: PinholeCamera, x: float, y: float, z_depth: float, center_offset: float) -> np.ndarray:
    camera_point = np.array(
        [
            (x + center_offset - camera.cx) * z_depth / camera.fx,
            (y + center_offset - camera.cy) * z_depth / camera.fy,
            z_depth,
        ],
        dtype=np.float64,
    )
    return camera.rotation.T @ (camera_point - camera.translation)


def _image_size(matches: dict[str, Any], key: str, fallback: tuple[int, int]) -> tuple[int, int]:
    value = matches.get(key)
    if isinstance(value, list) and len(value) == 2:
        width, height = int(value[0]), int(value[1])
        if width > 0 and height > 0:
            return width, height
    return fallback


def _scale_pixel(point: np.ndarray, source_size: tuple[int, int], target_size: tuple[int, int]) -> tuple[float, float]:
    source_width, source_height = source_size
    target_width, target_height = target_size
    return float(point[0]) * target_width / source_width, float(point[1]) * target_height / source_height


def _axis_matrix(name: str) -> np.ndarray:
    if name == "identity":
        return np.eye(3, dtype=np.float64)
    return np.diag([1.0, 1.0, -1.0])


def _load_view_points(
    spec: list[str],
    generated_cameras: dict[int, tuple[int, int, float, float, float, float]],
    generated_images: dict[str, tuple[int, np.ndarray, np.ndarray]],
    args: argparse.Namespace,
) -> ViewPoints:
    matches_path, unity_depth_path, unity_camera_path, generated_depth_path = map(Path, spec)
    matches = _read_json(matches_path)
    points0 = np.asarray(matches.get("mkpts0", []), dtype=np.float64)
    points1 = np.asarray(matches.get("mkpts1", []), dtype=np.float64)
    if points0.size == 0:
        points0 = np.empty((0, 2), dtype=np.float64)
    if points1.size == 0:
        points1 = np.empty((0, 2), dtype=np.float64)
    if points0.ndim != 2 or points0.shape[1:] != (2,) or points1.shape != points0.shape:
        raise ValueError(f"matches.json 点数组格式错误: {matches_path}")
    count = len(points0)
    inliers = np.asarray(matches.get("inliers", np.ones(count)), dtype=bool).reshape(-1)
    if len(inliers) != count:
        raise ValueError(f"matches.json inliers 数量错误: {matches_path}")
    confidence = np.asarray(matches.get("confidence", np.ones(count)), dtype=np.float64).reshape(-1)
    if len(confidence) != count:
        raise ValueError(f"matches.json confidence 数量错误: {matches_path}")
    confidence = np.nan_to_num(confidence, nan=0.0, posinf=0.0, neginf=0.0)

    unity_camera, pixel_origin = _load_unity_camera(unity_camera_path)
    if pixel_origin != "top_left":
        raise ValueError(f"仅支持 top_left Unity 深度: {unity_camera_path}")
    generated_camera = _find_generated_camera(generated_depth_path, generated_cameras, generated_images)
    unity_depth = _read_depth(unity_depth_path, unity_camera)
    generated_depth = _read_depth(generated_depth_path, generated_camera)
    scene_image_size = _image_size(matches, "image0_size", (unity_camera.width, unity_camera.height))
    generated_image_size = _image_size(matches, "image1_size", (generated_camera.width, generated_camera.height))
    geometric_indices = np.flatnonzero(inliers)
    candidate_indices = _balanced_indices(
        geometric_indices,
        points0,
        confidence,
        scene_image_size,
        args.max_matches_per_view,
        args.spatial_grid_size,
    )
    axis = _axis_matrix(args.generated_axis)

    source_points: list[np.ndarray] = []
    target_points: list[np.ndarray] = []
    point_confidence: list[float] = []
    accepted_indices: list[int] = []
    scene_pixels: list[tuple[float, float]] = []
    generated_pixels: list[tuple[float, float]] = []
    for index in candidate_indices:
        ux, uy = _scale_pixel(points0[index], scene_image_size, (unity_camera.width, unity_camera.height))
        gx, gy = _scale_pixel(points1[index], generated_image_size, (generated_camera.width, generated_camera.height))
        unity_value = _sample_depth(unity_depth, ux, uy, args.min_depth, args.max_depth_relative_spread)
        generated_value = _sample_depth(generated_depth, gx, gy, args.min_depth, args.max_depth_relative_spread)
        if unity_value is None or generated_value is None:
            continue
        target = _unity_backproject(unity_camera, ux, uy, unity_value, args.pixel_center_offset)
        source = axis @ _generated_backproject(generated_camera, gx, gy, generated_value, args.pixel_center_offset)
        if np.all(np.isfinite(source)) and np.all(np.isfinite(target)):
            source_points.append(source)
            target_points.append(target)
            point_confidence.append(max(float(confidence[index]), 1e-12))
            accepted_indices.append(int(index))
            scene_pixels.append((ux, uy))
            generated_pixels.append((gx, gy))

    name = generated_depth_path.stem
    empty = np.empty((0, 3), dtype=np.float64)
    empty_pixels = np.empty((0, 2), dtype=np.float64)
    file_paths = {
        "matches": matches_path,
        "unity_depth": unity_depth_path,
        "unity_camera": unity_camera_path,
        "generated_depth": generated_depth_path,
    }
    image0 = Path(str(matches.get("image0", "")))
    image1 = Path(str(matches.get("image1", "")))
    if image0.is_file():
        file_paths["scene_image"] = image0
    if image1.is_file():
        file_paths["generated_image"] = image1
    return ViewPoints(
        name=name,
        source=np.stack(source_points) if source_points else empty,
        target=np.stack(target_points) if target_points else empty.copy(),
        confidence=np.asarray(point_confidence, dtype=np.float64),
        input_matches=count,
        geometric_inliers=int(inliers.sum()),
        depth_valid=len(source_points),
        files={key: str(path.resolve()) for key, path in file_paths.items()},
        file_digests={key: _file_digest(path) for key, path in file_paths.items()},
        geometric_scene_pixels=points0[geometric_indices].copy() if len(geometric_indices) else empty_pixels,
        geometric_generated_pixels=points1[geometric_indices].copy() if len(geometric_indices) else empty_pixels.copy(),
        match_indices=np.asarray(accepted_indices, dtype=np.int64),
        scene_pixels=np.asarray(scene_pixels, dtype=np.float64).reshape((-1, 2)) if scene_pixels else empty_pixels.copy(),
        generated_pixels=np.asarray(generated_pixels, dtype=np.float64).reshape((-1, 2)) if generated_pixels else empty_pixels.copy(),
    )


def _umeyama(
    source: np.ndarray,
    target: np.ndarray,
    weights: Optional[np.ndarray],
    with_scale: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 3:
        raise ValueError("Umeyama 至少需要 3 对三维点")
    if weights is None:
        weights = np.ones(len(source), dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.clip(weights, 0.0, None)
    if not np.any(weights > 0):
        weights = np.ones(len(source), dtype=np.float64)
    weights /= weights.sum()
    source_mean = np.sum(source * weights[:, None], axis=0)
    target_mean = np.sum(target * weights[:, None], axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = (target_centered * weights[:, None]).T @ source_centered
    left, singular, right_t = np.linalg.svd(covariance)
    correction = np.ones(3, dtype=np.float64)
    if np.linalg.det(left @ right_t) < 0:
        correction[-1] = -1.0
    rotation = left @ np.diag(correction) @ right_t
    variance = float(np.sum(weights * np.sum(source_centered * source_centered, axis=1)))
    if variance <= 1e-15:
        raise ValueError("源三维点退化，无法求相似变换")
    scale = float(np.dot(singular, correction) / variance) if with_scale else 1.0
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("求得的 scale 无效")
    translation = target_mean - scale * (rotation @ source_mean)
    return rotation, translation, scale


def _residuals(source: np.ndarray, target: np.ndarray, rotation: np.ndarray, translation: np.ndarray, scale: float) -> np.ndarray:
    transformed = (scale * (rotation @ source.T)).T + translation
    return np.linalg.norm(transformed - target, axis=1)


def _robust_similarity(
    source: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
    threshold: float,
    iterations: int,
    min_inliers: int,
    seed: int,
    with_scale: bool,
    view_ids: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    if len(source) < max(3, min_inliers):
        raise ValueError(f"有效三维对应点不足: {len(source)}，至少需要 {max(3, min_inliers)}")
    if threshold <= 0 or iterations < 1:
        raise ValueError("RANSAC threshold/iterations 必须为正数")
    rng = np.random.default_rng(seed)
    probabilities = np.clip(confidence, 0.0, None)
    probabilities = probabilities / probabilities.sum() if probabilities.sum() > 0 else None
    best_mask: Optional[np.ndarray] = None
    if view_ids is None:
        view_ids = np.zeros(len(source), dtype=np.int64)
    else:
        view_ids = np.asarray(view_ids, dtype=np.int64).reshape(-1)
        if len(view_ids) != len(source):
            raise ValueError("view_ids 与对应点数量不一致")
    view_count = int(view_ids.max()) + 1 if len(view_ids) else 1
    best_score: Optional[tuple[int, int, int, float]] = None
    sample_size = 3
    for _ in range(iterations):
        indices = rng.choice(len(source), size=sample_size, replace=False, p=probabilities)
        if np.linalg.matrix_rank(source[indices] - source[indices].mean(axis=0)) < 2:
            continue
        try:
            rotation, translation, scale = _umeyama(source[indices], target[indices], None, with_scale)
        except (ValueError, np.linalg.LinAlgError):
            continue
        residual = _residuals(source, target, rotation, translation, scale)
        mask = residual <= threshold
        count = int(mask.sum())
        median = float(np.median(residual[mask])) if count else float("inf")
        counts = np.bincount(view_ids[mask], minlength=view_count)
        supported = int(np.count_nonzero(counts >= 3))
        minimum = int(counts.min()) if len(counts) else 0
        score = (supported, minimum, count, -median)
        if best_score is None or score > best_score:
            best_score = score
            best_mask = mask
    if best_mask is None or int(best_mask.sum()) < min_inliers:
        best_count = int(best_mask.sum()) if best_mask is not None else 0
        raise ValueError(f"RANSAC 内点不足: {best_count}，至少需要 {min_inliers}")

    mask = best_mask
    for _ in range(5):
        rotation, translation, scale = _umeyama(source[mask], target[mask], confidence[mask], with_scale)
        residual = _residuals(source, target, rotation, translation, scale)
        updated = residual <= threshold
        if int(updated.sum()) < min_inliers or np.array_equal(updated, mask):
            break
        mask = updated
    rotation, translation, scale = _umeyama(source[mask], target[mask], confidence[mask], with_scale)
    residual = _residuals(source, target, rotation, translation, scale)
    return rotation, translation, scale, mask, residual


def _vector_json(value: np.ndarray) -> dict[str, float]:
    return {axis: float(value[index]) for index, axis in enumerate("xyz")}


def _fit_views(views: list[ViewPoints], args: argparse.Namespace, seed: int) -> SimilarityFit:
    active = [view for view in views if view.depth_valid]
    if not active:
        raise ValueError("没有可用的双侧深度对应点")
    source = np.concatenate([view.source for view in active], axis=0)
    target = np.concatenate([view.target for view in active], axis=0)
    confidence = np.concatenate([view.confidence for view in active], axis=0)
    view_ids = np.concatenate(
        [np.full(len(view.source), index, dtype=np.int64) for index, view in enumerate(active)],
        axis=0,
    )
    rotation, translation, scale, inliers, residuals = _robust_similarity(
        source,
        target,
        confidence,
        args.ransac_threshold,
        args.ransac_iterations,
        args.min_inliers,
        seed,
        not args.without_scale,
        view_ids,
    )
    return SimilarityFit(rotation, translation, scale, inliers, residuals)


def _transform_json(fit: SimilarityFit) -> dict[str, Any]:
    quaternion = Rotation.from_matrix(fit.rotation).as_quat()
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = fit.scale * fit.rotation
    matrix[:3, 3] = fit.translation
    accepted = fit.residuals[fit.inliers]
    return {
        "position": _vector_json(fit.translation),
        "rotation": {
            "x": float(quaternion[0]),
            "y": float(quaternion[1]),
            "z": float(quaternion[2]),
            "w": float(quaternion[3]),
        },
        "uniformScale": float(fit.scale),
        "matrix4x4": matrix.tolist(),
        "input3dPairs": int(len(fit.inliers)),
        "inlierCount": int(fit.inliers.sum()),
        "inlierRatio": float(fit.inliers.mean()) if len(fit.inliers) else 0.0,
        "residualMean": float(accepted.mean()) if len(accepted) else None,
        "residualMedian": float(np.median(accepted)) if len(accepted) else None,
        "residualMax": float(accepted.max()) if len(accepted) else None,
    }


def _score_transform(view: ViewPoints, fit: SimilarityFit, threshold: float) -> dict[str, Any]:
    if not view.depth_valid:
        return {"supportCount": 0, "supportRatio": 0.0, "residualMedian": None, "residualMean": None}
    residuals = _residuals(view.source, view.target, fit.rotation, fit.translation, fit.scale)
    mask = residuals <= threshold
    accepted = residuals[mask]
    return {
        "supportCount": int(mask.sum()),
        "supportRatio": float(mask.mean()),
        "residualMedian": float(np.median(accepted)) if len(accepted) else None,
        "residualMean": float(accepted.mean()) if len(accepted) else None,
    }


def _local_masks(views: list[ViewPoints], fit: SimilarityFit) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    start = 0
    for view in [item for item in views if item.depth_valid]:
        end = start + view.depth_valid
        result[view.name] = fit.inliers[start:end]
        start = end
    for view in views:
        result.setdefault(view.name, np.zeros(view.depth_valid, dtype=bool))
    return result


def _build_validation(
    args: argparse.Namespace,
    views: list[ViewPoints],
    joint: SimilarityFit,
) -> tuple[dict[str, Any], dict[str, SimilarityFit], dict[str, np.ndarray]]:
    independent: dict[str, SimilarityFit] = {}
    independent_json: dict[str, Any] = {}
    cross_matrix: dict[str, dict[str, Any]] = {}
    for index, view in enumerate(views):
        if view.depth_valid < max(3, args.min_inliers):
            independent_json[view.name] = {"status": "unavailable", "reason": "insufficient_depth_valid_points"}
            continue
        try:
            fit = _fit_views([view], args, args.seed + 1000 + index)
        except ValueError as exc:
            independent_json[view.name] = {"status": "failed", "reason": str(exc)}
            continue
        independent[view.name] = fit
        independent_json[view.name] = {"status": "ready", **_transform_json(fit)}
        cross_matrix[view.name] = {
            target.name: _score_transform(target, fit, args.ransac_threshold)
            for target in views
        }

    leave_one_out: dict[str, Any] = {}
    for index, held_out in enumerate(views):
        training = [view for view in views if view is not held_out and view.depth_valid]
        if not training:
            leave_one_out[held_out.name] = {"status": "unavailable", "reason": "no_training_views"}
            continue
        try:
            fit = _fit_views(training, args, args.seed + 2000 + index)
            leave_one_out[held_out.name] = {
                "status": "ready",
                "trainingViews": [view.name for view in training],
                "heldOut": _score_transform(held_out, fit, args.ransac_threshold),
                "transform": _transform_json(fit),
            }
        except ValueError as exc:
            leave_one_out[held_out.name] = {
                "status": "failed",
                "trainingViews": [view.name for view in training],
                "reason": str(exc),
            }

    disagreements = []
    names = list(independent)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            left, right = independent[left_name], independent[right_name]
            relative = left.rotation @ right.rotation.T
            angle = math.degrees(Rotation.from_matrix(relative).magnitude())
            disagreements.append(
                {
                    "views": [left_name, right_name],
                    "scaleRatio": float(max(left.scale, right.scale) / min(left.scale, right.scale)),
                    "rotationDegrees": float(angle),
                    "translationDistance": float(np.linalg.norm(left.translation - right.translation)),
                }
            )

    joint_masks = _local_masks(views, joint)
    joint_views: dict[str, Any] = {}
    rejection_reasons: list[str] = []
    for view in views:
        mask = joint_masks[view.name]
        count = int(mask.sum())
        ratio = float(mask.mean()) if len(mask) else 0.0
        residuals = _residuals(view.source, view.target, joint.rotation, joint.translation, joint.scale) if view.depth_valid else np.empty(0)
        accepted = residuals[mask] if len(residuals) else np.empty(0)
        joint_views[view.name] = {
            "depthValid": view.depth_valid,
            "poseInliers": count,
            "poseInlierRatio": ratio,
            "poseResidualMedian": float(np.median(accepted)) if len(accepted) else None,
        }
        if not args.no_quality_gate:
            if count < args.min_view_inliers:
                rejection_reasons.append(f"{view.name}:joint_inliers={count}<{args.min_view_inliers}")
            if ratio < args.min_view_inlier_ratio:
                rejection_reasons.append(f"{view.name}:joint_ratio={ratio:.6g}<{args.min_view_inlier_ratio:.6g}")

    primary_view = args.primary_view_name.strip().lower() if args.primary_view_name else None
    corroborating_views: list[str] = []
    if len(views) < 2 and not (args.allow_single_view or args.no_quality_gate):
        rejection_reasons.append("single_view_not_allowed")
    if len(views) >= 2 and not args.no_quality_gate:
        if primary_view:
            primary_scores = cross_matrix.get(primary_view)
            if primary_scores is None:
                rejection_reasons.append(f"{primary_view}:primary_fit_unavailable")
            else:
                for view in views:
                    if view.name == primary_view:
                        continue
                    score = primary_scores.get(view.name, {})
                    count = int(score.get("supportCount", 0))
                    ratio = float(score.get("supportRatio", 0.0))
                    if count >= args.min_cross_view_inliers and ratio >= args.min_cross_view_ratio:
                        corroborating_views.append(view.name)
                if not corroborating_views:
                    rejection_reasons.append(f"{primary_view}:no_corroborating_view")
        else:
            for view in views:
                result = leave_one_out.get(view.name, {})
                held_out = result.get("heldOut", {}) if result.get("status") == "ready" else {}
                count = int(held_out.get("supportCount", 0))
                ratio = float(held_out.get("supportRatio", 0.0))
                if count < args.min_cross_view_inliers:
                    rejection_reasons.append(f"{view.name}:cross_inliers={count}<{args.min_cross_view_inliers}")
                if ratio < args.min_cross_view_ratio:
                    rejection_reasons.append(f"{view.name}:cross_ratio={ratio:.6g}<{args.min_cross_view_ratio:.6g}")

    validation = {
        "status": "rejected" if rejection_reasons else "ready",
        "qualityGateEnabled": not args.no_quality_gate,
        "policy": "primary_with_corroboration" if primary_view else "all_leave_one_out",
        "primaryView": primary_view,
        "corroboratingViews": corroborating_views,
        "thresholds": {
            "minViewInliers": args.min_view_inliers,
            "minViewInlierRatio": args.min_view_inlier_ratio,
            "minCrossViewInliers": args.min_cross_view_inliers,
            "minCrossViewRatio": args.min_cross_view_ratio,
        },
        "rejectionReasons": rejection_reasons,
        "jointPerView": joint_views,
        "independentFits": independent_json,
        "crossValidation": cross_matrix,
        "leaveOneOut": leave_one_out,
        "transformDisagreement": disagreements,
    }
    return validation, independent, joint_masks


def _draw_match_panel(
    image0: np.ndarray,
    image1: np.ndarray,
    points0: np.ndarray,
    points1: np.ndarray,
    accepted: Optional[np.ndarray],
    title: str,
    width: int = 520,
    height: int = 250,
    limit: int = 80,
) -> np.ndarray:
    import cv2

    image_height = height - 34
    half = width // 2
    left = cv2.resize(image0, (half, image_height), interpolation=cv2.INTER_AREA)
    right = cv2.resize(image1, (width - half, image_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[34:, :half] = left
    canvas[34:, half:] = right
    cv2.putText(canvas, title, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 1, cv2.LINE_AA)
    if points0.ndim != 2 or points1.shape != points0.shape or not len(points0):
        return canvas
    count = min(len(points0), max(0, limit))
    indices = np.linspace(0, len(points0) - 1, count, dtype=np.int64) if count else np.empty(0, dtype=np.int64)
    h0, w0 = image0.shape[:2]
    h1, w1 = image1.shape[:2]
    for index in indices:
        x0 = int(round(float(points0[index, 0]) * half / max(1, w0)))
        y0 = 34 + int(round(float(points0[index, 1]) * image_height / max(1, h0)))
        x1 = half + int(round(float(points1[index, 0]) * (width - half) / max(1, w1)))
        y1 = 34 + int(round(float(points1[index, 1]) * image_height / max(1, h1)))
        if accepted is None:
            color = (0, 210, 255)
        else:
            color = (60, 220, 80) if bool(accepted[index]) else (50, 70, 230)
        cv2.line(canvas, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)
        cv2.circle(canvas, (x0, y0), 2, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (x1, y1), 2, color, -1, cv2.LINE_AA)
    return canvas


def _write_diagnostic_images(
    output_dir: Path,
    views: list[ViewPoints],
    independent: dict[str, SimilarityFit],
    joint_masks: dict[str, np.ndarray],
    validation: dict[str, Any],
) -> None:
    try:
        import cv2
    except ImportError:
        return
    view_canvases: list[np.ndarray] = []
    for view in views:
        image0 = cv2.imread(view.files.get("scene_image", ""), cv2.IMREAD_COLOR)
        image1 = cv2.imread(view.files.get("generated_image", ""), cv2.IMREAD_COLOR)
        if image0 is None or image1 is None:
            image0 = np.zeros((256, 256, 3), dtype=np.uint8)
            image1 = np.zeros((256, 256, 3), dtype=np.uint8)
        independent_mask = independent[view.name].inliers if view.name in independent else np.zeros(view.depth_valid, dtype=bool)
        panels = [
            _draw_match_panel(
                image0,
                image1,
                view.geometric_scene_pixels,
                view.geometric_generated_pixels,
                None,
                f"{view.name.upper()} RAW GEOMETRIC ({view.geometric_inliers})",
            ),
            _draw_match_panel(
                image0,
                image1,
                view.scene_pixels,
                view.generated_pixels,
                independent_mask,
                f"{view.name.upper()} INDEPENDENT 3D ({int(independent_mask.sum())}/{view.depth_valid})",
            ),
            _draw_match_panel(
                image0,
                image1,
                view.scene_pixels,
                view.generated_pixels,
                joint_masks[view.name],
                f"{view.name.upper()} JOINT 3D ({int(joint_masks[view.name].sum())}/{view.depth_valid})",
            ),
        ]
        canvas = np.vstack(panels)
        cv2.imwrite(str(output_dir / f"{view.name}_validation.png"), canvas)
        view_canvases.append(canvas)
    if not view_canvases:
        return
    height = max(canvas.shape[0] for canvas in view_canvases)
    normalized = [
        cv2.copyMakeBorder(canvas, 0, height - canvas.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        for canvas in view_canvases
    ]
    summary = np.hstack(normalized)
    text_height = 250
    text_panel = np.zeros((text_height, summary.shape[1], 3), dtype=np.uint8)
    status_color = (60, 220, 80) if validation["status"] == "ready" else (50, 70, 230)
    cv2.putText(text_panel, f"MULTIVIEW GATE: {validation['status'].upper()}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, status_color, 2, cv2.LINE_AA)
    x = 12
    y = 58
    for source_name, row in validation.get("crossValidation", {}).items():
        values = "  ".join(
            f"{target}:{int(score.get('supportCount', 0))}/{float(score.get('supportRatio', 0.0)):.3f}"
            for target, score in row.items()
        )
        cv2.putText(text_panel, f"fit {source_name} -> {values}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1, cv2.LINE_AA)
        y += 24
    for held_name, result in validation.get("leaveOneOut", {}).items():
        held = result.get("heldOut", {})
        line = f"leave-out {held_name}: {int(held.get('supportCount', 0))}/{float(held.get('supportRatio', 0.0)):.3f}"
        cv2.putText(text_panel, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1, cv2.LINE_AA)
        y += 24
    reasons = validation.get("rejectionReasons", [])
    if reasons:
        reason_text = "; ".join(reasons)
        for start in range(0, len(reason_text), 120):
            cv2.putText(text_panel, reason_text[start : start + 120], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, status_color, 1, cv2.LINE_AA)
            y += 20
            if y >= text_height - 8:
                break
    cv2.imwrite(str(output_dir / "multiview_summary.png"), np.vstack((summary, text_panel)))


def _write_result(
    args: argparse.Namespace,
    views: list[ViewPoints],
    joint: SimilarityFit,
    validation: dict[str, Any],
    independent: dict[str, SimilarityFit],
    joint_masks: dict[str, np.ndarray],
) -> dict[str, Any]:
    rotation, translation, scale = joint.rotation, joint.translation, joint.scale
    quaternion = Rotation.from_matrix(rotation).as_quat()
    euler_zxy = Rotation.from_matrix(rotation).as_euler("zxy", degrees=True)
    euler_unity = np.array([euler_zxy[1], euler_zxy[2], euler_zxy[0]], dtype=np.float64)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = translation
    view_results = []
    diagnostics_views = []
    for view in views:
        local_mask = joint_masks[view.name]
        local_residuals = _residuals(view.source, view.target, rotation, translation, scale) if view.depth_valid else np.empty(0)
        accepted = local_residuals[local_mask]
        view_results.append(
            {
                "name": view.name,
                "inputMatches": view.input_matches,
                "geometricInliers": view.geometric_inliers,
                "depthValid": view.depth_valid,
                "poseInliers": int(local_mask.sum()),
                "poseResidualMedian": float(np.median(accepted)) if len(accepted) else None,
                "files": view.files,
                "fileDigests": view.file_digests,
            }
        )
        independent_mask = independent[view.name].inliers if view.name in independent else np.zeros(view.depth_valid, dtype=bool)
        diagnostics_views.append(
            {
                "name": view.name,
                "geometricScenePixels": view.geometric_scene_pixels.tolist(),
                "geometricGeneratedPixels": view.geometric_generated_pixels.tolist(),
                "depthValidMatchIndices": view.match_indices.tolist(),
                "depthValidScenePixels": view.scene_pixels.tolist(),
                "depthValidGeneratedPixels": view.generated_pixels.tolist(),
                "independentInliers": independent_mask.astype(int).tolist(),
                "jointInliers": local_mask.astype(int).tolist(),
                "jointResiduals": local_residuals.tolist(),
                "files": view.files,
            }
        )
    accepted_residuals = joint.residuals[joint.inliers]
    camera_digests = {
        "cameras": _file_digest(args.generated_cameras),
        "images": _file_digest(args.generated_images),
    }
    output = {
        "schemaVersion": 2,
        "status": validation["status"],
        "transformDirection": "generated_world_to_unity_world",
        "position": _vector_json(translation),
        "rotation": {
            "x": float(quaternion[0]),
            "y": float(quaternion[1]),
            "z": float(quaternion[2]),
            "w": float(quaternion[3]),
        },
        "rotationEuler": _vector_json(euler_unity),
        "scale": {"x": float(scale), "y": float(scale), "z": float(scale)},
        "uniformScale": float(scale),
        "matrix4x4": matrix.tolist(),
        "fit": {
            "input3dPairs": int(len(joint.inliers)),
            "inlierCount": int(joint.inliers.sum()),
            "inlierRatio": float(joint.inliers.mean()),
            "thresholdUnityUnits": float(args.ransac_threshold),
            "residualMean": float(accepted_residuals.mean()),
            "residualMedian": float(np.median(accepted_residuals)),
            "residualMax": float(accepted_residuals.max()),
            "withScale": not args.without_scale,
            "seed": args.seed,
            "spatialGridSize": args.spatial_grid_size,
            "maxDepthRelativeSpread": args.max_depth_relative_spread,
        },
        "validation": validation,
        "coordinateContract": {
            "generatedAxis": args.generated_axis,
            "unityWorld": "left_handed_y_up_z_forward",
            "unityDepth": "radial_distance",
            "generatedDepth": "camera_z",
            "pixelOrigin": "top_left",
            "pixelCenterOffset": args.pixel_center_offset,
        },
        "generatedCameraFiles": {
            "cameras": str(args.generated_cameras.resolve()),
            "images": str(args.generated_images.resolve()),
            "digests": camera_digests,
        },
        "provenance": {
            "runId": args.run_id,
            "candidateId": args.candidate_id,
            "sourceFileDigests": {
                view.name: view.file_digests for view in views
            },
        },
        "views": view_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.diagnostics_dir:
        args.diagnostics_dir.mkdir(parents=True, exist_ok=True)
        diagnostics = {
            "schemaVersion": 2,
            "status": validation["status"],
            "runId": args.run_id,
            "candidateId": args.candidate_id,
            "pose": str(args.output.resolve()),
            "validation": validation,
            "jointFit": _transform_json(joint),
            "views": diagnostics_views,
        }
        summary_json = args.diagnostics_dir / "multiview_summary.json"
        summary_json.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_diagnostic_images(args.diagnostics_dir, views, independent, joint_masks, validation)
        output["diagnostics"] = {
            "json": str(summary_json.resolve()),
            "image": str((args.diagnostics_dir / "multiview_summary.png").resolve()),
        }
        args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def _write_unfit_rejection(args: argparse.Namespace, views: list[ViewPoints], reason: str) -> dict[str, Any]:
    if "RANSAC" in reason:
        reason_code = "ransac_inliers_below_minimum"
    elif "双侧深度" in reason or "depth" in reason.lower():
        reason_code = "no_depth_valid_correspondences"
    elif "退化" in reason or "degenerate" in reason.lower():
        reason_code = "degenerate_correspondences"
    else:
        reason_code = "fit_unavailable"
    rejection_reasons = [f"pose_unavailable:{reason_code}"]
    joint_views: dict[str, Any] = {}
    independent_fits: dict[str, SimilarityFit] = {}
    independent: dict[str, Any] = {}
    cross_validation: dict[str, dict[str, Any]] = {}
    for index, view in enumerate(views):
        if view.depth_valid < max(3, args.min_inliers):
            independent[view.name] = {
                "status": "unavailable",
                "reason": f"depth_valid={view.depth_valid}<{max(3, args.min_inliers)}",
            }
            continue
        try:
            fit = _fit_views([view], args, args.seed + 1000 + index)
        except ValueError as exc:
            independent[view.name] = {"status": "failed", "reason": str(exc)}
            continue
        independent_fits[view.name] = fit
        independent[view.name] = {"status": "ready", **_transform_json(fit)}
        cross_validation[view.name] = {
            target.name: _score_transform(target, fit, args.ransac_threshold)
            for target in views
        }

    leave_one_out: dict[str, Any] = {}
    for index, held_out in enumerate(views):
        training = [view for view in views if view is not held_out and view.depth_valid]
        if not training:
            leave_one_out[held_out.name] = {"status": "unavailable", "reason": "no_training_views"}
            continue
        try:
            fit = _fit_views(training, args, args.seed + 2000 + index)
            leave_one_out[held_out.name] = {
                "status": "ready",
                "trainingViews": [view.name for view in training],
                "heldOut": _score_transform(held_out, fit, args.ransac_threshold),
                "transform": _transform_json(fit),
            }
        except ValueError as exc:
            leave_one_out[held_out.name] = {
                "status": "failed",
                "trainingViews": [view.name for view in training],
                "reason": str(exc),
            }

    disagreements = []
    names = list(independent_fits)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            left, right = independent_fits[left_name], independent_fits[right_name]
            relative = left.rotation @ right.rotation.T
            disagreements.append(
                {
                    "views": [left_name, right_name],
                    "scaleRatio": float(max(left.scale, right.scale) / min(left.scale, right.scale)),
                    "rotationDegrees": float(math.degrees(Rotation.from_matrix(relative).magnitude())),
                    "translationDistance": float(np.linalg.norm(left.translation - right.translation)),
                }
            )

    joint_masks: dict[str, np.ndarray] = {}
    view_results = []
    diagnostics_views = []
    for view in views:
        joint_views[view.name] = {
            "depthValid": view.depth_valid,
            "poseInliers": 0,
            "poseInlierRatio": 0.0,
            "poseResidualMedian": None,
        }
        joint_masks[view.name] = np.zeros(view.depth_valid, dtype=bool)
        view_results.append(
            {
                "name": view.name,
                "inputMatches": view.input_matches,
                "geometricInliers": view.geometric_inliers,
                "depthValid": view.depth_valid,
                "poseInliers": 0,
                "poseResidualMedian": None,
                "files": view.files,
                "fileDigests": view.file_digests,
            }
        )
        diagnostics_views.append(
            {
                "name": view.name,
                "geometricScenePixels": view.geometric_scene_pixels.tolist(),
                "geometricGeneratedPixels": view.geometric_generated_pixels.tolist(),
                "depthValidMatchIndices": view.match_indices.tolist(),
                "depthValidScenePixels": view.scene_pixels.tolist(),
                "depthValidGeneratedPixels": view.generated_pixels.tolist(),
                "independentInliers": (
                    independent_fits[view.name].inliers.astype(int).tolist()
                    if view.name in independent_fits else np.zeros(view.depth_valid, dtype=int).tolist()
                ),
                "jointInliers": np.zeros(view.depth_valid, dtype=int).tolist(),
                "jointResiduals": [],
                "files": view.files,
            }
        )
    validation = {
        "status": "rejected",
        "qualityGateEnabled": not args.no_quality_gate,
        "thresholds": {
            "minViewInliers": args.min_view_inliers,
            "minViewInlierRatio": args.min_view_inlier_ratio,
            "minCrossViewInliers": args.min_cross_view_inliers,
            "minCrossViewRatio": args.min_cross_view_ratio,
        },
        "rejectionReasons": rejection_reasons,
        "rejectionDetails": [reason],
        "jointPerView": joint_views,
        "independentFits": independent,
        "crossValidation": cross_validation,
        "leaveOneOut": leave_one_out,
        "transformDisagreement": disagreements,
    }
    output = {
        "schemaVersion": 2,
        "status": "rejected",
        "transformDirection": "generated_world_to_unity_world",
        "fit": None,
        "validation": validation,
        "coordinateContract": {
            "generatedAxis": args.generated_axis,
            "unityWorld": "left_handed_y_up_z_forward",
            "unityDepth": "radial_distance",
            "generatedDepth": "camera_z",
            "pixelOrigin": "top_left",
            "pixelCenterOffset": args.pixel_center_offset,
        },
        "generatedCameraFiles": {
            "cameras": str(args.generated_cameras.resolve()),
            "images": str(args.generated_images.resolve()),
            "digests": {
                "cameras": _file_digest(args.generated_cameras),
                "images": _file_digest(args.generated_images),
            },
        },
        "provenance": {
            "runId": args.run_id,
            "candidateId": args.candidate_id,
            "sourceFileDigests": {view.name: view.file_digests for view in views},
        },
        "views": view_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.diagnostics_dir:
        args.diagnostics_dir.mkdir(parents=True, exist_ok=True)
        diagnostics = {
            "schemaVersion": 2,
            "status": "rejected",
            "runId": args.run_id,
            "candidateId": args.candidate_id,
            "pose": str(args.output.resolve()),
            "validation": validation,
            "jointFit": None,
            "views": diagnostics_views,
        }
        summary_json = args.diagnostics_dir / "multiview_summary.json"
        summary_json.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_diagnostic_images(args.diagnostics_dir, views, independent_fits, joint_masks, validation)
        output["diagnostics"] = {
            "json": str(summary_json.resolve()),
            "image": str((args.diagnostics_dir / "multiview_summary.png").resolve()),
        }
        args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def main() -> int:
    args = parse_args()
    if (
        args.max_matches_per_view < 0
        or args.min_inliers < 3
        or args.min_view_inliers < 0
        or args.min_cross_view_inliers < 0
        or not 0 <= args.min_view_inlier_ratio <= 1
        or not 0 <= args.min_cross_view_ratio <= 1
        or args.spatial_grid_size < 1
        or args.max_depth_relative_spread < 0
    ):
        raise SystemExit("pose 匹配、深度、视角门禁或网格参数无效")
    generated_cameras = _read_colmap_cameras(args.generated_cameras)
    generated_images = _read_colmap_images(args.generated_images)
    views = [
        _load_view_points(spec, generated_cameras, generated_images, args)
        for spec in args.view
    ]
    if args.primary_view_name:
        primary = args.primary_view_name.strip().lower()
        if not primary or primary not in {view.name.lower() for view in views}:
            raise SystemExit(f"primary-view-name 不在输入视图中: {args.primary_view_name!r}")
        args.primary_view_name = primary
    if not any(view.depth_valid for view in views):
        reason = "所有视角都没有可用的双侧深度对应点"
        _write_unfit_rejection(args, views, reason)
        print("POSE_ESTIMATE_REJECTED", reason, args.output, flush=True)
        return 0 if args.exit_zero_on_rejected else 2
    try:
        joint = _fit_views(views, args, args.seed)
    except ValueError as exc:
        _write_unfit_rejection(args, views, str(exc))
        print("POSE_ESTIMATE_REJECTED", str(exc), args.output, flush=True)
        return 0 if args.exit_zero_on_rejected else 2
    validation, independent, joint_masks = _build_validation(args, views, joint)
    output = _write_result(args, views, joint, validation, independent, joint_masks)
    if output["status"] == "ready":
        print("POSE_ESTIMATE_READY", int(joint.inliers.sum()), f"scale={joint.scale:.9g}", args.output, flush=True)
        return 0
    print("POSE_ESTIMATE_REJECTED", ";".join(validation["rejectionReasons"]), args.output, flush=True)
    return 0 if args.exit_zero_on_rejected else 2


if __name__ == "__main__":
    raise SystemExit(main())
