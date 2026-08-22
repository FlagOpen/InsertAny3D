"""Deterministic mesh-to-surface-Gaussian conversion for Hunyuan outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from plyfile import PlyData, PlyElement

try:
    from ..contracts import CoordinateContract, sha256_file, write_json
except ImportError:  # direct ``python path/to/mesh_to_gaussian.py`` execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from model_center.contracts import CoordinateContract, sha256_file, write_json

_SH_C0 = 0.28209479177387814


def _mesh_geometries(mesh: trimesh.Trimesh | trimesh.Scene) -> list[trimesh.Trimesh]:
    if isinstance(mesh, trimesh.Trimesh):
        return [mesh]
    geometries: list[trimesh.Trimesh] = []
    for node_name in mesh.graph.nodes_geometry:
        transform, geometry_name = mesh.graph.get(node_name)
        geometry = mesh.geometry[geometry_name]
        transformed = geometry.copy()
        if transform is not None:
            transformed.apply_transform(transform)
        geometries.append(transformed)
    return geometries


def _face_colors(mesh: trimesh.Trimesh, faces: np.ndarray) -> np.ndarray:
    visual = mesh.visual
    if hasattr(visual, "to_color"):
        try:
            visual = visual.to_color()
        except Exception:
            pass
    if hasattr(visual, "face_colors") and len(visual.face_colors) >= len(mesh.faces):
        colors = np.asarray(visual.face_colors[faces], dtype=np.float64)
    elif hasattr(visual, "vertex_colors") and len(visual.vertex_colors) >= len(mesh.vertices):
        colors = np.asarray(visual.vertex_colors[mesh.faces[faces]], dtype=np.float64).mean(axis=1)
    else:
        colors = np.full((len(faces), 4), 255.0, dtype=np.float64)
    if colors.shape[1] == 3:
        colors = np.column_stack([colors, np.full(len(colors), 255.0)])
    return np.clip(colors[:, :4] / 255.0, 0.0, 1.0)


def _matrix_to_quat(matrix: np.ndarray) -> np.ndarray:
    # Return wxyz Gaussian convention.
    result = np.empty((len(matrix), 4), dtype=np.float64)
    for i, m in enumerate(matrix):
        trace = float(np.trace(m))
        if trace > 0:
            root = math.sqrt(max(trace + 1.0, 1e-12)) * 2
            q = np.array([0.25 * root, (m[2, 1] - m[1, 2]) / root, (m[0, 2] - m[2, 0]) / root, (m[1, 0] - m[0, 1]) / root])
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            root = math.sqrt(max(1 + m[0, 0] - m[1, 1] - m[2, 2], 1e-12)) * 2
            q = np.array([(m[2, 1] - m[1, 2]) / root, 0.25 * root, (m[0, 1] + m[1, 0]) / root, (m[0, 2] + m[2, 0]) / root])
        elif m[1, 1] > m[2, 2]:
            root = math.sqrt(max(1 + m[1, 1] - m[0, 0] - m[2, 2], 1e-12)) * 2
            q = np.array([(m[0, 2] - m[2, 0]) / root, (m[0, 1] + m[1, 0]) / root, 0.25 * root, (m[1, 2] + m[2, 1]) / root])
        else:
            root = math.sqrt(max(1 + m[2, 2] - m[0, 0] - m[1, 1], 1e-12)) * 2
            q = np.array([(m[1, 0] - m[0, 1]) / root, (m[0, 2] + m[2, 0]) / root, (m[1, 2] + m[2, 1]) / root, 0.25 * root])
        result[i] = q / max(float(np.linalg.norm(q)), 1e-12)
    return result


def _sample_triangle(triangle: np.ndarray, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if count <= 0:
        return np.empty((0, 3)), np.empty((0, 3))
    # A deterministic low-discrepancy sequence avoids dependence on global RNG.
    index = np.arange(count, dtype=np.float64) + 0.5
    u = (index * 0.7548776662466927 + (seed % 997) / 997.0) % 1.0
    v = (index * 0.5698402909980532 + (seed % 991) / 991.0) % 1.0
    root = np.sqrt(u)
    bary = np.column_stack([1.0 - root, root * (1.0 - v), root * v])
    points = bary @ triangle
    edge0 = triangle[1] - triangle[0]
    edge1 = triangle[2] - triangle[0]
    normal = np.cross(edge0, edge1)
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        normals = np.tile(np.array([0.0, 1.0, 0.0]), (count, 1))
    else:
        normals = np.tile(normal / norm, (count, 1))
    return points, normals


def convert_mesh_to_gaussian(
    input_mesh: Path,
    output_ply: Path,
    density: float = 32.0,
    thickness: float = 0.002,
    max_points: int = 250000,
    seed: int = 1,
    contract: CoordinateContract | None = None,
    metadata_path: Path | None = None,
) -> dict[str, Any]:
    if density <= 0 or thickness <= 0 or max_points < 1:
        raise ValueError("density/thickness/max_points must be positive")
    if not input_mesh.is_file():
        raise FileNotFoundError(f"mesh does not exist: {input_mesh}")
    loaded = trimesh.load(str(input_mesh), force="scene", process=False)
    geometries = _mesh_geometries(loaded)
    if not geometries:
        raise ValueError(f"mesh contains no geometry: {input_mesh}")
    source_vertices = [np.asarray(mesh.vertices, dtype=np.float64) for mesh in geometries if len(mesh.vertices)]
    if not source_vertices:
        raise ValueError(f"mesh contains no vertices: {input_mesh}")
    combined_vertices = np.concatenate(source_vertices, axis=0)
    source_lower = combined_vertices.min(axis=0)
    source_upper = combined_vertices.max(axis=0)
    center = (source_lower + source_upper) * 0.5
    extent = float(np.max(source_upper - source_lower))
    if not math.isfinite(extent) or extent <= 1e-8:
        raise ValueError("mesh has zero extent")
    all_points: list[np.ndarray] = []
    all_normals: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    all_scales: list[np.ndarray] = []
    for geometry_index, mesh in enumerate(geometries):
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if len(vertices) == 0 or len(faces) == 0:
            continue
        triangles = vertices[faces]
        areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) * 0.5
        face_colors = _face_colors(mesh, np.arange(len(faces)))
        for face_index, (triangle, area) in enumerate(zip(triangles, areas)):
            if area <= 1e-12:
                continue
            normalized_area = float(area) / (extent * extent)
            count = max(1, int(math.ceil(normalized_area * density * density)))
            points, normals = _sample_triangle(triangle, count, seed + geometry_index * 1000003 + face_index)
            edge_lengths = [float(np.linalg.norm(triangle[1] - triangle[0])), float(np.linalg.norm(triangle[2] - triangle[0])), float(np.linalg.norm(triangle[2] - triangle[1]))]
            tangent_scale = max(
                min(edge_lengths) / extent / max(math.sqrt(count), 1.0) * 0.75,
                1e-5,
            )
            scales = np.column_stack([
                np.full(count, tangent_scale),
                np.full(count, tangent_scale),
                np.full(count, thickness),
            ])
            all_points.append(points)
            all_normals.append(normals)
            all_colors.append(np.tile(face_colors[face_index], (count, 1)))
            all_scales.append(scales)
            if sum(len(item) for item in all_points) >= max_points:
                break
        if sum(len(item) for item in all_points) >= max_points:
            break
    if not all_points:
        raise ValueError("mesh has no non-degenerate triangles")
    points = np.concatenate(all_points, axis=0)[:max_points]
    normals = np.concatenate(all_normals, axis=0)[:max_points]
    colors = np.concatenate(all_colors, axis=0)[:max_points]
    scales = np.concatenate(all_scales, axis=0)[:max_points]
    points = (points - center) / extent
    scales[:, 2] = thickness
    normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    if contract is not None:
        matrix = np.asarray(contract.axis_matrix, dtype=np.float64)
        if np.linalg.det(matrix) < 0:
            raise ValueError("mesh converter requires a proper rotation contract")
        points = (points @ matrix.T + np.asarray(contract.translation)) * contract.uniform_scale
        normals = normals @ matrix.T
        scales *= contract.uniform_scale
    # Stable tangent frame: choose the least parallel world axis.
    reference = np.zeros_like(normals)
    axis_index = np.argmin(np.abs(normals), axis=1)
    reference[np.arange(len(normals)), axis_index] = 1.0
    tangent = np.cross(reference, normals)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    bitangent = np.cross(normals, tangent)
    frames = np.stack([tangent, bitangent, normals], axis=2)
    quaternions = _matrix_to_quat(frames)
    rgb = colors[:, :3]
    f_dc = (rgb - 0.5) / _SH_C0
    alpha = np.clip(colors[:, 3], 1e-3, 1.0 - 1e-3)
    opacity = np.log(alpha / (1.0 - alpha))
    dtype = [(name, "f4") for name in (
        "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
        "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    )]
    values = np.empty(len(points), dtype=dtype)
    values["x"], values["y"], values["z"] = points[:, 0], points[:, 1], points[:, 2]
    values["nx"], values["ny"], values["nz"] = normals[:, 0], normals[:, 1], normals[:, 2]
    values["f_dc_0"], values["f_dc_1"], values["f_dc_2"] = f_dc[:, 0], f_dc[:, 1], f_dc[:, 2]
    values["opacity"] = opacity
    values["scale_0"], values["scale_1"], values["scale_2"] = np.log(np.maximum(scales[:, 0], 1e-7)), np.log(np.maximum(scales[:, 1], 1e-7)), np.log(np.maximum(scales[:, 2], 1e-7))
    for index, name in enumerate(("rot_0", "rot_1", "rot_2", "rot_3")):
        values[name] = quaternions[:, index]
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(values, "vertex")], text=False).write(str(output_ply))
    result = {
        "schemaVersion": 1,
        "representation": "surface_splats",
        "inputMesh": str(input_mesh.resolve()),
        "inputSha256": sha256_file(input_mesh),
        "outputPly": str(output_ply.resolve()),
        "pointCount": int(len(points)),
        "density": density,
        "thickness": thickness,
        "maxPoints": max_points,
        "seed": seed,
        "sourceBoundsCenter": center.tolist(),
        "sourceBoundsExtent": extent,
        "sourceBoundsMin": source_lower.tolist(),
        "sourceBoundsMax": source_upper.tolist(),
        "coordinateContract": contract.to_dict() if contract else None,
        "materialStrategy": "texture_to_vertex_or_face_average_rgba",
    }
    if metadata_path is not None:
        write_json(metadata_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a mesh/GLB to standard surface Gaussian PLY")
    parser.add_argument("--input-mesh", required=True, type=Path)
    parser.add_argument("--output-ply", required=True, type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--density", type=float, default=32.0)
    parser.add_argument("--thickness", type=float, default=0.002)
    parser.add_argument("--max-points", type=int, default=250000)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = convert_mesh_to_gaussian(
        args.input_mesh,
        args.output_ply,
        density=args.density,
        thickness=args.thickness,
        max_points=args.max_points,
        seed=args.seed,
        metadata_path=args.metadata,
    )
    print("MESH_TO_GAUSSIAN_READY", result["pointCount"], args.output_ply, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
