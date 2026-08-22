"""Apply a provider coordinate contract to a Gaussian PLY.

The transformer handles positions, log-scales and quaternion orientations
together.  It deliberately accepts only proper rotations for the generic path;
reflection-based legacy imports remain owned by the existing TRELLIS adapter.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from plyfile import PlyData, PlyElement

from ..contracts import CoordinateContract, write_json


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    # Gaussian PLY convention is w, x, y, z.
    q = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    q = q / np.maximum(norm, 1e-12)
    w, x, y, z = [q[:, index] for index in range(4)]
    result = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    result[:, 0, 0] = 1 - 2 * (y * y + z * z)
    result[:, 0, 1] = 2 * (x * y - z * w)
    result[:, 0, 2] = 2 * (x * z + y * w)
    result[:, 1, 0] = 2 * (x * y + z * w)
    result[:, 1, 1] = 1 - 2 * (x * x + z * z)
    result[:, 1, 2] = 2 * (y * z - x * w)
    result[:, 2, 0] = 2 * (x * z - y * w)
    result[:, 2, 1] = 2 * (y * z + x * w)
    result[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return result


def _matrix_to_quat(matrix: np.ndarray) -> np.ndarray:
    output = np.empty((matrix.shape[0], 4), dtype=np.float64)
    for index, m in enumerate(matrix):
        trace = float(np.trace(m))
        if trace > 0:
            root = math.sqrt(max(trace + 1.0, 1e-12)) * 2
            w = 0.25 * root
            x = (m[2, 1] - m[1, 2]) / root
            y = (m[0, 2] - m[2, 0]) / root
            z = (m[1, 0] - m[0, 1]) / root
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            root = math.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 1e-12)) * 2
            w = (m[2, 1] - m[1, 2]) / root
            x = 0.25 * root
            y = (m[0, 1] + m[1, 0]) / root
            z = (m[0, 2] + m[2, 0]) / root
        elif m[1, 1] > m[2, 2]:
            root = math.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 1e-12)) * 2
            w = (m[0, 2] - m[2, 0]) / root
            x = (m[0, 1] + m[1, 0]) / root
            y = 0.25 * root
            z = (m[1, 2] + m[2, 1]) / root
        else:
            root = math.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 1e-12)) * 2
            w = (m[1, 0] - m[0, 1]) / root
            x = (m[0, 2] + m[2, 0]) / root
            y = (m[1, 2] + m[2, 1]) / root
            z = 0.25 * root
        q = np.array([w, x, y, z], dtype=np.float64)
        output[index] = q / max(float(np.linalg.norm(q)), 1e-12)
    return output


def transform_gaussian_ply(
    input_path: Path,
    output_path: Path,
    contract: CoordinateContract,
    metadata_path: Path | None = None,
) -> dict[str, Any]:
    if not input_path.is_file():
        raise FileNotFoundError(f"Gaussian PLY does not exist: {input_path}")
    matrix = np.asarray(contract.axis_matrix, dtype=np.float64)
    determinant = float(np.linalg.det(matrix))
    if determinant < 0:
        raise ValueError("generic Gaussian transform does not support reflection axis matrices")
    data = PlyData.read(str(input_path))
    if not data.elements or data.elements[0].name != "vertex":
        raise ValueError(f"Gaussian PLY has no vertex element: {input_path}")
    vertex = data.elements[0].data
    names = {item.name for item in data.elements[0].properties}
    required = {"x", "y", "z"}
    if not required.issubset(names):
        raise ValueError(f"Gaussian PLY missing position fields: {sorted(required - names)}")
    xyz = np.column_stack([np.asarray(vertex[name], dtype=np.float64) for name in ("x", "y", "z")])
    xyz = (xyz @ matrix.T + np.asarray(contract.translation, dtype=np.float64)) * float(contract.uniform_scale)

    scales = None
    rotations = None
    scale_names = [f"scale_{index}" for index in range(3)]
    rotation_names = [f"rot_{index}" for index in range(4)]
    if all(name in names for name in scale_names):
        scales = np.column_stack([np.exp(np.asarray(vertex[name], dtype=np.float64)) for name in scale_names])
        scales *= float(contract.uniform_scale)
    if all(name in names for name in rotation_names):
        rotations = _quat_to_matrix(np.column_stack([np.asarray(vertex[name], dtype=np.float64) for name in rotation_names]))
        rotations = np.einsum("ij,njk->nik", matrix, rotations)
        if np.any(np.linalg.det(rotations) < 0.0):
            raise ValueError("transformed Gaussian orientations contain a reflection")
        quaternions = _matrix_to_quat(rotations)

    dtype = []
    for prop in data.elements[0].properties:
        property_dtype = prop.dtype() if callable(prop.dtype) else prop.dtype
        dtype.append((prop.name, property_dtype))
    output = np.empty(len(vertex), dtype=dtype)
    for name in names:
        output[name] = vertex[name]
    for index, name in enumerate(("x", "y", "z")):
        output[name] = xyz[:, index].astype(output[name].dtype)
    if scales is not None:
        for index, name in enumerate(scale_names):
            output[name] = np.log(np.maximum(scales[:, index], 1e-7)).astype(output[name].dtype)
    if rotations is not None:
        for index, name in enumerate(rotation_names):
            output[name] = quaternions[:, index].astype(output[name].dtype)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    elements = [PlyElement.describe(output, "vertex")]
    # Preserve optional PLY elements (faces, cameras, comments from a provider)
    # instead of silently dropping diagnostics while replacing the Gaussian
    # vertex table.
    elements.extend(element for element in data.elements[1:])
    PlyData(elements, text=False, comments=list(data.comments)).write(str(output_path))
    result = {
        "schemaVersion": 1,
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "pointCount": int(len(output)),
        "coordinateContract": contract.to_dict(),
    }
    if metadata_path is not None:
        write_json(metadata_path, result)
    return result
