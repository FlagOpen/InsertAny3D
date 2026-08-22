"""Stable JSON-friendly contracts shared by model providers and the pipeline."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _matrix3(value: Any, name: str) -> tuple[tuple[float, float, float], ...]:
    if not isinstance(value, Sequence) or len(value) != 3:
        raise ValueError(f"{name} must be a 3x3 matrix")
    rows: list[tuple[float, float, float]] = []
    for row in value:
        if not isinstance(row, Sequence) or len(row) != 3:
            raise ValueError(f"{name} must be a 3x3 matrix")
        rows.append(tuple(_finite_float(item, name) for item in row))
    return tuple(rows)


def _vector3(value: Any, name: str, default: tuple[float, float, float] | None = None) -> tuple[float, float, float]:
    if value is None and default is not None:
        return default
    if not isinstance(value, Sequence) or len(value) != 3:
        raise ValueError(f"{name} must contain three values")
    return tuple(_finite_float(item, name) for item in value)  # type: ignore[return-value]


def _det3(matrix: tuple[tuple[float, float, float], ...]) -> float:
    a, b, c = matrix
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


@dataclass(frozen=True)
class CoordinateContract:
    """Provider-native to generated-world coordinate declaration.

    ``axis_matrix`` is intentionally explicit.  It must be a proper rotation
    for Gaussian orientation conversion; reflections belong in an explicit
    provider profile and are rejected by the generic PLY transformer unless a
    provider supplies a handedness-specific implementation.
    """

    source_frame: str = "provider_native"
    target_frame: str = "generated_world"
    axis_matrix: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    uniform_scale: float = 1.0
    handedness: str = "right"
    up_axis: str = "y"
    forward_axis: str = "z"
    origin: str = "object_center"
    normalization_mode: str = "provider_native"
    render_defaults: Mapping[str, float] = field(default_factory=dict)
    unity_generated_axis: str = "legacy-flip-z"

    def __post_init__(self) -> None:
        matrix = _matrix3(self.axis_matrix, "axis_matrix")
        object.__setattr__(self, "axis_matrix", matrix)
        object.__setattr__(self, "translation", _vector3(self.translation, "translation"))
        scale = _finite_float(self.uniform_scale, "uniform_scale")
        if scale <= 0:
            raise ValueError("uniform_scale must be positive")
        object.__setattr__(self, "uniform_scale", scale)
        determinant = _det3(matrix)
        if abs(abs(determinant) - 1.0) > 1e-3:
            raise ValueError(f"axis_matrix must be orthonormal (det={determinant:.6g})")
        rows = matrix
        for index in range(3):
            norm = sum(rows[index][axis] * rows[index][axis] for axis in range(3))
            if abs(norm - 1.0) > 1e-3:
                raise ValueError("axis_matrix rows must have unit length")
            for other in range(index):
                dot = sum(rows[index][axis] * rows[other][axis] for axis in range(3))
                if abs(dot) > 1e-3:
                    raise ValueError("axis_matrix rows must be orthogonal")
        if self.handedness not in {"right", "left"}:
            raise ValueError("handedness must be 'right' or 'left'")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CoordinateContract":
        normalization = value.get("normalization", {})
        unity = value.get("unityImport", {})
        render = value.get("renderDefaults", {})
        if not isinstance(normalization, Mapping) or not isinstance(unity, Mapping) or not isinstance(render, Mapping):
            raise ValueError("coordinate contract nested fields must be objects")
        return cls(
            source_frame=str(value.get("sourceFrame", "provider_native")),
            target_frame=str(value.get("targetFrame", "generated_world")),
            axis_matrix=_matrix3(
                value.get(
                    "axisMatrix",
                    ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                ),
                "axisMatrix",
            ),
            translation=_vector3(value.get("translation"), "translation", (0.0, 0.0, 0.0)),
            uniform_scale=_finite_float(normalization.get("scale", 1.0), "normalization.scale"),
            handedness=str(value.get("handedness", "right")),
            up_axis=str(value.get("upAxis", "y")),
            forward_axis=str(value.get("forwardAxis", "z")),
            origin=str(value.get("origin", "object_center")),
            normalization_mode=str(normalization.get("mode", "provider_native")),
            render_defaults={str(k): _finite_float(v, f"renderDefaults.{k}") for k, v in render.items()},
            unity_generated_axis=str(unity.get("generatedAxis", "legacy-flip-z")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sourceFrame": self.source_frame,
            "targetFrame": self.target_frame,
            "axisMatrix": [list(row) for row in self.axis_matrix],
            "translation": list(self.translation),
            "handedness": self.handedness,
            "upAxis": self.up_axis,
            "forwardAxis": self.forward_axis,
            "origin": self.origin,
            "normalization": {"mode": self.normalization_mode, "scale": self.uniform_scale},
            "renderDefaults": dict(self.render_defaults),
            "unityImport": {"generatedAxis": self.unity_generated_axis},
        }

    def apply_point(self, point: Sequence[float]) -> tuple[float, float, float]:
        x, y, z = _vector3(point, "point")
        matrix = self.axis_matrix
        transformed = (
            matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z,
            matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z,
            matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z,
        )
        return tuple((item + offset) * self.uniform_scale for item, offset in zip(transformed, self.translation))


@dataclass(frozen=True)
class MaskArtifact:
    path: str
    image_path: str
    engine: str
    prompt: str = ""
    width: int = 0
    height: int = 0
    mask_sha256: str = ""
    image_sha256: str = ""
    threshold: float | None = None
    detections: tuple[Mapping[str, Any], ...] = ()
    instance_index: int | None = None
    human_confirmed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "imagePath": self.image_path,
            "engine": self.engine,
            "prompt": self.prompt,
            "width": self.width,
            "height": self.height,
            "maskSha256": self.mask_sha256,
            "imageSha256": self.image_sha256,
            "threshold": self.threshold,
            "detections": list(self.detections),
            "instanceIndex": self.instance_index,
            "humanConfirmed": self.human_confirmed,
        }


@dataclass(frozen=True)
class GenerationRequest:
    input_image: Path
    output_dir: Path
    provider: str
    profile: str = "default"
    model: str | None = None
    seed: int = 1
    input_mask: Path | None = None
    options: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "inputImage": str(self.input_image),
            "outputDir": str(self.output_dir),
            "provider": self.provider,
            "profile": self.profile,
            "model": self.model,
            "seed": self.seed,
            "inputMask": str(self.input_mask) if self.input_mask else None,
            "options": dict(self.options),
        }


@dataclass(frozen=True)
class RenderRequest:
    """Provider-neutral render inputs consumed by the existing renderer."""

    input_ply: Path
    output_dir: Path
    mode: str = "anchor"
    resolution: int = 1024
    fov_degrees: float = 53.1301023542
    distance: float = 1.5
    near: float = 0.8
    far: float = 1.6
    yaw_degrees: float = 0.0
    pitch_degrees: float = 12.0
    side_angle_degrees: float = 24.0
    view_names: str = "left,center,right"
    radius: float = 1.5
    latitudes: str = "10,20,30"
    views_per_latitude: int = 30

    def __post_init__(self) -> None:
        if self.mode not in {"anchor", "sphere"}:
            raise ValueError("render mode must be 'anchor' or 'sphere'")
        if self.resolution < 1 or self.distance <= 0 or self.radius <= 0:
            raise ValueError("render resolution/distance/radius must be positive")
        if self.near <= 0 or self.far <= self.near:
            raise ValueError("render near/far must satisfy 0 < near < far")
        if self.views_per_latitude < 1:
            raise ValueError("views_per_latitude must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "inputPly": str(self.input_ply),
            "outputDir": str(self.output_dir),
            "mode": self.mode,
            "resolution": self.resolution,
            "fovDegrees": self.fov_degrees,
            "distance": self.distance,
            "near": self.near,
            "far": self.far,
            "yawDegrees": self.yaw_degrees,
            "pitchDegrees": self.pitch_degrees,
            "sideAngleDegrees": self.side_angle_degrees,
            "viewNames": self.view_names,
            "radius": self.radius,
            "latitudes": self.latitudes,
            "viewsPerLatitude": self.views_per_latitude,
        }


@dataclass(frozen=True)
class GeneratedAsset:
    gaussian_ply: Path
    provider: str
    model: str | None
    representation: str
    coordinate_contract: CoordinateContract
    metadata_path: Path | None = None
    source_mesh: Path | None = None
    weight_fingerprint: str | None = None
    input_mask: Path | None = None
    input_image_sha256: str | None = None
    model_revision: str | None = None
    seed: int | None = None
    converter_metadata: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "gaussianPly": str(self.gaussian_ply),
            "provider": self.provider,
            "model": self.model,
            "representation": self.representation,
            "coordinateContract": self.coordinate_contract.to_dict(),
            "metadataPath": str(self.metadata_path) if self.metadata_path else None,
            "sourceMesh": str(self.source_mesh) if self.source_mesh else None,
            "weightFingerprint": self.weight_fingerprint,
            "inputMask": str(self.input_mask) if self.input_mask else None,
            "inputImageSha256": self.input_image_sha256,
            "modelRevision": self.model_revision,
            "seed": self.seed,
            "converterMetadata": str(self.converter_metadata) if self.converter_metadata else None,
        }


@dataclass(frozen=True)
class EnvironmentReport:
    provider: str
    python: str
    available: bool
    requirements: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    blocked_reasons: tuple[str, ...] = ()
    minimum_vram_gb: float | None = None
    recommended_vram_gb: float | None = None
    maximum_vram_gb: float | None = None
    runtime_versions: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "python": self.python,
            "available": self.available,
            "requirements": list(self.requirements),
            "missing": list(self.missing),
            "warnings": list(self.warnings),
            "blockedReasons": list(self.blocked_reasons),
            "minimumVramGb": self.minimum_vram_gb,
            "recommendedVramGb": self.recommended_vram_gb,
            "maximumVramGb": self.maximum_vram_gb,
            "runtimeVersions": dict(self.runtime_versions),
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
