"""Build commands for the existing TRELLIS Gaussian renderers."""

from __future__ import annotations

from pathlib import Path

from ..contracts import CoordinateContract, RenderRequest


def _tools_root(project_root: Path) -> Path:
    candidate = project_root / "tools"
    return candidate if candidate.is_dir() else project_root / "codex_remote_tools"


def build_render_command(
    request: RenderRequest,
    project_root: Path,
    python: Path,
    contract: CoordinateContract | None = None,
) -> list[str]:
    """Return a fully explicit command, including provider near/far defaults."""

    if contract is not None:
        defaults = contract.render_defaults
        near = request.near if request.near is not None else float(defaults.get("near", 0.8))
        far = request.far if request.far is not None else float(defaults.get("far", 1.6))
    else:
        near, far = request.near, request.far
    tools = _tools_root(project_root)
    if request.mode == "sphere":
        return [
            str(python), str(tools / "render_trellis_3dgs.py"),
            "--input-ply", str(request.input_ply), "--output-dir", str(request.output_dir),
            "--resolution", str(request.resolution), "--radius", str(request.radius),
            "--fov", str(request.fov_degrees), "--near", str(near), "--far", str(far),
            "--latitudes", request.latitudes, "--views-per-latitude", str(request.views_per_latitude),
        ]
    return [
        str(python), str(tools / "render_trellis_views.py"),
        "--input-ply", str(request.input_ply), "--output-dir", str(request.output_dir),
        "--resolution", str(request.resolution), "--fov", str(request.fov_degrees),
        "--yaw-degrees", str(request.yaw_degrees), "--pitch-degrees", str(request.pitch_degrees),
        "--distance", str(request.distance), "--near", str(near), "--far", str(far),
        "--side-angle-degrees", str(request.side_angle_degrees), "--view-names", request.view_names,
    ]
