"""Hunyuan3D provider and its independent runtime declaration."""

from __future__ import annotations

from pathlib import Path

from .base import BaseProvider, ProviderSpec
from ..contracts import CoordinateContract, GenerationRequest


class HunyuanProvider(BaseProvider):
    spec = ProviderSpec(
        name="hunyuan",
        representation="surface_splats",
        runtime_python=Path("third_party/Hunyuan3D-2/.venv/bin/python"),
        source_root=Path("third_party/Hunyuan3D-2"),
        coordinate_contract=CoordinateContract(
            source_frame="hunyuan_mesh_native",
            normalization_mode="aabb_max_extent",
            render_defaults={"distance": 1.5, "near": 0.05, "far": 3.0},
            unity_generated_axis="legacy-flip-z",
        ),
        required_modules=("torch", "hy3dgen", "trimesh"),
        minimum_vram_gb=16,
        license_name="Tencent Hunyuan 3D 2.0 Community License",
    )

    def generation_command(self, request: GenerationRequest, project_root: Path) -> list[str]:
        request = self.prepare_request(request)
        command = self._common_runner_args(request, project_root)
        options = request.options
        for key, flag in (
            ("model_path", "--model-path"),
            ("shape_subfolder", "--shape-subfolder"),
        ):
            if options.get(key) is not None:
                command.extend([flag, str(options[key])])
        if bool(options.get("texture")):
            command.append("--texture")
        return command
