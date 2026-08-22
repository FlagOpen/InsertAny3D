"""TRELLIS provider metadata and command planning."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .base import BaseProvider, ProviderSpec
from ..contracts import CoordinateContract, GenerationRequest


class TrellisProvider(BaseProvider):
    spec = ProviderSpec(
        name="trellis",
        representation="native_gaussian",
        runtime_python=Path("third_party/TRELLIS/.venv/bin/python"),
        source_root=Path("third_party/TRELLIS"),
        coordinate_contract=CoordinateContract(
            source_frame="trellis_native",
            normalization_mode="trellis_normalized",
            render_defaults={"distance": 1.5, "near": 0.8, "far": 1.6},
            unity_generated_axis="legacy-flip-z",
        ),
        required_modules=("torch", "trellis", "plyfile"),
        minimum_vram_gb=16,
        license_name="MIT (code) / model-specific terms",
    )

    def generation_command(self, request: GenerationRequest, project_root: Path) -> list[str]:
        request = self.prepare_request(request)
        tool_root = project_root / "tools"
        if not tool_root.is_dir():
            tool_root = project_root / "codex_remote_tools"
        command = [
            str(self.spec.runtime_python),
            str(tool_root / "generate_trellis_asset.py"),
            "--input-image",
            str(request.input_image),
            "--output-dir",
            str(request.output_dir),
            "--seed",
            str(request.seed),
        ]
        if request.model:
            command.extend(["--model", request.model])
        for key, flag in (
            ("sparse_steps", "--sparse-steps"),
            ("slat_steps", "--slat-steps"),
            ("sparse_cfg", "--sparse-cfg"),
            ("slat_cfg", "--slat-cfg"),
        ):
            if key in request.options and request.options[key] is not None:
                command.extend([flag, str(request.options[key])])
        return command
