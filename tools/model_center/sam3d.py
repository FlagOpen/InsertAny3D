"""SAM 3D Objects provider using the existing TRELLIS Python runtime."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .base import BaseProvider, ProviderSpec
from ..contracts import CoordinateContract, GenerationRequest


class Sam3DProvider(BaseProvider):
    spec = ProviderSpec(
        name="sam3d",
        representation="native_gaussian",
        runtime_python=Path("third_party/TRELLIS/.venv/bin/python"),
        source_root=Path("third_party/SAM3D-Objects"),
        coordinate_contract=CoordinateContract(
            source_frame="sam3d_native",
            normalization_mode="sam3d_native",
            render_defaults={"distance": 1.5, "near": 0.8, "far": 1.6},
            unity_generated_axis="legacy-flip-z",
        ),
        required_modules=(
            "torch", "torchvision", "sam3d_objects", "pytorch3d", "gsplat", "moge",
            "utils3d", "kaolin", "plyfile", "omegaconf", "hydra", "seaborn",
        ),
        minimum_vram_gb=32,
        license_name="SAM License",
    )

    def prepare_request(self, request: GenerationRequest) -> GenerationRequest:
        request = super().prepare_request(request)
        if request.input_mask is None or not request.input_mask.is_file():
            raise ValueError("sam3d requires a single-object input mask")
        return request

    def generation_command(self, request: GenerationRequest, project_root: Path) -> list[str]:
        request = self.prepare_request(request)
        command = self._common_runner_args(request, project_root)
        options = request.options
        for key, flag in (
            ("model_dir", "--model-dir"),
            ("config_path", "--config-path"),
            ("source_root", "--source-root"),
        ):
            if options.get(key):
                command.extend([flag, str(options[key])])
        decoder = options.get("sam3d_decoder", options.get("decoder"))
        if decoder:
            command.extend(["--sam3d-decoder", str(decoder)])
        downsample = options.get("sam3d_downsample_ss_dist", options.get("downsample_ss_dist"))
        if downsample is not None:
            command.extend(["--sam3d-downsample-ss-dist", str(int(downsample))])
        if options.get("sam3d_load_unused_decoders", options.get("load_unused_decoders", False)):
            command.append("--sam3d-load-unused-decoders")
        depth_model_path = options.get("sam3d_depth_model_path", options.get("depth_model_path"))
        if depth_model_path:
            command.extend(["--sam3d-depth-model-path", str(depth_model_path)])
        if options.get("compile"):
            command.append("--compile")
        return command
