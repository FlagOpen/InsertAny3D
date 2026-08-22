"""Provider interface and command planning primitives."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from ..contracts import CoordinateContract, EnvironmentReport, GenerationRequest


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    representation: str
    runtime_python: Path
    source_root: Path
    coordinate_contract: CoordinateContract
    required_modules: tuple[str, ...] = ()
    minimum_vram_gb: float | None = None
    recommended_vram_gb: float | None = None
    license_name: str | None = None

    def environment_report(self) -> EnvironmentReport:
        missing: list[str] = []
        for module in self.required_modules:
            try:
                __import__(module)
            except Exception:
                missing.append(module)
        warnings: list[str] = []
        if not self.runtime_python.is_file():
            missing.append(str(self.runtime_python))
        return EnvironmentReport(
            provider=self.name,
            python=str(self.runtime_python),
            available=not missing,
            requirements=self.required_modules,
            missing=tuple(missing),
            warnings=tuple(warnings),
            minimum_vram_gb=self.minimum_vram_gb,
            recommended_vram_gb=self.recommended_vram_gb,
        )


class ModelProvider(Protocol):
    spec: ProviderSpec

    def prepare_request(self, request: GenerationRequest) -> GenerationRequest: ...

    def generation_command(self, request: GenerationRequest, project_root: Path) -> list[str]: ...

    def coordinate_contract(self, request: GenerationRequest | None = None) -> CoordinateContract: ...

    def environment_report(self) -> EnvironmentReport: ...


class BaseProvider:
    spec: ProviderSpec

    def prepare_request(self, request: GenerationRequest) -> GenerationRequest:
        if request.provider != self.spec.name:
            raise ValueError(f"request provider {request.provider!r} does not match {self.spec.name!r}")
        if not request.input_image.is_file():
            raise FileNotFoundError(f"provider input image does not exist: {request.input_image}")
        request.output_dir.mkdir(parents=True, exist_ok=True)
        return request

    def coordinate_contract(self, request: GenerationRequest | None = None) -> CoordinateContract:
        return self.spec.coordinate_contract

    def environment_report(self) -> EnvironmentReport:
        # The report is intentionally conservative: imports are checked in the
        # current interpreter, while the runner performs a second check inside
        # the provider's selected runtime.
        return self.spec.environment_report()

    def _runner(self, project_root: Path) -> Path:
        tool_root = project_root / "tools"
        if not tool_root.is_dir():
            tool_root = project_root / "codex_remote_tools"
        return tool_root / "model_center" / "provider_runner.py"

    def _common_runner_args(self, request: GenerationRequest, project_root: Path) -> list[str]:
        command = [
            str(self.spec.runtime_python),
            str(self._runner(project_root)),
            "--provider",
            self.spec.name,
            "--input-image",
            str(request.input_image),
            "--output-dir",
            str(request.output_dir),
            "--seed",
            str(request.seed),
        ]
        if request.model:
            command.extend(["--model", request.model])
        if request.input_mask:
            command.extend(["--input-mask", str(request.input_mask)])
        return command
