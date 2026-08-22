"""Provider registry with lazy, explicit model selection."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from .contracts import EnvironmentReport, GenerationRequest, write_json
from .providers import HunyuanProvider, ModelProvider, Sam3DProvider, TrellisProvider


_PROVIDER_TYPES = {
    "trellis": TrellisProvider,
    "sam3d": Sam3DProvider,
    "hunyuan": HunyuanProvider,
}


def provider_names() -> tuple[str, ...]:
    return tuple(_PROVIDER_TYPES)


def get_provider(name: str, project_root: Path) -> ModelProvider:
    key = str(name or "").strip().lower()
    if key not in _PROVIDER_TYPES:
        raise ValueError(f"unknown model provider {name!r}; choose one of {', '.join(provider_names())}")
    provider = _PROVIDER_TYPES[key]()
    # Provider declarations are relative to the project so they remain useful
    # in local tests and in the uploaded remote tools tree.
    spec = provider.spec
    provider.spec = replace(
        spec,
        runtime_python=(project_root / spec.runtime_python).resolve(),
        source_root=(project_root / spec.source_root).resolve(),
    )
    return provider


def provider_environment_report(provider: ModelProvider) -> EnvironmentReport:
    runtime = provider.spec.runtime_python
    if not runtime.is_file():
        return EnvironmentReport(
            provider=provider.spec.name,
            python=str(runtime),
            available=False,
            requirements=provider.spec.required_modules,
            missing=(str(runtime),),
            minimum_vram_gb=provider.spec.minimum_vram_gb,
            recommended_vram_gb=provider.spec.recommended_vram_gb,
        )
    command = [
        str(runtime),
        "-c",
        (
            "import importlib.util, json, os, sys; "
            "mods=json.loads(sys.argv[1]); "
            "sys.path.insert(0, sys.argv[2]); "
            "os.environ.setdefault('LIDRA_SKIP_INIT','true'); "
            "os.environ.setdefault('CONDA_PREFIX', os.environ.get('CUDA_HOME') or '/usr/local/cuda'); "
            "missing=[m for m in mods if importlib.util.find_spec(m) is None]; "
            "versions={'python':sys.version.split()[0]}; max_vram=None; "
            "exec(\"import torch\\nversions['torch']=torch.__version__\\nversions['cuda']=str(torch.version.cuda)\\nmax_vram=max((torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())),default=0)/(1024**3) if torch.cuda.is_available() else None\") if 'torch' not in missing else None; "
            "print(json.dumps({'missing':missing,'versions':versions,'maximumVramGb':max_vram}))"
        ),
        json.dumps(list(provider.spec.required_modules)),
        str(provider.spec.source_root),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return EnvironmentReport(
            provider=provider.spec.name,
            python=str(runtime),
            available=False,
            requirements=provider.spec.required_modules,
            missing=(f"runtime probe: {exc}",),
            minimum_vram_gb=provider.spec.minimum_vram_gb,
            recommended_vram_gb=provider.spec.recommended_vram_gb,
        )
    if completed.returncode != 0:
        return EnvironmentReport(
            provider=provider.spec.name,
            python=str(runtime),
            available=False,
            requirements=provider.spec.required_modules,
            missing=(f"runtime probe exit {completed.returncode}",),
            warnings=(completed.stderr.strip()[-500:],) if completed.stderr.strip() else (),
            minimum_vram_gb=provider.spec.minimum_vram_gb,
            recommended_vram_gb=provider.spec.recommended_vram_gb,
        )
    try:
        value = json.loads(completed.stdout.strip())
    except json.JSONDecodeError:
        value = {"missing": list(provider.spec.required_modules)}
    missing = tuple(str(item) for item in value.get("missing", []))
    warnings: list[str] = []
    blocked_reasons: list[str] = []
    maximum_vram = value.get("maximumVramGb")
    try:
        maximum_vram = float(maximum_vram) if maximum_vram is not None else None
    except (TypeError, ValueError):
        maximum_vram = None
    if provider.spec.minimum_vram_gb:
        warnings.append(f"tested adapter profile requires at least {provider.spec.minimum_vram_gb:g} GiB VRAM")
        if maximum_vram is not None and maximum_vram + 1e-6 < provider.spec.minimum_vram_gb:
            blocked_reasons.append(
                f"maximum single-device VRAM {maximum_vram:.1f} GiB is below required "
                f"{provider.spec.minimum_vram_gb} GiB"
            )
    if provider.spec.recommended_vram_gb:
        warnings.append(f"upstream recommends {provider.spec.recommended_vram_gb:g} GiB VRAM")
    versions = value.get("versions", {})
    if not isinstance(versions, Mapping):
        versions = {}
    return EnvironmentReport(
        provider=provider.spec.name,
        python=str(runtime),
        available=not missing and not blocked_reasons,
        requirements=provider.spec.required_modules,
        missing=missing,
        warnings=tuple(warnings),
        blocked_reasons=tuple(blocked_reasons),
        minimum_vram_gb=provider.spec.minimum_vram_gb,
        recommended_vram_gb=provider.spec.recommended_vram_gb,
        maximum_vram_gb=maximum_vram,
        runtime_versions={str(key): str(item) for key, item in versions.items()},
    )


def environment_report(name: str, project_root: Path) -> EnvironmentReport:
    return provider_environment_report(get_provider(name, project_root))


def write_environment_report(names: list[str], project_root: Path, output: Path) -> dict[str, Any]:
    reports = [environment_report(name, project_root).to_dict() for name in names]
    value = {
        "schemaVersion": 1,
        "platform": platform.platform(),
        "cudaVisibleDevices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "providers": reports,
    }
    write_json(output, value)
    return value


def build_generation_command(request: GenerationRequest, project_root: Path) -> list[str]:
    provider = get_provider(request.provider, project_root)
    prepared = provider.prepare_request(request)
    return provider.generation_command(prepared, project_root)
