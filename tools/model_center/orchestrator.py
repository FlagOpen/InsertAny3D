"""Small provider-neutral orchestration facade used by tests and callers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .config import ProviderProfile, cache_key, resolve_profile
from .contracts import CoordinateContract, GeneratedAsset, GenerationRequest
from .providers.base import ModelProvider
from .registry import get_provider, provider_environment_report
from .transforms.base import apply_contract


@dataclass(frozen=True)
class ProviderPlan:
    request: GenerationRequest
    profile: ProviderProfile
    provider: ModelProvider
    command: tuple[str, ...]
    candidate_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.to_dict(),
            "profile": self.profile.to_dict(),
            "provider": self.provider.spec.name,
            "command": list(self.command),
            "candidateId": self.candidate_id,
        }


class ModelOrchestrator:
    def __init__(self, project_root: Path):
        self.project_root = project_root.resolve()

    def plan(
        self,
        request: GenerationRequest,
        profile: str | Path | Mapping[str, Any] = "default",
        overrides: Mapping[str, Any] | None = None,
        model_revision: str | None = None,
        weight_fingerprint: str | None = None,
    ) -> ProviderPlan:
        resolved = resolve_profile(request.provider, profile, overrides)
        provider = get_provider(request.provider, self.project_root)
        merged = request.__class__(**{**request.__dict__, "options": resolved.merged_options()})
        command = provider.generation_command(merged, self.project_root)
        candidate = cache_key(
            merged.input_image,
            merged.provider,
            model_revision,
            weight_fingerprint,
            resolved.name,
            merged.seed,
            merged.input_mask,
            merged.options,
        )
        return ProviderPlan(merged, resolved, provider, tuple(command), candidate)

    def environment(self, provider: str):
        return provider_environment_report(get_provider(provider, self.project_root))

    def canonicalize(
        self,
        input_ply: Path,
        output_ply: Path,
        contract: CoordinateContract,
        metadata_path: Path | None = None,
        **asset_metadata: Any,
    ) -> GeneratedAsset:
        result = apply_contract(input_ply, output_ply, contract, metadata_path)
        return GeneratedAsset(
            gaussian_ply=output_ply,
            provider=str(asset_metadata.get("provider", "unknown")),
            model=asset_metadata.get("model"),
            representation=str(asset_metadata.get("representation", "native_gaussian")),
            coordinate_contract=contract,
            metadata_path=metadata_path,
            source_mesh=asset_metadata.get("source_mesh"),
            weight_fingerprint=asset_metadata.get("weight_fingerprint"),
            input_mask=asset_metadata.get("input_mask"),
            input_image_sha256=asset_metadata.get("input_image_sha256"),
            model_revision=asset_metadata.get("model_revision"),
            seed=asset_metadata.get("seed"),
            converter_metadata=asset_metadata.get("converter_metadata"),
        )
