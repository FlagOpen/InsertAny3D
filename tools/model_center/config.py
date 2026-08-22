"""Provider/profile configuration and deterministic cache identity helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .contracts import sha256_file


@dataclass(frozen=True)
class SegmentationProfile:
    engine: str = "legacy"
    input_mask_policy: str = "optional"
    fallback: str = "none"

    def __post_init__(self) -> None:
        if not self.engine.strip():
            raise ValueError("segmentation engine must not be empty")
        if self.input_mask_policy not in {"required", "optional", "provider_local"}:
            raise ValueError("input_mask_policy must be required, optional, or provider_local")

    def to_dict(self) -> dict[str, str]:
        return {
            "engine": self.engine,
            "inputMaskPolicy": self.input_mask_policy,
            "fallback": self.fallback,
        }


@dataclass(frozen=True)
class ProviderProfile:
    provider: str
    name: str = "default"
    options: Mapping[str, Any] = field(default_factory=dict)
    segmentation: SegmentationProfile = field(default_factory=SegmentationProfile)

    def __post_init__(self) -> None:
        provider = self.provider.strip().lower()
        name = self.name.strip()
        if not provider:
            raise ValueError("provider must not be empty")
        if not name:
            raise ValueError("profile name must not be empty")
        if not isinstance(self.options, Mapping):
            raise ValueError("provider profile options must be an object")
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "options", dict(self.options))

    def merged_options(self, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
        result = dict(self.options)
        if overrides is not None:
            if not isinstance(overrides, Mapping):
                raise ValueError("provider option overrides must be an object")
            result.update(overrides)
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "profile": self.name,
            "options": dict(self.options),
            "segmentation": self.segmentation.to_dict(),
        }


def _provider_option_defaults(provider: str) -> dict[str, Any]:
    """Return quality-preserving runtime defaults for each provider."""

    if provider == "sam3d":
        # The normal decoder remains the production representation.  The
        # adapter only avoids loading outputs that the integration contract
        # never consumes; downsample=1 is the upstream default.
        return {
            "decoder": "gaussian",
            "downsample_ss_dist": 1,
            "load_unused_decoders": False,
            "sequential_offload": True,
            "spconv_algo": "native",
        }
    return {}


def _provider_defaults(provider: str) -> SegmentationProfile:
    if provider == "sam3d":
        return SegmentationProfile(engine="legacy", input_mask_policy="required", fallback="none")
    if provider == "hunyuan":
        return SegmentationProfile(engine="alpha", input_mask_policy="provider_local", fallback="rembg")
    return SegmentationProfile(engine="legacy", input_mask_policy="optional", fallback="none")


def resolve_profile(
    provider: str,
    profile: str | Path | Mapping[str, Any] = "default",
    overrides: Mapping[str, Any] | None = None,
) -> ProviderProfile:
    """Resolve a named or JSON profile without silently changing providers."""

    from .registry import provider_names

    provider_name = str(provider).strip().lower()
    if provider_name not in provider_names():
        raise ValueError(f"unknown model provider {provider!r}; choose one of {', '.join(provider_names())}")
    profile_name = "default"
    values: Mapping[str, Any] = {}
    if isinstance(profile, Mapping):
        values = profile
        profile_name = str(values.get("profile", values.get("name", "default")))
    elif isinstance(profile, Path) or (isinstance(profile, str) and Path(profile).is_file()):
        path = Path(profile)
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, Mapping):
            raise ValueError(f"profile JSON must be an object: {path}")
        values = loaded
        profile_name = str(values.get("profile", values.get("name", path.stem)))
    else:
        profile_name = str(profile).strip() or "default"
    configured_provider = values.get("provider")
    if configured_provider is not None and str(configured_provider).strip().lower() != provider_name:
        raise ValueError(f"profile provider {configured_provider!r} does not match {provider_name!r}")
    options = values.get("options", {})
    if not isinstance(options, Mapping):
        raise ValueError("profile options must be an object")
    options = {**_provider_option_defaults(provider_name), **dict(options)}
    if overrides is not None:
        options = {**dict(options), **dict(overrides)}
    segmentation_value = values.get("segmentation", {})
    if not isinstance(segmentation_value, Mapping):
        raise ValueError("profile segmentation must be an object")
    defaults = _provider_defaults(provider_name)
    segmentation = SegmentationProfile(
        engine=str(segmentation_value.get("engine", defaults.engine)),
        input_mask_policy=str(segmentation_value.get("inputMaskPolicy", segmentation_value.get("input_mask_policy", defaults.input_mask_policy))),
        fallback=str(segmentation_value.get("fallback", defaults.fallback)),
    )
    return ProviderProfile(provider_name, profile_name, options, segmentation)


def cache_key(
    input_image: Path,
    provider: str,
    model_revision: str | None,
    weight_fingerprint: str | None,
    profile: str,
    seed: int,
    input_mask: Path | None = None,
    options: Mapping[str, Any] | None = None,
) -> str:
    """Return the stable candidate identity required by Phase 4."""

    payload = {
        "inputSha256": sha256_file(input_image) if input_image.is_file() else None,
        "inputMaskSha256": sha256_file(input_mask) if input_mask and input_mask.is_file() else None,
        "provider": provider,
        "modelRevision": model_revision,
        "weightFingerprint": weight_fingerprint,
        "profile": profile,
        "seed": int(seed),
        "options": dict(options or {}),
    }
    return "candidate-" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()[:16]
