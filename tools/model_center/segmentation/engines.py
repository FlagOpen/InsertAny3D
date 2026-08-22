"""Named segmentation policies used by the provider-neutral mask manager."""

from __future__ import annotations

from typing import Final

ENGINE_NAMES: Final[frozenset[str]] = frozenset(
    {"provided_mask", "alpha", "legacy", "langsam", "grounded_sam", "rembg"}
)


def validate_engine(name: str) -> str:
    value = str(name or "").strip().lower()
    if value not in ENGINE_NAMES:
        raise ValueError(f"unknown segmentation engine {name!r}; choose one of {', '.join(sorted(ENGINE_NAMES))}")
    return value


def describe_engine(name: str) -> dict[str, str]:
    value = validate_engine(name)
    if value == "provided_mask":
        return {"engine": value, "source": "caller", "fallback": "none"}
    if value == "alpha":
        return {"engine": value, "source": "input_alpha_channel", "fallback": "none"}
    if value == "rembg":
        return {"engine": value, "source": "provider_local_background_removal", "fallback": "none"}
    return {"engine": value, "source": "tools/auto_segment.py", "fallback": "none"}
