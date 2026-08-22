"""Provider-neutral model integration center for the server workflow.

The package intentionally lives in ``codex_remote_tools`` because that tree is
the authoritative payload copied to the remote ``tools`` directory.  It does
not import any third-party model at module import time; provider runtimes are
loaded lazily by :mod:`model_center.registry`.
"""

from .contracts import (
    CoordinateContract,
    EnvironmentReport,
    GenerationRequest,
    GeneratedAsset,
    MaskArtifact,
    RenderRequest,
)

__all__ = [
    "CoordinateContract",
    "EnvironmentReport",
    "GenerationRequest",
    "GeneratedAsset",
    "MaskArtifact",
    "RenderRequest",
]
