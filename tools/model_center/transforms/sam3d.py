"""SAM3D coordinate profile adapter."""

from __future__ import annotations

from ..contracts import CoordinateContract
from ..providers.sam3d import Sam3DProvider


def coordinate_contract() -> CoordinateContract:
    return Sam3DProvider().spec.coordinate_contract
