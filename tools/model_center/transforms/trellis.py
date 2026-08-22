"""TRELLIS coordinate profile adapter."""

from __future__ import annotations

from ..providers.trellis import TrellisProvider
from ..contracts import CoordinateContract


def coordinate_contract() -> CoordinateContract:
    return TrellisProvider().spec.coordinate_contract
