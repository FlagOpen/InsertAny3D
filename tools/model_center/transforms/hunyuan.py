"""Hunyuan coordinate profile adapter."""

from __future__ import annotations

from ..contracts import CoordinateContract
from ..providers.hunyuan import HunyuanProvider


def coordinate_contract() -> CoordinateContract:
    return HunyuanProvider().spec.coordinate_contract
