"""Common Gaussian transform entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..contracts import CoordinateContract
from .gaussian_ply import transform_gaussian_ply


def apply_contract(
    input_ply: Path,
    output_ply: Path,
    contract: CoordinateContract,
    metadata_path: Path | None = None,
) -> dict[str, Any]:
    return transform_gaussian_ply(input_ply, output_ply, contract, metadata_path)
