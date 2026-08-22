#!/usr/bin/env python3
"""Render a Gaussian PLY to the sphere/NVS debug layout.

This is the sphere counterpart to ``render_trellis_views.py``.  It lives in
the authoritative runtime tree so the upload step cannot silently retain a
different remote implementation.  Near/far are explicit provider contract
inputs rather than renderer defaults.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("SPCONV_ALGO", "native")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRELLIS_ROOT = PROJECT_ROOT / "third_party" / "TRELLIS"
sys.path.insert(0, str(TRELLIS_ROOT))

from trellis.utils.insertany3d_render_utils import (  # noqa: E402
    _extract_3dgs_result,
    _save_result,
    load_gaussian,
)
from trellis.utils.render_utils import (  # noqa: E402
    render_frames,
    yaw_pitch_r_fov_to_extrinsics_intrinsics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 Gaussian PLY 渲染为 sphere/NVS 数据目录")
    parser.add_argument("--input-ply", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--radius", type=float, default=1.5)
    parser.add_argument("--fov", type=float, default=53.1301023542)
    parser.add_argument("--near", type=float, default=0.8)
    parser.add_argument("--far", type=float, default=1.6)
    parser.add_argument("--latitudes", default="10,20,30")
    parser.add_argument("--views-per-latitude", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.radius <= 0 or args.resolution < 1 or args.near <= 0 or args.far <= args.near:
        raise SystemExit("radius/resolution/near/far 参数无效")
    latitudes = [float(value.strip()) for value in args.latitudes.split(",") if value.strip()]
    if not latitudes or args.views_per_latitude < 1:
        raise SystemExit("latitudes/views-per-latitude 参数无效")
    gaussian = load_gaussian(str(args.input_ply))
    yaws: list[float] = []
    pitches: list[float] = []
    for latitude in latitudes:
        yaws.extend(2.0 * math.pi * index / args.views_per_latitude for index in range(args.views_per_latitude))
        pitches.extend([math.radians(latitude)] * args.views_per_latitude)
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(
        yaws, pitches, args.radius, args.fov
    )
    rendered = render_frames(
        gaussian,
        extrinsics,
        intrinsics,
        {
            "resolution": args.resolution,
            "near": args.near,
            "far": args.far,
            "bg_color": (0, 0, 0),
        },
    )
    scene_path = args.output_dir / "source"
    model_path = args.output_dir / "model"
    _save_result(
        str(scene_path),
        rendered["color"],
        rendered["depth"],
        rendered["absdepth"],
        rendered["extr"],
        rendered["intr"],
        args.resolution,
    )
    _extract_3dgs_result(str(scene_path), str(model_path), str(args.input_ply), use_colmap=False)
    image_count = len(list((scene_path / "images").glob("*.png")))
    cameras_path = scene_path / "sparse" / "0" / "cameras.txt"
    points_path = model_path / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    if image_count == 0 or not cameras_path.stat().st_size or not points_path.stat().st_size:
        raise RuntimeError("3DGS sphere 渲染输出不完整")
    print("TRELLIS_3DGS_RENDER_READY", image_count, args.output_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
