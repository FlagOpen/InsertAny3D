#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path


os.environ.setdefault("SPCONV_ALGO", "native")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRELLIS_ROOT = PROJECT_ROOT / "third_party" / "TRELLIS"
sys.path.insert(0, str(TRELLIS_ROOT))

from trellis.utils.insertany3d_render_utils import (
    _extract_3dgs_result,
    _save_result,
    load_gaussian,
)
from trellis.utils.render_utils import render_sphere


def parse_args():
    parser = argparse.ArgumentParser(description="将 TRELLIS PLY 渲染为 3DGS 数据目录")
    parser.add_argument("--input-ply", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--radius", type=float, default=1.5)
    parser.add_argument("--fov", type=float, default=53.1301023542)
    parser.add_argument("--latitudes", default="10,20,30")
    parser.add_argument("--views-per-latitude", type=int, default=30)
    return parser.parse_args()


def main():
    args = parse_args()
    latitudes = [float(value) for value in args.latitudes.split(",")]
    scene_path = args.output_dir / "source"
    model_path = args.output_dir / "model"

    gaussian = load_gaussian(str(args.input_ply))
    rendered = render_sphere(
        sample=gaussian,
        r=args.radius,
        latitudes_deg=latitudes,
        fov=args.fov,
        resolution=args.resolution,
        nviews_one_lat=args.views_per_latitude,
    )
    _save_result(
        str(scene_path),
        rendered["color"],
        rendered["depth"],
        rendered["absdepth"],
        rendered["extr"],
        rendered["intr"],
        args.resolution,
    )
    _extract_3dgs_result(
        str(scene_path), str(model_path), str(args.input_ply), use_colmap=False
    )

    image_count = len(list((scene_path / "images").glob("*.png")))
    cameras_path = scene_path / "sparse" / "0" / "cameras.txt"
    points_path = model_path / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    if image_count == 0 or not cameras_path.stat().st_size or not points_path.stat().st_size:
        raise RuntimeError("3DGS 渲染输出不完整")
    print("TRELLIS_3DGS_RENDER_READY", image_count, args.output_dir)


if __name__ == "__main__":
    main()
