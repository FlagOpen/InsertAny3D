#!/usr/bin/env python3
"""Generate a Gaussian/mesh asset from one or more edited images with TRELLIS."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRELLIS_ROOT = PROJECT_ROOT / "third_party" / "TRELLIS"
sys.path.insert(0, str(TRELLIS_ROOT))
os.environ.setdefault("SPCONV_ALGO", "native")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从编辑图片生成 TRELLIS Gaussian PLY/GLB")
    parser.add_argument("--input-image", required=True, type=Path, nargs="+")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default=os.environ.get("TRELLIS_MODEL", "microsoft/TRELLIS-image-large"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--sparse-steps", type=int)
    parser.add_argument("--slat-steps", type=int)
    parser.add_argument("--sparse-cfg", type=float)
    parser.add_argument("--slat-cfg", type=float)
    parser.add_argument("--multi-mode", choices=("stochastic", "multidiffusion"), default="stochastic")
    parser.add_argument("--no-glb", action="store_true")
    parser.add_argument("--require-glb", action="store_true")
    return parser.parse_args()


def _sampler_params(args: argparse.Namespace) -> tuple[dict, dict]:
    sparse, slat = {}, {}
    if args.sparse_steps is not None:
        sparse["steps"] = args.sparse_steps
    if args.sparse_cfg is not None:
        sparse["cfg_strength"] = args.sparse_cfg
    if args.slat_steps is not None:
        slat["steps"] = args.slat_steps
    if args.slat_cfg is not None:
        slat["cfg_strength"] = args.slat_cfg
    return sparse, slat


def main() -> int:
    args = parse_args()
    for path in args.input_image:
        if not path.is_file():
            raise SystemExit(f"输入图片不存在: {path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.utils import postprocessing_utils

    print("TRELLIS_LOADING", args.model, flush=True)
    pipeline = TrellisImageTo3DPipeline.from_pretrained(args.model)
    pipeline.cuda()
    images = []
    for index, path in enumerate(args.input_image):
        processed = pipeline.preprocess_image(Image.open(path))
        processed_path = args.output_dir / f"processed_{index:02d}.png"
        processed.save(processed_path)
        images.append(processed)

    sparse_params, slat_params = _sampler_params(args)
    common = {
        "seed": args.seed,
        "formats": ["gaussian", "mesh"],
        "preprocess_image": False,
        "sparse_structure_sampler_params": sparse_params,
        "slat_sampler_params": slat_params,
    }
    print("TRELLIS_GENERATING", len(images), flush=True)
    if len(images) == 1:
        outputs = pipeline.run(images[0], **common)
    else:
        outputs = pipeline.run_multi_image(images, mode=args.multi_mode, **common)

    ply_path = args.output_dir / "sample.ply"
    outputs["gaussian"][0].save_ply(str(ply_path))
    glb_path = args.output_dir / "sample.glb"
    glb_error = None
    if not args.no_glb:
        try:
            glb = postprocessing_utils.to_glb(
                outputs["gaussian"][0], outputs["mesh"][0], simplify=0.95, texture_size=1024, verbose=False
            )
            glb.export(str(glb_path))
        except Exception as exc:  # GLB extraction is optional for the matching path.
            glb_error = f"{type(exc).__name__}: {exc}"
            if args.require_glb:
                raise
            print("TRELLIS_GLB_WARNING", glb_error, flush=True)

    metadata = {
        "model": args.model,
        "seed": args.seed,
        "input_images": [str(p) for p in args.input_image],
        "sample_ply": str(ply_path),
        "sample_glb": str(glb_path) if glb_path.exists() else None,
        "glb_error": glb_error,
        "sparse_params": sparse_params,
        "slat_params": slat_params,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    del outputs, pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("TRELLIS_ASSET_READY", ply_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
