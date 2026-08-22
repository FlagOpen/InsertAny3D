#!/usr/bin/env python3
"""Batch GroundingDINO + SAM anchor masks for multiview pose estimation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.ops import box_convert


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAGS_ROOT = PROJECT_ROOT / "third_party" / "SAGS"
GROUNDING_ROOT = SAGS_ROOT / "gaussiansplatting" / "dependencies" / "GroundingDINO"
SAM_ROOT = SAGS_ROOT / "gaussiansplatting" / "dependencies" / "sam_ckpt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="一次加载模型，分割三视图两侧的锚点实例")
    parser.add_argument(
        "--view",
        action="append",
        nargs=3,
        required=True,
        metavar=("NAME", "SCENE_IMAGE", "GENERATED_IMAGE"),
    )
    parser.add_argument("--prompt", required=True, help="简短英文锚点类别，例如 tractor")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--grounding-config",
        type=Path,
        default=GROUNDING_ROOT / "groundingdino/config/GroundingDINO_SwinT_OGC.py",
    )
    parser.add_argument(
        "--grounding-weights",
        type=Path,
        default=GROUNDING_ROOT / "weights/groundingdino_swint_ogc.pth",
    )
    parser.add_argument(
        "--sam-weights",
        type=Path,
        default=SAM_ROOT / "sam_vit_h_4b8939.pth",
    )
    return parser.parse_args()


def _segment(
    image_path: Path,
    output_dir: Path,
    prompt: str,
    grounding_model: Any,
    predictor: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from groundingdino.util.inference import load_image, predict

    started = time.time()
    image_rgb, image_tensor = load_image(str(image_path))
    boxes, scores, phrases = predict(
        grounding_model,
        image_tensor,
        prompt,
        args.box_threshold,
        args.text_threshold,
        device=args.device,
    )
    if len(boxes) == 0:
        raise RuntimeError(f"{image_path} 没有检测到锚点 {prompt!r}")

    height, width = image_rgb.shape[:2]
    boxes_xyxy = box_convert(
        boxes * torch.tensor([width, height, width, height]),
        in_fmt="cxcywh",
        out_fmt="xyxy",
    ).numpy()
    predictor.set_image(image_rgb)
    boxes_tensor = torch.as_tensor(boxes_xyxy, dtype=torch.float32, device=args.device)
    transformed = predictor.transform.apply_boxes_torch(boxes_tensor, image_rgb.shape[:2])
    masks_tensor, mask_scores_tensor, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed,
        multimask_output=False,
    )
    masks = masks_tensor[:, 0].detach().cpu().numpy()
    mask_scores = mask_scores_tensor[:, 0].detach().cpu().numpy()
    merged = np.any(masks, axis=0).astype(np.uint8) * 255

    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(merged, mode="L").save(output_dir / "mask.png")
    Image.fromarray(np.dstack([image_rgb, merged]), mode="RGBA").save(output_dir / "cutout.png")
    annotated = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    detections = []
    for box, phrase, score, mask_score in zip(
        boxes_xyxy, phrases, scores.tolist(), mask_scores.tolist()
    ):
        x1, y1, x2, y2 = np.round(box).astype(int)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"{phrase} {score:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        detections.append(
            {
                "phrase": phrase,
                "boxXyxy": [float(value) for value in box],
                "groundingScore": float(score),
                "maskScore": float(mask_score),
            }
        )
    cv2.imwrite(str(output_dir / "annotated.png"), annotated)
    (output_dir / "detections.json").write_text(
        json.dumps(detections, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "input": str(image_path.resolve()),
        "mask": str((output_dir / "mask.png").resolve()),
        "detections": detections,
        "maskPixels": int(np.count_nonzero(merged)),
        "seconds": time.time() - started,
    }


def main() -> int:
    args = parse_args()
    if not args.prompt.strip():
        raise SystemExit("--prompt 不能为空")
    if args.box_threshold <= 0 or args.text_threshold <= 0:
        raise SystemExit("检测阈值必须为正数")

    from groundingdino.util.inference import load_model
    from segment_anything import SamPredictor, sam_model_registry

    grounding_model = load_model(
        str(args.grounding_config), str(args.grounding_weights), args.device
    )
    sam = sam_model_registry["vit_h"](checkpoint=str(args.sam_weights)).to(args.device)
    predictor = SamPredictor(sam)
    manifest = {
        "schemaVersion": 1,
        "prompt": args.prompt.strip(),
        "boxThreshold": args.box_threshold,
        "textThreshold": args.text_threshold,
        "views": [],
    }
    for name, scene_value, generated_value in args.view:
        scene_path = Path(scene_value)
        generated_path = Path(generated_value)
        if not scene_path.is_file() or not generated_path.is_file():
            raise FileNotFoundError(f"{name} 锚点分割输入不存在")
        item = {"name": name}
        item["scene"] = _segment(
            scene_path, args.output_dir / name / "scene", args.prompt.strip(),
            grounding_model, predictor, args,
        )
        item["generated"] = _segment(
            generated_path, args.output_dir / name / "generated", args.prompt.strip(),
            grounding_model, predictor, args,
        )
        manifest["views"].append(item)
        print("ANCHOR_MASK_VIEW_READY", name, flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("ANCHOR_MASKS_READY", manifest_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
