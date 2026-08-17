#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.ops import box_convert


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAGS_ROOT = PROJECT_ROOT / "third_party" / "SAGS"
GROUNDING_ROOT = (
    SAGS_ROOT / "gaussiansplatting" / "dependencies" / "GroundingDINO"
)
SAM_ROOT = SAGS_ROOT / "gaussiansplatting" / "dependencies" / "sam_ckpt"


def parse_args():
    parser = argparse.ArgumentParser(description="使用 GroundingDINO 和 SAM 分割图片")
    parser.add_argument("--input", required=True, type=Path, help="输入图片")
    parser.add_argument("--prompt", required=True, help="英文文本提示")
    parser.add_argument("--output-dir", required=True, type=Path, help="输出目录")
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--grounding-config",
        type=Path,
        default=GROUNDING_ROOT
        / "groundingdino/config/GroundingDINO_SwinT_OGC.py",
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


def main():
    args = parse_args()

    from groundingdino.util.inference import load_image, load_model, predict
    from segment_anything import SamPredictor, sam_model_registry

    image_rgb, image_tensor = load_image(str(args.input))
    grounding_model = load_model(
        str(args.grounding_config), str(args.grounding_weights), args.device
    )
    boxes, scores, phrases = predict(
        grounding_model,
        image_tensor,
        args.prompt,
        args.box_threshold,
        args.text_threshold,
        device=args.device,
    )
    if len(boxes) == 0:
        raise RuntimeError(f"没有检测到与提示词 {args.prompt!r} 对应的目标")

    height, width = image_rgb.shape[:2]
    boxes_xyxy = box_convert(
        boxes * torch.tensor([width, height, width, height]),
        in_fmt="cxcywh",
        out_fmt="xyxy",
    ).numpy()

    sam = sam_model_registry["vit_h"](checkpoint=str(args.sam_weights)).to(args.device)
    predictor = SamPredictor(sam)
    predictor.set_image(image_rgb)
    masks, mask_scores, _ = predictor.predict(
        box=boxes_xyxy,
        multimask_output=False,
    )
    merged_mask = np.any(masks, axis=0).astype(np.uint8) * 255

    args.output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(merged_mask, mode="L").save(args.output_dir / "mask.png")

    cutout = np.dstack([image_rgb, merged_mask])
    Image.fromarray(cutout, mode="RGBA").save(args.output_dir / "cutout.png")

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
                "box_xyxy": [float(value) for value in box],
                "grounding_score": float(score),
                "mask_score": float(mask_score),
            }
        )
    cv2.imwrite(str(args.output_dir / "annotated.png"), annotated)
    (args.output_dir / "detections.json").write_text(
        json.dumps(detections, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        "SAGS_SEGMENTATION_READY",
        len(detections),
        int(np.count_nonzero(merged_mask)),
        args.output_dir,
    )


if __name__ == "__main__":
    main()
