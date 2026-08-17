#!/usr/bin/env python3
"""Text-guided 2D segmentation with automatic SAGS click-point output.

The primary engine is LangSAM.  ``--engine auto`` falls back to the project's
GroundingDINO + SAM entry point when LangSAM weights or its optional runtime
are unavailable.  The generated ``points.json`` can be consumed as positive
clicks by the existing SAGS workflow.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]


_CN_ALIASES = {
    "消防车": "fire engine",
    "救护车": "ambulance",
    "直升机": "helicopter",
    "邮箱": "mailbox",
    "邮筒": "mailbox",
    "墙": "wall",
    "海报": "poster",
    "灯箱": "lightbox",
    "路灯": "street lamp",
    "汽车": "car",
    "车辆": "vehicle",
    "人": "person",
    "椅子": "chair",
    "桌子": "table",
    "植物": "plant",
    "树": "tree",
}


def rewrite_task_prompt(task_prompt: str) -> tuple[str, str]:
    """Turn a task sentence into a short detector phrase.

    This is intentionally deterministic: it does not silently spend an LLM
    request for every image.  A caller can pass ``--prompt`` to bypass it.
    The original task and rewritten phrase are both recorded in the manifest.
    """
    original = " ".join(task_prompt.strip().split())
    if not original:
        raise ValueError("task prompt 为空")
    value = original
    for source, target in _CN_ALIASES.items():
        value = value.replace(source, target)
    # Chinese task descriptions commonly put the target after 一个/一只/一辆.
    cn_match = re.search(r"(?:一个|一只|一辆|一盏|一张|一件)\s*([^，。；,.;]+)", value)
    if cn_match:
        value = cn_match.group(1)
    # Remove action and spatial scaffolding that is not useful to GroundingDINO.
    value = re.sub(
        r"\b(?:insert|add|place|put|generate|create|edit|替换|插入|添加|放置|生成|创建)\b",
        " ",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"\b(?:into|onto|on|near|beside|behind|in front of|at)\b", " ", value, flags=re.I)
    value = re.sub(r"\s+", " ", value).strip(" ,.;，。；")
    if not value:
        value = original
    return original, value


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _normalise_mask(mask: Any, height: int, width: int) -> np.ndarray:
    arr = _as_numpy(mask)
    while arr.ndim > 2:
        arr = arr[0]
    if arr.shape != (height, width):
        arr = cv2.resize(arr.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
    return arr > 0.5


def _point_from_mask(mask: np.ndarray) -> tuple[int, int]:
    binary = mask.astype(np.uint8)
    if not np.any(binary):
        return 0, 0
    # The maximum distance point is more stable than a box center for thin or
    # concave objects and is a good positive prompt for SAM.
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    if float(distance.max()) > 0:
        y, x = np.unravel_index(int(distance.argmax()), distance.shape)
        return int(x), int(y)
    ys, xs = np.where(binary > 0)
    return int(round(float(xs.mean()))), int(round(float(ys.mean())))


def _points_from_mask(mask: np.ndarray, count: int) -> list[tuple[int, int]]:
    """Choose dispersed, interior positive prompts from one target mask."""
    binary = mask.astype(np.uint8)
    if not np.any(binary):
        return []
    if count <= 1:
        return [_point_from_mask(mask)]

    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    if float(distance.max()) <= 0:
        return [_point_from_mask(mask)]

    height, width = binary.shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    selected: list[tuple[int, int]] = []
    nearest_squared = np.full((height, width), np.inf, dtype=np.float32)
    ys, xs = np.where(binary > 0)
    diagonal = float(np.hypot(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))
    minimum_spacing = max(4.0, diagonal * 0.08)

    for _ in range(count):
        if not selected:
            score = distance
        else:
            last_x, last_y = selected[-1]
            nearest_squared = np.minimum(nearest_squared, (xx - last_x) ** 2 + (yy - last_y) ** 2)
            # Distance-to-boundary keeps prompts safely inside the mask while
            # farthest-point spacing spreads them across distinct body parts.
            score = distance * np.sqrt(nearest_squared)
        score = np.where(binary > 0, score, -1.0)
        y, x = np.unravel_index(int(score.argmax()), score.shape)
        if selected:
            nearest = min(float(np.hypot(x - px, y - py)) for px, py in selected)
            if nearest < minimum_spacing:
                break
        selected.append((int(x), int(y)))
    return selected


def _load_langsam(args: argparse.Namespace):
    from lang_sam import LangSAM

    return LangSAM(sam_type=args.sam_type)


def _run_langsam(args: argparse.Namespace, image: Image.Image, prompt: str):
    model = _load_langsam(args)
    results = model.predict(
        [image],
        [prompt],
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )
    return results[0] if results else {}


def _run_legacy(args: argparse.Namespace, output_dir: Path, prompt: str) -> dict[str, Any]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "segment_image.py"),
        "--input",
        str(args.input),
        "--prompt",
        prompt,
        "--output-dir",
        str(output_dir),
        "--box-threshold",
        str(args.box_threshold),
        "--text-threshold",
        str(args.text_threshold),
        "--device",
        args.device,
    ]
    subprocess.run(command, check=True)
    detections_path = output_dir / "detections.json"
    detections = json.loads(detections_path.read_text(encoding="utf-8")) if detections_path.exists() else []
    image = np.asarray(Image.open(args.input).convert("RGB"))
    mask = np.asarray(Image.open(output_dir / "mask.png").convert("L")) > 0
    points = [_point_from_mask(mask)] if np.any(mask) else []
    return {"detections": detections, "masks": [mask], "boxes": [], "points": points, "engine": "legacy"}


def _prompt_slug(prompt: str, index: int) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "_", prompt.strip()).strip("_")
    return f"{index:02d}_{value or 'target'}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LangSAM 文本分割并生成 SAGS 正点击点")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--prompt",
        action="append",
        help="已整理好的英文检测短语；可重复，多个结果会取并集",
    )
    group.add_argument("--task-prompt", help="原始任务描述；使用确定性规则改写")
    parser.add_argument("--engine", choices=("langsam", "legacy", "auto"), default="auto")
    # The installed LangSAM checkout uses the SAM 2.1 key (the upstream
    # package's constructor default is stale and would raise a KeyError).
    parser.add_argument("--sam-type", default="sam2.1_hiera_small")
    parser.add_argument("--box-threshold", type=float, default=0.30)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument(
        "--points-per-mask",
        type=int,
        default=4,
        help="每个检测 mask 生成的分散内部正点击点数；人物等多部件目标建议 3-5",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"输入图片不存在: {args.input}")
    if args.points_per_mask < 1:
        raise SystemExit("--points-per-mask 必须大于 0")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.prompt:
        detector_prompts = [value.strip() for value in args.prompt if value.strip()]
        if not detector_prompts:
            raise SystemExit("--prompt 不能为空")
        original_prompt: str | list[str] = detector_prompts[0] if len(detector_prompts) == 1 else detector_prompts
        rewrite_method = "explicit"
    else:
        original_prompt, detector_prompt = rewrite_task_prompt(args.task_prompt)
        detector_prompts = [detector_prompt]
        rewrite_method = "heuristic"

    image = Image.open(args.input).convert("RGB")
    image_rgb = np.asarray(image)
    height, width = image_rgb.shape[:2]
    engine_errors: list[str] = []
    prompt_results: list[tuple[str, dict[str, Any]]] = []
    multi_prompt = len(detector_prompts) > 1
    for index, detector_prompt in enumerate(detector_prompts):
        result: dict[str, Any]
        part_output = args.output_dir / "parts" / _prompt_slug(detector_prompt, index) if multi_prompt else args.output_dir
        if args.engine in ("langsam", "auto"):
            try:
                raw = _run_langsam(args, image, detector_prompt)
                boxes = _as_numpy(raw.get("boxes", []))
                scores = _as_numpy(raw.get("scores", []))
                mask_values = raw.get("masks", [])
                prompt_masks = [_normalise_mask(item, height, width) for item in mask_values]
                if not prompt_masks:
                    raise RuntimeError("LangSAM 没有返回 mask")
                result = {"boxes": boxes, "scores": scores, "masks": prompt_masks, "engine": "langsam"}
            except Exception as exc:
                engine_errors.append(f"{detector_prompt}: {type(exc).__name__}: {exc}")
                if args.engine == "langsam":
                    raise
                result = _run_legacy(args, part_output, detector_prompt)
        else:
            result = _run_legacy(args, part_output, detector_prompt)
        prompt_results.append((detector_prompt, result))

    masks = [mask for _, result in prompt_results for mask in result.get("masks", [])]
    if not masks or not any(np.any(mask) for mask in masks):
        raise SystemExit(f"没有检测到目标，提示词: {detector_prompts!r}")
    merged = np.any(np.stack(masks), axis=0).astype(np.uint8) * 255
    Image.fromarray(merged, mode="L").save(args.output_dir / "mask.png")
    Image.fromarray(np.dstack([image_rgb, merged]), mode="RGBA").save(args.output_dir / "cutout.png")

    detections = []
    for index, mask in enumerate(masks):
        detections.append({"index": index, "prompt": None, "box_xyxy": None, "score": None,
                           "mask_pixels": int(np.count_nonzero(mask))})
    points = []
    annotated = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    detection_index = 0
    for detector_prompt, result in prompt_results:
        result_masks = result.get("masks", [])
        boxes = result.get("boxes", [])
        scores = result.get("scores", [])
        boxes_np = _as_numpy(boxes) if len(boxes) else np.empty((0, 4))
        scores_np = _as_numpy(scores).reshape(-1) if len(scores) else np.empty((0,))
        for local_index, mask in enumerate(result_masks):
            mask_points = _points_from_mask(mask, args.points_per_mask)
            box = boxes_np[local_index].tolist() if local_index < len(boxes_np) else None
            score = float(scores_np[local_index]) if local_index < len(scores_np) else None
            detections[detection_index].update({
                "prompt": detector_prompt,
                "box_xyxy": box,
                "score": score,
                "point_count": len(mask_points),
            })
            for point_index, (x, y) in enumerate(mask_points):
                points.append({"x": x, "y": y, "label": 1, "detection_index": detection_index,
                               "point_index": point_index, "prompt": detector_prompt})
                cv2.circle(annotated, (x, y), 7, (0, 255, 255), -1)
                cv2.putText(annotated, f"{detection_index + 1}.{point_index + 1}", (x + 9, y + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
            if local_index < len(boxes_np):
                x1, y1, x2, y2 = np.round(boxes_np[local_index]).astype(int).tolist()
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            detection_index += 1
    cv2.imwrite(str(args.output_dir / "annotated.png"), annotated)
    (args.output_dir / "detections.json").write_text(json.dumps(detections, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "points.json").write_text(json.dumps(points, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "input": str(args.input),
        "engine": prompt_results[0][1].get("engine", args.engine),
        "engines": [result.get("engine", args.engine) for _, result in prompt_results],
        "original_prompt": original_prompt,
        "detector_prompt": detector_prompts[0] if len(detector_prompts) == 1 else detector_prompts,
        "rewrite_method": rewrite_method,
        "fallback_error": engine_errors or None,
        "points_per_mask": args.points_per_mask,
        "point_count": len(points),
        "mask_pixels": int(np.count_nonzero(merged)),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("AUTO_SEGMENT_READY", manifest["engine"], len(points), int(np.count_nonzero(merged)), args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
