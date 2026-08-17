#!/usr/bin/env python3
"""Run the upstream GIM matcher on arbitrary image paths.

The upstream demo is kept untouched and uses fixed example paths.  This
adapter reuses its model/preprocessing/visualisation helpers while exposing a
stable CLI and machine-readable match JSON for the InsertAny3D pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
GIM_ROOT = PROJECT_ROOT / "third_party" / "gim"
CALLER_CWD = Path.cwd()
sys.path.insert(0, str(GIM_ROOT))
os.chdir(GIM_ROOT)

import cv2
import numpy as np
import torch

import demo as gim_demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GIM 任意两张图片匹配")
    parser.add_argument("--image0", required=True, type=Path)
    parser.add_argument("--image1", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default="gim_roma", choices=("gim_dkm", "gim_roma", "gim_loftr", "gim_lightglue"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--max-matches", type=int, default=5000)
    parser.add_argument("--resize-max", type=int)
    parser.add_argument("--ransac-threshold", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--mask0", type=Path, help="image0 的允许区域二值 mask；非零区域有效")
    parser.add_argument("--mask1", type=Path, help="image1 的允许区域二值 mask；非零区域有效")
    parser.add_argument("--roi0", nargs=3, type=float, metavar=("CX", "CY", "RADIUS"), help="image0 原图像素坐标中的圆形 ROI")
    parser.add_argument("--roi1", nargs=3, type=float, metavar=("CX", "CY", "RADIUS"), help="image1 原图像素坐标中的圆形 ROI")
    parser.add_argument(
        "--max-aligned-displacement",
        type=float,
        default=0.0,
        help="相机已对齐时允许的最大对应点像素位移；0 表示关闭",
    )
    parser.add_argument("--auto-mask1-nonblack", action="store_true", help="自动排除 image1 的黑色/透明背景")
    parser.add_argument("--foreground-threshold", type=int, default=8, help="自动前景 mask 的 RGB 最大值阈值")
    parser.add_argument("--allow-empty", action="store_true", help="过滤后无匹配时写出零匹配诊断并返回成功，由多视角门禁拒绝")
    return parser.parse_args()


def _mask_at_points(path: Path, points: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"无法读取 mask: {path}")
    width, height = image_size
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    x = np.clip(np.rint(points[:, 0]).astype(np.int64), 0, width - 1)
    y = np.clip(np.rint(points[:, 1]).astype(np.int64), 0, height - 1)
    inside = (
        (points[:, 0] >= 0)
        & (points[:, 0] < width)
        & (points[:, 1] >= 0)
        & (points[:, 1] < height)
    )
    return inside & (mask[y, x] > 0)


def _roi_at_points(points: np.ndarray, roi: list[float] | None) -> np.ndarray:
    if roi is None:
        return np.ones(len(points), dtype=bool)
    cx, cy, radius = roi
    if radius <= 0:
        raise ValueError("ROI radius 必须大于 0")
    return np.square(points[:, 0] - cx) + np.square(points[:, 1] - cy) <= radius * radius


def _aligned_displacement_at_points(
    points0: np.ndarray,
    image0_size: tuple[int, int],
    points1: np.ndarray,
    image1_size: tuple[int, int],
    maximum: float,
) -> np.ndarray:
    """Keep near-diagonal matches after both cameras share one projection."""
    if maximum <= 0:
        return np.ones(len(points0), dtype=bool)
    width0, height0 = image0_size
    width1, height1 = image1_size
    if min(width0, height0, width1, height1) <= 0:
        raise ValueError("图像尺寸无效，不能计算对齐位移")
    points1_in_image0 = points1 * np.array([width0 / width1, height0 / height1], dtype=np.float64)
    displacement_squared = np.sum(np.square(points0 - points1_in_image0), axis=1)
    return displacement_squared <= maximum * maximum


def _nonblack_at_points(path: Path, points: np.ndarray, image_size: tuple[int, int], threshold: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    width, height = image_size
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    if image.ndim == 2:
        foreground = image > threshold
    else:
        foreground = np.max(image[:, :, :3], axis=2) > threshold
        if image.shape[2] == 4:
            foreground &= image[:, :, 3] > threshold
    x = np.clip(np.rint(points[:, 0]).astype(np.int64), 0, width - 1)
    y = np.clip(np.rint(points[:, 1]).astype(np.int64), 0, height - 1)
    inside = (
        (points[:, 0] >= 0)
        & (points[:, 0] < width)
        & (points[:, 1] >= 0)
        & (points[:, 1] < height)
    )
    return inside & foreground[y, x]


def load_model(model_name: str, checkpoint: Path | None, device: str):
    detector = None
    if model_name == "gim_dkm":
        model = gim_demo.DKMv3(weights=None, h=672, w=896)
        ckpt_name = "gim_dkm_100h.ckpt"
    elif model_name == "gim_roma":
        model = gim_demo.RoMa(img_size=[672])
        ckpt_name = "gim_roma_100h.ckpt"
    elif model_name == "gim_loftr":
        model = gim_demo.LoFTR(gim_demo.lower_config(gim_demo.get_cfg_defaults())["loftr"])
        ckpt_name = "gim_loftr_50h.ckpt"
    else:
        detector = gim_demo.SuperPoint({
            "max_num_keypoints": 2048,
            "force_num_keypoints": True,
            "detection_threshold": 0.0,
            "nms_radius": 3,
            "trainable": False,
        })
        model = gim_demo.LightGlue({"filter_threshold": 0.1, "flash": False, "checkpointed": True})
        ckpt_name = "gim_lightglue_100h.ckpt"
    checkpoint = checkpoint or (GIM_ROOT / "weights" / ckpt_name)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"GIM 权重不存在: {checkpoint}")
    state = torch.load(str(checkpoint), map_location="cpu")
    state = state.get("state_dict", state)
    if model_name in ("gim_dkm", "gim_roma", "gim_loftr"):
        for key in list(state.keys()):
            if model_name == "gim_dkm" and "encoder.net.fc" in key:
                state.pop(key)
                continue
            if key.startswith("model."):
                state[key.replace("model.", "", 1)] = state.pop(key)
        model.load_state_dict(state)
    else:
        detector_state = torch.load(str(checkpoint), map_location="cpu")
        detector_state = detector_state.get("state_dict", detector_state)
        for key in list(detector_state.keys()):
            if key.startswith("model."):
                detector_state.pop(key)
            elif key.startswith("superpoint."):
                detector_state[key.replace("superpoint.", "", 1)] = detector_state.pop(key)
        detector.load_state_dict(detector_state)
        state = torch.load(str(checkpoint), map_location="cpu")
        state = state.get("state_dict", state)
        for key in list(state.keys()):
            if key.startswith("superpoint."):
                state.pop(key)
            elif key.startswith("model."):
                state[key.replace("model.", "", 1)] = state.pop(key)
        model.load_state_dict(state)
    if detector is not None:
        detector = detector.eval().to(device)
    return model.eval().to(device), detector


def _fallback_side_by_side(image0: torch.Tensor, image1: torch.Tensor) -> np.ndarray:
    left = (image0[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    right = (image1[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    height = max(left.shape[0], right.shape[0])
    canvas = np.zeros((height, left.shape[1] + right.shape[1], 3), dtype=np.uint8)
    canvas[: left.shape[0], : left.shape[1]] = left
    canvas[: right.shape[0], left.shape[1] :] = right
    return canvas[..., ::-1]


def main() -> int:
    args = parse_args()
    # demo.py resolves its weights relative to the GIM checkout.  Preserve the
    # caller's meaning for relative CLI paths before doing that chdir.
    args.image0 = (CALLER_CWD / args.image0).resolve() if not args.image0.is_absolute() else args.image0
    args.image1 = (CALLER_CWD / args.image1).resolve() if not args.image1.is_absolute() else args.image1
    args.output_dir = (CALLER_CWD / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    if args.checkpoint and not args.checkpoint.is_absolute():
        args.checkpoint = (CALLER_CWD / args.checkpoint).resolve()
    for field in ("mask0", "mask1"):
        value = getattr(args, field)
        if value is not None and not value.is_absolute():
            setattr(args, field, (CALLER_CWD / value).resolve())
    if not args.image0.is_file() or not args.image1.is_file():
        raise SystemExit("GIM 输入图片不存在")
    if (args.mask0 and not args.mask0.is_file()) or (args.mask1 and not args.mask1.is_file()):
        raise SystemExit("GIM 输入 mask 不存在")
    if not 0 <= args.foreground_threshold <= 255 or args.max_aligned_displacement < 0:
        raise SystemExit("foreground-threshold 必须在 0..255，max-aligned-displacement 不能为负")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    cv2.setRNGSeed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, detector = load_model(args.model, args.checkpoint, device)

    image0_np = gim_demo.read_image(args.image0)
    image1_np = gim_demo.read_image(args.image1)
    image0, scale0 = gim_demo.preprocess(image0_np, resize_max=args.resize_max)
    image1, scale1 = gim_demo.preprocess(image1_np, resize_max=args.resize_max)
    image0 = image0.to(device)[None]
    image1 = image1.to(device)[None]
    data = {"color0": image0, "color1": image1, "image0": image0, "image1": image1}
    b_ids = None
    mconf = None
    kpts0 = None
    kpts1 = None
    points_are_original = False

    if args.model in ("gim_dkm", "gim_roma"):
        width, height = (672, 896) if args.model == "gim_dkm" else (672, 672)
        orig_width0, orig_height0, pad_left0, pad_right0, pad_top0, pad_bottom0 = gim_demo.get_padding_size(image0, width, height)
        orig_width1, orig_height1, pad_left1, pad_right1, pad_top1, pad_bottom1 = gim_demo.get_padding_size(image1, width, height)
        image0_p = torch.nn.functional.pad(image0, (pad_left0, pad_right0, pad_top0, pad_bottom0))
        image1_p = torch.nn.functional.pad(image1, (pad_left1, pad_right1, pad_top1, pad_bottom1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dense_matches, dense_certainty = model.match(image0_p, image1_p)
            sparse_matches, mconf = model.sample(dense_matches, dense_certainty, args.max_matches)
        h0, w0 = image0_p.shape[-2:]
        h1, w1 = image1_p.shape[-2:]
        kpts0 = torch.stack((w0 * (sparse_matches[:, 0] + 1) / 2, h0 * (sparse_matches[:, 1] + 1) / 2), dim=-1)
        kpts1 = torch.stack((w1 * (sparse_matches[:, 2] + 1) / 2, h1 * (sparse_matches[:, 3] + 1) / 2), dim=-1)
        b_ids = torch.where(mconf[None])[0]
        kpts0 -= kpts0.new_tensor((pad_left0, pad_top0))[None]
        kpts1 -= kpts1.new_tensor((pad_left1, pad_top1))[None]
        valid = (kpts0[:, 0] > 0) & (kpts0[:, 1] > 0) & (kpts1[:, 0] > 0) & (kpts1[:, 1] > 0)
        valid &= (kpts0[:, 0] <= orig_width0 - 1) & (kpts1[:, 0] <= orig_width1 - 1)
        valid &= (kpts0[:, 1] <= orig_height0 - 1) & (kpts1[:, 1] <= orig_height1 - 1)
        kpts0, kpts1, mconf, b_ids = kpts0[valid], kpts1[valid], mconf[valid], b_ids[valid]
    elif args.model == "gim_loftr":
        data.update({"color0": image0, "color1": image1})
        with torch.no_grad():
            model(data)
        kpts0, kpts1, b_ids, mconf = data["mkpts0_f"], data["mkpts1_f"], data["m_bids"], data["mconf"]
    else:
        gray0 = gim_demo.preprocess(gim_demo.read_image(args.image0, grayscale=True), grayscale=True)[0].to(device)[None]
        gray1 = gim_demo.preprocess(gim_demo.read_image(args.image1, grayscale=True), grayscale=True)[0].to(device)[None]
        data.update({"gray0": gray0, "gray1": gray1})
        data.update({"size0": torch.tensor(gray0.shape[-2:][::-1])[None], "size1": torch.tensor(gray1.shape[-2:][::-1])[None]})
        data.update({"scale0": torch.tensor(scale0).to(device)[None], "scale1": torch.tensor(scale1).to(device)[None]})
        with torch.no_grad():
            pred = {}
            pred.update({k + "0": v for k, v in detector({"image": gray0}).items()})
            pred.update({k + "1": v for k, v in detector({"image": gray1}).items()})
            pred.update(model({**pred, **data, "image_size0": data["size0"], "image_size1": data["size1"]}))
        kpts0_all = torch.cat([kp * s for kp, s in zip(pred["keypoints0"], data["scale0"][:, None])])
        kpts1_all = torch.cat([kp * s for kp, s in zip(pred["keypoints1"], data["scale1"][:, None])])
        m_bids = torch.nonzero(pred["keypoints0"].sum(dim=2) > -1)[:, 0]
        matches = pred["matches"]
        kpts0 = torch.cat([kpts0_all[m_bids == bid][matches[bid][..., 0]] for bid in range(image0.size(0))])
        kpts1 = torch.cat([kpts1_all[m_bids == bid][matches[bid][..., 1]] for bid in range(image0.size(0))])
        b_ids = torch.cat([m_bids[m_bids == bid][matches[bid][..., 0]] for bid in range(image0.size(0))])
        mconf = torch.cat(pred["scores"])
        points_are_original = True

    if kpts0 is None or len(kpts0) == 0:
        raise SystemExit("GIM 没有生成匹配点")
    processed_points0 = kpts0.detach().cpu().numpy()
    processed_points1 = kpts1.detach().cpu().numpy()
    if points_are_original:
        original_points0, original_points1 = processed_points0.copy(), processed_points1.copy()
    else:
        original_points0 = processed_points0 * np.asarray(scale0, dtype=np.float64)[None]
        original_points1 = processed_points1 * np.asarray(scale1, dtype=np.float64)[None]
    raw_match_count = len(processed_points0)
    valid = np.ones(raw_match_count, dtype=bool)
    filter_stats: dict[str, int] = {"rawMatches": raw_match_count}
    image0_size = (int(image0_np.shape[1]), int(image0_np.shape[0]))
    image1_size = (int(image1_np.shape[1]), int(image1_np.shape[0]))

    def apply_filter(name: str, mask: np.ndarray) -> None:
        nonlocal valid
        if len(mask) != raw_match_count:
            raise ValueError(f"{name} filter 数量错误")
        valid &= mask
        filter_stats[name] = int(valid.sum())

    if args.roi0 is not None:
        apply_filter("afterRoi0", _roi_at_points(original_points0, args.roi0))
    if args.roi1 is not None:
        apply_filter("afterRoi1", _roi_at_points(original_points1, args.roi1))
    if args.mask0 is not None:
        apply_filter("afterMask0", _mask_at_points(args.mask0, original_points0, image0_size))
    if args.mask1 is not None:
        apply_filter("afterMask1", _mask_at_points(args.mask1, original_points1, image1_size))
    if args.auto_mask1_nonblack:
        apply_filter(
            "afterGeneratedForeground",
            _nonblack_at_points(args.image1, original_points1, image1_size, args.foreground_threshold),
        )
    if args.max_aligned_displacement > 0:
        apply_filter(
            "afterAlignedDisplacement",
            _aligned_displacement_at_points(
                original_points0,
                image0_size,
                original_points1,
                image1_size,
                args.max_aligned_displacement,
            ),
        )
    selected = np.flatnonzero(valid)
    if not len(selected):
        reason = f"GIM 匹配全部被 ROI/mask 过滤: {filter_stats}"
        match_path = args.output_dir / "match.png"
        warp_path = args.output_dir / "warp.png"
        fallback = _fallback_side_by_side(image0, image1)
        cv2.imwrite(str(match_path), fallback)
        cv2.imwrite(str(warp_path), fallback)
        result = {
            "status": "empty",
            "reason": reason,
            "image0": str(args.image0), "image1": str(args.image1), "model": args.model, "seed": args.seed,
            "device": device, "raw_match_count": raw_match_count, "match_count": 0, "inlier_count": 0,
            "inlier_ratio": 0.0, "confidence_mean": 0.0, "confidence": [],
            "image0_size": list(image0_size), "image1_size": list(image1_size),
            "processed_image0_size": [int(image0.shape[-1]), int(image0.shape[-2])],
            "processed_image1_size": [int(image1.shape[-1]), int(image1.shape[-2])],
            "mkpts0": [], "mkpts1": [], "inliers": [], "geometry": {},
            "filters": {
                "counts": filter_stats,
                "roi0": args.roi0,
                "roi1": args.roi1,
                "mask0": str(args.mask0) if args.mask0 else None,
                "mask1": str(args.mask1) if args.mask1 else None,
                "autoMask1Nonblack": args.auto_mask1_nonblack,
                "foregroundThreshold": args.foreground_threshold,
                "maxAlignedDisplacementPixels": args.max_aligned_displacement,
            },
            "match_image": str(match_path), "warp_image": str(warp_path),
        }
        (args.output_dir / "matches.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print("GIM_MATCH_EMPTY", reason, args.output_dir, flush=True)
        return 0 if args.allow_empty else 2
    selected_tensor = torch.as_tensor(selected, dtype=torch.long, device=kpts0.device)
    kpts0, kpts1 = kpts0[selected_tensor], kpts1[selected_tensor]
    mconf, b_ids = mconf[selected_tensor], b_ids[selected_tensor]
    processed_points0 = processed_points0[selected]
    processed_points1 = processed_points1[selected]
    points0 = original_points0[selected]
    points1 = original_points1[selected]
    if len(processed_points0) >= 8:
        _, inlier_mask = cv2.findFundamentalMat(processed_points0, processed_points1, cv2.USAC_MAGSAC, ransacReprojThreshold=args.ransac_threshold, confidence=0.999999, maxIters=10000)
        inliers = (inlier_mask.reshape(-1) > 0) if inlier_mask is not None else np.ones(len(processed_points0), dtype=bool)
    else:
        inliers = np.ones(len(processed_points0), dtype=bool)

    data.update({"hw0_i": image0.shape[-2:], "hw1_i": image1.shape[-2:], "mkpts0_f": kpts0, "mkpts1_f": kpts1, "m_bids": b_ids, "mconf": mconf, "inliers": inliers})
    match_image = gim_demo.fast_make_matching_figure(data, b_id=0)
    overlay = gim_demo.fast_make_matching_overlay(data, b_id=0)
    match_image = cv2.addWeighted(match_image, 0.5, overlay, 0.5, 0)
    match_path = args.output_dir / "match.png"
    cv2.imwrite(str(match_path), match_image[..., ::-1])

    geom_info = {}
    try:
        geom_info = gim_demo.compute_geom(data)
    except Exception as exc:
        print("GIM_GEOMETRY_WARNING", type(exc).__name__, exc, flush=True)
    warp_path = args.output_dir / "warp.png"
    if "Homography" in geom_info and "Fundamental" in geom_info:
        warped = gim_demo.wrap_images(image0, image1, geom_info, "Homography")
        cv2.imwrite(str(warp_path), warped)
    else:
        cv2.imwrite(str(warp_path), _fallback_side_by_side(image0, image1))

    confidence = mconf.detach().cpu().numpy().reshape(-1) if hasattr(mconf, "detach") else np.asarray(mconf).reshape(-1)
    result = {
        "image0": str(args.image0), "image1": str(args.image1), "model": args.model, "seed": args.seed,
        "device": device, "raw_match_count": raw_match_count, "match_count": int(len(points0)), "inlier_count": int(inliers.sum()),
        "inlier_ratio": float(inliers.mean()) if len(inliers) else 0.0,
        "confidence_mean": float(confidence.mean()) if len(confidence) else 0.0,
        "confidence": confidence.tolist(),
        "image0_size": list(image0_size),
        "image1_size": list(image1_size),
        "processed_image0_size": [int(image0.shape[-1]), int(image0.shape[-2])],
        "processed_image1_size": [int(image1.shape[-1]), int(image1.shape[-2])],
        "mkpts0": points0.tolist(), "mkpts1": points1.tolist(),
        "inliers": inliers.astype(int).tolist(), "geometry": geom_info,
        "filters": {
            "counts": filter_stats,
            "roi0": args.roi0,
            "roi1": args.roi1,
            "mask0": str(args.mask0) if args.mask0 else None,
            "mask1": str(args.mask1) if args.mask1 else None,
            "autoMask1Nonblack": args.auto_mask1_nonblack,
            "foregroundThreshold": args.foreground_threshold,
            "maxAlignedDisplacementPixels": args.max_aligned_displacement,
        },
        "match_image": str(match_path), "warp_image": str(warp_path),
    }
    (args.output_dir / "matches.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("GIM_MATCH_READY", len(points0), int(inliers.sum()), args.output_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
