#!/usr/bin/env python3
"""Headless adapter for the existing SAGS ``app_text`` implementation.

The upstream file is a Gradio application and leaves its text callback empty.
This adapter injects one or more positive 2D points, then reuses its actual
3D prompt projection, SAM multi-view propagation, voting and decomposition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import types
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAGS_ROOT = PROJECT_ROOT / "third_party" / "SAGS"
CALLER_CWD = Path.cwd()


def _install_gradio_stubs() -> None:
    """Allow importing app_text in the shared env without starting a UI."""
    if "gradio" not in sys.modules:
        gradio = types.ModuleType("gradio")
        for name in ("Progress", "Request", "SelectData"):
            setattr(gradio, name, type(name, (), {}))
        # These names are only evaluated when create_gradio_interface is called.
        for name in ("Blocks", "Row", "Column", "Image", "Button", "Dropdown", "Textbox", "Gallery", "Radio", "DownloadButton"):
            setattr(gradio, name, type(name, (), {}))
        gradio.update = lambda **kwargs: kwargs
        sys.modules["gradio"] = gradio
    if "gradio_litmodel3d" not in sys.modules:
        lit = types.ModuleType("gradio_litmodel3d")
        lit.LitModel3D = type("LitModel3D", (), {})
        sys.modules["gradio_litmodel3d"] = lit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用自动 2D 点提示驱动 SAGS 多视角 3D 分割")
    parser.add_argument("--model-dir", required=True, type=Path, help="3DGS model 目录，包含 cfg_args 和 point_cloud")
    parser.add_argument("--points-json", required=True, type=Path, help="auto_segment.py 生成的 points.json")
    parser.add_argument("--mask", required=True, type=Path, help="与 points 所在视角对齐的完整二值 mask.png")
    parser.add_argument("--output-ply", required=True, type=Path)
    parser.add_argument("--view-name", help="点所在视角名称，默认使用第一个训练视角")
    parser.add_argument("--sam-checkpoint", type=Path, default=SAGS_ROOT / "gaussiansplatting/dependencies/sam_ckpt/sam_vit_h_4b8939.pth")
    parser.add_argument("--sam-arch", default="vit_h")
    parser.add_argument("--mask-id", type=int, default=-1, help="-1 自动选择每视角 SAM 候选；0..2 固定候选用于对照")
    parser.add_argument("--threshold", type=float, default=0.5, help="只在可见视角中计算的正标签比例")
    parser.add_argument("--min-votes", type=int, default=2, help="Gaussian 至少被多少个可见视角判为正类")
    parser.add_argument(
        "--vote-mode",
        choices=("majority", "union"),
        default="majority",
        help="多视角标签融合方式；union 保留任一视角命中的 Gaussian",
    )
    parser.add_argument("--gd-interval", type=int, default=-1, help="Gaussian Decomposition 间隔；三视角默认关闭，1 表示每个视角执行")
    parser.add_argument("--preview", type=Path, help="可选：保存当前视角点预览图")
    parser.add_argument("--diagnose-only", action="store_true", help="只执行多视角 mask/vote 并输出计数，不写前景 PLY")
    parser.add_argument("--force-seed-radius", type=int, default=2, help="将投影点击点回填到每视角 SAM mask 的半径；0 表示只回填一个像素")
    parser.add_argument("--no-force-seed", action="store_true", help="关闭投影点击点回填")
    parser.add_argument("--visibility-depth-tolerance", type=float, default=0.02, help="中心深度 z-buffer 的相对容差")
    parser.add_argument("--no-center-mask-hard", action="store_true", help="不把输入 mask 投影作为最终 Gaussian 的硬约束")
    parser.add_argument("--diagnostics-dir", type=Path, help="保存逐视角 SAM 候选、选中 mask 和投票统计")
    return parser.parse_args()


def _read_points(path: Path) -> tuple[list[list[int]], list[int]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("points", [])
    points: list[list[int]] = []
    labels: list[int] = []
    for item in value:
        if isinstance(item, dict):
            points.append([int(round(item["x"])), int(round(item["y"]))])
            labels.append(1 if int(item.get("label", 1)) > 0 else 0)
        else:
            points.append([int(round(item[0])), int(round(item[1]))])
            labels.append(1 if len(item) < 3 or int(item[2]) > 0 else 0)
    if not points or not any(labels):
        raise ValueError("points.json 没有正点击点")
    return points, labels


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_mask(path: Path, width: int, height: int) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.count_nonzero(left | right)
    return float(np.count_nonzero(left & right) / union) if union else 1.0


def main() -> int:
    args = parse_args()
    # app_text is imported from the SAGS checkout and needs that cwd for its
    # relative dependencies; resolve user paths before changing directories.
    for field in ("model_dir", "points_json", "mask", "output_ply", "preview", "diagnostics_dir"):
        value = getattr(args, field)
        if value is not None and not value.is_absolute():
            setattr(args, field, (CALLER_CWD / value).resolve())
    if not args.sam_checkpoint.is_absolute():
        args.sam_checkpoint = (CALLER_CWD / args.sam_checkpoint).resolve()
    if not args.model_dir.is_dir():
        raise SystemExit(f"SAGS model 目录不存在: {args.model_dir}")
    if not args.points_json.is_file() or not args.mask.is_file() or not args.sam_checkpoint.is_file():
        raise SystemExit("points.json、mask.png 或 SAM checkpoint 不存在")
    if args.mask_id < -1 or args.mask_id > 2 or not 0 <= args.threshold <= 1:
        raise SystemExit("mask-id 或 threshold 参数无效")
    if args.min_votes < 1 or args.visibility_depth_tolerance < 0 or args.gd_interval == 0 or args.gd_interval < -1:
        raise SystemExit("min-votes、visibility-depth-tolerance 或 gd-interval 参数无效")
    points, labels = _read_points(args.points_json)
    positive_points = [point for point, label in zip(points, labels) if label > 0]
    args.output_ply.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = args.diagnostics_dir or args.output_ply.parent / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    _install_gradio_stubs()
    os.chdir(SAGS_ROOT)
    sys.path.insert(0, str(SAGS_ROOT))
    import torch
    import app_text

    print("SAGS_LOADING_SAM", args.sam_checkpoint, flush=True)
    predictor = app_text.load_sam(args.sam_arch, str(args.sam_checkpoint))
    # app_text's nested segmentation closure accidentally references a module
    # global named predictor instead of self.predictor.
    app_text.predictor = predictor
    # Construct empty state first; app_text assumes these fields exist when a
    # model is loaded from the UI callback.
    tool = app_text.GradioAnnotationTool(model_path=None, predictor=predictor)
    tool.load_gaussian_scene(str(args.model_dir.resolve()))
    # Upstream stores one globally selected candidate under maskid.  The
    # adapter selects a candidate independently for every view and stores the
    # resulting set under slot 0.
    tool.maskid = 0
    tool.segmode = 0
    if args.view_name:
        if args.view_name not in tool.images:
            raise SystemExit(f"视角 {args.view_name!r} 不在训练相机中: {sorted(tool.images)}")
        view_name = args.view_name
    else:
        view_name = sorted(tool.images, key=lambda x: int(x) if str(x).isdigit() else str(x))[0]
    tool.current_image = view_name
    tool.seg2d_mark[view_name]["points"] = positive_points
    tool._update_3d_prompts()
    prompt_3d = tool.seg2d_mark[view_name]["prompts_3D"]
    if prompt_3d.numel() == 0:
        raise RuntimeError("2D 点没有投影到任何 3D Gaussian")
    if args.preview:
        preview = tool._render_original()
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.preview), preview)

    tool.args.gd_interval = args.gd_interval
    cameras = list(tool.scene.getTrainCameras())
    source_index = next(index for index, camera in enumerate(cameras) if camera.image_name == view_name)
    source_image = tool.images[view_name]
    input_mask_np = _load_mask(args.mask, source_image.shape[1], source_image.shape[0])
    input_mask = torch.from_numpy(input_mask_np.astype(np.uint8)).to("cuda").long()
    xyz = tool.scene.gaussians.get_xyz
    _, center_constraint_indices = app_text.mask_inverse(xyz, cameras[source_index], input_mask)
    center_constraint = torch.zeros(xyz.shape[0], dtype=torch.bool, device="cuda")
    center_constraint[center_constraint_indices.to("cuda")] = True
    cv2.imwrite(str(diagnostics_dir / "input_mask.png"), input_mask_np.astype(np.uint8) * 255)
    candidate_records: list[dict] = []
    cached = False

    def projected_visibility(camera):
        xyz_h = torch.cat((xyz, torch.ones((xyz.shape[0], 1), dtype=xyz.dtype, device=xyz.device)), dim=1)
        camera_points = xyz_h @ camera.world_view_transform[:, :3]
        depth = camera_points[:, 2]
        projected = app_text.project_to_2d(camera, xyz_h).long()
        height, width = int(camera.image_height), int(camera.image_width)
        valid = (
            (depth > 0)
            & (projected[:, 0] >= 0)
            & (projected[:, 0] < width)
            & (projected[:, 1] >= 0)
            & (projected[:, 1] < height)
        )
        linear = projected[:, 1] * width + projected[:, 0]
        front = torch.full((height * width,), float("inf"), dtype=depth.dtype, device=depth.device)
        if bool(valid.any()):
            front.scatter_reduce_(0, linear[valid], depth[valid], reduce="amin", include_self=True)
        tolerance = torch.clamp(front[linear.clamp(0, height * width - 1)].abs() * args.visibility_depth_tolerance, min=1e-5)
        visible = valid & (depth <= front[linear.clamp(0, height * width - 1)] + tolerance)
        return projected, visible

    def visible_mask_inverse(camera, sam_mask):
        projected, visible = projected_visibility(camera)
        point_mask = torch.full((xyz.shape[0],), -1, dtype=torch.long, device="cuda")
        point_mask[visible] = sam_mask[projected[visible, 1], projected[visible, 0]].long()
        return point_mask, int(visible.sum().item())

    def projected_center_prior(camera) -> np.ndarray:
        height, width = int(camera.image_height), int(camera.image_width)
        projected, visible = projected_visibility(camera)
        selected = visible & center_constraint
        prior = np.zeros((height, width), dtype=np.uint8)
        if bool(selected.any()):
            pixels = projected[selected].detach().cpu().numpy()
            prior[pixels[:, 1], pixels[:, 0]] = 1
            # Gaussian centers are point samples; a small dilation recovers a
            # stable projected silhouette without pretending to rerender SAM.
            prior = cv2.dilate(prior, np.ones((5, 5), dtype=np.uint8), iterations=2)
        return prior > 0

    def predict_candidates(camera, image_name, projected_points, projected_labels):
        height, width = int(camera.image_height), int(camera.image_width)
        valid = (
            (projected_points[:, 0] >= 0)
            & (projected_points[:, 0] < width)
            & (projected_points[:, 1] >= 0)
            & (projected_points[:, 1] < height)
        )
        projected_points = projected_points[valid]
        projected_labels = projected_labels[valid]
        if projected_points.numel() == 0:
            raise RuntimeError(f"{image_name} 没有可见的 SAM 提示点")
        predictor.features = tool.sam_features[image_name]
        masks_batch, scores_batch, _ = predictor.predict_torch(
            point_coords=projected_points[None].float(),
            point_labels=projected_labels[None].long(),
            multimask_output=True,
        )
        return masks_batch[0].bool(), scores_batch[0], projected_points, projected_labels

    def adaptive_get_mask(progress=None):
        nonlocal cached
        if cached:
            return None
        sam_masks = []
        multiview = []
        sam_all = {}
        radius = max(0, args.force_seed_radius)
        for index, camera in enumerate(cameras):
            image_name = camera.image_name
            if index == source_index:
                projected = torch.tensor(points, dtype=torch.float32, device="cuda")
                projected_labels = torch.tensor(labels, dtype=torch.long, device="cuda")
            else:
                projected = app_text.project_to_2d(camera, prompt_3d).float()
                projected_labels = torch.ones(projected.shape[0], dtype=torch.long, device="cuda")
            candidates, sam_scores, valid_prompts, valid_prompt_labels = predict_candidates(camera, image_name, projected, projected_labels)
            candidate_np = candidates.detach().cpu().numpy()
            projected_prior_np = input_mask_np if index == source_index else projected_center_prior(camera)
            projected_prior_ious = [_mask_iou(mask, projected_prior_np) for mask in candidate_np]
            if args.mask_id >= 0:
                selected_id = args.mask_id
            elif index == source_index:
                selected_id = int(np.argmax(projected_prior_ious))
            else:
                ranks = []
                for candidate_index, candidate in enumerate(candidates):
                    coords = valid_prompts.round().long()
                    coverage = float(candidate[coords[:, 1], coords[:, 0]].float().mean().item())
                    ranks.append(
                        (
                            projected_prior_ious[candidate_index],
                            coverage,
                            float(sam_scores[candidate_index].item()),
                            -int(candidate.sum().item()),
                            candidate_index,
                        )
                    )
                selected_id = max(ranks)[-1]
            selected = input_mask.clone() if index == source_index else candidates[selected_id].long()
            if not args.no_force_seed:
                for point in valid_prompts[valid_prompt_labels > 0].round().long():
                    x, y = int(point[0].item()), int(point[1].item())
                    x1, x2 = max(0, x - radius), min(selected.shape[1], x + radius + 1)
                    y1, y2 = max(0, y - radius), min(selected.shape[0], y + radius + 1)
                    selected[y1:y2, x1:x2] = 1
            selected_np = selected.detach().cpu().numpy().astype(bool)
            point_mask, visible_count = visible_mask_inverse(camera, selected)
            sam_masks.append(selected)
            multiview.append(point_mask.unsqueeze(-1))
            sam_all[image_name] = candidates.long()
            view_dir = diagnostics_dir / image_name
            view_dir.mkdir(parents=True, exist_ok=True)
            projected_prior_path = view_dir / "projected_input_mask.png"
            cv2.imwrite(str(projected_prior_path), projected_prior_np.astype(np.uint8) * 255)
            candidate_info = []
            for candidate_index, candidate in enumerate(candidate_np):
                candidate_path = view_dir / f"candidate_{candidate_index}.png"
                cv2.imwrite(str(candidate_path), candidate.astype(np.uint8) * 255)
                candidate_info.append(
                    {
                        "id": candidate_index,
                        "samScore": float(sam_scores[candidate_index].item()),
                        "pixels": int(np.count_nonzero(candidate)),
                        "inputMaskIoU": _mask_iou(candidate, input_mask_np) if index == source_index else None,
                        "projectedInputMaskIoU": projected_prior_ious[candidate_index],
                        "file": str(candidate_path.resolve()),
                    }
                )
            selected_path = view_dir / "selected.png"
            cv2.imwrite(str(selected_path), selected_np.astype(np.uint8) * 255)
            candidate_records.append(
                {
                    "view": image_name,
                    "selected": "provided_mask" if index == source_index else selected_id,
                    "selectedFile": str(selected_path.resolve()),
                    "selectedPixels": int(np.count_nonzero(selected_np)),
                    "projectedInputMask": str(projected_prior_path.resolve()),
                    "projectedInputMaskPixels": int(np.count_nonzero(projected_prior_np)),
                    "selectedProjectedInputMaskIoU": _mask_iou(selected_np, projected_prior_np),
                    "visibleGaussians": visible_count,
                    "positiveGaussians": int((point_mask == 1).sum().item()),
                    "promptPixels": valid_prompts.detach().cpu().tolist(),
                    "candidates": candidate_info,
                }
            )
        tool.record["prompts"] = prompt_3d
        tool.record["sam_all"] = sam_all
        tool.record["mvmask"][0] = {"multiview": multiview, "sam_masks": sam_masks}
        cached = True
        return None

    vote_diagnostics: dict[str, int | float] = {}

    def visibility_ensemble(threshold=0.5):
        masks = torch.cat(tool.record["mvmask"][0]["multiview"], dim=1)
        valid = masks >= 0
        positive = masks == 1
        valid_count = valid.sum(dim=1)
        positive_count = positive.sum(dim=1)
        ratio = positive_count.float() / valid_count.clamp_min(1).float()
        if args.vote_mode == "union":
            selected = positive_count > 0
        else:
            selected = (valid_count >= args.min_votes) & (positive_count >= args.min_votes) & (ratio >= threshold)
        if not args.no_center_mask_hard:
            selected &= center_constraint
        labels_out = selected.long()
        indices = torch.where(selected)[0].detach().cpu()
        vote_diagnostics.update(
            {
                "gaussianCount": int(masks.shape[0]),
                "visibleInAtLeastOneView": int((valid_count > 0).sum().item()),
                "visibleInMinVotes": int((valid_count >= args.min_votes).sum().item()),
                "positiveBeforeCenterConstraint": int(((positive_count >= args.min_votes) & (ratio >= threshold)).sum().item()),
                "centerConstraintPositive": int(center_constraint.sum().item()),
                "votedPositive": int(indices.numel()),
            }
        )
        return labels_out, indices

    tool._get_mask = adaptive_get_mask
    tool.ensemble = visibility_ensemble
    print("SAGS_SEGMENTING", view_name, len(positive_points), tuple(prompt_3d.shape), flush=True)
    tool._get_mask()
    _, final_indices = tool.ensemble(args.threshold)
    diagnostic = {
        "schemaVersion": 2,
        "inputMask": str(args.mask.resolve()),
        "inputMaskSha256": _sha256(args.mask),
        "pointsJson": str(args.points_json.resolve()),
        "points": [{"x": point[0], "y": point[1], "label": label} for point, label in zip(points, labels)],
        "sourceView": view_name,
        "maskSelection": "auto_per_view" if args.mask_id < 0 else f"fixed_{args.mask_id}",
        "threshold": args.threshold,
        "minVotes": args.min_votes,
        "centerMaskHard": not args.no_center_mask_hard,
        "visibilityDepthTolerance": args.visibility_depth_tolerance,
        "gdInterval": args.gd_interval,
        "views": candidate_records,
        "vote": vote_diagnostics,
    }
    diagnostics_json = diagnostics_dir / "sags_diagnostics.json"
    diagnostics_json.write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.diagnose_only:
        print("SAGS_DIAGNOSTIC", json.dumps(vote_diagnostics), flush=True)
        return 0
    result_path = tool.seg_gaussian(threshold=args.threshold)
    if not result_path or not Path(result_path).is_file():
        raise RuntimeError(f"SAGS 没有生成结果: {result_path}")
    shutil.copy2(result_path, args.output_ply)
    manifest = {
        "model_dir": str(args.model_dir.resolve()),
        "points_json": str(args.points_json.resolve()),
        "mask": str(args.mask.resolve()),
        "mask_sha256": _sha256(args.mask),
        "view_name": view_name,
        "points_2d": [{"point": point, "label": label} for point, label in zip(points, labels)],
        "prompt_3d_count": int(prompt_3d.shape[0]),
        "mask_id": args.mask_id,
        "threshold": args.threshold,
        "min_votes": args.min_votes,
        "center_mask_hard": not args.no_center_mask_hard,
        "visibility_depth_tolerance": args.visibility_depth_tolerance,
        "gd_interval": args.gd_interval,
        "diagnostics": str(diagnostics_json.resolve()),
        "candidate_selection": candidate_records,
        "vote": vote_diagnostics,
        "output_ply": str(args.output_ply.resolve()),
    }
    args.output_ply.with_suffix(".json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("SAGS_TEXT_READY", args.output_ply, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
