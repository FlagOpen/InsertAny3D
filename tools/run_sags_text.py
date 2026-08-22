#!/usr/bin/env python3
"""Headless adapter for the existing SAGS ``app_text`` implementation.

The upstream file is a Gradio application and leaves its text callback empty.
This adapter supports the historical center-point projection mode and an
independent annotation mode in which every rendered camera contributes its own
mask before visibility-aware Gaussian voting.  It then reuses SAGS' actual
3D prompt projection, SAM feature loading and decomposition code.
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
    parser.add_argument("--points-json", type=Path, help="auto_segment.py 生成的 points.json（旧单中心模式必填）")
    parser.add_argument("--mask", type=Path, help="与 points 所在视角对齐的完整二值 mask.png（旧单中心模式必填）")
    parser.add_argument(
        "--view-annotation",
        action="append",
        nargs=3,
        metavar=("VIEW", "MASK", "POINTS"),
        help="独立视角标注；可重复，参数依次为视角名、mask.png、points.json",
    )
    parser.add_argument(
        "--annotation-mode",
        choices=("legacy", "independent"),
        default="legacy",
        help="legacy 从中心点投影到其他视角；independent 使用每个视角自己的 mask",
    )
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
    parser.add_argument(
        "--independent-min-prior-coverage",
        type=float,
        default=0.25,
        help="独立标注在非源视角中至少覆盖中心 Gaussian 几何先验的比例；0 关闭门控",
    )
    center_mask_group = parser.add_mutually_exclusive_group()
    center_mask_group.add_argument(
        "--no-center-mask-hard",
        dest="no_center_mask_hard",
        action="store_true",
        help="不把输入 mask 投影作为最终 Gaussian 的硬约束",
    )
    center_mask_group.add_argument(
        "--center-mask-hard",
        dest="no_center_mask_hard",
        action="store_false",
        help="把输入 mask 投影作为最终 Gaussian 的硬约束",
    )
    parser.set_defaults(no_center_mask_hard=None)
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


def _normalise_view_name(value: str) -> str:
    name = Path(str(value).strip()).stem
    if not name or name in {".", ".."} or Path(name).name != name:
        raise ValueError(f"视角名无效: {value!r}")
    return name


def _read_view_annotations(values: list[list[str]] | None, caller_cwd: Path) -> dict[str, dict[str, Path]]:
    """Parse repeated ``VIEW MASK POINTS`` records without changing legacy inputs."""

    result: dict[str, dict[str, Path]] = {}
    for record in values or []:
        if len(record) != 3:
            raise ValueError("--view-annotation 必须包含 VIEW、MASK、POINTS 三个参数")
        view_name = _normalise_view_name(record[0])
        if view_name in result:
            raise ValueError(f"--view-annotation 重复指定视角: {view_name}")
        mask_path = Path(record[1])
        points_path = Path(record[2])
        if not mask_path.is_absolute():
            mask_path = caller_cwd / mask_path
        if not points_path.is_absolute():
            points_path = caller_cwd / points_path
        result[view_name] = {
            "mask": mask_path.resolve(),
            "points": points_path.resolve(),
        }
    return result


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
    annotation_map = _read_view_annotations(args.view_annotation, CALLER_CWD)
    # The historical adapter keeps the center mask as a hard constraint.  A
    # set of independently labelled views must opt out by default, otherwise
    # unseen backsides can never be recovered from the other views.
    if args.no_center_mask_hard is None:
        args.no_center_mask_hard = args.annotation_mode == "independent"
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
    if (args.points_json is None) != (args.mask is None):
        raise SystemExit("--points-json 与 --mask 必须同时提供")
    for path, label in ((args.points_json, "points.json"), (args.mask, "mask.png")):
        if path is not None and not path.is_file():
            raise SystemExit(f"{label} 不存在: {path}")
    for view_name, annotation in annotation_map.items():
        for key, label in (("mask", "mask.png"), ("points", "points.json")):
            if not annotation[key].is_file():
                raise SystemExit(f"视角 {view_name} 的 {label} 不存在: {annotation[key]}")
    if args.annotation_mode == "independent" and not annotation_map:
        raise SystemExit("independent 模式至少需要一个 --view-annotation")
    if not args.sam_checkpoint.is_file():
        raise SystemExit(f"SAM checkpoint 不存在: {args.sam_checkpoint}")
    if args.mask_id < -1 or args.mask_id > 2 or not 0 <= args.threshold <= 1:
        raise SystemExit("mask-id 或 threshold 参数无效")
    if (
        args.min_votes < 1 or args.visibility_depth_tolerance < 0
        or not 0 <= args.independent_min_prior_coverage <= 1
        or args.gd_interval == 0 or args.gd_interval < -1
    ):
        raise SystemExit("min-votes、visibility-depth-tolerance、independent-prior-coverage 或 gd-interval 参数无效")

    requested_view_name = _normalise_view_name(args.view_name) if args.view_name else None
    if annotation_map and (requested_view_name is None or requested_view_name not in annotation_map):
        requested_view_name = "center" if "center" in annotation_map else sorted(annotation_map)[0]
    source_annotation = annotation_map.get(requested_view_name) if requested_view_name else None
    source_points_path = args.points_json or (source_annotation or {}).get("points")
    source_mask_path = args.mask or (source_annotation or {}).get("mask")
    if source_points_path is None or source_mask_path is None:
        raise SystemExit("没有找到源视角的 points.json 和 mask.png")
    points, labels = _read_points(source_points_path)
    positive_points = [point for point, label in zip(points, labels) if label > 0]
    annotation_points: dict[str, tuple[list[list[int]], list[int]]] = {}
    for annotation_name, annotation in annotation_map.items():
        annotation_points[annotation_name] = _read_points(annotation["points"])
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
    if requested_view_name:
        if requested_view_name not in tool.images:
            raise SystemExit(f"视角 {requested_view_name!r} 不在训练相机中: {sorted(tool.images)}")
        view_name = requested_view_name
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
    input_mask_np = _load_mask(source_mask_path, source_image.shape[1], source_image.shape[0])
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
            # This prior comes from the center annotation and the Gaussian
            # z-buffer. Independent masks use it only as a conservative
            # false-positive gate; the accepted mask itself stays independent.
            geometric_prior_np = projected_center_prior(camera)
            # Explicit per-view annotations are the new ring-view path.  They
            # are already produced by GroundingDINO/SAM (or LangSAM), so
            # running another projected-point SAM pass would throw away the
            # independent evidence and reintroduce center-view bias.
            annotation = annotation_map.get(image_name)
            if index == source_index and source_mask_path is not None and source_points_path is not None:
                annotation = {"mask": source_mask_path, "points": source_points_path}
            if args.annotation_mode == "independent" and annotation is None:
                raise RuntimeError(f"independent 模式缺少视角标注: {image_name}")

            independent_rejected = False
            annotation_prior_coverage: float | None = None
            if annotation is not None:
                annotation_mask_np = _load_mask(annotation["mask"], int(camera.image_width), int(camera.image_height))
                annotation_view_points, annotation_view_labels = annotation_points.get(image_name, (points, labels))
                valid_prompts = torch.tensor(annotation_view_points, dtype=torch.float32, device="cuda")
                valid_prompt_labels = torch.tensor(annotation_view_labels, dtype=torch.long, device="cuda")
                selected = torch.from_numpy(annotation_mask_np.astype(np.uint8)).to("cuda").long()
                candidates = selected.bool().unsqueeze(0)
                sam_scores = torch.ones((1,), dtype=torch.float32, device="cuda")
                candidate_np = candidates.detach().cpu().numpy()
                projected_prior_np = annotation_mask_np
                projected_prior_ious = [1.0]
                selected_id = "independent_mask"
                annotation_source = "provided_annotation"
                if args.annotation_mode == "independent" and index != source_index:
                    annotation_pixels = int(np.count_nonzero(annotation_mask_np))
                    prior_pixels = int(np.count_nonzero(geometric_prior_np))
                    overlap_pixels = int(np.count_nonzero(annotation_mask_np & geometric_prior_np))
                    annotation_prior_coverage = float(overlap_pixels / annotation_pixels) if annotation_pixels else 0.0
                    if (
                        args.independent_min_prior_coverage > 0
                        and (prior_pixels == 0 or annotation_prior_coverage < args.independent_min_prior_coverage)
                    ):
                        # A detector can confidently label the anchor when the
                        # requested object is fully occluded. Treat that view
                        # as unknown instead of adding false Gaussian labels.
                        independent_rejected = True
                        annotation_source = "provided_annotation_rejected_geometry_prior"
            else:
                if index == source_index:
                    projected = torch.tensor(points, dtype=torch.float32, device="cuda")
                    projected_labels = torch.tensor(labels, dtype=torch.long, device="cuda")
                else:
                    projected = app_text.project_to_2d(camera, prompt_3d).float()
                    projected_labels = torch.ones(projected.shape[0], dtype=torch.long, device="cuda")
                candidates, sam_scores, valid_prompts, valid_prompt_labels = predict_candidates(
                    camera, image_name, projected, projected_labels
                )
                candidate_np = candidates.detach().cpu().numpy()
                projected_prior_np = input_mask_np if index == source_index else geometric_prior_np
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
                annotation_source = "center_mask" if index == source_index else "projected_points"
            if independent_rejected:
                selected.zero_()
            if not independent_rejected and not args.no_force_seed:
                for point, label in zip(valid_prompts.round().long(), valid_prompt_labels):
                    x, y = int(point[0].item()), int(point[1].item())
                    x1, x2 = max(0, x - radius), min(selected.shape[1], x + radius + 1)
                    y1, y2 = max(0, y - radius), min(selected.shape[0], y + radius + 1)
                    selected[y1:y2, x1:x2] = 1 if int(label.item()) > 0 else 0
            selected_np = selected.detach().cpu().numpy().astype(bool)
            if independent_rejected:
                point_mask = torch.full((xyz.shape[0],), -1, dtype=torch.long, device="cuda")
                visible_count = 0
            else:
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
            geometric_prior_path = view_dir / "geometric_center_prior.png"
            cv2.imwrite(str(geometric_prior_path), geometric_prior_np.astype(np.uint8) * 255)
            candidate_records.append(
                {
                    "view": image_name,
                    "selected": selected_id,
                    "annotationSource": annotation_source,
                    "selectedFile": str(selected_path.resolve()),
                    "selectedPixels": int(np.count_nonzero(selected_np)),
                    "projectedInputMask": str(projected_prior_path.resolve()),
                    "projectedInputMaskPixels": int(np.count_nonzero(projected_prior_np)),
                    "selectedProjectedInputMaskIoU": _mask_iou(selected_np, projected_prior_np),
                    "geometricCenterPrior": str(geometric_prior_path.resolve()),
                    "geometricCenterPriorPixels": int(np.count_nonzero(geometric_prior_np)),
                    "annotationPriorCoverage": annotation_prior_coverage,
                    "annotationPriorGate": "rejected" if independent_rejected else "accepted",
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
        "annotationMode": args.annotation_mode,
        "inputMask": str(source_mask_path.resolve()),
        "inputMaskSha256": _sha256(source_mask_path),
        "pointsJson": str(source_points_path.resolve()),
        "points": [{"x": point[0], "y": point[1], "label": label} for point, label in zip(points, labels)],
        "sourceView": view_name,
        "viewAnnotations": {
            name: {key: str(path.resolve()) for key, path in value.items()}
            for name, value in annotation_map.items()
        },
        "maskSelection": "auto_per_view" if args.mask_id < 0 else f"fixed_{args.mask_id}",
        "threshold": args.threshold,
        "minVotes": args.min_votes,
        "centerMaskHard": not args.no_center_mask_hard,
        "independentMinPriorCoverage": args.independent_min_prior_coverage,
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
        "annotation_mode": args.annotation_mode,
        "points_json": str(source_points_path.resolve()),
        "mask": str(source_mask_path.resolve()),
        "mask_sha256": _sha256(source_mask_path),
        "view_name": view_name,
        "view_annotations": {
            name: {key: str(path.resolve()) for key, path in value.items()}
            for name, value in annotation_map.items()
        },
        "points_2d": [{"point": point, "label": label} for point, label in zip(points, labels)],
        "prompt_3d_count": int(prompt_3d.shape[0]),
        "mask_id": args.mask_id,
        "threshold": args.threshold,
        "min_votes": args.min_votes,
        "center_mask_hard": not args.no_center_mask_hard,
        "independent_min_prior_coverage": args.independent_min_prior_coverage,
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
