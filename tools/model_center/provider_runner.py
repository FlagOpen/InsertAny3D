#!/usr/bin/env python3
"""Run one provider in its declared runtime.

This process boundary is important: SAM3D is deliberately loaded in the
TRELLIS environment, while Hunyuan keeps its existing independent venv.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


TOOL_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TOOL_ROOT.parent

# The upstream notebook module assumes a conda environment and eagerly imports
# an optional ``sam3d_objects.init`` module that is not shipped in the source
# checkout.  Keep those assumptions local to this provider subprocess; the
# TRELLIS process and its CUDA/PyTorch installation remain untouched.
os.environ.setdefault("LIDRA_SKIP_INIT", "true")
os.environ.setdefault("CONDA_PREFIX", os.environ.get("CUDA_HOME") or "/usr/local/cuda")


def _model_revision(model_path: str) -> str | None:
    path = Path(model_path)
    parts = path.parts
    if "snapshots" in parts:
        index = parts.index("snapshots")
        if index + 1 < len(parts) and parts[index + 1]:
            return parts[index + 1]
    for candidate in (path / "refs" / "main", path.parent / "refs" / "main"):
        if candidate.is_file():
            value = candidate.read_text(encoding="utf-8").strip()
            if value:
                return value
    return None


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _load_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L"), dtype=np.uint8)
    if (mask.shape[1], mask.shape[0]) != size:
        raise ValueError(f"mask size {mask.shape[1]}x{mask.shape[0]} does not match image {size[0]}x{size[1]}")
    return mask > 0


def _resolve_sam_config(model_dir: Path, explicit: Path | None) -> Path:
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.extend(
        [
            model_dir / "checkpoints" / "pipeline.yaml",
            model_dir / "checkpoints" / "hf" / "pipeline.yaml",
            model_dir / "pipeline.yaml",
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("SAM3D pipeline.yaml not found; checked: " + ", ".join(str(item) for item in candidates))


def _sam_weight_descriptor(model_dir: Path) -> dict[str, Any]:
    candidates = (
        model_dir.parent / "modelscope_weights_manifest.json",
        model_dir / "weights_manifest.json",
    )
    for path in candidates:
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            return {
                "weightsManifest": str(path.resolve()),
                "weightFingerprint": value.get("weightFingerprint"),
                "weightRevision": value.get("revision"),
                "weightSource": value.get("source"),
                "weightModelId": value.get("modelId"),
            }
    return {}


def _sam_depth_weight_descriptor(depth_model_path: Path | None) -> dict[str, Any]:
    manifest_path = PROJECT_ROOT / "codex_ops" / "sam3d_moge_weights_manifest.json"
    if not manifest_path.is_file():
        return {
            "depthModelPath": str(depth_model_path) if depth_model_path else None,
            "depthModelBytes": depth_model_path.stat().st_size if depth_model_path and depth_model_path.is_file() else None,
        }
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"depthWeightsManifest": str(manifest_path.resolve())}
    return {
        "depthWeightsManifest": str(manifest_path.resolve()),
        "depthWeightSource": value.get("source"),
        "depthWeightModelId": value.get("modelId"),
        "depthWeightRevision": value.get("revision"),
        "depthWeightSha256": value.get("sha256"),
        "depthWeightBytes": value.get("bytes"),
        "depthModelPath": str(depth_model_path) if depth_model_path else None,
    }


def _materialize_sam_runtime_config(
    base_config: Path,
    output_dir: Path,
    *,
    decoder: str,
    downsample_ss_dist: int | None,
    compile_model: bool,
    load_unused_decoders: bool,
    depth_model_path: Path | None,
) -> tuple[Path, dict[str, Any]]:
    """Create an adapter-owned SAM3D config without editing third_party.

    The upstream pipeline resolves its checkpoint paths relative to the
    pipeline file.  Runtime configs live beside the generated sample, so all
    path-valued fields are made absolute before applying the decoder profile.
    This also lets us disable decoders that cannot contribute to the required
    3DGS output while leaving the upstream checkout untouched.
    """

    if decoder not in {"gaussian", "gaussian_4"}:
        raise ValueError(f"unsupported SAM3D decoder: {decoder}")

    from omegaconf import OmegaConf  # type: ignore[import-not-found]

    config = OmegaConf.load(str(base_config))
    config_root = base_config.parent.resolve()
    for key in list(config.keys()):
        if not (str(key).endswith("_config_path") or str(key).endswith("_ckpt_path")):
            continue
        value = config.get(key)
        if not isinstance(value, str) or not value or "${" in value:
            continue
        path = Path(value)
        if not path.is_absolute():
            config[key] = str((config_root / path).resolve())

    original_gs4_config = config.get("slat_decoder_gs_4_config_path")
    original_gs4_ckpt = config.get("slat_decoder_gs_4_ckpt_path")
    if not load_unused_decoders:
        if decoder == "gaussian_4":
            if not original_gs4_config or not original_gs4_ckpt:
                raise FileNotFoundError("SAM3D gaussian_4 decoder paths are missing from pipeline.yaml")
            config["slat_decoder_gs_config_path"] = original_gs4_config
            config["slat_decoder_gs_ckpt_path"] = original_gs4_ckpt

        # InferencePipeline loads every decoder declared by the config even
        # when only one output format is requested.  The provider monkeypatch
        # below turns these null entries into identity modules.
        config["slat_decoder_gs_4_config_path"] = None
        config["slat_decoder_gs_4_ckpt_path"] = None
        config["slat_decoder_mesh_config_path"] = None
        config["slat_decoder_mesh_ckpt_path"] = None
        config["decode_formats"] = ["gaussian"]

    if downsample_ss_dist is not None:
        if downsample_ss_dist < 0:
            raise ValueError("SAM3D downsample_ss_dist must be >= 0")
        config["downsample_ss_dist"] = int(downsample_ss_dist)
    config["compile_model"] = bool(compile_model)

    # MoGe is a separate public checkpoint and is not included in the SAM3D
    # ModelScope snapshot.  Prefer an adapter-managed local file when one is
    # present, while retaining the upstream repository id as a fallback.
    discovered_depth_model = depth_model_path
    if discovered_depth_model is None:
        candidate = base_config.parent.parent / "moge-vitl-model.pt"
        if candidate.is_file():
            discovered_depth_model = candidate
    if discovered_depth_model is not None:
        discovered_depth_model = discovered_depth_model.expanduser().resolve()
        if not discovered_depth_model.is_file():
            raise FileNotFoundError(f"SAM3D depth model does not exist: {discovered_depth_model}")
        depth_model = config.get("depth_model")
        if depth_model is not None and depth_model.get("model") is not None:
            depth_model["model"]["pretrained_model_name_or_path"] = str(discovered_depth_model)

    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir = output_dir / "sam3d_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_config = runtime_dir / "pipeline.yaml"
    OmegaConf.save(config, str(runtime_config))
    decode_formats = config.get("decode_formats")
    return runtime_config, {
        "sam3dDecoder": decoder,
        "sam3dDownsampleSsDist": config.get("downsample_ss_dist"),
        "sam3dLoadUnusedDecoders": bool(load_unused_decoders),
        "sam3dDecodeFormats": list(decode_formats) if decode_formats is not None else None,
        "sam3dRuntimeConfig": str(runtime_config.resolve()),
        "sam3dBaseConfig": str(base_config.resolve()),
        "sam3dDepthModelPath": str(discovered_depth_model) if discovered_depth_model else None,
    }


def _patch_sam_optional_decoder_initializers() -> None:
    """Allow null optional decoder paths in the adapter-owned config.

    ``InferencePipeline`` puts all decoder return values in a ModuleDict, so
    returning Identity modules keeps its contract intact.  This patch is
    process-local and is only installed by the SAM3D provider subprocess.
    """

    import torch
    from sam3d_objects.pipeline.inference_pipeline import InferencePipeline  # type: ignore[import-not-found]

    if getattr(InferencePipeline, "_insertany3d_optional_decoders_patched", False):
        return

    original_gs = InferencePipeline.init_slat_decoder_gs
    original_mesh = InferencePipeline.init_slat_decoder_mesh

    def init_gs(self: Any, config_path: Any, ckpt_path: Any, *args: Any, **kwargs: Any) -> Any:
        if config_path is None or ckpt_path is None:
            return torch.nn.Identity()
        return original_gs(self, config_path, ckpt_path, *args, **kwargs)

    def init_mesh(self: Any, config_path: Any, ckpt_path: Any, *args: Any, **kwargs: Any) -> Any:
        if config_path is None or ckpt_path is None:
            return torch.nn.Identity()
        return original_mesh(self, config_path, ckpt_path, *args, **kwargs)

    InferencePipeline.init_slat_decoder_gs = init_gs  # type: ignore[method-assign]
    InferencePipeline.init_slat_decoder_mesh = init_mesh  # type: ignore[method-assign]
    InferencePipeline._insertany3d_optional_decoders_patched = True


def _patch_sam_torch_hub_offline() -> None:
    """Use an existing torch hub checkout for DINOv2 when available.

    SAM3D's DINO embedder calls ``torch.hub.load`` with a GitHub source even
    though the weights and repository are commonly already cached.  A failed
    metadata request should not prevent an otherwise offline generation run.
    The original loader remains the fallback when the cache is absent.
    """

    import torch

    if getattr(torch.hub, "_insertany3d_offline_dino_patched", False):
        return
    original_load = torch.hub.load
    cached_repo = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"

    def load(repo_or_dir: Any, model: Any, *args: Any, **kwargs: Any) -> Any:
        source = kwargs.get("source", "github")
        if (
            source == "github"
            and str(repo_or_dir) == "facebookresearch/dinov2"
            and cached_repo.is_dir()
        ):
            kwargs = dict(kwargs)
            kwargs["source"] = "local"
            repo_or_dir = str(cached_repo)
        return original_load(repo_or_dir, model, *args, **kwargs)

    torch.hub.load = load  # type: ignore[method-assign]
    torch.hub._insertany3d_offline_dino_patched = True


def _sam_cuda_snapshot(torch: Any, label: str) -> dict[str, Any]:
    event: dict[str, Any] = {"label": label, "timestamp": time.time()}
    if not torch.cuda.is_available():
        event["cuda"] = False
        return event
    try:
        torch.cuda.synchronize()
        device = torch.cuda.current_device()
        event.update(
            {
                "cuda": True,
                "device": int(device),
                "allocatedBytes": int(torch.cuda.memory_allocated(device)),
                "reservedBytes": int(torch.cuda.memory_reserved(device)),
                "peakAllocatedBytes": int(torch.cuda.max_memory_allocated(device)),
                "peakReservedBytes": int(torch.cuda.max_memory_reserved(device)),
            }
        )
    except Exception as exc:  # pragma: no cover - defensive for failed CUDA contexts
        event["error"] = f"{type(exc).__name__}: {exc}"
    return event


def _tensor_row_count(value: Any) -> int | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        dims = tuple(int(item) for item in shape)
    except (TypeError, ValueError):
        return None
    if len(dims) >= 3 and dims[0] == 1:
        return dims[1]
    return dims[0] if dims else None


def _offload_sam_finished_stage(pipeline: Any, stage: str, torch: Any, report: dict[str, Any]) -> None:
    models = getattr(pipeline, "models", {})
    conditioners = getattr(pipeline, "condition_embedders", {})

    def release(mapping: Any, key: str, label: str) -> None:
        try:
            if key in mapping:
                del mapping[key]
                report.setdefault("releasedModules", []).append(label)
        except (KeyError, TypeError, AttributeError) as exc:
            report.setdefault("releaseErrors", {})[label] = f"{type(exc).__name__}: {exc}"

    if stage == "compute_pointmap":
        depth_model = getattr(pipeline, "depth_model", None)
        if depth_model is not None and getattr(depth_model, "model", None) is not None:
            depth_model.model = None
            report.setdefault("releasedModules", []).append("depth_model")
    elif stage == "sample_sparse_structure":
        release(models, "ss_generator", "ss_generator")
        release(models, "ss_decoder", "ss_decoder")
        release(conditioners, "ss_condition_embedder", "ss_condition_embedder")
    elif stage == "sample_slat":
        release(models, "slat_generator", "slat_generator")
        release(conditioners, "slat_condition_embedder", "slat_condition_embedder")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _attach_sam_stage_probe(
    pipeline: Any,
    torch: Any,
    report: dict[str, Any],
    output_dir: Path,
    *,
    sequential_offload: bool,
) -> None:
    """Record stage peaks and optionally release models whose stage is done."""

    def snapshot(label: str) -> None:
        report.setdefault("events", {})[label] = _sam_cuda_snapshot(torch, label)

    for method_name in ("compute_pointmap", "sample_sparse_structure", "sample_slat", "decode_slat"):
        original = getattr(pipeline, method_name, None)
        if original is None:
            continue

        def make_wrapper(name: str, bound_method: Any) -> Any:
            def wrapped(*args: Any, **kwargs: Any) -> Any:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                value = bound_method(*args, **kwargs)
                elapsed = time.perf_counter() - started
                snapshot(name)
                report.setdefault("durationsSec", {})[name] = elapsed
                if name == "sample_sparse_structure" and isinstance(value, dict):
                    report["stage1"] = {
                        "coordsOriginal": _tensor_row_count(value.get("coords_original")),
                        "coordsAfterDownsample": _tensor_row_count(value.get("coords")),
                    }
                if sequential_offload:
                    _offload_sam_finished_stage(pipeline, name, torch, report)
                    report.setdefault("events", {})[f"{name}AfterOffload"] = _sam_cuda_snapshot(
                        torch, f"{name}AfterOffload"
                    )
                _write_sam_memory_report(output_dir, report)
                return value

            return wrapped

        setattr(pipeline, method_name, make_wrapper(method_name, original))


def _write_sam_memory_report(output_dir: Path, report: dict[str, Any]) -> Path:
    path = output_dir / "sam3d_memory_report.json"
    _write_json(path, report)
    return path


def run_sam3d(args: argparse.Namespace) -> dict[str, Any]:
    if args.input_mask is None or not args.input_mask.is_file():
        raise ValueError("SAM3D requires --input-mask")
    if args.sam3d_spconv_algo:
        # This must be set before importing any SAM3D sparse module.  spconv
        # 2.3.6 auto-selects an implicit GEMM kernel that raises SIGFPE on the
        # tested RTX 3090/CUDA 12.1 runtime; Native is numerically equivalent.
        os.environ["SPCONV_ALGO"] = args.sam3d_spconv_algo
    source_root = Path(args.source_root) if args.source_root else PROJECT_ROOT / "third_party" / "SAM3D-Objects"
    model_dir = Path(args.model_dir) if args.model_dir else source_root / "checkpoints" / "modelscope"
    config_path = _resolve_sam_config(model_dir, Path(args.config_path) if args.config_path else None)
    notebook_root = source_root / "notebook"
    if not notebook_root.is_dir():
        raise FileNotFoundError(f"SAM3D notebook source not found: {notebook_root}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime_config, runtime_meta = _materialize_sam_runtime_config(
        config_path,
        output_dir,
        decoder=args.sam3d_decoder,
        downsample_ss_dist=args.sam3d_downsample_ss_dist,
        compile_model=bool(args.compile),
        load_unused_decoders=bool(args.sam3d_load_unused_decoders),
        depth_model_path=Path(args.sam3d_depth_model_path) if args.sam3d_depth_model_path else None,
    )
    if not args.sam3d_load_unused_decoders:
        sys.path.insert(0, str(source_root))
        _patch_sam_optional_decoder_initializers()
    sys.path.insert(0, str(notebook_root))
    sys.path.insert(0, str(source_root))
    from inference import Inference  # type: ignore[import-not-found]

    import torch

    _patch_sam_torch_hub_offline()

    memory_report: dict[str, Any] = {
        **runtime_meta,
        "sam3dSequentialOffload": bool(args.sam3d_sequential_offload),
        "sam3dSpconvAlgo": os.environ.get("SPCONV_ALGO", "auto"),
        "events": {},
        "durationsSec": {},
        "status": "loading",
    }

    with Image.open(args.input_image) as image:
        rgb = np.asarray(image.convert("RGB"))
    mask = _load_mask(args.input_mask, (rgb.shape[1], rgb.shape[0]))
    print("SAM3D_LOADING", runtime_config, flush=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        inference = Inference(str(runtime_config), compile=bool(args.compile))
        memory_report["status"] = "loaded"
        memory_report["events"]["afterModelLoad"] = _sam_cuda_snapshot(torch, "afterModelLoad")
        _write_sam_memory_report(output_dir, memory_report)
        pipeline = getattr(inference, "_pipeline", None)
        if pipeline is not None:
            _attach_sam_stage_probe(
                pipeline,
                torch,
                memory_report,
                output_dir,
                sequential_offload=bool(args.sam3d_sequential_offload),
            )
        print("SAM3D_GENERATING", flush=True)
        output = inference(rgb, mask, seed=args.seed)
        memory_report["status"] = "generated"
        memory_report["events"]["afterGeneration"] = _sam_cuda_snapshot(torch, "afterGeneration")
        memory_report["durationsSec"]["total"] = time.perf_counter() - started
    except Exception as exc:
        memory_report["status"] = "error"
        memory_report["error"] = f"{type(exc).__name__}: {exc}"
        memory_report["events"]["error"] = _sam_cuda_snapshot(torch, "error")
        memory_report["durationsSec"]["total"] = time.perf_counter() - started
        _write_sam_memory_report(output_dir, memory_report)
        raise

    gaussian = None
    if isinstance(output, dict):
        gaussian = output.get("gs")
        if gaussian is None:
            gaussian = output.get("gs_4")
    if gaussian is None or not hasattr(gaussian, "save_ply"):
        memory_report["status"] = "error"
        memory_report["error"] = "RuntimeError: SAM3D output does not contain a Gaussian object with save_ply()"
        _write_sam_memory_report(output_dir, memory_report)
        raise RuntimeError("SAM3D output does not contain a Gaussian object with save_ply()")

    for attribute in ("get_xyz", "means", "xyz"):
        value = getattr(gaussian, attribute, None)
        count = _tensor_row_count(value)
        if count is not None:
            memory_report["output"] = {"gaussianCount": count, "attribute": attribute}
            break
    ply_path = output_dir / "sample.ply"
    gaussian.save_ply(str(ply_path))
    memory_report["samplePlyBytes"] = ply_path.stat().st_size if ply_path.is_file() else None
    events = tuple(memory_report.get("events", {}).values())
    memory_report["overallPeak"] = {
        "allocatedBytes": max((int(event.get("peakAllocatedBytes", 0)) for event in events), default=0),
        "reservedBytes": max((int(event.get("peakReservedBytes", 0)) for event in events), default=0),
    }
    memory_report_path = _write_sam_memory_report(output_dir, memory_report)
    return {
        "provider": "sam3d",
        "representation": "native_gaussian",
        "model": str(model_dir.resolve()),
        "configPath": str(config_path),
        **runtime_meta,
        "sam3dSequentialOffload": bool(args.sam3d_sequential_offload),
        "sam3dSpconvAlgo": os.environ.get("SPCONV_ALGO", "auto"),
        "inputImage": str(Path(args.input_image).resolve()),
        "inputMask": str(args.input_mask.resolve()),
        "samplePly": str(ply_path.resolve()),
        "memoryReport": str(memory_report_path.resolve()),
        "seed": args.seed,
        "coordinateContractStatus": "gpu_render_smoke_complete_axis_pending",
        "coordinateContract": {
            "sourceFrame": "sam3d_native",
            "targetFrame": "generated_world",
            "axisMatrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "translation": [0, 0, 0],
            "handedness": "right",
            "upAxis": "y",
            "forwardAxis": "z",
            "origin": "object_center",
            "normalization": {"mode": "sam3d_native", "scale": 1.0},
            "renderDefaults": {"distance": 1.5, "near": 0.8, "far": 1.6},
            "unityImport": {"generatedAxis": "legacy-flip-z"},
        },
        **_sam_weight_descriptor(model_dir),
        **_sam_depth_weight_descriptor(
            Path(str(runtime_meta["sam3dDepthModelPath"])) if runtime_meta.get("sam3dDepthModelPath") else None
        ),
    }


def run_hunyuan(args: argparse.Namespace) -> dict[str, Any]:
    model_path = args.model_path or os.environ.get("HUNYUAN_MODEL_PATH", "tencent/Hunyuan3D-2")
    shape_subfolder = args.shape_subfolder or "hunyuan3d-dit-v2-0"
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline  # type: ignore[import-not-found]

    image = Image.open(args.input_image).convert("RGBA")
    if args.input_mask:
        mask = _load_mask(args.input_mask, image.size)
        rgba = np.asarray(image).copy()
        rgba[..., 3] = mask.astype(np.uint8) * 255
        image = Image.fromarray(rgba, mode="RGBA")
        segmentation_engine = "provided_mask"
    elif image.getextrema()[3][0] < 255:
        segmentation_engine = "input_alpha"
    else:
        from hy3dgen.rembg import BackgroundRemover  # type: ignore[import-not-found]

        image = BackgroundRemover()(image)
        segmentation_engine = "hunyuan_rembg"
    print("HUNYUAN_LOADING", model_path, shape_subfolder, flush=True)
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path, subfolder=shape_subfolder)
    print("HUNYUAN_GENERATING", flush=True)
    mesh = pipeline(image=image)[0]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = output_dir / "source_mesh.glb"
    mesh.export(str(mesh_path))
    if args.texture:
        # Texture synthesis is intentionally a separate opt-in stage.  The
        # shape output remains available if the optional paint dependency fails.
        from hy3dgen.texgen import Hunyuan3DPaintPipeline  # type: ignore[import-not-found]

        paint = Hunyuan3DPaintPipeline.from_pretrained(model_path)
        mesh = paint(mesh, image=image)
        mesh.export(str(mesh_path))
    return {
        "provider": "hunyuan",
        "representation": "mesh",
        "model": model_path,
        "weightSource": "local_snapshot" if Path(model_path).is_dir() else "huggingface_or_modelscope_id",
        "weightRevision": _model_revision(model_path),
        "weightIdentifier": f"{model_path}@{_model_revision(model_path) or 'unresolved'}:{shape_subfolder}",
        "shapeSubfolder": shape_subfolder,
        "inputImage": str(Path(args.input_image).resolve()),
        "inputMask": str(args.input_mask.resolve()) if args.input_mask else None,
        "segmentationEngine": segmentation_engine,
        "sourceMesh": str(mesh_path.resolve()),
        "seed": args.seed,
        "texture": bool(args.texture),
        "coordinateContract": {
            "sourceFrame": "hunyuan_mesh_native",
            "targetFrame": "generated_world",
            "axisMatrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "translation": [0, 0, 0],
            "handedness": "right",
            "upAxis": "y",
            "forwardAxis": "z",
            "origin": "object_center",
            "normalization": {"mode": "aabb_max_extent", "scale": 1.0},
            "renderDefaults": {"distance": 1.5, "near": 0.05, "far": 3.0},
            "unityImport": {"generatedAxis": "legacy-flip-z"},
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a selected InsertAny3D model provider")
    parser.add_argument("--provider", required=True, choices=("sam3d", "hunyuan"))
    parser.add_argument("--input-image", required=True, type=Path)
    parser.add_argument("--input-mask", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--model")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--config-path", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--sam3d-decoder", choices=("gaussian", "gaussian_4"), default="gaussian")
    parser.add_argument(
        "--sam3d-downsample-ss-dist",
        "--downsample-ss-dist",
        dest="sam3d_downsample_ss_dist",
        type=int,
    )
    parser.add_argument(
        "--sam3d-load-unused-decoders",
        action="store_true",
        help="keep upstream mesh and gaussian_4 decoders loaded (diagnostic baseline)",
    )
    parser.add_argument("--sam3d-depth-model-path", type=Path)
    parser.add_argument(
        "--sam3d-sequential-offload",
        action="store_true",
        help="release completed SAM3D stage modules without changing generated values",
    )
    parser.add_argument("--sam3d-spconv-algo", choices=("auto", "native", "implicit_gemm"))
    parser.add_argument("--model-path")
    parser.add_argument("--shape-subfolder")
    parser.add_argument("--texture", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input_image.is_file():
        raise SystemExit(f"input image does not exist: {args.input_image}")
    result = run_sam3d(args) if args.provider == "sam3d" else run_hunyuan(args)
    output_dir = Path(args.output_dir)
    _write_json(output_dir / "provider_manifest.json", result)
    if args.provider == "sam3d":
        print("SAM3D_PROVIDER_READY", result["samplePly"], flush=True)
    else:
        print("HUNYUAN_PROVIDER_READY", result["sourceMesh"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
