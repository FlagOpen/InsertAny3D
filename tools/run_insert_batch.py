#!/usr/bin/env python3
"""Run independent InsertAny3D server tasks serially.

The batch file owns scheduling only.  Each task is still executed by the
single-task ``run_insert_pipeline.py`` process and receives its own directory,
log, and manifest.  This keeps a failed task from poisoning the next GPU job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PIPELINE = PROJECT_ROOT / "tools" / "run_insert_pipeline.py"
DEFAULT_PYTHON = PROJECT_ROOT / "third_party" / "TRELLIS" / ".venv" / "bin" / "python"
CORE_KEYS = {
    "task_id", "input_image", "run_root", "output_dir", "pipeline_args", "prompts",
    "task_manifest", "unity_manifest",
    "input_image_manifest",
    "prompt", "task_prompt", "object_prompt", "provider_options",
}
LIST_FLAGS = {
    "scene_images": "--scene-image",
    "scene_depths": "--scene-depth",
    "scene_cameras": "--scene-camera",
    "scene_masks": "--scene-mask",
    "generated_masks": "--generated-mask",
    "trellis_mask_prompts": "--trellis-mask-prompt",
}

STRICT_EDIT_PROMPT = (
    "STRICT INSERTION-ONLY EDITING CONTRACT (highest priority): Treat the input image as an immutable base plate, "
    "not a scene to recreate. Keep the exact camera, framing, resolution, background, lighting, colors, style, and "
    "every existing object. Preservation has priority over the requested interaction: if they conflict, reinterpret "
    "the new object's pose or action and leave the original scene unchanged. The locked anchor must retain exactly "
    "the same image-space position, silhouette, geometry, scale, orientation, articulation, open/closed state, "
    "attachment state, color, texture, visibility, and ground contact. Never move, rotate, deform, open, close, lift, "
    "remove, duplicate, replace, or reveal hidden parts of any existing object. Interaction words such as hold, "
    "operate, repair, inspect, or touch describe only how the newly inserted object should adapt; they never authorize "
    "changing the anchor. An existing object resting on the ground must remain at the same location on the ground: "
    "make a new person bend, crouch, or reach to it rather than lifting it. An existing closed hood, lid, panel, or "
    "door must remain closed: make a new person interact with its visible exterior rather than opening it or inventing "
    "an interior. Only change pixels inside the new object's necessary silhouette, physically required occlusion, and "
    "minimal contact shadow. Do not repaint, relight, restyle, crop, shift, or regenerate the rest of the image. When "
    "uncertain, copy the input unchanged."
)
STRICT_ANCHOR_PROMPT = (
    "This is the locked registration anchor. Its exact visible geometry, pose, state, image-space location, and ground "
    "contact override any requested interaction. A grounded anchor stays grounded and a closed part stays closed."
)
LEGACY_EDIT_PROMPTS = {
    "Keep the original anchor object and surrounding scene unchanged. Add the requested object at the marked location "
    "with consistent scale, lighting, and perspective.",
    "Perform a strict local insertion edit. Treat the provided image as an immutable base plate, not a scene to "
    "regenerate. Preserve the camera, framing, resolution, background, lighting, colors, style, and every existing "
    "object. Change only the pixels required for the newly inserted object, its necessary occlusion, and a minimal "
    "contact shadow. The locked anchor must retain exactly the same position, silhouette, geometry, scale, orientation, "
    "articulation, open/closed state, attachment state, color, texture, and ground contact as in the input image. Never "
    "move, rotate, deform, open, close, lift, remove, duplicate, replace, or reveal hidden parts of the anchor. Make the "
    "new object adapt its position, pose, and action to the anchor's existing state; never alter the anchor to make the "
    "requested interaction easier. Outside the inserted object's necessary local edit region, reproduce the input image "
    "unchanged.",
}
LEGACY_ANCHOR_PROMPTS = {
    "Preserve the anchor object's geometry and visible context so it remains a reliable registration reference.",
    "This is the locked anchor used for registration. Preserve its exact visible geometry, pose, state, and image-space "
    "location from the input image.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按 job JSON 串行执行独立 InsertAny3D 任务")
    parser.add_argument("--jobs", required=True, type=Path, help="批量任务 JSON")
    parser.add_argument("--python", dest="python_path", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--pipeline", type=Path, default=DEFAULT_PIPELINE)
    parser.add_argument("--run-root", type=Path, help="覆盖 JSON 中的 run_root")
    parser.add_argument("--cuda-device", help="覆盖所有任务的 CUDA_VISIBLE_DEVICES")
    parser.add_argument("--fail-fast", action="store_true", help="首个失败后停止，不再启动后续任务")
    parser.add_argument("--skip-ready", action="store_true", help="跳过已有 manifest status=ready 的任务")
    parser.add_argument(
        "--require-edit-manifest",
        action="store_true",
        help="要求 edited/edit_manifest.json 为真实模型编辑且哈希、prompt 均匹配",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印命令并生成计划，不运行模型")
    return parser.parse_args()


def load_job_file(path: Path) -> tuple[Path | None, dict[str, Any], list[dict[str, Any]]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return None, {}, value
    if not isinstance(value, dict) or not isinstance(value.get("tasks"), list):
        raise ValueError("jobs JSON 必须是任务数组，或包含 tasks 数组的对象")
    defaults = value.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("defaults 必须是对象")
    run_root = Path(value["run_root"]) if value.get("run_root") else None
    return run_root, defaults, value["tasks"]


def validate_task_id(task_id: str) -> str:
    value = str(task_id).strip()
    if not value or value in {".", ".."} or Path(value).name != value or "/" in value or "\\" in value:
        raise ValueError(f"task_id 不是安全的目录名: {task_id!r}")
    return value


def combine_prompt(default: Any, user: Any) -> str:
    prefix = str(default or "").strip()
    addition = str(user or "").strip()
    if not prefix:
        return addition
    if not addition:
        return prefix
    return prefix + "\n" + addition


def migrate_default_prompt(value: Any, legacy_values: set[str], replacement: str) -> Any:
    text = str(value).strip() if value is not None else None
    return replacement if value is None or text in legacy_values else value


def prompt_values(defaults: dict[str, Any], task: dict[str, Any]) -> dict[str, str]:
    base = defaults.get("prompts", {})
    override = task.get("prompts", {})
    if not isinstance(base, dict) or not isinstance(override, dict):
        raise ValueError("prompts 必须是对象")

    def get(name: str, legacy_name: str) -> Any:
        if name in override:
            return override[name]
        if name in task:
            return task[name]
        if legacy_name in task:
            return task[legacy_name]
        if name in base:
            return base[name]
        if legacy_name in base:
            return base[legacy_name]
        if legacy_name in defaults:
            return defaults[legacy_name]
        return ""

    edit_default = get("edit_default", "default_edit_prompt")
    edit_user = get("edit_user", "edit_prompt")
    object_default = get("object_default", "default_object_prompt")
    object_user = get("object_user", "object_prompt")
    anchor_default = get("anchor_default", "default_anchor_prompt")
    anchor_user = get("anchor_user", "anchor_prompt")
    edit_default = migrate_default_prompt(edit_default, LEGACY_EDIT_PROMPTS, STRICT_EDIT_PROMPT)
    anchor_default = migrate_default_prompt(anchor_default, LEGACY_ANCHOR_PROMPTS, STRICT_ANCHOR_PROMPT)
    result = {
        "edit_default": str(edit_default or "").strip(),
        "edit_user": str(edit_user or "").strip(),
        "edit_effective": combine_prompt(edit_default, edit_user),
        "object_default": str(object_default or "").strip(),
        "object_user": str(object_user or "").strip(),
        "object_effective": combine_prompt(object_default, object_user),
        "anchor_default": str(anchor_default or "").strip(),
        "anchor_user": str(anchor_user or "").strip(),
        "anchor_effective": combine_prompt(anchor_default, anchor_user),
    }
    result["image_edit_effective"] = image_edit_prompt(result)
    return result


def image_edit_prompt(prompts: dict[str, str]) -> str:
    sections = []
    if prompts.get("edit_default"):
        sections.append(prompts["edit_default"])
    if prompts.get("anchor_effective"):
        sections.append("LOCKED ANCHOR (must remain unchanged):\n" + prompts["anchor_effective"])
    if prompts.get("edit_user"):
        sections.append("REQUESTED NEW-OBJECT INSERTION:\n" + prompts["edit_user"])
    return "\n\n".join(sections)


def enrich_from_unity_manifest(task: dict[str, Any]) -> dict[str, Any]:
    """Import prompt fields from a Unity task_manifest when requested."""
    manifest_path = task.get("unity_manifest") or task.get("task_manifest")
    if not manifest_path:
        return dict(task)
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Unity task manifest 不存在: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Unity task manifest 不是对象: {path}")
    result = dict(value)
    result.update(task)
    manifest_prompts = {
        "edit_default": value.get("defaultEditPrompt", ""),
        "edit_user": value.get("editPrompt", ""),
        "object_default": value.get("defaultObjectPrompt", ""),
        "object_user": value.get("objectPrompt", ""),
        "anchor_default": value.get("defaultAnchorPrompt", ""),
        "anchor_user": value.get("anchorPrompt", ""),
    }
    prompt_override = task.get("prompts", {})
    if prompt_override:
        if not isinstance(prompt_override, dict):
            raise ValueError("prompts 必须是对象")
        manifest_prompts.update(prompt_override)
    for canonical, legacy_name in (
        ("edit_user", "edit_prompt"),
        ("object_user", "object_prompt"),
        ("anchor_user", "anchor_prompt"),
    ):
        if legacy_name in task:
            manifest_prompts[canonical] = task[legacy_name]
    result["prompts"] = manifest_prompts
    if "trellis_mask_prompts" not in result and value.get("trellisMaskPrompts"):
        result["trellis_mask_prompts"] = value["trellisMaskPrompts"]
    if "anchor_mask_prompt" not in result and value.get("anchorMaskPrompt"):
        result["anchor_mask_prompt"] = value["anchorMaskPrompt"]
    if "task_id" not in task and value.get("taskId"):
        result["task_id"] = value["taskId"]
    return result


def option_flag(key: str) -> str:
    return key if key.startswith("--") else "--" + key.replace("_", "-")


def append_option(command: list[str], key: str, value: Any) -> None:
    if value is None:
        return
    if key == "provider_options":
        if not isinstance(value, dict):
            raise ValueError("provider_options 必须是对象")
        command.extend(["--provider-options-json", json.dumps(value, ensure_ascii=False, separators=(",", ":"))])
        return
    if key in CORE_KEYS or key == "prompts":
        return
    if key in LIST_FLAGS:
        flag = LIST_FLAGS[key]
        if not isinstance(value, list):
            raise ValueError(f"{key} 必须是数组")
        for item in value:
            command.extend([flag, str(item)])
        return
    if key == "gim_pairs":
        if not isinstance(value, list):
            raise ValueError("gim_pairs 必须是二维数组")
        for pair in value:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError("每个 gim_pairs 元素必须包含两张图片")
            command.extend(["--gim-pair", str(pair[0]), str(pair[1])])
        return
    flag = option_flag(key)
    if isinstance(value, bool):
        if value:
            command.append(flag)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            command.extend([flag, str(item)])
        return
    if isinstance(value, dict):
        raise ValueError(f"不支持把嵌套对象直接作为 pipeline 参数: {key}")
    command.extend([flag, str(value)])


def has_prompt_flag(args: list[str]) -> bool:
    return "--prompt" in args or "--task-prompt" in args


def build_command(
    task: dict[str, Any],
    defaults: dict[str, Any],
    run_root: Path,
    python_path: Path,
    pipeline: Path,
) -> tuple[list[str], dict[str, str]]:
    task_id = validate_task_id(task.get("task_id", ""))
    input_image = task.get("input_image")
    if not input_image:
        raise ValueError(f"{task_id} 缺少 input_image")

    command = [
        str(python_path),
        str(pipeline),
        "--run-root", str(run_root),
        "--task-id", task_id,
        "--input-image", str(input_image),
    ]
    unity_manifest = task.get("unity_manifest") or task.get("task_manifest")
    if unity_manifest:
        command.extend(["--unity-manifest", str(unity_manifest)])
    merged = dict(defaults)
    task_options = task.get("options", {})
    if not isinstance(task_options, dict):
        raise ValueError(f"{task_id} 的 options 必须是对象")
    merged.update(task_options)
    # Keep common per-task paths convenient at the top level while retaining
    # ``options`` for less common pipeline flags.
    for key in (
        "input_ply", "scene_images", "scene_depths", "scene_cameras", "scene_masks", "generated_masks", "gim_pairs",
        "trellis_input", "trellis_mask_prompts", "anchor_mask_prompt", "seg_engine", "render_resolution", "render_mode", "render_fov", "render_distance",
        "render_side_angle_degrees", "render_yaw_degrees", "render_pitch_degrees",
        "run_sags", "skip_gim", "skip_pose", "coarse_pose_view_names", "pose_view_names",
        "disable_camera_refinement", "gim_anchor_roi_radius", "gim_aligned_max_displacement",
        "sags_points_json", "sags_mask", "sags_view_name", "sags_view_mode", "sags_yaw_offsets", "sags_view_names", "sags_points_per_mask", "sags_force_seed_radius", "sags_no_force_seed",
        "sags_mask_id", "sags_threshold", "sags_min_votes", "sags_independent_min_prior_coverage", "sags_visibility_depth_tolerance", "sags_gd_interval",
        "model_provider", "model_profile", "provider_options", "model_input_mask", "model_mask_prompt",
        "model_dir", "model_config_path", "hunyuan_python", "hunyuan_model_path", "hunyuan_shape_subfolder",
        "hunyuan_texture", "mesh_to_gaussian_density", "mesh_to_gaussian_thickness", "mesh_to_gaussian_max_points",
        "sparse_steps", "slat_steps", "sparse_cfg", "slat_cfg",
    ):
        if key in task:
            merged[key] = task[key]
    for key, value in merged.items():
        append_option(command, key, value)

    prompts = prompt_values(defaults, task)
    explicit_args = task.get("pipeline_args", [])
    if not isinstance(explicit_args, list) or not all(isinstance(item, (str, int, float)) for item in explicit_args):
        raise ValueError(f"{task_id} 的 pipeline_args 必须是字符串数组")
    explicit_args = [str(item) for item in explicit_args]
    command.extend(explicit_args)
    if not has_prompt_flag(command):
        object_prompt = str(task.get("prompt", merged.get("prompt", defaults.get("prompt", ""))) or "").strip()
        if not object_prompt:
            object_prompt = prompts["object_effective"]
        task_prompt = str(
            task.get("task_prompt", merged.get("task_prompt", defaults.get("task_prompt", "")))
            or ""
        ).strip()
        if object_prompt:
            command.extend(["--prompt", object_prompt])
        elif task_prompt:
            command.extend(["--task-prompt", task_prompt])

    return command, prompts


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_pipeline_status(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value.get("status") if isinstance(value, dict) else None


def read_pipeline_details(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(value, dict):
        return {}
    stages = value.get("stages", {})
    if not isinstance(stages, dict):
        stages = {}
    failed = [(name, item) for name, item in stages.items()
              if isinstance(item, dict) and item.get("status") in {"failed", "rejected", "blocked"}]
    stage_name, stage_value = failed[0] if failed else (None, {})
    if stage_name is None and stages:
        # A running manifest is useful when the worker is interrupted before
        # it can write a terminal record.  The bridge uses this to show the
        # last known server stage while the task is still recoverable.
        pending = [
            (name, item) for name, item in stages.items()
            if isinstance(item, dict) and item.get("status") in {"running", "blocked"}
        ]
        if pending:
            stage_name, stage_value = pending[-1]
    failure_error = stage_value.get("error") if isinstance(stage_value, dict) else None
    if not failure_error:
        failure_error = value.get("fatal_error")
    failure_log = stage_value.get("log") if isinstance(stage_value, dict) else None
    return {
        "pipeline_manifest": str(path.resolve()),
        "pipeline_status": value.get("status"),
        "failed_stages": value.get("failed_stages", []),
        "rejected_stages": value.get("rejected_stages", []),
        "failure_stage": stage_name,
        "failure_stage_status": stage_value.get("status") if isinstance(stage_value, dict) else None,
        "failure_error": failure_error,
        "failure_log": failure_log,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_edit_provenance(
    task: dict[str, Any],
    prompts: dict[str, str],
    required: bool,
) -> dict[str, Any] | None:
    image_path = Path(str(task.get("input_image", "")))
    configured = task.get("input_image_manifest")
    manifest_path = Path(str(configured)) if configured else image_path.with_name("edit_manifest.json")
    if not manifest_path.is_file():
        if required:
            raise FileNotFoundError(f"缺少真实图片编辑来源清单: {manifest_path}")
        return None
    if not image_path.is_file():
        raise FileNotFoundError(f"编辑图片不存在: {image_path}")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"图片编辑来源清单不是对象: {manifest_path}")
    if value.get("status") != "ready" or value.get("provenanceType") != "model_image_edit":
        raise ValueError(f"图片不是已完成的模型编辑结果: {manifest_path}")
    generator = str(value.get("generator", ""))
    if not generator.startswith("apiyi-gemini-"):
        raise ValueError(f"不接受未知或 fallback 图片来源 {generator!r}: {manifest_path}")
    expected_output_hash = str(value.get("output", {}).get("sha256", ""))
    actual_output_hash = sha256_file(image_path)
    if not expected_output_hash or expected_output_hash != actual_output_hash:
        raise ValueError(f"edited 图片哈希与来源清单不一致: {image_path}")
    effective_prompt = prompts.get("image_edit_effective", "")
    expected_prompt_hash = hashlib.sha256(effective_prompt.encode("utf-8")).hexdigest()
    manifest_prompt_hash = str(value.get("prompt", {}).get("sha256", ""))
    if not manifest_prompt_hash or manifest_prompt_hash != expected_prompt_hash:
        raise ValueError(f"edited 图片使用的 prompt 与当前 task 不一致: {manifest_path}")
    return {
        "manifest": str(manifest_path.resolve()),
        "generator": generator,
        "model": value.get("request", {}).get("model"),
        "input_sha256": value.get("input", {}).get("sha256"),
        "output_sha256": actual_output_hash,
        "prompt_sha256": manifest_prompt_hash,
    }


def run_one(
    task: dict[str, Any],
    defaults: dict[str, Any],
    run_root: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    task_id = validate_task_id(task.get("task_id", ""))
    task_root = run_root / task_id
    manifest_path = task_root / "manifest.json"
    if args.skip_ready and read_pipeline_status(manifest_path) == "ready":
        return {"task_id": task_id, "status": "skipped", "reason": "existing manifest is ready"}

    command, prompts = build_command(task, defaults, run_root, args.python_path, args.pipeline)
    edit_provenance = validate_edit_provenance(task, prompts, args.require_edit_manifest)
    task_root.mkdir(parents=True, exist_ok=True)
    prompt_path = task_root / "prompts.json"
    write_json(prompt_path, {"task_id": task_id, **prompts})
    log_path = task_root / "logs" / "batch.log"
    printable = " ".join(shlex.quote(part) for part in command)
    started = time.time()
    record: dict[str, Any] = {
        "task_id": task_id,
        "status": "planned" if args.dry_run else "running",
        "command": command,
        "log": str(log_path),
        "prompts": str(prompt_path),
        "edit_provenance": edit_provenance,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    print(f"[batch:{task_id}] {printable}", flush=True)
    if args.dry_run:
        record["status"] = "dry-run"
        return record

    env = os.environ.copy()
    cuda_device = args.cuda_device or defaults.get("cuda_device") or task.get("cuda_device")
    if cuda_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    env["PYTHONUNBUFFERED"] = "1"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"COMMAND: {printable}\n")
        log.write(f"CUDA_VISIBLE_DEVICES: {env.get('CUDA_VISIBLE_DEVICES', '')}\n\n")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(f"[batch:{task_id}] {line}", end="", flush=True)
            log.write(line)
        return_code = process.wait()

    pipeline_status = read_pipeline_status(manifest_path)
    pipeline_details = read_pipeline_details(manifest_path)
    if pipeline_status == "rejected":
        task_status = "rejected"
    elif return_code == 0 and pipeline_status not in {"failed", "blocked"}:
        task_status = "ready"
    else:
        task_status = "failed"
    record.update(
        {
            "return_code": return_code,
            "pipeline_status": pipeline_status,
            **pipeline_details,
            "status": task_status,
            "duration_seconds": round(time.time() - started, 3),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    return record


def main() -> int:
    args = parse_args()
    job_root, defaults, tasks = load_job_file(args.jobs)
    run_root = args.run_root or job_root
    if run_root is None:
        raise SystemExit("jobs JSON 需要 run_root，或命令行传 --run-root")
    run_root = run_root.resolve()
    if not isinstance(tasks, list) or not tasks:
        raise SystemExit("tasks 不能为空")
    if not args.dry_run:
        if not args.python_path.is_file():
            raise SystemExit(f"Python 不存在: {args.python_path}")
        if not args.pipeline.is_file():
            raise SystemExit(f"pipeline 不存在: {args.pipeline}")

    records: list[dict[str, Any]] = []
    failed = False
    rejected = False
    for task in tasks:
        if not isinstance(task, dict):
            raise SystemExit("tasks 的每一项必须是对象")
        try:
            record = run_one(enrich_from_unity_manifest(task), defaults, run_root, args)
        except Exception as exc:
            task_id = str(task.get("task_id", "<unknown>"))
            error = f"{type(exc).__name__}: {exc}"
            record = {"task_id": task_id, "status": "failed", "error": error,
                      "pipeline_status": "failed", "failure_stage": "batch",
                      "failure_stage_status": "failed", "failure_error": error}
            # Do not let a malformed task id turn error handling into a path
            # traversal.  Valid task ids have already passed validate_task_id
            # in run_one; this branch also handles failures before that call.
            safe_task = task_id and Path(task_id).name == task_id and task_id not in {".", ".."} \
                and "/" not in task_id and "\\" not in task_id
            if safe_task:
                write_json(
                    run_root / task_id / "manifest.json",
                    {
                        "schemaVersion": 2,
                        "status": "failed",
                        "provider": task.get("model_provider", defaults.get("model_provider", "trellis")),
                        "failed_stages": ["batch"],
                        "fatal_error": error,
                        "stages": {"batch": {"status": "failed", "error": error}},
                        "updatedAtUtc": datetime.now(timezone.utc).isoformat(),
                    },
                )
            print(f"[batch:{task_id}] ERROR: {record['error']}", file=sys.stderr, flush=True)
        records.append(record)
        failed = failed or record.get("status") == "failed"
        rejected = rejected or record.get("status") == "rejected"
        write_json(
            run_root / "batch_manifest.json",
            {
                "schemaVersion": 1,
                "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
                "jobsFile": str(args.jobs.resolve()),
                "runRoot": str(run_root),
                "taskCount": len(records),
                "tasks": records,
                "failedTasks": [item["task_id"] for item in records if item.get("status") == "failed"],
                "rejectedTasks": [item["task_id"] for item in records if item.get("status") == "rejected"],
            },
        )
        if failed and args.fail_fast:
            break

    outcome = "FAILED" if failed else "REJECTED" if rejected else "READY"
    print(f"INSERT_BATCH_{outcome} {run_root} {len(records)}/{len(tasks)} tasks", flush=True)
    return 1 if failed else 2 if rejected else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit("已取消")
