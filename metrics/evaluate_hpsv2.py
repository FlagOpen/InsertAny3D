#!/usr/bin/env python3
"""Run the unsupervised HPSv2 metric on one InsertAny3D task/run.

This command consumes existing Step 6 PNGs only. It deliberately has no Unity
or renderer dependency. The task description in task_manifest.json is used as
the text prompt unless --prompt is supplied.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from utils.view_selection import select_six_views


ROOT = Path(__file__).resolve().parent
HPS_ROOT = ROOT / "HPSv2"
if str(HPS_ROOT) not in sys.path:
    sys.path.insert(0, str(HPS_ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="InsertAny3D 单 run/task HPSv2 无监督评测")
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--prompt")
    parser.add_argument("--hps-version", default="v2.0", choices=("v2.0", "v2.1"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--clip-checkpoint", type=Path)
    parser.add_argument("--pitches", nargs=2, type=float, metavar=("LOW", "HIGH"))
    parser.add_argument("--views", type=int, help="每个俯视角的总 view 数，用于校验文件名")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def task_manifest(run_root: Path, task_id: str) -> dict:
    path = run_root / task_id / "task_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"缺少任务 manifest: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"任务 manifest 不是 JSON 对象: {path}")
    return value


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_root = args.run_root.expanduser().resolve()
    manifest = task_manifest(run_root, args.task_id)
    prompt = (args.prompt or manifest.get("taskDescription") or manifest.get("effectiveEditPrompt") or "").strip()
    if not prompt:
        raise ValueError("没有评测 prompt；请在 task_manifest.json 中提供 taskDescription 或传入 --prompt")
    selected = select_six_views(run_root, args.task_id, pitches=args.pitches, views=args.views)
    image_paths = [item["path"] for item in selected]
    if args.clip_checkpoint:
        os.environ["HPSV2_CLIP_CHECKPOINT"] = str(args.clip_checkpoint)
    if args.checkpoint:
        os.environ["HPSV2_CHECKPOINT"] = str(args.checkpoint)
    from hpsv2.img_score import score
    scores = score(image_paths, prompt, str(args.checkpoint) if args.checkpoint else None, args.hps_version)
    records = [{**item, "score": float(value)} for item, value in zip(selected, scores)]
    by_pitch = {}
    for item in records:
        by_pitch.setdefault(str(item["pitch"]), []).append(item["score"])
    output = args.output or (run_root / args.task_id / "metrics" / "hpsv2.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schemaVersion": 1,
        "metric": "hpsv2",
        "evaluationType": "unsupervised",
        "runRoot": str(run_root), "taskId": args.task_id,
        "hpsVersion": args.hps_version, "prompt": prompt,
        "selection": "center view_000, left/right adjacent circular views at low/high pitch",
        "views": records,
        "mean": sum(item["score"] for item in records) / len(records),
        "meanByPitch": {key: sum(values) / len(values) for key, values in by_pitch.items()},
        "createdAtUtc": datetime.now(timezone.utc).isoformat(),
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ready", "output": str(output), "mean": result["mean"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(f"HPSV2_METRIC_ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
