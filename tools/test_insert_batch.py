#!/usr/bin/env python3
"""Dependency-free checks for the scene batch command builder."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import run_insert_batch as batch


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        manifest = Path(directory) / "task_manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "taskId": "Task_001",
                    "defaultObjectPrompt": "",
                    "objectPrompt": "red mailbox",
                    "defaultEditPrompt": "Keep anchor",
                    "editPrompt": "Add mailbox",
                }
            ),
            encoding="utf-8",
        )
        task = batch.enrich_from_unity_manifest(
            {
                "task_id": "Task_001",
                "input_image": "edited.png",
                "unity_manifest": str(manifest),
                "object_prompt": "blue chair",
                "trellis_mask_prompts": ["blue chair", "wooden table"],
                "options": {
                    "coarse_pose_view_names": "left",
                    "gim_aligned_max_displacement": 128,
                    "pose_primary_view_name": "left",
                },
            }
        )
        command, prompts = batch.build_command(
            task,
            {"seg_engine": "legacy", "render_resolution": 1024},
            Path("/runs/scene"),
            Path("/bin/echo"),
            Path("/bin/true"),
        )
        assert prompts["edit_effective"] == "Keep anchor\nAdd mailbox"
        assert prompts["image_edit_effective"] == "Keep anchor\n\nREQUESTED NEW-OBJECT INSERTION:\nAdd mailbox"
        assert prompts["object_effective"] == "blue chair"
        assert command[command.index("--prompt") + 1] == "blue chair"
        assert command[command.index("--unity-manifest") + 1] == str(manifest)
        assert "--seg-engine" in command and "--render-resolution" in command
        assert command[command.index("--coarse-pose-view-names") + 1] == "left"
        assert command[command.index("--gim-aligned-max-displacement") + 1] == "128"
        assert command[command.index("--pose-primary-view-name") + 1] == "left"
        mask_values = [command[index + 1] for index, value in enumerate(command) if value == "--trellis-mask-prompt"]
        assert mask_values == ["blue chair", "wooden table"]
        provider_task = dict(task)
        provider_task.update(
            {
                "model_provider": "hunyuan",
                "model_profile": "shape",
                "provider_options": {"shape_subfolder": "hunyuan3d-dit-v2-0"},
                "hunyuan_texture": True,
            }
        )
        provider_command, _ = batch.build_command(
            provider_task,
            {},
            Path("/runs/provider"),
            Path("/bin/echo"),
            Path("/bin/true"),
        )
        assert provider_command[provider_command.index("--model-provider") + 1] == "hunyuan"
        assert provider_command[provider_command.index("--model-profile") + 1] == "shape"
        provider_options = json.loads(provider_command[provider_command.index("--provider-options-json") + 1])
        assert provider_options == {"shape_subfolder": "hunyuan3d-dit-v2-0"}
        assert "--hunyuan-texture" in provider_command
        migrated_task = batch.enrich_from_unity_manifest(
            {
                "task_id": "Task_001",
                "input_image": "edited.png",
                "unity_manifest": str(manifest),
                "prompts": {
                    "edit_default": next(iter(sorted(batch.LEGACY_EDIT_PROMPTS))),
                    "anchor_default": next(iter(sorted(batch.LEGACY_ANCHOR_PROMPTS))),
                },
            }
        )
        _, migrated_prompts = batch.build_command(
            migrated_task,
            {"seg_engine": "legacy"},
            Path("/runs/scene"),
            Path("/bin/echo"),
            Path("/bin/true"),
        )
        assert migrated_prompts["edit_default"] == batch.STRICT_EDIT_PROMPT
        assert migrated_prompts["anchor_default"] == batch.STRICT_ANCHOR_PROMPT
        assert batch.migrate_default_prompt(None, batch.LEGACY_EDIT_PROMPTS, batch.STRICT_EDIT_PROMPT) == batch.STRICT_EDIT_PROMPT
        image = Path(directory) / "edited.png"
        image.write_bytes(b"real edited image")
        edit_manifest = Path(directory) / "edit_manifest.json"
        edit_manifest.write_text(
            json.dumps(
                {
                    "status": "ready",
                    "provenanceType": "model_image_edit",
                    "generator": "apiyi-gemini-generateContent",
                    "input": {"sha256": "source"},
                    "prompt": {
                        "sha256": hashlib.sha256(prompts["image_edit_effective"].encode("utf-8")).hexdigest()
                    },
                    "request": {"model": "gemini-3.1-flash-image-preview"},
                    "output": {"sha256": batch.sha256_file(image)},
                }
            ),
            encoding="utf-8",
        )
        task["input_image"] = str(image)
        task["input_image_manifest"] = str(edit_manifest)
        provenance = batch.validate_edit_provenance(task, prompts, required=True)
        assert provenance["generator"] == "apiyi-gemini-generateContent"
        assert provenance["output_sha256"] == batch.sha256_file(image)
    print("INSERT_BATCH_TEST_READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
