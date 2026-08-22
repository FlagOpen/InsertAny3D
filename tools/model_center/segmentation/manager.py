"""Segmentation orchestration independent of any generation provider."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

from ..contracts import MaskArtifact, sha256_file, write_json


@dataclass(frozen=True)
class MaskManagerConfig:
    engine: str = "legacy"
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    fallback: str = "none"
    human_confirmed: bool = False


class MaskManager:
    """Create a single-object or union mask and record its provenance.

    The existing ``auto_segment.py`` remains the implementation of the
    GroundingDINO/SAM/LangSAM engines.  This wrapper owns only selection,
    validation and manifest writing, so model providers cannot silently choose
    a different segmentation policy.
    """

    def __init__(self, tool_root: Path, config: MaskManagerConfig | None = None):
        self.tool_root = tool_root
        self.config = config or MaskManagerConfig()

    @staticmethod
    def _image_size(path: Path) -> tuple[int, int]:
        try:
            from PIL import Image

            with Image.open(path) as image:
                return int(image.width), int(image.height)
        except Exception:
            return 0, 0

    def _artifact(self, mask: Path, image: Path, engine: str, prompt: str) -> MaskArtifact:
        if not mask.is_file():
            raise FileNotFoundError(f"segmentation mask does not exist: {mask}")
        if not image.is_file():
            raise FileNotFoundError(f"segmentation input image does not exist: {image}")
        width, height = self._image_size(image)
        return MaskArtifact(
            path=str(mask.resolve()),
            image_path=str(image.resolve()),
            engine=engine,
            prompt=prompt,
            width=width,
            height=height,
            mask_sha256=sha256_file(mask),
            image_sha256=sha256_file(image),
            threshold=self.config.box_threshold,
            human_confirmed=self.config.human_confirmed,
        )

    def use_provided(self, image: Path, mask: Path, output_dir: Path, prompt: str = "") -> tuple[Path, MaskArtifact]:
        output_dir.mkdir(parents=True, exist_ok=True)
        if not mask.is_file():
            raise FileNotFoundError(f"segmentation mask does not exist: {mask}")
        image_size = self._image_size(image)
        mask_size = self._image_size(mask)
        if image_size != (0, 0) and mask_size != (0, 0) and image_size != mask_size:
            raise ValueError(
                f"provided mask size {mask_size[0]}x{mask_size[1]} does not match image "
                f"{image_size[0]}x{image_size[1]}"
            )
        target = output_dir / "mask.png"
        if mask.resolve() != target.resolve():
            shutil.copy2(mask, target)
        artifact = self._artifact(target, image, "provided_mask", prompt)
        write_json(output_dir / "mask_manifest.json", {"schemaVersion": 1, **artifact.to_dict()})
        return target, artifact

    def generate(
        self,
        image: Path,
        prompts: Sequence[str],
        output_dir: Path,
        python: Path | None = None,
    ) -> tuple[Path, MaskArtifact]:
        normalized = [" ".join(str(item).split()) for item in prompts if str(item).strip()]
        if not normalized:
            raise ValueError("at least one segmentation prompt is required")
        output_dir.mkdir(parents=True, exist_ok=True)
        executable = python or Path(sys.executable)
        command = [
            str(executable),
            str(self.tool_root / "auto_segment.py"),
            "--input",
            str(image),
            "--output-dir",
            str(output_dir),
            "--engine",
            self.config.engine,
        ]
        for prompt in normalized:
            command.extend(["--prompt", prompt])
        command.extend(["--box-threshold", str(self.config.box_threshold), "--text-threshold", str(self.config.text_threshold)])
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"segmentation failed with exit code {completed.returncode}")
        mask = output_dir / "mask.png"
        artifact = self._artifact(mask, image, self.config.engine, " | ".join(normalized))
        detections_path = output_dir / "detections.json"
        detections: list[dict[str, Any]] = []
        if detections_path.is_file():
            try:
                value = json.loads(detections_path.read_text(encoding="utf-8"))
                if isinstance(value, list):
                    detections = [item for item in value if isinstance(item, dict)]
            except json.JSONDecodeError:
                pass
        artifact = replace(artifact, detections=tuple(detections))
        write_json(output_dir / "mask_manifest.json", {"schemaVersion": 1, **artifact.to_dict()})
        return mask, artifact

    def resolve(
        self,
        image: Path,
        output_dir: Path,
        provided_mask: Path | None = None,
        prompts: Sequence[str] = (),
        python: Path | None = None,
    ) -> tuple[Path, MaskArtifact]:
        if provided_mask is not None:
            return self.use_provided(image, provided_mask, output_dir, " | ".join(str(p) for p in prompts))
        return self.generate(image, prompts, output_dir, python=python)
