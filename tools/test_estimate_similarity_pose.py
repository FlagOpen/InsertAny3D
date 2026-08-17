#!/usr/bin/env python3
"""Synthetic end-to-end check for estimate_similarity_pose.py."""

from __future__ import annotations

import json
import base64
import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


WIDTH = 256
HEIGHT = 256
FX = 120.0
FY = 120.0
CX = WIDTH / 2
CY = HEIGHT / 2
OFFSET = 0.5


def generated_point(pixel: tuple[int, int], depth: float) -> np.ndarray:
    x, y = pixel
    return np.array([(x + OFFSET - CX) * depth / FX, (y + OFFSET - CY) * depth / FY, depth])


def unity_project(point: np.ndarray) -> tuple[np.ndarray, float]:
    x, y, z = point
    pixel = np.array([FX * x / z + CX - OFFSET, CY - FY * y / z - OFFSET])
    return pixel, float(np.linalg.norm(point))


def write_depth(path: Path, samples: list[tuple[np.ndarray, float]]) -> None:
    depth = np.zeros((HEIGHT, WIDTH), dtype="<f4")
    for pixel, value in samples:
        x0, y0 = int(np.floor(pixel[0])), int(np.floor(pixel[1]))
        x1, y1 = min(x0 + 1, WIDTH - 1), min(y0 + 1, HEIGHT - 1)
        depth[y0, x0] = value
        depth[y0, x1] = value
        depth[y1, x0] = value
        depth[y1, x1] = value
    depth.tofile(path)


def main() -> int:
    tool = Path(__file__).with_name("estimate_similarity_pose.py")
    rotation = Rotation.from_euler("xyz", [8.0, -13.0, 17.0], degrees=True).as_matrix()
    scale = 1.65
    translation = np.array([0.35, -0.22, 1.8])
    pixels = [
        (48, 52), (82, 50), (116, 48), (150, 52), (188, 55),
        (55, 88), (91, 91), (127, 86), (163, 90), (198, 94),
        (52, 132), (88, 128), (124, 135), (160, 130), (196, 136),
        (58, 176), (94, 172), (130, 180), (166, 174), (194, 178),
    ]
    source = [generated_point(pixel, 2.2 + 0.025 * index) for index, pixel in enumerate(pixels)]
    target = [scale * (rotation @ point) + translation for point in source]

    with tempfile.TemporaryDirectory(prefix="pose_estimator_test_") as temp_value:
        root = Path(temp_value)
        cameras = root / "cameras.txt"
        images = root / "images.txt"
        cameras.write_text(
            "# synthetic\n1 PINHOLE 256 256 120 120 128 128\n",
            encoding="utf-8",
        )
        images.write_text(
            "# synthetic\n"
            "1 1 0 0 0 0 0 0 1 left.png\n\n"
            "2 1 0 0 0 0 0 0 1 right.png\n\n",
            encoding="utf-8",
        )
        camera_json = {
            "schemaVersion": 1,
            "width": WIDTH,
            "height": HEIGHT,
            "intrinsics": {"fx": FX, "fy": FY, "cx": CX, "cy": CY},
            "cameraToWorld": {
                "position": {"x": 0, "y": 0, "z": 0},
                "rotationXyzw": {"x": 0, "y": 0, "z": 0, "w": 1},
            },
            "depthMetadata": {"type": "radial_distance"},
            "pixelOrigin": "top_left",
        }
        camera_path = root / "unity.camera.json"
        camera_path.write_text(json.dumps(camera_json), encoding="utf-8")

        command = [
            sys.executable,
            str(tool),
            "--generated-cameras",
            str(cameras),
            "--generated-images",
            str(images),
            "--output",
            str(root / "pose.json"),
            "--diagnostics-dir",
            str(root / "diagnostics"),
            "--ransac-threshold",
            "0.005",
            "--ransac-iterations",
            "1000",
            "--min-inliers",
            "8",
            "--seed",
            "7",
        ]
        fixture_files = {}
        for view_index, name in enumerate(("left", "right")):
            indices = list(range(view_index, len(pixels), 2))
            generated_depth = root / f"{name}.raw"
            unity_depth = root / f"unity_{name}.raw"
            generated_samples = [(np.array(pixels[index], dtype=float), source[index][2]) for index in indices]
            projected = [unity_project(target[index]) for index in indices]
            unity_samples = [(pixel, radial) for pixel, radial in projected]
            write_depth(generated_depth, generated_samples)
            write_depth(unity_depth, unity_samples)
            matches = {
                "mkpts0": [pixel.tolist() for pixel, _ in projected],
                "mkpts1": [list(pixels[index]) for index in indices],
                "inliers": [1] * len(indices),
                "confidence": [1.0 - 0.01 * item for item in range(len(indices))],
                "image0_size": [WIDTH, HEIGHT],
                "image1_size": [WIDTH, HEIGHT],
            }
            matches_path = root / f"matches_{name}.json"
            matches_path.write_text(json.dumps(matches), encoding="utf-8")
            fixture_files[name] = {
                "matches": matches_path,
                "unity_depth": unity_depth,
                "generated_depth": generated_depth,
            }
            command.extend(["--view", str(matches_path), str(unity_depth), str(camera_path), str(generated_depth)])

        completed = subprocess.run(command, check=True, text=True, capture_output=True)
        result = json.loads((root / "pose.json").read_text(encoding="utf-8"))
        recovered = np.asarray(result["matrix4x4"], dtype=float)
        expected = np.eye(4)
        expected[:3, :3] = scale * rotation
        expected[:3, 3] = translation
        maximum_error = float(np.max(np.abs(recovered - expected)))
        if maximum_error > 1e-5:
            raise AssertionError(f"pose matrix max error {maximum_error}:\n{recovered}\n!=\n{expected}")
        if result["fit"]["inlierCount"] != len(pixels):
            raise AssertionError(result["fit"])
        if result["status"] != "ready" or not (root / "diagnostics" / "multiview_summary.json").is_file():
            raise AssertionError("multiview diagnostics missing")
        print(completed.stdout.strip())
        print("POSE_ESTIMATOR_SYNTHETIC_OK", f"max_error={maximum_error:.3g}")

        primary_command = list(command)
        primary_command[primary_command.index(str(root / "pose.json"))] = str(root / "pose_primary.json")
        primary_command[primary_command.index(str(root / "diagnostics"))] = str(root / "diagnostics_primary")
        primary_command.extend(["--primary-view-name", "left"])
        subprocess.run(primary_command, check=True, text=True, capture_output=True)
        primary_result = json.loads((root / "pose_primary.json").read_text(encoding="utf-8"))
        primary_validation = primary_result["validation"]
        if (
            primary_result["status"] != "ready"
            or primary_validation.get("policy") != "primary_with_corroboration"
            or primary_validation.get("primaryView") != "left"
            or "right" not in primary_validation.get("corroboratingViews", [])
        ):
            raise AssertionError(primary_validation)
        print("POSE_PRIMARY_CORROBORATION_OK", primary_validation["corroboratingViews"])

        # Both views remain individually solvable, but the right view now
        # describes a different translation.  A concatenation-only RANSAC
        # could silently select one side; the multiview gate must reject it.
        right_indices = list(range(1, len(pixels), 2))
        inconsistent_target = [target[index] + np.array([0.6, 0.0, 0.0]) for index in right_indices]
        inconsistent_projected = [unity_project(point) for point in inconsistent_target]
        bad_unity_depth = root / "unity_right_inconsistent.raw"
        write_depth(bad_unity_depth, inconsistent_projected)
        bad_matches = dict(json.loads(fixture_files["right"]["matches"].read_text(encoding="utf-8")))
        bad_matches["mkpts0"] = [pixel.tolist() for pixel, _ in inconsistent_projected]
        bad_matches_path = root / "matches_right_inconsistent.json"
        bad_matches_path.write_text(json.dumps(bad_matches), encoding="utf-8")
        bad_command = list(command)
        bad_command[bad_command.index(str(root / "pose.json"))] = str(root / "pose_rejected.json")
        bad_command[bad_command.index(str(root / "diagnostics"))] = str(root / "diagnostics_rejected")
        bad_command[bad_command.index(str(fixture_files["right"]["matches"]))] = str(bad_matches_path)
        bad_command[bad_command.index(str(fixture_files["right"]["unity_depth"]))] = str(bad_unity_depth)
        rejected = subprocess.run(bad_command, text=True, capture_output=True)
        rejected_result = json.loads((root / "pose_rejected.json").read_text(encoding="utf-8"))
        if rejected.returncode != 2 or rejected_result["status"] != "rejected":
            raise AssertionError((rejected.returncode, rejected.stdout, rejected.stderr, rejected_result.get("validation")))
        if not rejected_result["validation"]["rejectionReasons"]:
            raise AssertionError("rejected pose has no reason")
        print("POSE_MULTIVIEW_REJECTION_OK", rejected_result["validation"]["rejectionReasons"])

        empty_matches = dict(json.loads(fixture_files["right"]["matches"].read_text(encoding="utf-8")))
        empty_matches.update(
            {
                "status": "empty",
                "match_count": 0,
                "inlier_count": 0,
                "confidence": [],
                "mkpts0": [],
                "mkpts1": [],
                "inliers": [],
            }
        )
        empty_matches_path = root / "matches_right_empty.json"
        empty_matches_path.write_text(json.dumps(empty_matches), encoding="utf-8")
        empty_command = list(command)
        empty_command[empty_command.index(str(root / "pose.json"))] = str(root / "pose_empty_view.json")
        empty_command[empty_command.index(str(root / "diagnostics"))] = str(root / "diagnostics_empty_view")
        empty_command[empty_command.index(str(fixture_files["right"]["matches"]))] = str(empty_matches_path)
        empty = subprocess.run(empty_command, text=True, capture_output=True)
        empty_result = json.loads((root / "pose_empty_view.json").read_text(encoding="utf-8"))
        reasons = empty_result["validation"]["rejectionReasons"]
        if empty.returncode != 2 or empty_result["status"] != "rejected" or not any(reason.startswith("right:") for reason in reasons):
            raise AssertionError((empty.returncode, empty.stdout, empty.stderr, reasons))
        if not (root / "diagnostics_empty_view" / "multiview_summary.png").is_file():
            raise AssertionError("empty-view rejection diagnostics missing")
        print("POSE_EMPTY_VIEW_REJECTION_OK", reasons)

        # Exercise the public task orchestrator contract without invoking GPU
        # stages: existing matches/depth/cameras should produce 05_pose/pose.json.
        task = root / "Task_001"
        sparse = task / "03_rendered_3dgs" / "source" / "sparse" / "0"
        generated_images = task / "03_rendered_3dgs" / "source" / "images"
        generated_depths = task / "03_rendered_3dgs" / "source" / "depths" / "absdepth"
        sparse.mkdir(parents=True)
        generated_images.mkdir(parents=True)
        generated_depths.mkdir(parents=True)
        shutil.copy2(cameras, sparse / "cameras.txt")
        shutil.copy2(images, sparse / "images.txt")
        png = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=")
        pipeline_command = [
            sys.executable,
            str(tool.with_name("run_insert_pipeline.py")),
            "--input-image",
            str(root / "scene_left.png"),
            "--run-root",
            str(root),
            "--task-id",
            "Task_001",
            "--input-ply",
            str(root / "sample.ply"),
            "--skip-segmentation",
            "--skip-render",
            "--skip-gim",
            "--render-mode",
            "anchor",
            "--pose-view-names",
            "left,right",
            "--pose-generated-axis",
            "identity",
            "--pose-ransac-threshold",
            "0.005",
            "--pose-min-inliers",
            "8",
            "--trellis-python",
            sys.executable,
            "--gim-python",
            sys.executable,
        ]
        (root / "sample.ply").write_text("ply\n", encoding="ascii")
        for name in ("left", "right"):
            scene_image = root / f"scene_{name}.png"
            scene_image.write_bytes(png)
            (generated_images / f"{name}.png").write_bytes(png)
            shutil.copy2(fixture_files[name]["generated_depth"], generated_depths / f"{name}.raw")
            pair_dir = task / "04_gim" / f"pair_{0 if name == 'left' else 1:02d}"
            pair_dir.mkdir(parents=True)
            shutil.copy2(fixture_files[name]["matches"], pair_dir / "matches.json")
            pipeline_command.extend(["--scene-image", str(scene_image)])
        for name in ("left", "right"):
            pipeline_command.extend(["--scene-depth", str(fixture_files[name]["unity_depth"])])
        for _ in ("left", "right"):
            pipeline_command.extend(["--scene-camera", str(camera_path)])
        pipeline = subprocess.run(pipeline_command, check=True, text=True, capture_output=True)
        pipeline_result = json.loads((task / "05_pose" / "pose.json").read_text(encoding="utf-8"))
        pipeline_manifest = json.loads((task / "manifest.json").read_text(encoding="utf-8"))
        provenance = json.loads((task / "provenance.json").read_text(encoding="utf-8"))
        pipeline_matrix = np.asarray(pipeline_result["matrix4x4"], dtype=float)
        pipeline_error = float(np.max(np.abs(pipeline_matrix - expected)))
        if pipeline_error > 1e-5 or pipeline_manifest["stages"]["pose"]["status"] != "ready":
            raise AssertionError((pipeline_error, pipeline_manifest["stages"].get("pose")))
        expected_ply_hash = hashlib.sha256((root / "sample.ply").read_bytes()).hexdigest()
        recorded_ply = [
            artifact for artifact in provenance["artifacts"]
            if Path(artifact["path"]) == (root / "sample.ply").resolve()
        ]
        if len(recorded_ply) != 1 or recorded_ply[0]["sha256"] != expected_ply_hash:
            raise AssertionError("external input PLY is missing from provenance")
        print(pipeline.stdout.strip())
        print("INSERT_PIPELINE_POSE_SYNTHETIC_OK", f"max_error={pipeline_error:.3g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
