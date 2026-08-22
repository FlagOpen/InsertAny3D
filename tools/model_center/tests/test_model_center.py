#!/usr/bin/env python3
"""CPU-only contract and converter tests for the provider boundary."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

from model_center.contracts import CoordinateContract
from model_center.converters.mesh_to_gaussian import convert_mesh_to_gaussian
from model_center.config import cache_key, resolve_profile
from model_center.contracts import RenderRequest
from model_center.renderers.provider_render import build_render_command
from model_center.providers.hunyuan import HunyuanProvider
from model_center.providers.sam3d import Sam3DProvider
from model_center.providers.trellis import TrellisProvider
from model_center.registry import get_provider, provider_names
from model_center.segmentation.manager import MaskManager, MaskManagerConfig
from model_center.transforms.gaussian_ply import transform_gaussian_ply


class ModelCenterTests(unittest.TestCase):
    def test_registry_and_contract_round_trip(self) -> None:
        self.assertEqual(provider_names(), ("trellis", "sam3d", "hunyuan"))
        with self.assertRaises(ValueError):
            get_provider("unknown", Path("/tmp/project"))
        contract = CoordinateContract(
            source_frame="fixture",
            axis_matrix=((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            translation=(1.0, 2.0, 3.0),
            uniform_scale=2.0,
        )
        restored = CoordinateContract.from_dict(contract.to_dict())
        self.assertEqual(restored, contract)
        self.assertEqual(restored.apply_point((1.0, 0.0, 0.0)), (2.0, 6.0, 6.0))
        render = RenderRequest(input_ply=Path("input.ply"), output_dir=Path("render"), near=0.1, far=4.0)
        self.assertEqual(render.to_dict()["near"], 0.1)
        profile = resolve_profile("sam3d", "default", {"seed": 7})
        self.assertEqual(profile.segmentation.input_mask_policy, "required")
        self.assertEqual(profile.options["seed"], 7)
        self.assertEqual(profile.options["decoder"], "gaussian")
        self.assertEqual(profile.options["downsample_ss_dist"], 1)
        self.assertFalse(profile.options["load_unused_decoders"])
        self.assertTrue(profile.options["sequential_offload"])
        self.assertEqual(profile.options["spconv_algo"], "native")
        self.assertTrue(cache_key(Path("/missing.png"), "trellis", "r1", "w1", "default", 1).startswith("candidate-"))
        quality_key = cache_key(
            Path("/missing.png"), "sam3d", "r1", "w1", "default", 1,
            options={"decoder": "gaussian", "downsample_ss_dist": 1},
        )
        low_density_key = cache_key(
            Path("/missing.png"), "sam3d", "r1", "w1", "default", 1,
            options={"decoder": "gaussian_4", "downsample_ss_dist": 1},
        )
        self.assertNotEqual(quality_key, low_density_key)

    def test_sam3d_memory_profile_options_are_forwarded(self) -> None:
        provider = Sam3DProvider()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "image.png"
            mask_path = root / "mask.png"
            Image.new("RGB", (4, 4), (0, 0, 0)).save(image_path)
            Image.new("L", (4, 4), 255).save(mask_path)
            from model_center.contracts import GenerationRequest

            command = provider.generation_command(
                GenerationRequest(
                    input_image=image_path,
                    output_dir=root / "out",
                    provider="sam3d",
                    input_mask=mask_path,
                    options={
                        "decoder": "gaussian_4",
                        "downsample_ss_dist": 4,
                        "load_unused_decoders": True,
                        "sequential_offload": True,
                        "spconv_algo": "native",
                    },
                ),
                root,
            )
            self.assertEqual(command[command.index("--sam3d-decoder") + 1], "gaussian_4")
            self.assertEqual(command[command.index("--sam3d-downsample-ss-dist") + 1], "4")
            self.assertIn("--sam3d-load-unused-decoders", command)
            self.assertIn("--sam3d-sequential-offload", command)
            self.assertEqual(command[command.index("--sam3d-spconv-algo") + 1], "native")

    def test_gaussian_transform_changes_position_scale_and_rotation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.ply"
            output_path = root / "output.ply"
            values = np.zeros(
                1,
                dtype=[
                    ("x", "f4"), ("y", "f4"), ("z", "f4"),
                    ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
                    ("opacity", "f4"),
                    ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
                    ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
                ],
            )
            values["x"], values["y"], values["z"] = 1.0, 2.0, 3.0
            values["scale_0"], values["scale_1"], values["scale_2"] = math.log(1.0), math.log(2.0), math.log(3.0)
            values["rot_0"] = 1.0
            PlyData([PlyElement.describe(values, "vertex")], text=False).write(str(input_path))
            contract = CoordinateContract(
                source_frame="fixture",
                axis_matrix=((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
                translation=(1.0, 0.0, 0.0),
                uniform_scale=2.0,
            )
            result = transform_gaussian_ply(input_path, output_path, contract)
            self.assertEqual(result["pointCount"], 1)
            transformed = PlyData.read(str(output_path))["vertex"].data
            np.testing.assert_allclose(
                [transformed["x"][0], transformed["y"][0], transformed["z"][0]],
                [-2.0, 2.0, 6.0],
                atol=1e-5,
            )
            np.testing.assert_allclose(
                np.exp([transformed["scale_0"][0], transformed["scale_1"][0], transformed["scale_2"][0]]),
                [2.0, 4.0, 6.0],
                atol=1e-5,
            )
            quat = np.array([transformed[f"rot_{index}"][0] for index in range(4)])
            self.assertAlmostEqual(float(np.linalg.norm(quat)), 1.0, places=5)

    def test_mesh_converter_is_deterministic_and_standard(self) -> None:
        import trimesh

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mesh_path = root / "box.glb"
            first_path = root / "first.ply"
            second_path = root / "second.ply"
            mesh = trimesh.creation.box(extents=(1.0, 2.0, 3.0))
            mesh.visual.vertex_colors = np.tile(np.array([200, 100, 50, 255], dtype=np.uint8), (len(mesh.vertices), 1))
            mesh.export(str(mesh_path))
            first = convert_mesh_to_gaussian(mesh_path, first_path, density=3.0, max_points=500, seed=7)
            second = convert_mesh_to_gaussian(mesh_path, second_path, density=3.0, max_points=500, seed=7)
            self.assertEqual(first["pointCount"], second["pointCount"])
            self.assertEqual(first_path.read_bytes(), second_path.read_bytes())
            vertex = PlyData.read(str(first_path))["vertex"].data
            self.assertGreater(len(vertex), 0)
            names = {item.name for item in PlyData.read(str(first_path))["vertex"].properties}
            self.assertTrue({"x", "y", "z", "f_dc_0", "opacity", "scale_0", "rot_0"}.issubset(names))
            for name in names:
                self.assertTrue(np.isfinite(np.asarray(vertex[name])).all(), name)

    def test_provided_mask_is_dimension_checked_and_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "image.png"
            mask_path = root / "mask.png"
            Image.new("RGB", (8, 6), (20, 30, 40)).save(image_path)
            Image.new("L", (8, 6), 255).save(mask_path)
            target, artifact = MaskManager(Path("/tools"), MaskManagerConfig()).use_provided(
                image_path, mask_path, root / "managed", "object"
            )
            self.assertTrue(target.is_file())
            self.assertEqual(artifact.width, 8)
            self.assertEqual(artifact.height, 6)
            self.assertTrue((root / "managed" / "mask_manifest.json").is_file())
            Image.new("L", (7, 6), 255).save(mask_path)
            with self.assertRaises(ValueError):
                MaskManager(Path("/tools")).use_provided(image_path, mask_path, root / "bad")

    def test_hunyuan_texture_is_flag_without_value(self) -> None:
        provider = HunyuanProvider()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "image.png"
            Image.new("RGB", (4, 4), (0, 0, 0)).save(image_path)
            # Exercise the command shape without invoking the independent GPU runtime.
            from model_center.contracts import GenerationRequest

            command = provider.generation_command(
                GenerationRequest(
                    input_image=image_path,
                    output_dir=root / "out",
                    provider="hunyuan",
                    options={"texture": True},
                ),
                root,
            )
            self.assertIn("--texture", command)
            index = command.index("--texture")
            self.assertTrue(index == len(command) - 1 or not command[index + 1].startswith("True"))

    def test_trellis_sampler_options_are_forwarded(self) -> None:
        provider = TrellisProvider()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "image.png"
            Image.new("RGB", (4, 4), (0, 0, 0)).save(image_path)
            from model_center.contracts import GenerationRequest

            command = provider.generation_command(
                GenerationRequest(
                    input_image=image_path,
                    output_dir=root / "out",
                    provider="trellis",
                    options={"sparse_steps": 12, "slat_steps": 10, "sparse_cfg": 5.0, "slat_cfg": 4.0},
                ),
                root,
            )
            for flag, expected in (("--sparse-steps", "12"), ("--slat-steps", "10"), ("--sparse-cfg", "5.0"), ("--slat-cfg", "4.0")):
                self.assertEqual(command[command.index(flag) + 1], expected)

    def test_render_command_uses_contract_clipping(self) -> None:
        request = RenderRequest(
            input_ply=Path("asset.ply"), output_dir=Path("render"), mode="sphere",
            resolution=32, near=0.2, far=2.5, latitudes="10", views_per_latitude=2,
        )
        command = build_render_command(request, Path("/project"), Path("/python"))
        self.assertEqual(command[command.index("--near") + 1], "0.2")
        self.assertEqual(command[command.index("--far") + 1], "2.5")


if __name__ == "__main__":
    unittest.main()
