"""Known PNG fixtures test texture integrity, units and preflight isolation."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from src.adapters.generation.material_assets import verify_material_assets
from src.adapters.storage.local import file_hash
from src.core.scene import validate_scene_job
from src.pipeline.scene import run_scene
from src.synthesis.warehouse_scene import build_job
from src.validation.scene_dataset import export_dataset


class MaterialAssetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.assets = self.root / "assets"
        self.assets.mkdir()
        Image.fromarray(np.full((8, 8, 3), 128, dtype=np.uint8)).save(
            self.assets / "color.png"
        )
        Image.fromarray(np.full((8, 8), 32768, dtype=np.uint16)).save(
            self.assets / "rough.png"
        )
        maps = {
            role: dict(
                path=filename,
                sha256=file_hash(self.assets / filename),
                color_space=space,
            )
            for role, filename, space in (
                ("base_color", "color.png", "sRGB"),
                ("roughness", "rough.png", "Non-Color"),
            )
        }
        self.catalog = dict(
            schema_version="dubaidata.material-assets/1.0",
            assets={
                "fixture.wood": dict(
                    license="CC0-1.0",
                    source_page="https://example.invalid/known-fixture",
                    tile_size_metres=[2.0, 2.0, 2.0],
                    maps=maps,
                )
            },
        )
        self.job = build_job(
            dict(status="passed", split="train", manifest_sha256="a" * 64),
            "b" * 64,
            dict(
                width=32,
                height=18,
                intrinsics=[20, 20, 16, 9],
                near_metres=0.05,
                far_metres=60,
                model="pinhole_zero_distortion",
            ),
        )
        self.job["materials"]["wood"]["texture_set"] = dict(
            asset_id="fixture.wood",
            tile_size_metres=[2.0, 2.0, 2.0],
            base_color_mix=1.0,
            maps=copy.deepcopy(maps),
        )
        self.save_catalog()

    def save_catalog(self):
        path = self.assets / "catalog.json"
        path.write_text(json.dumps(self.catalog), encoding="utf-8")
        self.job["asset_catalog_sha256"] = file_hash(path)

    def test_known_color_and_16bit_roughness_remain_lossless(self):
        validate_scene_job(self.job)
        self.assertEqual(len(verify_material_assets(self.job, self.assets)), 2)
        with Image.open(self.assets / "rough.png") as image:
            self.assertEqual(int(np.asarray(image)[0, 0]), 32768)

    def test_corrupt_texture_fails_before_worker_or_output(self):
        manifest = self.root / "manifest.jsonl"
        manifest.write_text("known-fixture", encoding="utf-8")
        self.job["conditioning"]["manifest_sha256"] = file_hash(manifest)
        profile = self.root / "profile.json"
        profile.write_text(
            json.dumps(
                dict(
                    status="passed", split="train", manifest_sha256=file_hash(manifest)
                )
            ),
            encoding="utf-8",
        )
        self.job["conditioning"]["profile_sha256"] = file_hash(profile)
        fake_executable = self.root / "not_executed.exe"
        fake_executable.write_bytes(b"This file must never execute")
        (self.assets / "color.png").write_bytes(b"corrupted")
        output = self.root / "attempt"
        with self.assertRaisesRegex(ValueError, "Texture map checksum"):
            run_scene(
                self.job,
                dict(
                    blender_executable=str(fake_executable),
                    training_profile=str(profile),
                    source_manifest=str(manifest),
                    asset_root=str(self.assets),
                ),
                output,
                self.root,
            )
        self.assertFalse(output.exists())

    def test_catalog_tampering_or_missing_root_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "asset root"):
            verify_material_assets(self.job, None)
        (self.assets / "catalog.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "catalog checksum"):
            verify_material_assets(self.job, self.assets)

    def test_missing_license_and_unregistered_map_are_rejected(self):
        self.catalog["assets"]["fixture.wood"]["license"] = ""
        self.save_catalog()
        with self.assertRaisesRegex(ValueError, "license"):
            verify_material_assets(self.job, self.assets)
        self.catalog["assets"]["fixture.wood"]["license"] = "CC0-1.0"
        self.save_catalog()
        self.job["materials"]["wood"]["texture_set"]["maps"]["base_color"]["sha256"] = (
            "f" * 64
        )
        with self.assertRaisesRegex(ValueError, "registered asset"):
            verify_material_assets(self.job, self.assets)

    def test_unsafe_paths_color_spaces_and_scales_are_rejected(self):
        changes = (
            lambda texture: texture["maps"]["base_color"].update(path="../outside.png"),
            lambda texture: texture["maps"]["roughness"].update(color_space="sRGB"),
            lambda texture: texture.update(tile_size_metres=[0, 2, 2]),
            lambda texture: texture.update(base_color_mix=1.5),
        )
        for change in changes:
            with self.subTest(change=change):
                job = copy.deepcopy(self.job)
                change(job["materials"]["wood"]["texture_set"])
                with self.assertRaises(ValueError):
                    validate_scene_job(job)

    def test_response_comparison_is_explicit_and_bounded(self):
        self.job["render"].update(
            view_transform="AgX", response_candidates=["Standard"]
        )
        validate_scene_job(self.job)
        for value in (["AgX"], ["Standard", "Standard"], ["unknown"]):
            with self.subTest(value=value):
                self.job["render"]["response_candidates"] = value
                with self.assertRaises(ValueError):
                    validate_scene_job(self.job)

    def test_textured_export_requires_its_asset_catalog(self):
        output = self.root / "unbound_export"
        with self.assertRaisesRegex(ValueError, "asset root"):
            export_dataset(self.job, self.root / "worker", output)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
