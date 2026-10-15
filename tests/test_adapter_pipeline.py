"""Quality contracts for dataset independence, units, leakage and worker failures."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from jsonschema.exceptions import ValidationError

from src.core.contracts import SCHEMA_VERSION, validate_job, validate_record
from src.adapters.datasets.paired_images import PairedPngAdapter, read_pair
from src.adapters.execution.local import LocalExecutor
from src.adapters.generation.capabilities import BACKENDS
from src.adapters.storage.local import file_hash, source_path, reserve_output
from src.pipeline.cache import (
    prepare_cache,
    resized_camera,
    validate_membership,
    verify_npz_arrays,
)

PROJECT = Path(__file__).resolve().parents[1]


class AdapterPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.raw = self.base / "customer_inputs"
        self.raw.mkdir()
        self.rgb = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
        self.depth = np.array(
            [[0, 100, 200, 300], [400, 500, 600, 65535]], dtype=np.uint16
        )
        Image.fromarray(self.rgb).save(self.raw / "colour.png")
        Image.fromarray(self.depth).save(self.raw / "measured.png")
        self.record = {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": "customer_fixture",
            "sample_id": "capture-A",
            "group_id": "survey-A",
            "split": "train",
            "timestamp_seconds": 0.0,
            "timestamp_basis": "capture_relative",
            "image_size": [4, 2],
            "camera": {
                "sensor_id": "different_camera",
                "image_convention": "declared_pinhole",
                "intrinsics": [4.0, 4.0, 1.5, 0.5],
            },
            "modalities": {
                "rgb": {"status": "available", "path": "colour.png"},
                "depth": {
                    "status": "available",
                    "path": "measured.png",
                    "units": "centimetres",
                    "scale_to_metres": 0.01,
                    "invalid_raw_value": 65535,
                },
                "lidar": {
                    "status": "unsupported",
                    "reason": "Customer supplied no LiDAR",
                },
            },
            "source_metadata": {},
        }
        self.manifest = self.base / "manifest.jsonl"
        self.write_manifest([self.record])

    def tearDown(self):
        self.temporary.cleanup()

    def write_manifest(self, records):
        self.manifest.write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
        )

    def adapter(self):
        return PairedPngAdapter(self.raw, self.manifest, "customer_fixture")

    def job(self, **parameters):
        return {
            "schema_version": SCHEMA_VERSION,
            "job_id": "fixture_job",
            "kind": "prepare_rgb_depth_cache",
            "dataset_id": "customer_fixture",
            "source_manifest_sha256": file_hash(self.manifest),
            "parameters": {
                "resolution": [4, 2],
                "splits": ["train"],
                "depth_max_metres": 20,
                "cache_format": "canonical",
                **parameters,
            },
        }

    def test_other_customer_units_resolution_and_invalid_sentinel_round_trip(self):
        report = prepare_cache(self.adapter(), self.job(), self.base / "cache")
        with np.load(self.base / "cache/train.npz") as cache:
            np.testing.assert_array_equal(cache["rgb"][0], self.rgb)
            np.testing.assert_array_equal(cache["depth_raw"][0], self.depth)
            self.assertEqual(float(cache["depth_metres"][0, 0, 1]), 1.0)
            self.assertFalse(cache["depth_valid"][0, 0, 0])
            self.assertFalse(cache["depth_valid"][0, 1, 3])
            self.assertTrue(np.isnan(cache["depth_metres"][0, 1, 3]))
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["test_sensor_files_opened"], 0)

    def test_legacy_export_rejects_non_millimetre_source_without_quantization(self):
        before = (self.raw / "measured.png").read_bytes()
        with self.assertRaisesRegex(ValueError, "Legacy cache"):
            prepare_cache(
                self.adapter(),
                self.job(cache_format="legacy_mm"),
                self.base / "failed_cache",
            )
        self.assertFalse((self.base / "failed_cache/profile.json").exists())
        self.assertEqual(before, (self.raw / "measured.png").read_bytes())

    def test_test_split_is_rejected_before_decode(self):
        with patch.object(PairedPngAdapter, "read_rgb_depth") as decode:
            with self.assertRaises(ValidationError):
                prepare_cache(
                    self.adapter(), self.job(splits=["test"]), self.base / "bad"
                )
            decode.assert_not_called()

    def test_duplicate_ids_and_cross_split_groups_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_membership([self.record, self.record])
        other = copy.deepcopy(self.record)
        other.update(sample_id="capture-B", split="test")
        with self.assertRaisesRegex(ValueError, "crosses"):
            validate_membership([self.record, other])

    def test_path_escape_and_windows_drive_are_rejected(self):
        for path in (
            "../escape.png",
            "/escape.png",
            "C:/secret.png",
            "folder\\secret.png",
        ):
            with self.assertRaises(ValueError):
                source_path(self.raw, path)

    def test_source_manifest_hash_is_bound_to_job(self):
        job = self.job()
        job["source_manifest_sha256"] = "0" * 64
        with patch.object(PairedPngAdapter, "read_rgb_depth") as decode:
            with self.assertRaisesRegex(ValueError, "different dataset or manifest"):
                prepare_cache(self.adapter(), job, self.base / "bad")
            decode.assert_not_called()

    def test_declared_source_hash_detects_changed_payload(self):
        self.record["modalities"]["rgb"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "artifact hash"):
            read_pair(self.raw, self.record)

    def test_record_cannot_be_mutated_to_another_source(self):
        adapter = self.adapter()
        record = next(adapter.records())
        record["sample_id"] = "invented"
        with self.assertRaisesRegex(ValueError, "differs"):
            adapter.read_rgb_depth(record)
        self.assertEqual(next(adapter.records())["sample_id"], "capture-A")

    def test_native_shape_and_dtype_are_checked(self):
        self.record["image_size"] = [8, 4]
        with self.assertRaisesRegex(ValueError, "geometry"):
            read_pair(self.raw, self.record)
        self.record["image_size"] = [4, 2]
        Image.fromarray(self.rgb).save(self.raw / "measured.png")
        with self.assertRaisesRegex(ValueError, "uint16"):
            read_pair(self.raw, self.record)

    def test_nonfinite_metadata_and_missing_depth_units_are_rejected(self):
        changed = copy.deepcopy(self.record)
        changed["timestamp_seconds"] = float("nan")
        with self.assertRaises(ValueError):
            validate_record(changed)
        del changed["modalities"]["depth"]["units"]
        changed["timestamp_seconds"] = 0
        with self.assertRaises(ValueError):
            validate_record(changed)

    def test_resize_updates_intrinsics_and_preserves_source_metadata(self):
        camera = resized_camera(self.record, (2, 1))
        self.assertEqual(camera["intrinsics"], [2.0, 2.0, 0.5, 0.0])
        self.assertEqual(self.record["camera"]["intrinsics"], [4.0, 4.0, 1.5, 0.5])
        report = prepare_cache(
            self.adapter(), self.job(resolution=[2, 1]), self.base / "resized"
        )
        self.assertEqual(report["resolution"], [2, 1])

    def test_upscaling_and_aspect_warp_rejected_without_output(self):
        for resolution in ([8, 4], [2, 2]):
            with self.assertRaisesRegex(ValueError, "aspect ratio"):
                prepare_cache(
                    self.adapter(), self.job(resolution=resolution), self.base / "bad"
                )
            self.assertFalse((self.base / "bad").exists())

    def test_existing_and_input_outputs_are_protected(self):
        before = (self.raw / "colour.png").read_bytes()
        with self.assertRaises(ValueError):
            reserve_output(self.raw / "derivative", (self.raw,))
        with self.assertRaises(ValueError):
            reserve_output(self.base, (self.raw,))
        output = self.base / "existing"
        output.mkdir()
        with self.assertRaises(FileExistsError):
            reserve_output(output, (self.raw,))
        self.assertEqual(before, (self.raw / "colour.png").read_bytes())

    def test_unknown_and_machine_specific_job_fields_are_rejected(self):
        for field in ("python_executable", "root", "shell_command"):
            job = self.job()
            job[field] = "machine_specific"
            with self.assertRaises(ValidationError):
                validate_job(job)

    def test_unimplemented_engines_and_lidar_requests_fail_explicitly(self):
        with self.assertRaisesRegex(NotImplementedError, "lidar"):
            BACKENDS["blenderproc"].require({"rgb", "lidar"})
        with self.assertRaisesRegex(NotImplementedError, "pending"):
            BACKENDS["blenderproc"].require({"rgb"})
        with self.assertRaises(NotImplementedError):
            BACKENDS["isaac_sim"].require({"lidar"})

    def test_bounded_npz_verification_detects_value_or_schema_changes(self):
        path = self.base / "cache.npz"
        array = np.array([1.0, np.nan], dtype=np.float32)
        np.savez(path, depth=array)
        verify_npz_arrays(path, {"depth": array})
        with self.assertRaises(ValueError):
            verify_npz_arrays(
                path, {"depth": np.array([2.0, np.nan], dtype=np.float32)}
            )
        with self.assertRaises(ValueError):
            verify_npz_arrays(path, {"another_key": array})

    def test_worker_failure_cannot_create_success_result(self):
        runtime = {
            "dataset": {
                "adapter": "paired_png_manifest",
                "root": str(self.raw),
                "manifest": str(self.manifest),
            }
        }
        with patch(
            "src.adapters.execution.local.subprocess.run",
            return_value=SimpleNamespace(returncode=3),
        ):
            with self.assertRaisesRegex(RuntimeError, "Worker failed"):
                LocalExecutor(Path(sys.executable), PROJECT).run(
                    self.job(), runtime, self.base / "run"
                )
        result = json.loads((self.base / "run/result.json").read_text())
        self.assertEqual(result["status"], "failed")

    def test_real_local_worker_handles_another_customer_without_torwic(self):
        runtime = {
            "dataset": {
                "adapter": "paired_png_manifest",
                "root": str(self.raw),
                "manifest": str(self.manifest),
            }
        }
        result = LocalExecutor(Path(sys.executable), PROJECT).run(
            self.job(), runtime, self.base / "run", timeout_seconds=60
        )
        self.assertEqual(result["status"], "passed")
        self.assertTrue((self.base / "run/dataset/source_provenance.jsonl").exists())

    def test_torwic_adapter_import_has_no_prior_path_setup_dependency(self):
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from src.adapters.datasets.torwic import TorwicAdapter",
            ],
            cwd=PROJECT,
            check=True,
            timeout=30,
        )

    def test_core_imports_do_not_import_datasets_or_simulators(self):
        code = "import sys; import src.core.contracts; assert not any(name.startswith(('bpy','blenderproc','isaacsim','src.adapters.datasets','torch')) for name in sys.modules)"
        subprocess.run(
            [sys.executable, "-c", code], cwd=PROJECT, check=True, timeout=30
        )


if __name__ == "__main__":
    unittest.main()
