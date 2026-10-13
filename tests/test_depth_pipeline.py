"""Contracts for relocated depth workflows, import safety and retained measurements."""

import importlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training.runtime import PROJECT_ROOT, require_cached_ids, require_output
from src.training.depth_baseline import check_no_leakage
from src.training.depth_context import checks as context_checks, ContextModel
from src.training.depth_quality import QualityModel, corrupt
from src.inference.depth_quality import restore
from src.inference.depth_context import complete


class DepthPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_importing_audits_never_reads_data_or_writes_reports(self):
        names = (
            "src.validation.depth_baseline_diagnostics",
            "src.validation.depth_context_assessment",
            "src.validation.depth_stress_audit",
        )
        for name in names:
            with patch("numpy.load") as arrays, patch(
                "torch.load"
            ) as weights, patch.object(Path, "mkdir") as mkdir:
                module = importlib.import_module(name)
                importlib.reload(module)
                arrays.assert_not_called()
                weights.assert_not_called()
                mkdir.assert_not_called()

    def test_raw_inputs_and_existing_results_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "input"
            source.mkdir()
            with self.assertRaises(ValueError):
                require_output(source / "result.npz", sources=[source])
            with self.assertRaises(ValueError):
                require_output(PROJECT_ROOT / "data/raw/result.npz")
            existing = Path(temporary) / "result.npz"
            existing.write_bytes(b"preserve")
            with self.assertRaises(ValueError):
                require_output(existing)
            self.assertEqual(existing.read_bytes(), b"preserve")

    def test_training_ids_reject_duplicates_cross_splits_and_test(self):
        self.assertEqual(
            require_cached_ids(["Aisle_CCW_Run_1/000001"], "train"),
            ["Aisle_CCW_Run_1/000001"],
        )
        for ids in (
            [],
            ["Aisle_CCW_Run_1/000001"] * 2,
            ["Aisle_CCW/000001"],
            ["Aisle_CW_Run_1/000001"],
        ):
            with self.assertRaises(ValueError):
                require_cached_ids(ids, "train")

    def test_original_leakage_and_rgb_ablation_contracts_still_pass(self):
        check_no_leakage()
        context_checks()

    def test_quality_corruption_keeps_hidden_targets_out_of_inputs(self):
        rgb = np.zeros((180, 320, 3), dtype=np.uint8)
        depth = np.ones((180, 320), dtype=np.float32)
        donors = np.full((2, 180, 320), 1000, dtype=np.uint16)
        donors[:, 30:100, 60:200] = 0
        inputs, _, hidden, _, _ = corrupt(rgb, depth, donors, 0, 4242, "empirical")
        changed = depth.copy()
        changed[hidden] = 15
        other = corrupt(rgb, changed, donors, 0, 4242, "empirical")[0]
        np.testing.assert_array_equal(inputs, other)

    def test_quality_inference_preserves_observed_measurements_on_cpu(self):
        depth = np.full((20, 32), 2.0, dtype=np.float32)
        depth[5:12, 8:24] = 0
        model = QualityModel().eval()
        result, observed = restore(model, depth)
        np.testing.assert_allclose(result[observed], depth[observed], atol=1e-5)
        self.assertTrue(np.isfinite(result).all())
        with self.assertRaises(ValueError):
            restore(model, np.zeros_like(depth))

    def test_context_inference_retains_original_measurements(self):
        depth = np.full((20, 32), 3.0, dtype=np.float32)
        depth[5:12, 8:24] = 0
        rgb = np.zeros((20, 32, 3), dtype=np.uint8)
        result, observed = complete(
            ContextModel(False).eval(), rgb, depth, device="cpu"
        )
        np.testing.assert_allclose(result[observed], depth[observed], atol=1e-5)

    def test_exported_denoising_gate_fails_before_creating_output(self):
        from src.inference import depth_quality

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "prediction.npz"
            args = [
                "predict",
                "--checkpoint",
                str(Path(temporary) / "model.pt"),
                "--depth",
                str(Path(temporary) / "depth.png"),
                "--output",
                str(output),
                "--device",
                "cpu",
                "--denoise",
            ]
            with patch.object(sys, "argv", args), patch(
                "torch.load", return_value={"denoising_allowed": False}
            ):
                with self.assertRaisesRegex(ValueError, "Denoising is blocked"):
                    depth_quality.main()
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
