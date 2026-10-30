"""Verify explicit segmentation backbone contracts and incompatible weights."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from src.adapters.storage.local import file_hash
from src.training.goods_model import build_model, validate_configuration


class GoodsBackboneTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(
            (
                Path(__file__).parents[1]
                / "configs/benchmarks/warehouse_goods_negative_replay.json"
            ).read_text()
        )

    def test_resnet_and_mobile_are_explicit_two_class_segmenters(self):
        for name in ["deeplabv3_mobilenet_v3_large", "deeplabv3_resnet50"]:
            with self.subTest(model=name):
                config = copy.deepcopy(self.config)
                config["model"] = name
                validate_configuration(config)
                model = build_model(config)
                self.assertEqual(model.classifier[4].out_channels, 2)
                self.assertEqual(model.aux_classifier[4].out_channels, 2)
                self.assertEqual(
                    hasattr(model.backbone, "layer4"), name == "deeplabv3_resnet50"
                )

    def test_unknown_or_nonfixture_models_do_not_fall_back(self):
        for name in ["resnet50", "deeplabv3_resnet101", "fixture_conv", "unknown"]:
            with self.subTest(model=name):
                config = dict(self.config, model=name)
                with self.assertRaises(ValueError):
                    validate_configuration(config)
                with self.assertRaisesRegex(ValueError, "Unsupported goods model"):
                    build_model(config)

    def test_wrong_checkpoint_checksum_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "wrong.pth"
            checkpoint.write_bytes(b"not the declared pretrained weights")
            config = dict(self.config, model="deeplabv3_resnet50")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                build_model(config, checkpoint)

    def test_other_backbone_weights_cannot_be_accepted_as_initialization(self):
        import torch

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "mobile_weights.pth"
            mobile = build_model(self.config)
            torch.save(mobile.state_dict(), checkpoint)
            config = dict(
                self.config,
                model="deeplabv3_resnet50",
                initial_checkpoint_sha256=file_hash(checkpoint),
            )
            with self.assertRaisesRegex(ValueError, "beyond the declared class heads"):
                build_model(config, checkpoint)


if __name__ == "__main__":
    unittest.main()
