"""Check odd input geometry and the YOLO RGB/padding semantic contract."""
import importlib.util
import unittest

import torch

from src.training.additional_segmenters import VGG16UNet, YOLOSemanticAdapter


class AdditionalSegmenterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def test_vgg_unet_keeps_odd_native_grid(self):
        model = VGG16UNet().eval()
        with torch.inference_mode():
            logits = model(torch.zeros(1, 3, 65, 97))["out"]
        self.assertEqual(tuple(logits.shape), (1, 2, 65, 97))
        self.assertTrue(bool(torch.isfinite(logits).all()))

    @unittest.skipUnless(importlib.util.find_spec("ultralytics"), "Requires the isolated YOLO trial package")
    def test_yolo_uses_dense_logits_rgb_scaling_and_bottom_right_padding(self):
        model = YOLOSemanticAdapter().eval()
        rgb = torch.tensor([0.2, 0.4, 0.8])[None, :, None, None].expand(1, 3, 65, 97)
        inputs = (rgb - model.mean) / model.std
        captured = []
        hook = model.network.register_forward_pre_hook(lambda module, args: captured.append(args[0].clone()))
        with torch.inference_mode():
            logits = model(inputs)["out"]
        hook.remove()
        self.assertEqual(tuple(logits.shape), (1, 2, 65, 97))
        self.assertEqual(tuple(captured[0].shape), (1, 3, 96, 128))
        self.assertTrue(torch.allclose(captured[0][:, :, :65, :97], rgb, atol=1e-6))
        self.assertEqual(int(torch.count_nonzero(captured[0][:, :, 65:])), 0)
        self.assertEqual(int(torch.count_nonzero(captured[0][:, :, :, 97:])), 0)
        self.assertTrue(bool(torch.isfinite(logits).all()))

    def test_vgg_rejects_incomplete_encoder_weights(self):
        model = VGG16UNet()
        with self.assertRaisesRegex(ValueError, "checkpoint layout mismatch"):
            model.load_encoder({})


if __name__ == "__main__":
    unittest.main()
