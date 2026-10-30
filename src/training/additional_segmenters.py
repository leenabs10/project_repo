"""Experimental VGG U-Net and YOLO semantic adapters for canonical RGB tensors.

Inputs are N x 3 x H x W float32 ImageNet-normalized RGB. Outputs are dictionaries
containing N x 2 x H x W logits. Neither adapter downloads assets or starts jobs.
Class-output heads are initialized for background/goods; ignore ID 255 belongs
to the shared loss, never a third prediction class. Unsupported packages,
checkpoint bindings or tensor layouts fail rather than changing tasks.
"""
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional

from src.adapters.storage.local import file_hash


class DecoderBlock(nn.Module):
    """Upsample features to the skip geometry, preserving the input label grid."""

    def __init__(self, input_channels, skip_channels, output_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels + skip_channels, output_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, features, skip):
        features = functional.interpolate(features, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.layers(torch.cat([features, skip], dim=1))


class VGG16UNet(nn.Module):
    """Experimental U-Net: VGG16 convolutional encoder and four decoder stages.

    The encoder retains VGG16 convolutional blocks through relu5_3, with four
    pooling operations. Decoder widths are 256/128/64/32; two convolutions per
    stage, no decoder BatchNorm. Odd dimensions are matched to skip feature
    geometry, so outputs use the original H/W without resizing categorical IDs.
    This is a declared local architecture, not an SMP implementation or checkpoint.
    """

    def __init__(self):
        super().__init__()
        from torchvision.models import vgg16

        encoder = vgg16(weights=None).features
        layers = list(encoder.children())
        self.encoder = nn.ModuleList([
            nn.Sequential(*layers[start:end])
            for start, end in [(0, 4), (4, 9), (9, 16), (16, 23), (23, 30)]
        ])
        self.decoder = nn.ModuleList([
            DecoderBlock(512, 512, 256), DecoderBlock(256, 256, 128),
            DecoderBlock(128, 128, 64), DecoderBlock(64, 64, 32),
        ])
        self.classifier = nn.Conv2d(32, 2, 1)

    def forward(self, inputs):
        skips = []
        features = inputs
        for stage in self.encoder:
            features = stage(features)
            skips.append(features)
        for block, skip in zip(self.decoder, reversed(skips[:-1])):
            features = block(features, skip)
        return {"out": self.classifier(features)}

    def load_encoder(self, state):
        """Load all VGG features strictly; discard only its classification layers."""
        starts = [0, 4, 9, 16, 23]
        expected = set()
        for index, stage in enumerate(self.encoder):
            selected = {}
            for key, target in stage.state_dict().items():
                layer, parameter = key.split(".", 1)
                source = f"features.{starts[index] + int(layer)}.{parameter}"
                expected.add(source)
                if source not in state or state[source].shape != target.shape:
                    raise ValueError("VGG feature checkpoint layout mismatch")
                selected[key] = state[source]
            stage.load_state_dict(selected, strict=True)
        if set(key for key in state if not key.startswith("classifier.")) != expected:
            raise ValueError("Unexpected VGG pretrained tensors")


class YOLOSemanticAdapter(nn.Module):
    """Pinned YOLO26n semantic model, with explicit preprocessing and geometry.

    Undo canonical ImageNet normalization to obtain the model's RGB [0,1] input.
    Pad only bottom/right to stride 32 with black RGB, upsample the main logits
    to that padded size, and crop back to H/W. Auxiliary logits are not used,
    matching the shared primary-only cross-entropy protocol. Instance outputs
    are never substituted for dense semantic logits.
    """

    def __init__(self):
        super().__init__()
        try:
            import ultralytics
            from ultralytics.nn.tasks import SemanticSegmentationModel
            from ultralytics.utils import YAML
        except ImportError as error:
            raise ValueError("YOLO semantic trial requires isolated ultralytics 8.4.192") from error
        if ultralytics.__version__ != "8.4.192":
            raise ValueError("Unsupported YOLO semantic package version")
        definition = Path(ultralytics.__file__).parent / "cfg/models/26/yolo26-sem.yaml"
        configuration = deepcopy(YAML.load(definition))
        configuration.update(scale="n", nc=2)
        self.network = SemanticSegmentationModel(configuration, ch=3, nc=2, verbose=False)
        if type(self.network.model[-1]).__name__ != "SemanticSegment":
            raise ValueError("Expected the dense semantic YOLO head")
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def forward(self, inputs):
        height, width = inputs.shape[-2:]
        rgb = inputs * self.std + self.mean
        pad_height, pad_width = (-height) % 32, (-width) % 32
        rgb = functional.pad(rgb, (0, pad_width, 0, pad_height), value=0)
        output = self.network(rgb)
        logits = output[0] if isinstance(output, tuple) else output
        if not isinstance(logits, torch.Tensor) or logits.ndim != 4 or logits.shape[1] != 2:
            raise ValueError("YOLO did not return two-class semantic logits")
        logits = functional.interpolate(logits, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
        return {"out": logits[:, :, :height, :width]}

    def load_pretrained(self, saved):
        """Load converted tensor-only official weights, replacing four output tensors."""
        if saved.get("package_version") != "8.4.192":
            raise ValueError("YOLO checkpoint package binding mismatch")
        state = saved["state_dict"]
        target = self.network.state_dict()
        index = len(self.network.model) - 1
        heads = {f"model.{index}.{head}.1.{parameter}"
            for head in ["classifier", "aux_head"] for parameter in ["weight", "bias"]}
        if set(state) != set(target):
            raise ValueError("YOLO checkpoint architecture keys differ")
        mismatches = {key for key in state if state[key].shape != target[key].shape}
        if mismatches != heads:
            raise ValueError("YOLO checkpoint differs beyond declared class outputs")
        compatible = {key: value for key, value in state.items() if key not in heads}
        loaded = self.network.load_state_dict(compatible, strict=False)
        if set(loaded.missing_keys) != heads or loaded.unexpected_keys:
            raise ValueError("YOLO pretrained tensor transfer failed")


def build_additional_segmenter(config, checkpoint=None):
    """Build the declared experimental model with an optional SHA-bound local initializer."""
    name = config["model"]
    if name == "unet_vgg16":
        model = VGG16UNet()
    elif name == "yolo26n_semantic":
        model = YOLOSemanticAdapter()
    else:
        raise ValueError("Unsupported additional segmentation architecture")
    if checkpoint is not None:
        if file_hash(checkpoint) != config["initial_checkpoint_sha256"]:
            raise ValueError("Initialization checkpoint checksum mismatch")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if name == "unet_vgg16":
            model.load_encoder(saved)
        else:
            model.load_pretrained(saved)
    return model
