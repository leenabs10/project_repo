"""Goods model, loss, optimizer, and frozen-configuration contracts.

Model construction performs no downloads. PyTorch is imported only when a model,
optimizer, or loss is requested. Invalid configuration and checkpoint bindings
raise ValueError; this module never starts training or writes artifacts.
"""

import math

from src.adapters.storage.local import file_hash
from src.training.auxiliary_inputs import (
    validate_auxiliary_sampling,
    validate_auxiliary_settings,
)
from src.training.hybrid_inputs import validate_mixing
from src.training.scene_inputs import validate_scene_mixing


def validate_loss_configuration(settings):
    """Validate two-class cross-entropy without silently changing class costs.

    Optional weights are ordered [background, goods]. Both must be finite and
    positive. Ignore ID 255 never receives a weight or contributes supervision.
    """
    if (
        not isinstance(settings, dict)
        or set(settings) - {"name", "ignore_index", "class_weights"}
        or settings.get("name") != "cross_entropy"
        or type(settings.get("ignore_index")) is not int
        or settings["ignore_index"] != 255
    ):
        raise ValueError("Unsupported goods loss configuration")
    if "class_weights" in settings:
        weights = settings["class_weights"]
        if (
            not isinstance(weights, list)
            or len(weights) != 2
            or any(
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value <= 0
                for value in weights
            )
        ):
            raise ValueError("Class weights require two finite positive values")


def segmentation_loss(logits, targets, settings):
    """Return mean supervised CE and its normalization mass for epoch accounting.

    Logits are N x 2 x H x W; targets are N x H x W integer IDs 0/1/255.
    Weighted CE divides by the sum of valid target-class weights. Record that
    same denominator when aggregating batches; ignored logits have zero gradient.
    Missing weights preserve the existing unweighted objective exactly.
    """
    import torch
    from torch.nn import functional

    validate_loss_configuration(settings)
    weights = settings.get("class_weights")
    weight_tensor = (
        None
        if weights is None
        else torch.tensor(weights, dtype=logits.dtype, device=logits.device)
    )
    valid = targets != 255
    if not bool(valid.any()):
        raise ValueError("Training batch has no supervised pixels")
    mass = (
        float(valid.sum())
        if weight_tensor is None
        else float(weight_tensor[targets[valid]].sum())
    )
    return (
        functional.cross_entropy(
            logits, targets, weight=weight_tensor, ignore_index=255
        ),
        mass,
    )


def checkpoint_selection_metric(config):
    """Return the effective selection metric, preserving frozen legacy metadata.

    Fixed-final runs do not inspect development scores to select a checkpoint.
    Earlier frozen configurations named validation IoU despite that policy; accept
    those exact legacy declarations without rewriting saved experiment evidence.
    New fixed-final configurations explicitly use null for the selection metric.
    """
    policy = config.get("checkpoint_policy", "best_validation_iou")
    metric = config.get("selection_metric", "validation_goods_iou")
    if policy == "final_epoch" and metric in (None, "validation_goods_iou"):
        return None
    if policy == "best_validation_iou" and metric == "validation_goods_iou":
        return metric
    raise ValueError("Unsupported or contradictory checkpoint-selection declaration")


def validate_configuration(config, *, fixture=False):
    """Reject silent preprocessing/model changes and unsupported precision modes."""
    models = {"deeplabv3_mobilenet_v3_large"}
    if fixture:
        models.add("fixture_conv")
    if (
        config.get("model") not in models
        or config.get("experiment_kind")
        not in {"real_only", "real_plus_synthetic", "real_plus_scene"}
        or config.get("resampling")
        != {
            "rgb": "BOX",
            "target": "NEAREST",
            "evaluation_logits_to_native": "BILINEAR",
        }
        or config.get("optimizer", {}).get("name") != "AdamW"
        or config.get("batch_norm") != "frozen_running_statistics"
        or config.get("normalization")
        != {"scale": 255, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    ):
        raise ValueError("Unsupported goods benchmark configuration")
    validate_loss_configuration(config.get("loss"))
    validate_auxiliary_sampling(config)
    if config.get("checkpoint_policy", "best_validation_iou") not in {
        "best_validation_iou",
        "final_epoch",
    }:
        raise ValueError("Unsupported checkpoint-selection policy")
    checkpoint_selection_metric(config)
    auxiliary = config.get("auxiliary_training")
    if auxiliary is not None:
        validate_auxiliary_settings(auxiliary)
        if (
            config["experiment_kind"] not in {"real_only", "real_plus_scene"}
            or type(config.get("generation_seed")) is not int
            or config["generation_seed"] < 0
            or (
                config["experiment_kind"] == "real_plus_scene"
                and config.get("synthetic_mixing") != {"strategy": "auxiliary_only"}
            )
        ):
            raise ValueError(
                "Auxiliary runs require preserved real exposure and matched sources"
            )
    elif config.get("synthetic_mixing") == {"strategy": "auxiliary_only"}:
        raise ValueError(
            "Auxiliary-only scenes require an explicit bounded loss policy"
        )
    if config["experiment_kind"] != "real_only":
        if config["experiment_kind"] == "real_plus_scene":
            validate_scene_mixing(config.get("synthetic_mixing"))
        else:
            validate_mixing(config.get("synthetic_mixing"))
        if (
            type(config.get("generation_seed")) is not int
            or config["generation_seed"] < 0
        ):
            raise ValueError("Explicit nonnegative generation seed is required")
    elif "synthetic_mixing" in config:
        raise ValueError("Real-only configurations cannot declare synthetic mixing")
    required_packet = config.get("required_reference_packet_sha256")
    if required_packet is not None and (
        not isinstance(required_packet, str)
        or len(required_packet) != 64
        or any(character not in "0123456789abcdef" for character in required_packet)
    ):
        raise ValueError("Required reference packet must be a SHA256 digest")
    for field in ("training_resolution", "native_reference_resolution"):
        if len(config[field]) != 2 or any(
            type(value) is not int or value <= 0 for value in config[field]
        ):
            raise ValueError("Invalid benchmark image dimensions")
    if any(
        type(config[field]) is not int or config[field] <= 0
        for field in ("epochs", "batch_size")
    ):
        raise ValueError("Positive epoch/batch budgets are required")
    if not config["model_seeds"] or any(
        type(seed) is not int or seed < 0 for seed in config["model_seeds"]
    ):
        raise ValueError("Explicit nonnegative model seeds are required")
    if (
        any(
            not math.isfinite(config["optimizer"][name])
            or config["optimizer"][name] < 0
            for name in ("learning_rate", "weight_decay")
        )
        or config["optimizer"]["learning_rate"] == 0
    ):
        raise ValueError("Invalid optimizer parameters")
    multiplier = config["optimizer"].get("backbone_learning_rate_multiplier", 1.0)
    if (
        not isinstance(multiplier, (int, float))
        or isinstance(multiplier, bool)
        or not math.isfinite(multiplier)
        or not 0 < multiplier <= 1
    ):
        raise ValueError("Backbone learning-rate multiplier must be in (0, 1]")


def build_optimizer(model, config):
    """Build AdamW with an explicit optional lower rate for pretrained features.

    A small reference cohort can overfit when every pretrained layer receives
    the classification head's rate. Parameter groups change only update rates;
    targets, sampling, loss, epoch budget and initialization remain unchanged.
    The default multiplier preserves the original flat-rate baseline.
    """
    import torch

    settings = config["optimizer"]
    learning_rate = settings["learning_rate"]
    multiplier = settings.get("backbone_learning_rate_multiplier", 1.0)
    if multiplier == 1:
        parameters = model.parameters()
    else:
        backbone, head = [], []
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                (backbone if name.startswith("backbone.") else head).append(parameter)
        if not backbone or not head:
            raise ValueError(
                "Differential learning rates require backbone and head parameters"
            )
        parameters = [
            {"params": backbone, "lr": learning_rate * multiplier, "name": "backbone"},
            {"params": head, "lr": learning_rate, "name": "head"},
        ]
    return torch.optim.AdamW(
        parameters, lr=learning_rate, weight_decay=settings["weight_decay"]
    )


def build_model(config, checkpoint=None, *, fixture=False):
    """Create models without network access; replace only declared class heads."""
    import torch

    if config["model"] == "fixture_conv" and fixture:

        class FixtureModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = torch.nn.Conv2d(3, 2, 1)

            def forward(self, inputs):
                return {"out": self.head(inputs)}

        return FixtureModel()
    from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

    model = deeplabv3_mobilenet_v3_large(
        weights=None, weights_backbone=None, num_classes=2, aux_loss=True
    )
    if checkpoint is not None:
        if file_hash(checkpoint) != config["initial_checkpoint_sha256"]:
            raise ValueError("Initialization checkpoint checksum mismatch")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        target = model.state_dict()
        compatible = {
            name: value
            for name, value in state.items()
            if name in target and value.shape == target[name].shape
        }
        heads = {
            "classifier.4.weight",
            "classifier.4.bias",
            "aux_classifier.4.weight",
            "aux_classifier.4.bias",
        }
        result = model.load_state_dict(compatible, strict=False)
        if (
            set(result.missing_keys) != heads
            or result.unexpected_keys
            or set(state) - set(compatible) != heads
        ):
            raise ValueError("Initialization differs beyond the declared class heads")
    return model
