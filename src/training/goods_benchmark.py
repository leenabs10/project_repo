"""Goods training with qualified RGB conditions and fixed native evaluation."""

import argparse
from collections import Counter
import json
import logging
import math
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image

from src.adapters.storage.local import file_hash, reserve_output, source_path
from src.training.auxiliary_inputs import (
    balanced_auxiliary_loss,
    prepare_auxiliary_donors,
    select_auxiliary_donor,
    validate_auxiliary_sampling,
    validate_auxiliary_settings,
)
from src.training.scene_inputs import (
    load_scene_inputs,
    select_scene_training_sample,
    validate_scene_mixing,
)
from src.training.hybrid_inputs import (
    load_synthetic_inputs,
    select_training_input,
    validate_mixing,
)
from src.adapters.tasks.goods import (
    confusion_counts,
    load_definition,
    segmentation_metrics,
    validate_target,
)
from src.validation.artifacts import (
    artifact_hashes,
    verify_snapshot,
    write_implementation_snapshot,
    write_json,
)


def load_references(root, expected_sha, task, *, fixture=False):
    """Validate provenance, splits, identities and native targets before model I/O.

    Evaluation reads only the exported development cohort. There is no held-out
    test flag, source-dataset scan, or automatic pretrained-weight download.
    """
    root, task = Path(root), Path(task)
    profile = verify_snapshot(root, expected_sha)
    definition = load_definition(task)
    if profile.get("status") != "passed" or not profile.get(
        "benchmark_reference_ready"
    ):
        raise ValueError("Qualified references are required before benchmark execution")
    if (
        profile["protocol_sha256"] != file_hash(task)
        or profile["task_id"] != definition["task_id"]
    ):
        raise ValueError("Reference task/protocol mismatch")
    rows = [
        json.loads(line)
        for line in (root / "references.jsonl").read_text().splitlines()
    ]
    expected_origin = (
        "fixture_ground_truth" if fixture else definition["reference_origin"]
    )
    if fixture != (profile.get("purpose") == "known_fixture_only"):
        raise ValueError("Fixture evidence cannot be used as a real benchmark")
    seen, groups, rgb_splits = set(), {}, {}
    counts = Counter()
    for row in rows:
        if row["split"] not in {"train", "validation"}:
            raise ValueError("Held-out test references are not supported by this CLI")
        if row["sample_id"] in seen:
            raise ValueError("Duplicate reference sample")
        seen.add(row["sample_id"])
        if row["reference_origin"] != expected_origin or (
            not fixture and row.get("reviewer_kind", "human") != "human"
        ):
            raise ValueError(
                "AI/predicted targets cannot enter the real reference benchmark"
            )
        if (
            row["protocol_sha256"] != file_hash(task)
            or row["task_id"] != definition["task_id"]
        ):
            raise ValueError("Reference row task binding mismatch")
        group = row["group_id"]
        if group in groups and groups[group] != row["split"]:
            raise ValueError("A complete capture group cannot cross dataset splits")
        groups[group] = row["split"]
        digest = row["source_rgb_sha256"]
        if digest in rgb_splits and rgb_splits[digest] != row["split"]:
            raise ValueError("RGB payload is reused across train/development splits")
        rgb_splits[digest] = row["split"]
        rgb_path, target_path = source_path(root, row["rgb_path"]), source_path(
            root, row["target_path"]
        )
        if (
            file_hash(rgb_path) != digest
            or file_hash(target_path) != row["target_sha256"]
        ):
            raise ValueError("Reference payload hash mismatch")
        with Image.open(rgb_path) as image:
            if image.mode != "RGB" or list(image.size) != row["image_size"]:
                raise ValueError("Reference RGB geometry/encoding mismatch")
        with Image.open(target_path) as image:
            target = np.array(image)
        validate_target(
            target,
            tuple(reversed(row["image_size"])),
            definition["maximum_ignore_fraction"],
        )
        counts[row["split"]] += 1
    if dict(counts) != profile["reference_counts"] or any(
        counts[split] < minimum
        for split, minimum in definition["minimum_reference_frames"].items()
    ):
        raise ValueError("Reference cohort is incomplete or counts changed")
    if not fixture and profile.get("qualified_reference_counts", dict(counts)) != dict(
        counts
    ):
        raise ValueError("Qualified reference counts disagree with human provenance")
    return rows


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


def input_tensor(root, row, config):
    """Resize RGB only with BOX and normalize channels in float32."""
    import torch

    with Image.open(source_path(root, row["rgb_path"])) as image:
        rgb = np.array(
            image.resize(tuple(config["training_resolution"]), Image.Resampling.BOX),
            copy=True,
        )
    value = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255
    mean = torch.tensor(config["normalization"]["mean"])[:, None, None]
    std = torch.tensor(config["normalization"]["std"])[:, None, None]
    return (value - mean) / std


def target_tensor(root, row, config):
    """Nearest-neighbour target resize preserves integer IDs and ignore regions."""
    import torch

    with Image.open(source_path(root, row["target_path"])) as image:
        target = np.array(
            image.resize(
                tuple(config["training_resolution"]), Image.Resampling.NEAREST
            ),
            copy=True,
        )
    return torch.from_numpy(target.astype(np.int64))


def native_metrics(model, root, rows, config, device):
    """Upsample logits to native size before argmax; score nonignored pixels only."""
    import torch
    from torch.nn import functional as functional

    model.eval()
    total = {
        key: 0
        for key in (
            "true_positive",
            "false_positive",
            "false_negative",
            "true_negative",
            "valid_pixels",
        )
    }
    frames, native_pixels = [], 0
    with torch.no_grad():
        for row in rows:
            logits = model(input_tensor(root, row, config)[None].to(device))["out"]
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite evaluation logits")
            logits = functional.interpolate(
                logits,
                size=tuple(reversed(row["image_size"])),
                mode="bilinear",
                align_corners=False,
            )
            prediction = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
            with Image.open(source_path(root, row["target_path"])) as image:
                target = np.array(image)
            counts = confusion_counts(prediction, target)
            frames.append(
                {"sample_id": row["sample_id"], **segmentation_metrics(counts)}
            )
            for key in total:
                total[key] += counts[key]
            native_pixels += target.size
    defined = [frame["goods_iou"] for frame in frames if frame["goods_iou"] is not None]
    return {
        **segmentation_metrics(total),
        "confusion": total,
        "reference_coverage": total["valid_pixels"] / native_pixels,
        "frame_mean_goods_iou": sum(defined) / len(defined) if defined else None,
        "frames": frames,
        "score_resolution": "native",
        "test_sensor_files_opened": 0,
    }


def train_run(
    references,
    references_sha,
    task,
    configuration,
    checkpoint,
    output,
    seed,
    device="cuda",
    *,
    fixture=False,
    synthetic=None,
    synthetic_sha=None,
):
    """Train a fresh matched run; reject unqualified inputs before GPU allocation.

    Qualified conditions replace their own source RGB in existing training slots.
    Epochs, batches, source targets, shuffle RNG and native evaluation are shared
    with the real-only control. No additional optimization steps are introduced.
    """
    references, task, configuration = map(Path, (references, task, configuration))
    config = json.loads(configuration.read_text())
    validate_configuration(config, fixture=fixture)
    rows = load_references(references, references_sha, task, fixture=fixture)
    required_packet = config.get("required_reference_packet_sha256")
    if required_packet is not None:
        profile = verify_snapshot(references, references_sha)
        if profile.get("packet_profile_sha256") != required_packet:
            raise ValueError(
                "The required reviewed annotation revision is not qualified in this cohort"
            )
    alternatives = {}
    hybrid = config["experiment_kind"] != "real_only"
    scene_hybrid = config["experiment_kind"] == "real_plus_scene"
    if hybrid != (synthetic is not None and synthetic_sha is not None) or (
        not hybrid and (synthetic is not None or synthetic_sha is not None)
    ):
        raise ValueError("Hybrid runs require both admitted dataset path and checksum")
    if hybrid:
        synthetic = Path(synthetic)
        loader = load_scene_inputs if scene_hybrid else load_synthetic_inputs
        alternatives = loader(
            synthetic,
            synthetic_sha,
            references,
            references_sha,
            rows,
            config,
            fixture=fixture,
        )
    auxiliary_settings = config.get("auxiliary_training")
    auxiliary_donors = (
        prepare_auxiliary_donors(references, rows, synthetic, alternatives, config)
        if auxiliary_settings is not None
        else []
    )
    if seed not in config["model_seeds"]:
        raise ValueError("Seed is outside the frozen benchmark configuration")
    if any(row["image_size"] != config["native_reference_resolution"] for row in rows):
        raise ValueError("Native reference geometry differs from benchmark config")
    if config["model"] != "fixture_conv" and checkpoint is None:
        raise ValueError("An explicit local initialization checkpoint is required")
    if (
        checkpoint is not None
        and file_hash(checkpoint) != config["initial_checkpoint_sha256"]
    ):
        raise ValueError("Initialization checkpoint checksum mismatch")
    configuration_sha, task_sha = file_hash(configuration), file_hash(task)
    output = reserve_output(
        output,
        (
            references,
            task,
            configuration,
            Path(checkpoint) if checkpoint else references,
            Path(synthetic) if synthetic is not None else references,
            Path(__file__).parents[2] / "data/raw",
        ),
    )
    write_json(
        output / "profile.json", {"status": "running", "benchmark_complete": False}
    )
    start = time.perf_counter()
    try:
        import torch

        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model = build_model(config, checkpoint, fixture=fixture).to(device)
        optimizer = build_optimizer(model, config)
        training = [row for row in rows if row["split"] == "train"]
        validation = [row for row in rows if row["split"] == "validation"]
        generator = torch.Generator().manual_seed(seed)
        history, sampling = [], Counter()
        input_sampling, synthetic_presentations = Counter(), 0
        auxiliary_sampling, auxiliary_presentations = Counter(), 0
        auxiliary_presence_counts = Counter()
        updates, best_score, best_epoch = 0, -1.0, None
        for epoch in range(config["epochs"]):
            model.train()
            # Small cohorts should not update pretrained BN statistics from tiny
            # batches; affine parameters still receive ordinary gradients.
            for layer in model.modules():
                if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
                    layer.eval()
            order = torch.randperm(len(training), generator=generator).tolist()
            weighted_loss, supervised = 0.0, 0
            auxiliary_loss_sum, epoch_auxiliary_count = 0.0, 0
            for offset in range(0, len(order), config["batch_size"]):
                batch = [
                    training[index]
                    for index in order[offset : offset + config["batch_size"]]
                ]
                # A scene replaces RGB AND target. RGB-only derivatives retain
                # their original real target through the existing condition path.
                if auxiliary_settings is not None:
                    # Every original real slot remains present. The auxiliary
                    # stream adds gradients, never replaces RGB or real targets.
                    selected_inputs = [(references, row, False) for row in batch]
                elif scene_hybrid:
                    selected_inputs = [
                        select_scene_training_sample(
                            references, row, synthetic, alternatives, config, epoch
                        )
                        for row in batch
                    ]
                else:
                    selected_inputs = [
                        select_training_input(
                            references, row, alternatives, config, epoch
                        )
                        for row in batch
                    ]
                inputs = torch.stack(
                    [
                        input_tensor(root, row, config)
                        for root, row, _ in selected_inputs
                    ]
                ).to(device)
                input_sampling.update(row["sample_id"] for _, row, _ in selected_inputs)
                synthetic_presentations += sum(
                    changed for _, _, changed in selected_inputs
                )
                targets = torch.stack(
                    (
                        [
                            target_tensor(root, selected, config)
                            for root, selected, _ in selected_inputs
                        ]
                        if scene_hybrid
                        else [target_tensor(references, row, config) for row in batch]
                    )
                ).to(device)
                valid_count = int((targets != 255).sum())
                if not valid_count:
                    raise ValueError(
                        "Training batch has no supervised pixels after resizing"
                    )
                optimizer.zero_grad(set_to_none=True)
                logits = model(inputs)["out"]
                loss, loss_mass = segmentation_loss(logits, targets, config["loss"])
                if not torch.isfinite(loss) or not torch.isfinite(logits).all():
                    raise ValueError("Nonfinite training loss/output")
                loss.backward()
                if auxiliary_settings is not None:
                    donor = select_auxiliary_donor(
                        auxiliary_donors, config, updates + 1
                    )
                    if donor is not None:
                        donor_root, donor_row = donor
                        auxiliary_input = (
                            input_tensor(donor_root, donor_row, config)
                            .unsqueeze(0)
                            .to(device)
                        )
                        auxiliary_target = (
                            target_tensor(donor_root, donor_row, config)
                            .unsqueeze(0)
                            .to(device)
                        )
                        auxiliary_logits = model(auxiliary_input)["out"]
                        auxiliary_loss = balanced_auxiliary_loss(
                            auxiliary_logits,
                            auxiliary_target,
                            auxiliary_settings["foreground_fraction"],
                            missing_class_policy=auxiliary_settings.get(
                                "missing_class_policy", "reject"
                            ),
                        )
                        (auxiliary_settings["loss_weight"] * auxiliary_loss).backward()
                        auxiliary_sampling[donor_row["sample_id"]] += 1
                        if "auxiliary_target_kind" in donor_row:
                            auxiliary_presence_counts[
                                donor_row["auxiliary_target_kind"]
                            ] += 1
                        input_sampling[donor_row["sample_id"]] += 1
                        auxiliary_presentations += 1
                        synthetic_presentations += int(scene_hybrid)
                        auxiliary_loss_sum += float(auxiliary_loss.detach())
                        epoch_auxiliary_count += 1
                if any(
                    parameter.grad is not None
                    and not torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                ):
                    raise ValueError("Nonfinite training gradients")
                optimizer.step()
                updates += 1
                sampling.update(row["sample_id"] for row in batch)
                weighted_loss += float(loss.detach()) * loss_mass
                supervised += loss_mass
            # Fixed-final comparisons do not inspect development predictions
            # during earlier epochs; the legacy best-IoU policy is unchanged.
            evaluate_now = (
                config.get("checkpoint_policy", "best_validation_iou")
                == "best_validation_iou"
                or epoch + 1 == config["epochs"]
            )
            metrics = (
                native_metrics(model, references, validation, config, device)
                if evaluate_now
                else None
            )
            history.append(
                {
                    "epoch": epoch + 1,
                    "optimizer_updates": updates,
                    "train_cross_entropy": weighted_loss / supervised,
                    "validation": metrics,
                    **(
                        {
                            "auxiliary_presentations": epoch_auxiliary_count,
                            "auxiliary_balanced_loss": (
                                auxiliary_loss_sum / epoch_auxiliary_count
                                if epoch_auxiliary_count
                                else None
                            ),
                        }
                        if auxiliary_settings is not None
                        else {}
                    ),
                }
            )
            score = metrics["goods_iou"] if metrics is not None else None
            if evaluate_now and score is None:
                raise ValueError(
                    "Selection metric is undefined for this development cohort"
                )
            if score is not None and score > best_score:
                best_score, best_epoch = score, epoch + 1
                torch.save(
                    {
                        "state_dict": {
                            name: value.detach().cpu()
                            for name, value in model.state_dict().items()
                        },
                        "seed": seed,
                        "configuration_sha256": file_hash(configuration),
                        "reference_profile_sha256": references_sha,
                        "synthetic_profile_sha256": synthetic_sha,
                        "experiment_kind": config["experiment_kind"],
                        "protocol_sha256": file_hash(task),
                        "purpose": (
                            "fixture_only"
                            if fixture
                            else (
                                "hybrid_benchmark" if hybrid else "real_only_benchmark"
                            )
                        ),
                        "epoch": best_epoch,
                        "optimizer_updates": updates,
                    },
                    output / "best.pt",
                )
            # Keep bounded progress evidence so a long run can be inspected
            # without changing its sampling, optimizer or selection decisions.
            write_json(
                output / "progress.json",
                {
                    "seed": seed,
                    "epoch": epoch + 1,
                    "epochs": config["epochs"],
                    "optimizer_updates": updates,
                    "train_cross_entropy": weighted_loss / supervised,
                    "validation_goods_iou": score,
                    "best_epoch": best_epoch,
                    "best_validation_goods_iou": best_score,
                },
            )
            logging.getLogger(__name__).info(
                "Seed %s: epoch %s/%s, loss %.6f, development IoU %s",
                seed,
                epoch + 1,
                config["epochs"],
                weighted_loss / supervised,
                score,
            )
        # Reload the selected checkpoint, rather than reporting the last epoch.
        restored = build_model(config, fixture=fixture).to(device)
        saved = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
        restored.load_state_dict(saved["state_dict"], strict=True)
        metrics = native_metrics(restored, references, validation, config, device)
        if not math.isclose(
            metrics["goods_iou"], best_score, rel_tol=1e-7, abs_tol=1e-7
        ):
            raise ValueError("Selected checkpoint reload changed evaluation metrics")
        verify_snapshot(references, references_sha)
        if synthetic is not None:
            verify_snapshot(synthetic, synthetic_sha)
        if file_hash(configuration) != configuration_sha or file_hash(task) != task_sha:
            raise ValueError("Benchmark configuration/task changed during execution")
        if (
            checkpoint is not None
            and file_hash(checkpoint) != config["initial_checkpoint_sha256"]
        ):
            raise ValueError("Initialization checkpoint changed during execution")
        shutil_copy(configuration, output / "configuration.json")
        shutil_copy(task, output / "task.json")
        write_json(output / "history.json", history)
        write_json(output / "validation_metrics.json", metrics)
        write_json(output / "sampling.json", dict(sampling))
        write_json(output / "input_sampling.json", dict(input_sampling))
        if auxiliary_settings is not None:
            write_json(output / "auxiliary_sampling.json", dict(auxiliary_sampling))
        write_implementation_snapshot(output)
        report = {
            "status": "passed",
            "purpose": (
                "fixture_only"
                if fixture
                else ("hybrid_benchmark" if hybrid else "real_only_benchmark")
            ),
            "seed_run_complete": True,
            "benchmark_complete": False,
            "matched_comparison_complete": False,
            "synthetic_data_utility_measured": False,
            "held_out_test_evaluated": False,
            "seed": seed,
            "model": config["model"],
            "precision": "float32",
            "device": device,
            "reference_profile_sha256": references_sha,
            "protocol_sha256": file_hash(task),
            "configuration_sha256": file_hash(configuration),
            "best_checkpoint_sha256": file_hash(output / "best.pt"),
            "best_epoch": best_epoch,
            "optimizer_updates": updates,
            "expected_optimizer_updates": config["epochs"]
            * math.ceil(len(training) / config["batch_size"]),
            "real_training_frames": len(training),
            "development_frames": len(validation),
            "synthetic_training_frames": (
                len(alternatives)
                if scene_hybrid
                else sum(len(values) for values in alternatives.values())
            ),
            "real_presentations": sum(sampling.values())
            + auxiliary_presentations
            - synthetic_presentations,
            **(
                {
                    "primary_real_presentations": sum(sampling.values()),
                    "auxiliary_presentations": auxiliary_presentations,
                    "auxiliary_loss_weight": auxiliary_settings["loss_weight"],
                    "auxiliary_foreground_fraction": auxiliary_settings[
                        "foreground_fraction"
                    ],
                    **(
                        {
                            "auxiliary_target_presentations": dict(
                                auxiliary_presence_counts
                            )
                        }
                        if auxiliary_settings.get("missing_class_policy")
                        == "background_only_zero_foreground_mass"
                        else {}
                    ),
                }
                if auxiliary_settings is not None
                else {}
            ),
            "checkpoint_policy": config.get("checkpoint_policy", "best_validation_iou"),
            "mixing_unit": (
                "primary_real_plus_paired_auxiliary"
                if auxiliary_settings is not None
                else (
                    "RGB_and_rendered_target"
                    if scene_hybrid
                    else "RGB_only_real_target_unchanged"
                )
            ),
            "sampling_semantics": (
                "Every real source slot preserved; bounded auxiliary gradients share existing optimizer updates"
                if auxiliary_settings is not None
                else "Fixed real source slots; scene replacement reduces actual real presentations without adding updates"
            ),
            "synthetic_presentations": synthetic_presentations,
            "total_training_presentations": sum(sampling.values())
            + auxiliary_presentations,
            "synthetic_profile_sha256": synthetic_sha,
            "experiment_kind": config["experiment_kind"],
            "selected_validation_goods_iou": best_score,
            "strict_reload": "passed",
            "initial_checkpoint_sha256": file_hash(checkpoint) if checkpoint else None,
            "optimizer_parameter_groups": [
                {
                    "name": group.get("name", "all_parameters"),
                    "learning_rate": group["lr"],
                    "parameter_count": sum(
                        parameter.numel() for parameter in group["params"]
                    ),
                }
                for group in optimizer.param_groups
            ],
            "torch": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_name": (
                torch.cuda.get_device_name() if device == "cuda" else "cpu"
            ),
            "training_resolution": config["training_resolution"],
            "evaluation_resolution": config["native_reference_resolution"],
            "elapsed_seconds": time.perf_counter() - start,
            "peak_cuda_allocated_bytes": (
                torch.cuda.max_memory_allocated() if device == "cuda" else None
            ),
            "limitations": "Development selection is reused; video frames are dependent. Model-assisted references may retain anchoring bias. Fixture metrics are not customer accuracy or utility evidence.",
            "artifact_sha256": artifact_hashes(output),
        }
        write_json(output / "profile.json", report)
        return report
    except Exception as error:
        write_json(
            output / "profile.json",
            {"status": "failed", "error": str(error), "benchmark_complete": False},
        )
        raise


def shutil_copy(source, destination):
    """Preserve exact input bytes in a fresh run snapshot."""
    import shutil

    shutil.copyfile(source, destination)


def evaluate_run(
    run, run_sha, references, references_sha, output, device="cuda", *, fixture=False
):
    """Evaluate the selected checkpoint on the same locked development snapshot."""
    import torch

    run, references = Path(run), Path(references)
    profile = verify_snapshot(run, run_sha)
    allowed_purposes = (
        {"fixture_only"} if fixture else {"real_only_benchmark", "hybrid_benchmark"}
    )
    if profile["purpose"] not in allowed_purposes or profile["status"] != "passed":
        raise ValueError("Checkpoint purpose/status does not match this evaluation")
    if profile["reference_profile_sha256"] != references_sha:
        raise ValueError("Evaluation reference snapshot differs from model selection")
    rows = load_references(
        references, references_sha, run / "task.json", fixture=fixture
    )
    config = json.loads((run / "configuration.json").read_text())
    validate_configuration(config, fixture=fixture)
    expected_purpose = (
        "fixture_only"
        if fixture
        else (
            "hybrid_benchmark"
            if config["experiment_kind"] != "real_only"
            else "real_only_benchmark"
        )
    )
    if profile["purpose"] != expected_purpose:
        raise ValueError("Checkpoint purpose differs from configuration")
    saved = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    for key in (
        "reference_profile_sha256",
        "protocol_sha256",
        "configuration_sha256",
        "seed",
    ):
        if saved[key] != profile[key]:
            raise ValueError("Checkpoint metadata differs from verified run")
    if config["experiment_kind"] != "real_only" and (
        saved.get("synthetic_profile_sha256") != profile.get("synthetic_profile_sha256")
        or not saved.get("synthetic_profile_sha256")
        or saved.get("experiment_kind") != config["experiment_kind"]
    ):
        raise ValueError("Checkpoint synthetic provenance differs from its run")
    model = build_model(config, fixture=fixture).to(device)
    model.load_state_dict(saved["state_dict"], strict=True)
    output = reserve_output(
        output, (run, references, Path(__file__).parents[2] / "data/raw")
    )
    metrics = native_metrics(
        model,
        references,
        [row for row in rows if row["split"] == "validation"],
        config,
        device,
    )
    write_json(output / "metrics.json", metrics)
    result = {
        "status": "passed",
        "purpose": profile["purpose"],
        "split": "validation",
        "run_profile_sha256": run_sha,
        "reference_profile_sha256": references_sha,
        "test_sensor_files_opened": 0,
        "artifact_sha256": artifact_hashes(output),
    }
    write_json(output / "profile.json", result)
    return result


def fixture_run(output, *, production_model=False, checkpoint=None, device="cpu"):
    """Exercise the same train/save/reload/eval engine on generated known labels."""
    output = reserve_output(output, (Path(__file__).parents[2] / "data/raw",))
    task = json.loads(
        (
            Path(__file__).parents[2] / "configs/tasks/warehouse_goods_reviewed.json"
        ).read_text()
    )
    task["minimum_reference_frames"] = {"train": 4, "validation": 1}
    write_json(output / "task.json", task)
    config = json.loads(
        (
            Path(__file__).parents[2]
            / "configs/benchmarks/warehouse_goods_real_only.json"
        ).read_text()
    )
    config.update(epochs=1, batch_size=4, model_seeds=[17])
    if not production_model:
        config.update(
            model="fixture_conv",
            training_resolution=[16, 16],
            native_reference_resolution=[32, 32],
        )
    write_json(output / "config.json", config)
    root = output / "references"
    root.mkdir()
    width, height = config["native_reference_resolution"]
    rows = []
    for index, split in enumerate(["train"] * 4 + ["validation"]):
        target = np.zeros((height, width), np.uint8)
        target[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4] = 1
        target[: max(1, height // 20)] = 255
        rgb = np.zeros((height, width, 3), np.uint8)
        rgb[:] = [20 + index * 7, 65, 90]
        rgb[target == 1] = [190, 35 + index, 50]
        rgb_path, target_path = root / f"rgb_{index}.png", root / f"target_{index}.png"
        Image.fromarray(rgb).save(rgb_path)
        Image.fromarray(target).save(target_path)
        rows.append(
            {
                "sample_id": f"fixture_{split}/{index}",
                "split": split,
                "group_id": f"fixture_{split}",
                "task_id": task["task_id"],
                "protocol_sha256": file_hash(output / "task.json"),
                "source_rgb_sha256": file_hash(rgb_path),
                "rgb_path": rgb_path.name,
                "target_path": target_path.name,
                "target_sha256": file_hash(target_path),
                "reference_origin": "fixture_ground_truth",
                "image_size": [width, height],
            }
        )
    (root / "references.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    write_json(
        root / "profile.json",
        {
            "status": "passed",
            "purpose": "known_fixture_only",
            "benchmark_reference_ready": True,
            "task_id": task["task_id"],
            "protocol_sha256": file_hash(output / "task.json"),
            "reference_counts": {"train": 4, "validation": 1},
            "artifact_sha256": artifact_hashes(root),
        },
    )
    sha = file_hash(root / "profile.json")
    trained = train_run(
        root,
        sha,
        output / "task.json",
        output / "config.json",
        checkpoint,
        output / "training",
        17,
        device,
        fixture=True,
    )
    evaluate_run(
        output / "training",
        file_hash(output / "training/profile.json"),
        root,
        sha,
        output / "evaluation",
        device,
        fixture=True,
    )
    result = {
        "status": "passed",
        "purpose": "fixture_only",
        "customer_sensor_files_opened": 0,
        "production_model": production_model,
        "train_save_reload_evaluate": "passed",
        "optimizer_updates": trained["optimizer_updates"],
        "native_resolution": [width, height],
        "peak_cuda_allocated_bytes": trained["peak_cuda_allocated_bytes"],
        "artifact_sha256": artifact_hashes(output),
    }
    write_json(output / "profile.json", result)
    return result


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    for flag in ("references", "task", "configuration", "checkpoint", "output"):
        train.add_argument("--" + flag, type=Path, required=True)
    train.add_argument("--references-sha256", required=True)
    train.add_argument("--synthetic", type=Path)
    train.add_argument("--synthetic-sha256")
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    evaluate = commands.add_parser("evaluate")
    for flag in ("run", "references", "output"):
        evaluate.add_argument("--" + flag, type=Path, required=True)
    evaluate.add_argument("--run-sha256", required=True)
    evaluate.add_argument("--references-sha256", required=True)
    evaluate.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    fixture = commands.add_parser("fixture")
    fixture.add_argument("--output", type=Path, required=True)
    fixture.add_argument("--production-model", action="store_true")
    fixture.add_argument("--checkpoint", type=Path)
    fixture.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.command == "train":
        result = train_run(
            args.references,
            args.references_sha256,
            args.task,
            args.configuration,
            args.checkpoint,
            args.output,
            args.seed,
            args.device,
            synthetic=args.synthetic,
            synthetic_sha=args.synthetic_sha256,
        )
    elif args.command == "evaluate":
        result = evaluate_run(
            args.run,
            args.run_sha256,
            args.references,
            args.references_sha256,
            args.output,
            args.device,
        )
    else:
        result = fixture_run(
            args.output,
            production_model=args.production_model,
            checkpoint=args.checkpoint,
            device=args.device,
        )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "artifact_sha256"},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
