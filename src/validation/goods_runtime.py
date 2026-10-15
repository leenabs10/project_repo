"""Discarded fixture optimizer step and checkpoint reload for the goods runtime."""

import argparse
import io
import json
from pathlib import Path
import time

from src.adapters.storage.local import file_hash, reserve_output
from src.validation.artifacts import write_json


def check_runtime(configuration: Path, checkpoint: Path, output: Path) -> dict:
    """Exercise the configured resolution/batch on known fixture labels, not customer data.

    This is a runtime/memory check. The optimizer update and resulting checkpoint
    are discarded; no benchmark training, semantic accuracy or utility is measured.
    """
    configuration, checkpoint = Path(configuration), Path(checkpoint)
    config = json.loads(configuration.read_text(encoding="utf-8"))
    config_sha, checkpoint_sha = file_hash(configuration), file_hash(checkpoint)
    if checkpoint_sha != config["initial_checkpoint_sha256"]:
        raise ValueError("Initialization checkpoint checksum mismatch")
    if config["model"] != "deeplabv3_mobilenet_v3_large" or config["loss"] != {
        "name": "cross_entropy",
        "ignore_index": 255,
    }:
        raise ValueError("Unsupported goods runtime configuration")
    output = reserve_output(
        output, (configuration, checkpoint, Path(__file__).parents[2] / "data/raw")
    )
    write_json(
        output / "profile.json", {"status": "running", "benchmark_complete": False}
    )
    started = time.perf_counter()
    try:
        import torch
        from torch.nn import functional as functional
        from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

        if not torch.cuda.is_available():
            raise ValueError("Configured CUDA runtime is unavailable")
        torch.manual_seed(config["model_seeds"][0])
        device = torch.device("cuda")
        width, height = config["training_resolution"]
        batch_size = config["batch_size"]
        model = deeplabv3_mobilenet_v3_large(
            weights=None, weights_backbone=None, num_classes=2, aux_loss=True
        )
        source_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        target_state = model.state_dict()
        compatible = {
            key: value
            for key, value in source_state.items()
            if key in target_state and value.shape == target_state[key].shape
        }
        expected_heads = {
            "classifier.4.weight",
            "classifier.4.bias",
            "aux_classifier.4.weight",
            "aux_classifier.4.bias",
        }
        result = model.load_state_dict(compatible, strict=False)
        if (
            set(result.missing_keys) != expected_heads
            or result.unexpected_keys
            or set(source_state) - set(compatible) != expected_heads
        ):
            raise ValueError(
                "Pretrained initialization changed beyond the declared class heads"
            )
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model.to(device).train()
        # Known rectangular targets verify foreground/background/ignore loss handling.
        target = torch.zeros(
            (batch_size, height, width), dtype=torch.long, device=device
        )
        target[:, height // 4 : 3 * height // 4, width // 4 : 3 * width // 4] = 1
        target[:, : max(1, height // 20)] = 255
        inputs = torch.rand((batch_size, 3, height, width), device=device)
        mean = torch.tensor(config["normalization"]["mean"], device=device)[
            None, :, None, None
        ]
        std = torch.tensor(config["normalization"]["std"], device=device)[
            None, :, None, None
        ]
        inputs = (inputs - mean) / std
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["optimizer"]["learning_rate"],
            weight_decay=config["optimizer"]["weight_decay"],
        )
        logits = model(inputs)["out"]
        loss = functional.cross_entropy(logits, target, ignore_index=255)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite goods fixture loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(
            parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise ValueError("Nonfinite goods fixture gradients")
        optimizer.step()
        model.eval()
        with torch.no_grad():
            expected = model(inputs[:1])["out"].cpu()
        # The serialized state stays in memory and is never advertised as a trained model.
        buffer = io.BytesIO()
        torch.save(
            {key: value.detach().cpu() for key, value in model.state_dict().items()},
            buffer,
        )
        buffer.seek(0)
        restored = deeplabv3_mobilenet_v3_large(
            weights=None, weights_backbone=None, num_classes=2, aux_loss=True
        )
        restored.load_state_dict(
            torch.load(buffer, map_location="cpu", weights_only=True), strict=True
        )
        restored.to(device).eval()
        with torch.no_grad():
            prediction = restored(inputs[:1])["out"]
            native = functional.interpolate(
                prediction,
                size=tuple(reversed(config["native_reference_resolution"])),
                mode="bilinear",
                align_corners=False,
            )
        if (
            not torch.allclose(prediction.cpu(), expected, atol=1e-5, rtol=1e-5)
            or not torch.isfinite(native).all()
        ):
            raise ValueError(
                "Reloaded model output mismatch or nonfinite native logits"
            )
        torch.cuda.synchronize()
        if (
            file_hash(checkpoint) != checkpoint_sha
            or file_hash(configuration) != config_sha
        ):
            raise ValueError("Runtime inputs changed during verification")
        report = {
            "status": "passed",
            "purpose": "fixture_runtime_check_only",
            "benchmark_complete": False,
            "customer_sensor_files_opened": 0,
            "trained_candidate_saved": False,
            "model": config["model"],
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "resolution": [width, height],
            "batch_size": batch_size,
            "precision": "float32",
            "fixture_optimizer_steps": 1,
            "fixture_loss": float(loss.detach()),
            "pretrained_components_loaded": len(compatible),
            "reinitialized_class_head_tensors": sorted(expected_heads),
            "strict_reload_and_forward": "passed",
            "native_logit_resolution": config["native_reference_resolution"],
            "peak_cuda_allocated_bytes_including_reload": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes_including_reload": torch.cuda.max_memory_reserved(),
            "elapsed_seconds": time.perf_counter() - started,
            "configuration_sha256": config_sha,
            "initial_checkpoint_sha256": checkpoint_sha,
            "limitations": "One discarded fixture step verifies runtime compatibility, not a full-run memory bound, convergence, accuracy or synthetic-data utility.",
        }
        write_json(output / "profile.json", report)
        return report
    except Exception as error:
        write_json(
            output / "profile.json",
            {"status": "failed", "error": str(error), "benchmark_complete": False},
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            check_runtime(args.configuration, args.checkpoint, args.output), indent=2
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
