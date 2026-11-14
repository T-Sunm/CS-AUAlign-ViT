"""Pilot AU attention training for a few steps from the source checkpoint."""

import argparse
import math
from pathlib import Path
from types import SimpleNamespace

import torch
import train_adaptation
import yaml

from cs_aualign_vit.data.au_guidance import compute_guidance
from cs_aualign_vit.training.adaptation import AdaptationTrainer


def finite_parameters(module):
    return all(torch.isfinite(parameter).all().item() for parameter in module.parameters())


def post_update_metrics(trainer, source_batch, target_batch):
    source_images, _, source_au = source_batch
    target_images, _, target_au = target_batch
    valid = (
        source_au["au_valid"] & source_au["spatial_valid"]
        & target_au["au_valid"] & target_au["spatial_valid"]
    )
    guidance = compute_guidance(
        target_au["au"], source_au["au"],
        target_au["mask"], source_au["mask"], valid,
    ).to(trainer.device)
    trainer.model.eval()
    try:
        with torch.inference_mode(), torch.autocast(
            device_type=trainer.device.type, dtype=torch.float16, enabled=trainer.config.amp
        ):
            features = trainer.model.forward_features(
                source_images.to(trainer.device), target_images.to(trainer.device), guidance
            )
        feature_finite = all(
            torch.isfinite(value).all().item()
            for value in (
                features.source_features, features.target_features,
                features.target_to_source_features, features.source_to_target_features,
            )
        )
        return feature_finite, trainer.model.bcs_block1.last_guidance_metrics
    finally:
        trainer.model.train()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, default=Path("data/processed/raf-v5-deduplicated/processed.jsonl"))
    parser.add_argument("--source-checkpoint", type=Path, default=Path("runs/source-pretraining-val/best.pt"))
    parser.add_argument("--source-au-cache", type=Path, default=Path("runs/au_cache/raf_db_train.csv"))
    parser.add_argument("--target-au-cache", type=Path, default=Path("runs/au_cache/expw_train.csv"))
    parser.add_argument("--source-heatmap-cache", type=Path)
    parser.add_argument("--target-heatmap-cache", type=Path)
    parser.add_argument(
        "--heatmap-python", type=Path, default=Path("tools/au_heatmap/.venv/bin/python")
    )
    parser.add_argument(
        "--heatmap-checkpoint",
        type=Path,
        default=Path("tools/au_heatmap/checkpoints/disfa_adaptation_f0.pth"),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/training/ablation/e3_both.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--g-scale", type=float, required=True)
    parser.add_argument("--lambda-learning-rate", type=float, required=True)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        parser.error("steps and batch-size must be positive")
    if not all(math.isfinite(value) and value > 0 for value in (args.g_scale, args.lambda_learning_rate)):
        parser.error("g-scale and lambda-learning-rate must be finite positive numbers")
    if args.output.exists():
        parser.error("output directory already exists; choose a new pilot directory")
    if (args.source_heatmap_cache is None) != (args.target_heatmap_cache is None):
        parser.error("source and target heatmap caches must be provided together")

    config = yaml.safe_load(args.config.read_text())
    config["training"].update(max_epochs=1, batch_size=args.batch_size, str_epochs=[])
    config["au_guidance"].update(
        mode="dynamic", g_scale=args.g_scale,
        lambda_learning_rate=args.lambda_learning_rate,
    )
    config["au_spatial"] = {
        "mode": "heatmap" if args.source_heatmap_cache is not None else "landmark",
        "peak_threshold": 0.8,
    }
    train_adaptation.validate_config(config)
    run_args = SimpleNamespace(
        source_manifest=args.source_manifest,
        target_manifest=args.target_manifest,
        source_checkpoint=args.source_checkpoint,
        source_au_cache=args.source_au_cache,
        target_au_cache=args.target_au_cache,
        au_python=Path(".venv-au/bin/python"),
        source_heatmap_cache=args.source_heatmap_cache,
        target_heatmap_cache=args.target_heatmap_cache,
        heatmap_python=args.heatmap_python,
        heatmap_checkpoint=args.heatmap_checkpoint,
        config=SimpleNamespace(read_text=lambda: yaml.safe_dump(config)),
        output=args.output,
        resume=None,
    )

    def pilot_run(trainer, on_epoch_end=None, *, progress=False):
        trainer.refresh_supervision(0)
        block = trainer.model.bcs_block1
        if torch.count_nonzero(block.lambda_target_to_source) or torch.count_nonzero(block.lambda_source_to_target):
            raise RuntimeError("Pilot must start with both lambda vectors at zero")
        trainer.model.train()
        trainer.prototype_banks.train()
        trainer.adversarial_module.train()
        source_iterator = iter(trainer.source_loader)
        target_iterator = iter(trainer.target_loader)
        skipped_steps = 0
        for step in range(1, args.steps + 1):
            try:
                source_batch = next(source_iterator)
            except StopIteration:
                source_iterator = iter(trainer.source_loader)
                source_batch = next(source_iterator)
            target_batch = next(target_iterator)
            scaler_before = trainer.scaler.get_scale() if trainer.config.amp else None
            metrics = trainer.train_step(source_batch, target_batch)
            if trainer.global_step != step:
                raise RuntimeError("Pilot step counter differs from the requested sequence")
            lambdas = (block.lambda_target_to_source, block.lambda_source_to_target)
            gradients = [
                float(parameter.grad.detach().float().abs().max())
                if parameter.grad is not None else float("nan")
                for parameter in lambdas
            ]
            finite = (
                all(math.isfinite(value) for value in metrics.values() if isinstance(value, (int, float)))
                and all(finite_parameters(module) for module in (
                    trainer.model, trainer.prototype_banks, trainer.adversarial_module
                ))
            )
            gradient_finite = all(
                parameter.grad is None or torch.isfinite(parameter.grad).all().item()
                for module in (trainer.model, trainer.prototype_banks, trainer.adversarial_module)
                for parameter in module.parameters()
            )
            scaler_skipped = trainer.config.amp and trainer.scaler.get_scale() < scaler_before
            skipped_steps += int(scaler_skipped)
            feature_finite, attention = post_update_metrics(trainer, source_batch, target_batch)
            effective_lambda = [
                float(parameter.detach().abs().max() * args.g_scale) for parameter in lambdas
            ]
            print(
                f"step={step} loss={metrics['loss_total']:.5g} "
                f"lambda_grad_max={gradients} g_scale*lambda_max={effective_lambda} "
                f"post_features_finite={feature_finite} parameters_finite={finite} "
                f"gradients_finite={gradient_finite} scaler_skipped={scaler_skipped} "
                f"attention={attention}", flush=True,
            )
            if (not finite or not feature_finite
                    or (not gradient_finite and not scaler_skipped)
                    or any(not math.isfinite(value) for value in attention.values())):
                raise RuntimeError(f"Non-finite pilot state after step {step}; checkpoint not saved")
        print(f"AMP skipped {skipped_steps}/{args.steps} optimizer steps", flush=True)
        trainer.save_checkpoint()
        return []

    # Reuse production trainer setup and epoch-0 pseudo labels; stop before full-epoch work.
    original_run = AdaptationTrainer.run
    AdaptationTrainer.run = pilot_run
    try:
        print(f"pilot checkpoint: {train_adaptation.run(run_args)}")
    finally:
        AdaptationTrainer.run = original_run


if __name__ == "__main__":
    main()
