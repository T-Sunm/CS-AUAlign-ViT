"""Run USTST adaptation (Guo et al., MTAP 2023) from a source-pretraining checkpoint."""

import argparse
import json
import logging
import math
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml

from cs_aualign_vit.data.datasets import (
    FaceDataset,
    build_loader,
    build_source_loader,
    build_weak_strong_target_loader,
)
from cs_aualign_vit.data.manifests import CLASSES
from cs_aualign_vit.models import AdversarialAlignment, USTSTHeads, USTSTModel
from cs_aualign_vit.training import USTSTConfig, USTSTTrainer, evaluate_target
from cs_aualign_vit.training.self_training_resampling import SelfTrainingResampler

LOGGER = logging.getLogger(__name__)
CHECKPOINT_NAME = "adaptation.pt"
RUN_STATE_PATTERNS = ("adaptation*.pt", "best.pt", "metrics.jsonl", "test_results.json", "str")
REQUIRED = {
    "training": {
        "max_epochs",
        "batch_size",
        "learning_rate",
        "momentum",
        "weight_decay",
        "num_workers",
        "seed",
        "device",
        "amp",
        "kmeans_interval",
        "str_epochs",
    },
    "loss": {
        "confidence_threshold",
        "entropy_threshold",
        "similarity_margin",
        "lambda_c",
        "lambda_sim",
        "lambda_m",
        "lambda_a",
        "mmd_kernel_mul",
        "mmd_kernel_num",
    },
    "augmentation": {"randaugment_num_ops", "randaugment_magnitude"},
    "adversarial": {"hidden_dim", "dropout", "lambda_grl"},
    "str": {"confidence_threshold", "patch_scale_range"},
    "model": {"cross_start_block"},
    "preprocessing": {"image_size", "mean", "std"},
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/training/ustst.yaml"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--max-steps-per-epoch",
        type=int,
        help="Smoke testing only: truncate every epoch after this many paired steps.",
    )
    return parser.parse_args()


def validate_config(config):
    if not isinstance(config, dict) or any(
        not isinstance(config.get(section), dict) or keys - config[section].keys()
        for section, keys in REQUIRED.items()
    ):
        raise ValueError("Incomplete USTST configuration")
    training = config["training"]
    if training["batch_size"] < 1 or training["num_workers"] < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    if type(training["amp"]) is not bool:
        raise ValueError("training.amp must be true or false")
    augmentation = config["augmentation"]
    if augmentation["randaugment_num_ops"] < 0 or augmentation["randaugment_magnitude"] < 0:
        raise ValueError("RandAugment settings must be non-negative")
    return config


def resolve_output(output, resume, runs_dir=Path("runs/ustst")):
    if resume is not None:
        if output is not None and output.resolve() != resume.parent.resolve():
            raise ValueError("--output must be the directory containing --resume")
        return resume.parent
    if output is not None:
        # A fresh run must not overwrite or mix with an earlier run's results, nor
        # reuse its STR transactions. resolved_config.yaml alone (a run that
        # crashed before its first checkpoint) is fine.
        existing = [
            path.name
            for pattern in RUN_STATE_PATTERNS
            for path in (sorted(output.glob(pattern)) if output.is_dir() else ())
        ]
        if existing:
            raise FileExistsError(
                f"{output} already holds run state ({', '.join(existing)}); "
                f"pass --resume {output / CHECKPOINT_NAME} or choose a new --output"
            )
        output.mkdir(parents=True, exist_ok=True)
        return output
    runs_dir.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        candidate = runs_dir / f"run-{index:03d}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            index += 1


def persist_config(output, config):
    resolved = output / "resolved_config.yaml"
    if resolved.exists():
        if yaml.safe_load(resolved.read_text()) != config:
            raise ValueError("Resolved USTST configuration differs from the current run")
    else:
        resolved.write_text(yaml.safe_dump(config, sort_keys=False))


def read_metrics(path):
    """Rows of ``metrics.jsonl``; a partially written last row is dropped.

    The last row is appended after its checkpoint is saved, so a dropped row is
    recovered from the checkpoint on resume. A bad row elsewhere is corruption.
    """
    lines = path.read_text().splitlines()
    rows = []
    for number, line in enumerate(lines, start=1):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            if number == len(lines):
                LOGGER.warning("dropping partially written last row of %s: %r", path, line[:80])
                break
            raise ValueError(f"{path} line {number} is not valid JSON") from error
    return rows


def atomic_copy(source, destination):
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def load_into(path, model, heads, device):
    state = torch.load(path, map_location=device, weights_only=True)
    if not state.get("boundary_completed"):
        raise ValueError(f"{path} is not at a completed adaptation boundary")
    model.load_state_dict(state["model_state_dict"])
    heads.load_state_dict(state["prototype_state_dict"])
    return state


class _TruncatedLoader:
    """Smoke-test wrapper that keeps the loader interface the trainer validates."""

    def __init__(self, loader, max_steps):
        self.loader = loader
        self.max_steps = max_steps
        self.dataset = loader.dataset
        self.batch_size = loader.batch_size
        self.drop_last = loader.drop_last
        self.generator = loader.generator

    def __len__(self):
        return min(len(self.loader), self.max_steps)

    def __iter__(self):
        for step, batch in enumerate(self.loader):
            if step >= self.max_steps:
                return
            yield batch


def run(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    config = validate_config(yaml.safe_load(args.config.read_text()))
    training = config["training"]
    seed = training["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(training["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if training["amp"] and device.type != "cuda":
        raise ValueError("AMP requires a CUDA training device")

    output = resolve_output(args.output, args.resume)
    checkpoint_path = output / CHECKPOINT_NAME
    if args.resume is not None and args.resume.resolve() != checkpoint_path.resolve():
        # best.pt and boundary snapshots are read-only results; training always
        # continues from, and writes to, the rolling adaptation.pt.
        raise ValueError(f"--resume must be the run's rolling {CHECKPOINT_NAME}: {checkpoint_path}")
    persist_config(output, config)
    LOGGER.info("USTST output: %s", output)
    preprocessing = config["preprocessing"]
    source_state = torch.load(args.source_checkpoint, map_location="cpu", weights_only=True)
    if source_state.get("preprocessing_config") != preprocessing:
        raise ValueError("USTST preprocessing differs from the source checkpoint")
    loader_options = {
        "batch_size": training["batch_size"],
        "num_workers": training["num_workers"],
        "seed": seed,
        **preprocessing,
    }
    source_loader = build_source_loader([args.source_manifest], drop_last=True, **loader_options)
    target_loader = build_weak_strong_target_loader(
        args.target_manifest, **loader_options, **config["augmentation"]
    )
    max_steps = getattr(args, "max_steps_per_epoch", None)
    if max_steps is not None:
        if max_steps < 1:
            raise ValueError("--max-steps-per-epoch must be positive")
        target_loader = _TruncatedLoader(target_loader, max_steps)
    target_feature_loader = build_loader(args.target_manifest, "target_features", **loader_options)
    validation_loader = build_loader(
        args.target_manifest, "evaluation", split="validation", **loader_options
    )
    test_loader = build_loader(args.target_manifest, "evaluation", split="test", **loader_options)
    target_sample_ids = [
        sample["sample_id"]
        for sample in FaceDataset(args.target_manifest, "target_features").samples
    ]

    model = USTSTModel.from_source_checkpoint(
        args.source_checkpoint, cross_start_block=config["model"]["cross_start_block"]
    )
    heads = USTSTHeads.from_source_checkpoint(args.source_checkpoint, feature_dim=model.feature_dim)
    adversarial = AdversarialAlignment(input_dim=model.feature_dim, **config["adversarial"])
    optimizer = torch.optim.SGD(
        [p for module in (model, heads, adversarial) for p in module.parameters()],
        lr=training["learning_rate"],
        momentum=training["momentum"],
        weight_decay=training["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training["max_epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=training["amp"])
    patch_scale_range = config["str"]["patch_scale_range"]
    patch_scale_range = tuple(patch_scale_range) if patch_scale_range is not None else None
    ustst_config = USTSTConfig(
        seed=seed,
        max_epochs=training["max_epochs"],
        kmeans_interval=training["kmeans_interval"],
        str_epochs=tuple(training["str_epochs"]),
        amp=training["amp"],
        num_classes=len(CLASSES),
        str_threshold=config["str"]["confidence_threshold"],
        str_patch_scale_range=patch_scale_range,
        cross_start_block=config["model"]["cross_start_block"],
        max_steps_per_epoch=max_steps,
        **config["loss"],
    )
    resampler = SelfTrainingResampler(
        source_manifest=args.source_manifest,
        target_feature_loader=target_feature_loader,
        prototype_banks=heads,
        output_dir=output / "str",
        global_seed=seed,
        batch_size=training["batch_size"],
        device=device,
        num_workers=training["num_workers"],
        num_classes=len(CLASSES),
        confidence_threshold=config["str"]["confidence_threshold"],
        patch_scale_range=patch_scale_range,
    )
    trainer = USTSTTrainer(
        model=model,
        prototype_banks=heads,
        adversarial_module=adversarial,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        source_loader=source_loader,
        target_loader=target_loader,
        target_feature_loader=target_feature_loader,
        target_sample_ids=target_sample_ids,
        device=device,
        checkpoint_path=checkpoint_path,
        config=ustst_config,
        str_service=resampler,
        source_checkpoint=args.source_checkpoint,
    )
    if args.resume:
        trainer.load_checkpoint()

    metrics_path = output / "metrics.jsonl"
    best_validation_accuracy = float("-inf")
    last_reported_epoch = 0
    if args.resume and metrics_path.exists():
        kept = []
        for previous in read_metrics(metrics_path):
            if previous["epoch"] <= trainer.completed_epoch:
                kept.append(previous)
                last_reported_epoch = max(last_reported_epoch, previous["epoch"])
                best_validation_accuracy = max(
                    best_validation_accuracy, previous.get("validation_accuracy", float("-inf"))
                )
        # Drop rows of epochs that are re-run after the resumed checkpoint; replace the
        # file atomically so an interruption here cannot lose the earlier history.
        temporary = metrics_path.with_name(metrics_path.name + ".tmp")
        temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in kept))
        os.replace(temporary, metrics_path)

    def report_epoch(metrics):
        nonlocal best_validation_accuracy
        validation = evaluate_target(
            model, heads, validation_loader, device, num_classes=len(CLASSES)
        )
        metrics.update(
            validation_accuracy=validation["accuracy"],
            validation_per_class_accuracy=validation["per_class_accuracy"],
            validation_balanced_accuracy=validation.get("balanced_accuracy"),
            validation_macro_f1=validation.get("macro_f1"),
        )
        if torch.cuda.is_available() and device.type == "cuda":
            metrics["max_memory_allocated_gb"] = torch.cuda.max_memory_allocated(device) / 2**30
        if validation["accuracy"] > best_validation_accuracy:
            best_validation_accuracy = validation["accuracy"]
            atomic_copy(trainer.checkpoint_path, output / "best.pt")
        with metrics_path.open("a") as stream:
            stream.write(json.dumps(metrics, sort_keys=True) + "\n")
        LOGGER.info(
            "completed epoch %d/%d: loss=%.4f hard=%.3f soft=%.3f val_accuracy=%.4f best=%.4f",
            metrics["epoch"],
            training["max_epochs"],
            metrics.get("loss_total", math.nan),
            metrics.get("hard_ratio", math.nan),
            metrics.get("soft_ratio", math.nan),
            validation["accuracy"],
            best_validation_accuracy,
        )

    if args.resume and trainer.completed_epoch > last_reported_epoch:
        # The checkpoint is saved before the epoch callback; an interruption in
        # between leaves the saved epoch without validation and out of best.pt.
        if trainer.completed_epoch != last_reported_epoch + 1:
            raise RuntimeError(
                f"metrics.jsonl ends at epoch {last_reported_epoch} but the checkpoint is at "
                f"epoch {trainer.completed_epoch}"
            )
        trainer._finish_pending_boundary()  # best.pt must hold a completed boundary
        LOGGER.info("recovering validation of epoch %d after an interruption", trainer.completed_epoch)
        report_epoch({"epoch": trainer.completed_epoch, "recovered_after_interrupt": True})

    trainer.run(on_epoch_end=report_epoch, progress=True)
    if trainer.completed_epoch == training["max_epochs"]:
        write_test_results(output, model, heads, test_loader, device, max_steps)
    return trainer.checkpoint_path


def write_test_results(output, model, heads, test_loader, device, max_steps):
    """ExpW test for both reported selection rules, written once training ends."""
    results = {
        "max_steps_per_epoch": max_steps,
        "selection": {
            "best": "epoch with the highest ExpW validation accuracy (uses target labels)",
            "final": "last epoch, fixed in advance (no target labels)",
        },
    }
    for name, filename in (("final", CHECKPOINT_NAME), ("best", "best.pt")):
        state = load_into(output / filename, model, heads, device)
        results[name] = {
            "checkpoint": str(output / filename),
            "completed_epoch": state["completed_epoch"],
            **evaluate_target(model, heads, test_loader, device, num_classes=len(CLASSES)),
        }
    path = output / "test_results.json"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(results, indent=2, sort_keys=True))
    os.replace(temporary, path)
    LOGGER.info(
        "ExpW test: best(epoch %d)=%.4f final(epoch %d)=%.4f",
        results["best"]["completed_epoch"],
        results["best"]["accuracy"],
        results["final"]["completed_epoch"],
        results["final"]["accuracy"],
    )
    return results


def main():
    print(run(parse_args()))


if __name__ == "__main__":
    main()
