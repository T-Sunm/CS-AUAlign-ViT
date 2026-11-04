"""Run POST adaptation from a source-pretraining checkpoint."""

import argparse
import json
import logging
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml

from cs_aualign_vit.data.datasets import FaceDataset, build_loader, build_source_loader
from cs_aualign_vit.data.manifests import CLASSES
from cs_aualign_vit.models import AdversarialAlignment, POSTBCSModel, PrototypeBanks
from cs_aualign_vit.training import AdaptationConfig, AdaptationTrainer, evaluate_target
from cs_aualign_vit.training.self_training_resampling import SelfTrainingResampler

LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/training/adaptation.yaml"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def validate_config(config):
    sections = ("training", "loss", "adversarial", "str", "preprocessing")
    if any(not isinstance(config.get(section), dict) for section in sections):
        raise ValueError("Incomplete adaptation configuration")
    required = {
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
        "loss": {"alpha", "beta", "gamma"},
        "adversarial": {"hidden_dim", "dropout", "lambda_grl"},
        "str": {"confidence_threshold"},
        "preprocessing": {"image_size", "mean", "std"},
    }
    if any(required[name] - config[name].keys() for name in required):
        raise ValueError("Incomplete adaptation configuration")
    training = config["training"]
    if training["batch_size"] < 1 or training["num_workers"] < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    if type(training["amp"]) is not bool:
        raise ValueError("training.amp must be true or false")
    return config


def resolve_output(output, resume, runs_dir=Path("runs/adaptation")):
    if resume is not None:
        if output is not None and output.resolve() != resume.parent.resolve():
            raise ValueError("--output must be the directory containing --resume")
        return resume.parent
    if output is not None:
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

    preprocessing = config["preprocessing"]
    source_state = torch.load(args.source_checkpoint, map_location="cpu", weights_only=True)
    if source_state.get("preprocessing_config") != preprocessing:
        raise ValueError("Adaptation preprocessing differs from the source checkpoint")
    loader_options = {
        "batch_size": training["batch_size"],
        "num_workers": training["num_workers"],
        "seed": seed,
        **preprocessing,
    }
    source_loader = build_source_loader([args.source_manifest], drop_last=True, **loader_options)
    target_loader = build_loader(
        args.target_manifest, "target_adaptation", drop_last=True, **loader_options
    )
    target_feature_loader = build_loader(args.target_manifest, "target_features", **loader_options)
    validation_loader = build_loader(
        args.target_manifest, "evaluation", split="validation", **loader_options
    )
    target_sample_ids = [
        sample["sample_id"]
        for sample in FaceDataset(args.target_manifest, "target_features").samples
    ]

    model = POSTBCSModel.from_source_checkpoint(args.source_checkpoint)
    prototype_banks = PrototypeBanks.from_source_checkpoint(
        args.source_checkpoint, feature_dim=model.feature_dim
    )
    adversarial = AdversarialAlignment(input_dim=model.feature_dim, **config["adversarial"])
    optimizer = torch.optim.SGD(
        [*model.parameters(), *prototype_banks.parameters(), *adversarial.parameters()],
        lr=training["learning_rate"],
        momentum=training["momentum"],
        weight_decay=training["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training["max_epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=training["amp"])
    adaptation_config = AdaptationConfig(
        max_epochs=training["max_epochs"],
        kmeans_interval=training["kmeans_interval"],
        str_epochs=tuple(training["str_epochs"]),
        amp=training["amp"],
        num_classes=len(CLASSES),
        **config["loss"],
    )
    output = resolve_output(args.output, args.resume)
    LOGGER.info("adaptation output: %s", output)
    resampler = SelfTrainingResampler(
        source_manifest=args.source_manifest,
        target_feature_loader=target_feature_loader,
        prototype_banks=prototype_banks,
        output_dir=output / "str",
        global_seed=seed,
        batch_size=training["batch_size"],
        device=device,
        num_workers=training["num_workers"],
        num_classes=len(CLASSES),
        confidence_threshold=config["str"]["confidence_threshold"],
    )
    trainer = AdaptationTrainer(
        model=model,
        prototype_banks=prototype_banks,
        adversarial_module=adversarial,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        source_loader=source_loader,
        target_loader=target_loader,
        target_feature_loader=target_feature_loader,
        target_sample_ids=target_sample_ids,
        device=device,
        checkpoint_path=args.resume or output / "adaptation.pt",
        config=adaptation_config,
        str_service=resampler,
        source_checkpoint=args.source_checkpoint,
    )
    if args.resume:
        trainer.load_checkpoint()

    metrics_path = output / "metrics.jsonl"
    best_validation_accuracy = float("-inf")
    if args.resume and metrics_path.exists():
        for line in metrics_path.read_text().splitlines():
            previous = json.loads(line)
            if previous["epoch"] <= trainer.completed_epoch:
                best_validation_accuracy = max(
                    best_validation_accuracy, previous.get("validation_accuracy", float("-inf"))
                )

    def report_epoch(metrics):
        nonlocal best_validation_accuracy
        validation = evaluate_target(
            model, prototype_banks, validation_loader, device, num_classes=len(CLASSES)
        )
        metrics.update(
            validation_accuracy=validation["accuracy"],
            validation_per_class_accuracy=validation["per_class_accuracy"],
        )
        improved = validation["accuracy"] > best_validation_accuracy
        if improved:
            best_validation_accuracy = validation["accuracy"]
            shutil.copy2(trainer.checkpoint_path, output / "best.pt")
        with metrics_path.open("a") as stream:
            stream.write(json.dumps(metrics, sort_keys=True) + "\n")
        LOGGER.info(
            "completed epoch %d/%d: loss=%.4f val_accuracy=%.4f val_per_class=%s best=%.4f",
            metrics["epoch"],
            training["max_epochs"],
            metrics["loss_total"],
            validation["accuracy"],
            validation["per_class_accuracy"],
            best_validation_accuracy,
        )

    trainer.run(on_epoch_end=report_epoch, progress=True)
    return trainer.checkpoint_path


def main():
    print(run(parse_args()))


if __name__ == "__main__":
    main()
