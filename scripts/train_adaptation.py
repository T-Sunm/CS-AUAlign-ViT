"""Run POST adaptation from a source-pretraining checkpoint."""

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from cs_aualign_vit.data.datasets import FaceDataset, build_loader, build_source_loader
from cs_aualign_vit.data.manifests import CLASSES
from cs_aualign_vit.models import AdversarialAlignment, POSTBCSModel, PrototypeBanks
from cs_aualign_vit.training import AdaptationConfig, AdaptationTrainer
from cs_aualign_vit.training.self_training_resampling import SelfTrainingResampler

LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/training/adaptation.yaml"))
    parser.add_argument("--output", type=Path, required=True)
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
    args.output.mkdir(parents=True, exist_ok=True)
    resampler = SelfTrainingResampler(
        source_manifest=args.source_manifest,
        target_feature_loader=target_feature_loader,
        prototype_banks=prototype_banks,
        output_dir=args.output / "str",
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
        checkpoint_path=args.resume or args.output / "adaptation.pt",
        config=adaptation_config,
        str_service=resampler,
        source_checkpoint=args.source_checkpoint,
    )
    if args.resume:
        trainer.load_checkpoint()

    def report_epoch(metrics):
        with (args.output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics, sort_keys=True) + "\n")
        LOGGER.info(
            "completed epoch %d/%d: loss=%.4f",
            metrics["epoch"],
            training["max_epochs"],
            metrics["loss_total"],
        )

    trainer.run(on_epoch_end=report_epoch, progress=True)
    return trainer.checkpoint_path


def main():
    print(run(parse_args()))


if __name__ == "__main__":
    main()
