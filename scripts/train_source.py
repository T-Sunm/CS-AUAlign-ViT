"""Fine-tune Swin-Tiny and source prototypes on prepared RAF-DB."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from cs_aualign_vit.data.datasets import FaceDataset, _seed_worker
from cs_aualign_vit.data.manifests import CLASSES
from cs_aualign_vit.data.transforms import build_transform
from cs_aualign_vit.models import build_swin_t
from cs_aualign_vit.training import evaluate, load_checkpoint, save_checkpoint, train_epoch
from cs_aualign_vit.training.source_pretraining import stratified_split_indices


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/training/source_pretraining.yaml")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def validate_config(config):
    model, training, preprocessing = (
        config.get("model", {}),
        config.get("training", {}),
        config.get("preprocessing", {}),
    )
    required_training = {
        "epochs",
        "batch_size",
        "learning_rate",
        "momentum",
        "weight_decay",
        "num_workers",
        "seed",
        "device",
        "amp",
        "validation_ratio",
        "early_stopping_patience",
    }
    if model.get("backbone") != "swin_t" or model.get("num_classes") != len(CLASSES):
        raise ValueError("Task 2 requires swin_t and the canonical 7 classes")
    missing_training = required_training - training.keys()
    missing_preprocessing = {"image_size", "mean", "std"} - preprocessing.keys()
    if missing_training or missing_preprocessing:
        raise ValueError("Incomplete source pretraining configuration")
    if training["epochs"] <= 0:
        raise ValueError("training.epochs must be positive")
    if type(training["amp"]) is not bool:
        raise ValueError("training.amp must be true or false")
    if training["early_stopping_patience"] <= 0:
        raise ValueError("training.early_stopping_patience must be positive")
    if not 0 < training["validation_ratio"] < 1:
        raise ValueError("training.validation_ratio must be between zero and one")
    return model, training, preprocessing


def build_source_loaders(manifest, training, preprocessing):
    train_dataset = FaceDataset(
        manifest, "source_train", transform=build_transform(True, **preprocessing)
    )
    validation_dataset = FaceDataset(
        manifest, "source_train", transform=build_transform(False, **preprocessing)
    )
    train_indices, validation_indices = stratified_split_indices(
        train_dataset.samples, training["validation_ratio"], training["seed"]
    )
    generator = torch.Generator().manual_seed(training["seed"])
    common = {
        "batch_size": training["batch_size"],
        "num_workers": training["num_workers"],
        "worker_init_fn": _seed_worker,
    }
    train_loader = DataLoader(
        Subset(train_dataset, train_indices), shuffle=True, generator=generator, **common
    )
    validation_loader = DataLoader(
        Subset(validation_dataset, validation_indices), shuffle=False, **common
    )
    return train_loader, validation_loader


def main():
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    model_config, training, preprocessing = validate_config(config)
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
        raise ValueError("AMP requires training.device: cuda")
    train_loader, validation_loader = build_source_loaders(args.manifest, training, preprocessing)
    model = build_swin_t(model_config.get("weights"), model_config["num_classes"]).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=training["learning_rate"],
        momentum=training["momentum"],
        weight_decay=training["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training["epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=training["amp"])
    start_epoch = 0
    best_validation_accuracy = float("-inf")
    epochs_without_improvement = 0
    if args.resume:
        state = load_checkpoint(
            args.resume,
            model,
            optimizer,
            scheduler,
            scaler,
            map_location=device,
        )
        if (
            state["backbone_config"] != model_config
            or state["preprocessing_config"] != preprocessing
            or state["training_config"] != training
        ):
            raise ValueError("Resume checkpoint configuration differs from the current config")
        start_epoch = state["epoch"] + 1
        best_validation_accuracy = state["best_validation_accuracy"]
        epochs_without_improvement = state["epochs_without_improvement"]

    args.output.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output / "metrics.jsonl"
    for epoch in range(start_epoch, training["epochs"]):
        metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler=scaler,
            amp=training["amp"],
            progress=True,
            description=f"Epoch {epoch + 1}/{training['epochs']}",
        )
        validation_metrics = evaluate(model, validation_loader, device, amp=training["amp"])
        scheduler.step()
        metrics.update(
            epoch=epoch,
            learning_rate=optimizer.param_groups[0]["lr"],
            validation_loss=validation_metrics["loss"],
            validation_accuracy=validation_metrics["accuracy"],
        )
        improved = validation_metrics["accuracy"] > best_validation_accuracy
        if improved:
            best_validation_accuracy = validation_metrics["accuracy"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        with metrics_path.open("a") as stream:
            stream.write(json.dumps(metrics, sort_keys=True) + "\n")
        checkpoint_arguments = {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "scaler": scaler,
            "epoch": epoch,
            "model_config": model_config,
            "preprocessing": preprocessing,
            "training_config": training,
            "best_validation_accuracy": best_validation_accuracy,
            "epochs_without_improvement": epochs_without_improvement,
        }
        save_checkpoint(args.output / "last.pt", **checkpoint_arguments)
        if improved:
            save_checkpoint(args.output / "best.pt", **checkpoint_arguments)
        print(json.dumps(metrics, sort_keys=True), flush=True)
        if epochs_without_improvement >= training["early_stopping_patience"]:
            print(f"Early stopping after epoch {epoch + 1}", flush=True)
            break


if __name__ == "__main__":
    main()
