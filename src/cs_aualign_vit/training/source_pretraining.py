"""Minimal source-pretraining loop and versioned checkpoint contract."""

import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F

from cs_aualign_vit.data.manifests import CLASSES

CHECKPOINT_VERSION = 1


def stratified_split_indices(samples, validation_ratio, seed):
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between zero and one")
    by_class = defaultdict(list)
    for index, sample in enumerate(samples):
        by_class[sample["label"]].append((sample["sample_id"], index))
    train_indices, validation_indices = [], []
    for label, members in sorted(by_class.items()):
        if len(members) < 2:
            raise ValueError(f"Class {label} needs at least two source samples")
        members.sort()
        random.Random(f"{seed}:{label}").shuffle(members)
        validation_count = min(len(members) - 1, max(1, int(len(members) * validation_ratio + 0.5)))
        validation_indices.extend(index for _sample_id, index in members[:validation_count])
        train_indices.extend(index for _sample_id, index in members[validation_count:])
    return sorted(train_indices), sorted(validation_indices)


def train_epoch(
    model, loader, optimizer, device, *, scaler=None, amp=False, progress=False, description="Train"
):
    if amp and (device.type != "cuda" or scaler is None):
        raise ValueError("AMP requires a CUDA device and GradScaler")
    model.train()
    total_loss = total_correct = total_samples = 0
    total_batches = len(loader)
    for batch_index, (images, labels) in enumerate(loader, 1):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            _features, logits = model(images)
            loss = F.cross_entropy(logits, labels)
        if amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_samples += batch_size
        if progress:
            completed = int(30 * batch_index / total_batches)
            bar = "=" * completed + "." * (30 - completed)
            loss_so_far = total_loss / total_samples
            accuracy_so_far = total_correct / total_samples
            sys.stdout.write(
                f"\r{description} [{bar}] {batch_index}/{total_batches} "
                f"loss={loss_so_far:.4f} acc={accuracy_so_far:.4f}"
            )
            sys.stdout.flush()
    if progress:
        sys.stdout.write("\n")
    if not total_samples:
        raise ValueError("Source loader is empty")
    return {"loss": total_loss / total_samples, "accuracy": total_correct / total_samples}


@torch.no_grad()
def evaluate(model, loader, device, *, amp=False):
    model.eval()
    total_loss = total_correct = total_samples = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            _features, logits = model(images)
            loss = F.cross_entropy(logits, labels)
        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_samples += batch_size
    if not total_samples:
        raise ValueError("Validation loader is empty")
    return {"loss": total_loss / total_samples, "accuracy": total_correct / total_samples}


def save_checkpoint(
    path,
    *,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    model_config,
    preprocessing,
    training_config,
    best_validation_accuracy,
    epochs_without_improvement,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": CHECKPOINT_VERSION,
        "epoch": epoch,
        "backbone_state_dict": model.backbone.state_dict(),
        "source_prototypes": model.source_prototypes.detach().cpu(),
        "feature_dim": model.feature_dim,
        "class_mapping": list(CLASSES),
        "backbone_config": dict(model_config),
        "preprocessing_config": dict(preprocessing),
        "training_config": dict(training_config),
        "best_validation_accuracy": best_validation_accuracy,
        "epochs_without_improvement": epochs_without_improvement,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    if checkpoint.get("schema_version") != CHECKPOINT_VERSION:
        raise ValueError("Unsupported source checkpoint schema")
    if checkpoint.get("class_mapping") != list(CLASSES):
        raise ValueError("Checkpoint class mapping differs from the data pipeline")
    if checkpoint.get("feature_dim") != model.feature_dim:
        raise ValueError("Checkpoint feature dimension differs from the model")
    prototypes = checkpoint["source_prototypes"]
    if tuple(prototypes.shape) != tuple(model.source_prototypes.shape):
        raise ValueError("Checkpoint prototype shape differs from the model")
    model.backbone.load_state_dict(checkpoint["backbone_state_dict"])
    with torch.no_grad():
        model.source_prototypes.copy_(prototypes)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return checkpoint
