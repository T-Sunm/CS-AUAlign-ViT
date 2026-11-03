"""Synthetic fixtures for STR tests: tiny PNG crops, manifests, and an image model."""

import hashlib

import torch
from PIL import Image
from torch import nn

from cs_aualign_vit.data.datasets import FaceDataset, build_loader
from cs_aualign_vit.data.manifests import CLASSES, write_jsonl
from cs_aualign_vit.models import BCSFeatures, PrototypeBanks

GOOD_LANDMARKS = [[30, 30], [70, 30], [50, 50], [35, 75], [65, 75]]
BAD_LANDMARKS = [[3, 3], [3, 3], [3, 3], [2, 6], [4, 6]]  # eyes coincide -> s=0
COLORS = {0: (255, 0, 0), 1: (0, 255, 0)}


def _row(name, dataset, label, path, landmarks, group):
    return {
        "sample_id": name,
        "dataset": dataset,
        "split": "train",
        "label": label,
        "class_name": CLASSES[label],
        "group_id": group,
        "usable": True,
        "processed_path": path,
        "bbox": [0, 0, 100, 100],
        "landmarks_crop": landmarks,
        "landmarks_original": landmarks,
    }


def make_source(root, counts, bad=()):
    """counts: {label: n}. ``bad`` names donors that get unusable landmarks."""
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, n in counts.items():
        for i in range(n):
            name = f"raf{label}_{i}"
            Image.new("RGB", (100, 100), (10 * label, 10 * i + 5, 200)).save(root / f"{name}.png")
            lm = BAD_LANDMARKS if name in bad else GOOD_LANDMARKS
            rows.append(_row(name, "raf_db", label, f"{name}.png", lm, f"g{name}"))
    write_jsonl(root / "processed.jsonl", rows)
    return root / "processed.jsonl"


def make_target(root, colors, bad=()):
    """colors: list of class ids (image colour drives the tiny model's prediction)."""
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, label in enumerate(colors):
        name = f"expw_{i:03d}"
        Image.new("RGB", (100, 100), COLORS[label]).save(root / f"{name}.png")
        lm = BAD_LANDMARKS if name in bad else GOOD_LANDMARKS
        rows.append(_row(name, "expw", label, f"{name}.png", lm, f"t{name}"))
    write_jsonl(root / "processed.jsonl", rows)
    return root / "processed.jsonl"


def feature_loader(manifest, batch_size=2):
    return build_loader(manifest, "target_features", batch_size=batch_size)


class TinyImageModel(nn.Module):
    """Global-pools an image to RGB means, then a linear map to two features."""

    def __init__(self, scale=20.0):
        super().__init__()
        self.projection = nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(torch.tensor([[scale, 0, 0], [0, scale, 0]]))

    def forward_target_features(self, images):
        return self.projection(images.mean(dim=(2, 3)))

    def forward_features(self, source, target):
        s, t = self.forward_target_features(source), self.forward_target_features(target)
        return BCSFeatures(s, t, s + t, s - t)


def make_banks():
    return PrototypeBanks(torch.eye(2))


def save_snapshot(path, model, banks, epoch):
    torch.save(
        {
            "completed_epoch": epoch,
            "model_state_dict": model.state_dict(),
            "prototype_state_dict": banks.state_dict(),
        },
        path,
    )


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = ["FaceDataset"]
