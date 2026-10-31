"""Refresh target pseudo labels from native target features."""

import logging
import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F

LOGGER = logging.getLogger(__name__)


@dataclass
class TargetSupervision:
    pseudo_labels: torch.Tensor
    predicted_labels: torch.Tensor
    confidence: torch.Tensor
    trusted_mask: torch.Tensor
    sample_ids: list[str]
    epoch: int

    def __post_init__(self):
        size = len(self.sample_ids)
        fields = (
            ("pseudo_labels", self.pseudo_labels, torch.long),
            ("predicted_labels", self.predicted_labels, torch.long),
            ("confidence", self.confidence, torch.float32),
            ("trusted_mask", self.trusted_mask, torch.bool),
        )
        if len(set(self.sample_ids)) != size:
            raise ValueError("Target supervision sample_ids must be unique")
        for name, value, dtype in fields:
            if not isinstance(value, torch.Tensor) or value.shape != (size,) or value.dtype != dtype:
                raise ValueError(f"{name} must have shape [N] and dtype {dtype}")
        if type(self.epoch) is not int:
            raise TypeError("epoch must be an integer")

    def lookup(self, sample_ids):
        positions = {sample_id: index for index, sample_id in enumerate(self.sample_ids)}
        try:
            indices = torch.tensor([positions[sample_id] for sample_id in sample_ids], dtype=torch.long)
        except KeyError as error:
            raise KeyError(f"Unknown target sample_id: {error.args[0]}") from error
        return self.pseudo_labels[indices], self.trusted_mask[indices]

    def state_dict(self):
        return {
            "pseudo_labels": self.pseudo_labels,
            "predicted_labels": self.predicted_labels,
            "confidence": self.confidence,
            "trusted_mask": self.trusted_mask,
            "sample_ids": self.sample_ids,
            "epoch": self.epoch,
        }

    @classmethod
    def from_state_dict(cls, state, expected_sample_ids):
        if not isinstance(state, dict):
            raise TypeError("Target supervision state must be a dictionary")
        if state.get("sample_ids") != list(expected_sample_ids):
            raise ValueError("Saved target sample_ids differ from the adaptation dataset")
        required = {
            "pseudo_labels",
            "predicted_labels",
            "confidence",
            "trusted_mask",
            "sample_ids",
            "epoch",
        }
        if set(state) != required:
            raise ValueError("Target supervision state has unexpected fields")
        return cls(**state)


def refresh_target_supervision(
    feature_model,
    prototype_banks,
    loader,
    expected_sample_ids,
    *,
    epoch,
    num_classes=7,
    kmeans_iterations=1,
    trust_threshold=0.95,
    eps=1e-8,
    previous=None,
):
    """Extract target features and replace the complete pseudo-label snapshot."""
    expected_sample_ids = list(expected_sample_ids)
    if not expected_sample_ids or len(set(expected_sample_ids)) != len(expected_sample_ids):
        raise ValueError("expected_sample_ids must be a non-empty unique sequence")
    if type(epoch) is not int:
        raise TypeError("epoch must be an integer")
    if type(kmeans_iterations) is not int or kmeans_iterations < 1:
        raise ValueError("kmeans_iterations must be a positive integer")
    if type(num_classes) is not int or num_classes < 1:
        raise ValueError("num_classes must be a positive integer")
    if not math.isfinite(trust_threshold) or not 0 <= trust_threshold <= 1:
        raise ValueError("trust_threshold must be finite and between zero and one")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")

    prototypes = prototype_banks.target_prototypes
    if prototypes.ndim != 2 or prototypes.shape[0] != num_classes:
        raise ValueError("Target prototypes must have shape [num_classes, feature_dim]")
    device = prototypes.device
    was_training = feature_model.training
    feature_batches = []
    observed_ids = []
    try:
        feature_model.eval()
        with torch.no_grad():
            for images, sample_ids in loader:
                features = feature_model.forward_target_features(images.to(device))
                if features.ndim != 2 or features.shape[1] != prototypes.shape[1]:
                    raise ValueError("Target features do not match target prototype dimension")
                feature_batches.append(features.float())
                observed_ids.extend(sample_ids)
    finally:
        feature_model.train(was_training)

    if observed_ids != expected_sample_ids or len(set(observed_ids)) != len(observed_ids):
        raise ValueError("Feature loader sample_ids differ from expected_sample_ids")
    if not feature_batches:
        raise ValueError("Feature loader produced no target samples")
    features = torch.cat(feature_batches)
    prototypes = prototypes.detach().float()
    if not torch.isfinite(features).all() or not torch.isfinite(prototypes).all():
        raise ValueError("Target features and prototypes must be finite")

    feature_norms = features.norm(dim=1)
    prototype_norms = prototypes.norm(dim=1)
    if (feature_norms < eps).any() or (prototype_norms < eps).any():
        raise ValueError("Target features and prototypes must have non-zero norm")
    normalized_features = features / feature_norms[:, None]
    centers = prototypes / prototype_norms[:, None]
    empty_clusters = 0
    near_zero_clusters = 0
    for _ in range(kmeans_iterations):
        assignments = (normalized_features @ centers.T).argmax(dim=1)
        updated = centers.clone()
        for class_index in range(num_classes):
            members = normalized_features[assignments == class_index]
            if len(members) == 0:
                empty_clusters += 1
                continue
            total = members.sum(dim=0)
            norm = total.norm()
            if norm < eps:
                near_zero_clusters += 1
                continue
            updated[class_index] = total / norm
        centers = updated
    pseudo_labels = (normalized_features @ centers.T).argmax(dim=1)

    probabilities = F.softmax(features @ prototypes.T, dim=1)
    confidence, predicted_labels = probabilities.max(dim=1)
    trusted_mask = predicted_labels.eq(pseudo_labels) & confidence.gt(trust_threshold)
    result = TargetSupervision(
        pseudo_labels.cpu(),
        predicted_labels.cpu(),
        confidence.cpu(),
        trusted_mask.cpu(),
        observed_ids,
        epoch,
    )
    changed = None
    if previous is not None:
        if previous.sample_ids != observed_ids:
            raise ValueError("Previous target supervision uses different sample_ids")
        changed = int(result.pseudo_labels.ne(previous.pseudo_labels).sum())
    trusted_per_class = [
        int((result.trusted_mask & result.pseudo_labels.eq(index)).sum())
        for index in range(num_classes)
    ]
    trusted_rate_per_class = []
    for index, trusted_count in enumerate(trusted_per_class):
        class_count = int(result.pseudo_labels.eq(index).sum())
        trusted_rate_per_class.append(trusted_count / class_count if class_count else None)
    LOGGER.info(
        "target supervision refresh: trusted=%d/%d trusted_per_class=%s "
        "trusted_rate_per_class=%s agreement=%.4f "
        "changed=%s empty_clusters=%d near_zero_clusters=%d",
        int(result.trusted_mask.sum()),
        len(result.sample_ids),
        trusted_per_class,
        trusted_rate_per_class,
        float(result.predicted_labels.eq(result.pseudo_labels).float().mean()),
        changed,
        empty_clusters,
        near_zero_clusters,
    )
    return result


def target_supervision_diagnostics(result, true_labels):
    """Compute label-aware metrics without feeding them back into training."""
    true_labels = torch.as_tensor(true_labels, dtype=torch.long)
    if true_labels.shape != result.pseudo_labels.shape:
        raise ValueError("true_labels must have shape [N]")
    trusted = result.trusted_mask
    return {
        "pseudo_label_accuracy": float(result.pseudo_labels.eq(true_labels).float().mean()),
        "trusted_accuracy": (
            float(result.pseudo_labels[trusted].eq(true_labels[trusted]).float().mean())
            if trusted.any()
            else None
        ),
    }
