"""Prototype classification losses for POST adaptation."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class PrototypeLosses:
    l_s: torch.Tensor
    l_t: torch.Tensor
    l_t2s: torch.Tensor
    l_s2t: torch.Tensor
    terms: dict[str, torch.Tensor]
    num_trusted: int


def compute_prototype_losses(
    features,
    prototype_banks,
    source_labels,
    pseudo_labels,
    trusted_mask,
    *,
    num_classes=7,
):
    """Return the four unweighted prototype losses from POST equations 5-9."""
    if type(num_classes) is not int or num_classes < 1:
        raise ValueError("num_classes must be a positive integer")
    feature_tensors = (
        features.source_features,
        features.target_features,
        features.target_to_source_features,
        features.source_to_target_features,
    )
    shape = feature_tensors[0].shape
    if len(shape) != 2 or shape[0] == 0 or any(value.shape != shape for value in feature_tensors):
        raise ValueError("All four features must have the same non-empty shape [B, D]")

    source_prototypes = prototype_banks.source_prototypes
    target_prototypes = prototype_banks.target_prototypes
    if (
        source_prototypes.ndim != 2
        or source_prototypes.shape != target_prototypes.shape
        or source_prototypes.shape != (num_classes, shape[1])
    ):
        raise ValueError("Prototype banks must have shape [num_classes, feature_dim]")
    device = feature_tensors[0].device
    if any(value.device != device for value in (*feature_tensors, source_prototypes, target_prototypes)):
        raise ValueError("Features and prototype banks must be on the same device")

    labels = (source_labels, pseudo_labels, trusted_mask)
    if any(not isinstance(value, torch.Tensor) or value.shape != (shape[0],) for value in labels):
        raise ValueError("Labels and trusted_mask must have shape [B]")
    integer_dtypes = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
    if source_labels.dtype not in integer_dtypes or pseudo_labels.dtype not in integer_dtypes:
        raise TypeError("source_labels and pseudo_labels must have integer dtype")
    if trusted_mask.dtype != torch.bool:
        raise TypeError("trusted_mask must have bool dtype")

    source_labels = source_labels.to(device=device, dtype=torch.long)
    pseudo_labels = pseudo_labels.to(device=device, dtype=torch.long)
    trusted_mask = trusted_mask.to(device=device)
    if ((source_labels < 0) | (source_labels >= num_classes)).any():
        raise ValueError("source_labels contain a class outside [0, num_classes)")
    trusted_pseudo_labels = pseudo_labels[trusted_mask]
    if ((trusted_pseudo_labels < 0) | (trusted_pseudo_labels >= num_classes)).any():
        raise ValueError("Trusted pseudo_labels contain a class outside [0, num_classes)")

    feature_tensors = tuple(value.float() for value in feature_tensors)
    source_prototypes = source_prototypes.float()
    target_prototypes = target_prototypes.float()
    if not all(
        torch.isfinite(value).all()
        for value in (*feature_tensors, source_prototypes, target_prototypes)
    ):
        raise ValueError("Features and prototype banks must be finite")

    source, target, target_to_source, source_to_target = feature_tensors
    safe_pseudo_labels = pseudo_labels.clone()
    safe_pseudo_labels[~trusted_mask] = 0
    denominator = trusted_mask.sum().clamp_min(1)

    def unmasked(inputs, prototypes, labels):
        return F.cross_entropy(inputs @ prototypes.T, labels)

    def masked(inputs, prototypes):
        losses = F.cross_entropy(
            inputs @ prototypes.T, safe_pseudo_labels, reduction="none"
        )
        return (losses * trusted_mask).sum() / denominator

    with torch.autocast(device_type=device.type, enabled=False):
        terms = {
            "s": unmasked(source, source_prototypes, source_labels),
            "t": masked(target, target_prototypes),
            "t2s_cross": unmasked(target_to_source, source_prototypes, source_labels),
            "t2s_native": masked(target, source_prototypes),
            "s2t_cross": masked(source_to_target, target_prototypes),
            "s2t_native": unmasked(source, target_prototypes, source_labels),
        }
    return PrototypeLosses(
        l_s=terms["s"],
        l_t=terms["t"],
        l_t2s=terms["t2s_cross"] + terms["t2s_native"],
        l_s2t=terms["s2t_cross"] + terms["s2t_native"],
        terms=terms,
        num_trusted=int(trusted_mask.sum()),
    )
