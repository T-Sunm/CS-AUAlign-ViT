"""USTST objectives (Eqs. 3-13 of Guo et al., MTAP 2023).

Every function returns an unweighted scalar; the trainer applies the lambdas of
Eq. 14. Losses with a per-sample gate follow the "loss, or 0 otherwise" form of
Eqs. 8-9 and 11-12 and are averaged over the whole batch, so a batch with no
selected sample contributes exactly zero.
"""

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class TargetGates:
    """Per-sample target selections shared by L_t, L_c and L_sim."""

    predicted_labels: torch.Tensor
    confidence: torch.Tensor
    normalized_entropy: torch.Tensor
    hard_mask: torch.Tensor
    soft_mask: torch.Tensor
    agree_mask: torch.Tensor
    soft_enabled: bool


def target_gates(
    weak_logits,
    cluster_labels,
    *,
    confidence_threshold=0.95,
    entropy_threshold=0.5,
):
    """Build the hard and soft pseudo-label selections from weak-view predictions.

    * hard: confident prediction (``max p > thr``) that agrees with the k-means
      label (Eqs. 8 and 11).
    * soft: ambiguous ("compound") prediction whose normalised entropy
      ``H(p) / log K`` exceeds ``thr_e`` (Eqs. 9 and 12). Following Section 3.3,
      soft labels are only enabled once the batch already contains a prediction
      above the confidence threshold, which suppresses the uniformly flat
      predictions of an untrained model.
    """
    if weak_logits.ndim != 2:
        raise ValueError("weak_logits must have shape [B, K]")
    if cluster_labels.shape != weak_logits.shape[:1]:
        raise ValueError("cluster_labels must have shape [B]")
    if not 0 <= confidence_threshold <= 1 or not 0 <= entropy_threshold <= 1:
        raise ValueError("Thresholds must lie in [0, 1]")
    with torch.no_grad():
        probabilities = F.softmax(weak_logits.detach().float(), dim=1)
        confidence, predicted = probabilities.max(dim=1)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
        normalized_entropy = entropy / math.log(weak_logits.shape[1])
        cluster_labels = cluster_labels.to(predicted.device)
        agree = predicted.eq(cluster_labels)
        hard = agree & confidence.gt(confidence_threshold)
        soft_enabled = bool(confidence.gt(confidence_threshold).any())
        soft = normalized_entropy.gt(entropy_threshold) & soft_enabled
    return TargetGates(predicted, confidence, normalized_entropy, hard, soft, agree, soft_enabled)


def _masked_batch_mean(per_sample, mask):
    return (per_sample * mask.to(per_sample.dtype)).sum() / per_sample.shape[0]


def hard_pseudo_label_loss(logits, labels, mask):
    per_sample = F.cross_entropy(logits.float(), labels.to(logits.device), reduction="none")
    return _masked_batch_mean(per_sample, mask.to(logits.device))


def soft_pseudo_label_loss(logits, teacher_logits, mask):
    targets = F.softmax(teacher_logits.detach().float(), dim=1)
    per_sample = F.cross_entropy(logits.float(), targets, reduction="none")
    return _masked_batch_mean(per_sample, mask.to(logits.device))


def logit_distance(first, second):
    """Eq. 3: mean squared logit difference for every pair, shape [N1, N2]."""
    return (first.float()[:, None, :] - second.float()[None, :, :]).pow(2).mean(dim=-1)


def similarity_loss(source_logits, source_labels, target_logits, target_labels, target_mask, *, margin):
    """Eqs. 3-4: batch-hard cross-domain triplet loss on predicted logits.

    Each source image is an anchor. Among selected target images, the positive is
    the same-class image least similar to the anchor (largest Eq. 3 distance) and
    the negative is the different-class image most similar to it (smallest
    distance). The hinge of Eq. 4 uses squared L2 norms ``||f(A) - f(P)||^2``, i.e.
    ``n`` times the Eq. 3 mean; selection is unaffected by that constant.
    Anchors without both a positive and a negative are skipped. Returns the loss
    and the number of anchors that contributed.
    """
    if margin < 0:
        raise ValueError("margin must be non-negative")
    target_mask = target_mask.to(target_logits.device)
    if not target_mask.any():
        return source_logits.float().sum() * 0, 0
    candidates = target_logits[target_mask]
    labels = target_labels.to(target_logits.device)[target_mask]
    distance = logit_distance(source_logits, candidates)
    same = source_labels.to(labels.device)[:, None].eq(labels[None, :])
    valid = same.any(dim=1) & (~same).any(dim=1)
    if not valid.any():
        return source_logits.float().sum() * 0, 0
    hardest_positive = distance.masked_fill(~same, float("-inf")).amax(dim=1)
    hardest_negative = distance.masked_fill(same, float("inf")).amin(dim=1)
    num_classes = source_logits.shape[1]
    losses = F.relu(num_classes * (hardest_positive - hardest_negative) + margin)
    return losses[valid].mean(), int(valid.sum())


def gaussian_mmd(source, target, *, kernel_mul=2.0, kernel_num=5):
    """Eq. 5 with a multi-bandwidth Gaussian kernel (biased MMD^2 estimate).

    The bandwidth family is centred on the mean pairwise squared distance of the
    joint batch, as in DAN; the centre is treated as a constant.
    """
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("MMD inputs must have shape [B, m] with equal m")
    if kernel_num < 1 or kernel_mul <= 0:
        raise ValueError("kernel_num must be positive and kernel_mul positive")
    total = torch.cat((source, target)).float()
    distances = torch.cdist(total, total).pow(2)
    count = total.shape[0]
    bandwidth = distances.detach().sum() / max(count * count - count, 1)
    bandwidth = bandwidth.clamp_min(1e-12) / (kernel_mul ** (kernel_num // 2))
    kernels = sum(
        torch.exp(-distances / (bandwidth * kernel_mul**index)) for index in range(kernel_num)
    )
    n = source.shape[0]
    return (
        kernels[:n, :n].mean()
        + kernels[n:, n:].mean()
        - kernels[:n, n:].mean()
        - kernels[n:, :n].mean()
    )
