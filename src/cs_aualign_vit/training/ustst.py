"""USTST adaptation trainer (Guo et al., MTAP 2023) on the POST training scaffold.

The epoch loop, k-means refresh, STR boundary transactions, checkpointing and
resume are inherited unchanged from :class:`AdaptationTrainer`; only the update
step and the configuration differ. Eq. 14:

    L = L_s + L_t + lambda_c L_c + lambda_sim L_sim + lambda_m L_m + lambda_a L_a
"""

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from cs_aualign_vit.training.adaptation import AdaptationConfig, AdaptationTrainer
from cs_aualign_vit.training.ustst_losses import (
    gaussian_mmd,
    hard_pseudo_label_loss,
    similarity_loss,
    soft_pseudo_label_loss,
    target_gates,
)


@dataclass(frozen=True)
class USTSTConfig(AdaptationConfig):
    """USTST hyperparameters; defaults follow Section 4.2 of the paper.

    ``kmeans_interval`` defaults to 1 because USTST clusters the target domain
    before every epoch. The inherited POST weights ``alpha``/``beta``/``gamma``
    and AU fields are unused and must keep their disabled defaults.
    """

    kmeans_interval: int = 1
    confidence_threshold: float = 0.95
    entropy_threshold: float = 0.5
    similarity_margin: float = 0.2
    lambda_c: float = 0.01
    lambda_sim: float = 0.01
    lambda_m: float = 0.01
    lambda_a: float = 0.1
    mmd_kernel_mul: float = 2.0
    mmd_kernel_num: int = 5
    str_threshold: float = 0.2
    str_patch_scale_range: tuple[float, float] | None = (0.3, 0.5)
    cross_start_block: int = 0
    # Smoke runs truncate every epoch; recording it makes a resume of a smoke
    # checkpoint as a full run fail the configuration check.
    max_steps_per_epoch: int | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.max_steps_per_epoch is not None and (
            type(self.max_steps_per_epoch) is not int or self.max_steps_per_epoch < 1
        ):
            raise ValueError("max_steps_per_epoch must be a positive integer or None")
        if type(self.cross_start_block) is not int or not 0 <= self.cross_start_block < 12:
            raise ValueError("cross_start_block must be an integer in [0, 11]")
        if self.str_patch_scale_range is not None and (
            len(self.str_patch_scale_range) != 2
            or not 0 < self.str_patch_scale_range[0] <= self.str_patch_scale_range[1]
        ):
            raise ValueError("str_patch_scale_range must be (low, high) with 0 < low <= high")
        if self.au_mode != "off" or self.au_alignment_enabled:
            raise ValueError("USTST does not use AU guidance or AU alignment")
        for name in ("confidence_threshold", "entropy_threshold", "str_threshold"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        for name in ("similarity_margin", "lambda_c", "lambda_sim", "lambda_m", "lambda_a"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.mmd_kernel_mul <= 0 or type(self.mmd_kernel_num) is not int or self.mmd_kernel_num < 1:
            raise ValueError("MMD kernel settings must be positive")


class USTSTTrainer(AdaptationTrainer):
    """Train the CST with the USTST objective.

    ``prototype_banks`` is a :class:`USTSTHeads`; target batches are
    ``((weak_images, strong_images), sample_ids)``. The k-means snapshot from
    ``refresh_target_supervision`` supplies the clustering pseudo-labels
    ``y_hat_k``; its POST trust mask is ignored.
    """

    def __init__(self, *, config=None, **kwargs):
        config = config or USTSTConfig()
        if not isinstance(config, USTSTConfig):
            raise TypeError("USTSTTrainer requires a USTSTConfig")
        super().__init__(config=config, **kwargs)

    def train_step(self, source_batch, target_batch):
        if self.supervision is None:
            raise RuntimeError("Target supervision must be refreshed before training")
        config = self.config
        source_images, source_labels = source_batch[:2]
        target_views, sample_ids = target_batch[:2]
        if not isinstance(target_views, (list, tuple)) or len(target_views) != 2:
            raise ValueError("USTST target batches must contain (weak, strong) image views")
        weak_images, strong_images = target_views
        if not len(source_images) == len(weak_images) == len(strong_images):
            raise ValueError("Paired source and target batches must have the same size")
        cluster_labels, _ = self.supervision.lookup(list(sample_ids))
        source_images = source_images.to(self.device)
        source_labels = source_labels.to(self.device)
        weak_images = weak_images.to(self.device)
        strong_images = strong_images.to(self.device)
        cluster_labels = cluster_labels.to(self.device)
        heads = self.prototype_banks

        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=config.amp):
            features = self.model.forward_features(source_images, weak_images, strong_images)
            source_logits = heads.logits(features.source_features)
            weak_logits = heads.logits(features.target_weak_features)
            strong_logits = heads.logits(features.target_strong_features)
            cross_logits = heads.cross_logits(features.cross_features)
            adversarial_loss = self.adversarial_module(
                features.source_features, features.target_weak_features
            )
        if not isinstance(adversarial_loss, torch.Tensor) or adversarial_loss.ndim:
            raise TypeError("Adversarial module must return one scalar loss tensor")

        # Losses run in float32 outside autocast: MMD and the triplet distances
        # are sensitive to half precision.
        source_logits = source_logits.float()
        weak_logits = weak_logits.float()
        strong_logits = strong_logits.float()
        cross_logits = cross_logits.float()
        gates = target_gates(
            weak_logits,
            cluster_labels,
            confidence_threshold=config.confidence_threshold,
            entropy_threshold=config.entropy_threshold,
        )
        loss_s = F.cross_entropy(source_logits, source_labels)
        loss_th = hard_pseudo_label_loss(strong_logits, gates.predicted_labels, gates.hard_mask)
        loss_ts = soft_pseudo_label_loss(strong_logits, weak_logits, gates.soft_mask)
        loss_ch = hard_pseudo_label_loss(cross_logits, gates.predicted_labels, gates.hard_mask)
        loss_cs = soft_pseudo_label_loss(cross_logits, weak_logits, gates.soft_mask)
        loss_t = loss_th + loss_ts
        loss_c = loss_ch + loss_cs
        loss_sim, sim_anchors = similarity_loss(
            source_logits,
            source_labels,
            weak_logits,
            gates.predicted_labels,
            gates.agree_mask,
            margin=config.similarity_margin,
        )
        loss_m = gaussian_mmd(
            features.source_features.float(),
            features.target_weak_features.float(),
            kernel_mul=config.mmd_kernel_mul,
            kernel_num=config.mmd_kernel_num,
        )
        loss_a = adversarial_loss.float()
        total = (
            loss_s
            + loss_t
            + config.lambda_c * loss_c
            + config.lambda_sim * loss_sim
            + config.lambda_m * loss_m
            + config.lambda_a * loss_a
        )
        if not torch.isfinite(total):
            raise FloatingPointError(f"Non-finite USTST loss at step {self.global_step}")

        if config.amp:
            self.scaler.scale(total).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total.backward()
            self.optimizer.step()
        self.global_step += 1
        batch = len(source_images)
        return {
            "loss_total": float(total.detach()),
            "loss_s": float(loss_s.detach()),
            "loss_t": float(loss_t.detach()),
            "loss_th": float(loss_th.detach()),
            "loss_ts": float(loss_ts.detach()),
            "loss_c": float(loss_c.detach()),
            "loss_ch": float(loss_ch.detach()),
            "loss_cs": float(loss_cs.detach()),
            "loss_sim": float(loss_sim.detach()),
            "loss_m": float(loss_m.detach()),
            "loss_ad": float(loss_a.detach()),
            "hard_ratio": float(gates.hard_mask.sum()) / batch,
            "soft_ratio": float(gates.soft_mask.sum()) / batch,
            "soft_enabled_ratio": float(gates.soft_enabled),
            "cluster_agreement": float(gates.agree_mask.float().mean()),
            "target_confidence": float(gates.confidence.mean()),
            "target_normalized_entropy": float(gates.normalized_entropy.mean()),
            "sim_anchor_ratio": sim_anchors / batch,
            "source_accuracy": float(source_logits.argmax(dim=1).eq(source_labels).float().mean()),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            **getattr(self.adversarial_module, "last_metrics", {}),
        }
