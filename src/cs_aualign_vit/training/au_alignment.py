"""AU-weighted batch-level alignment over native Stage-4 tokens."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AUAlignmentResult:
    loss: torch.Tensor
    source_region_features: torch.Tensor
    target_region_features: torch.Tensor
    source_centroids: torch.Tensor
    target_centroids: torch.Tensor
    source_n_eff: torch.Tensor
    target_n_eff: torch.Tensor
    source_weight_sums: torch.Tensor
    target_weight_sums: torch.Tensor
    valid_aus: torch.Tensor
    distances: torch.Tensor
    loss_components: torch.Tensor


def _domain_statistics(tokens, intensities, masks, au_valid, spatial_valid, tau_mask, epsilon):
    if tokens.ndim != 3 or tokens.shape[1] != 49:
        raise ValueError("Native tokens must have shape [B, 49, D]")
    batch = tokens.shape[0]
    if intensities.shape != (batch, 12) or masks.shape != (batch, 49, 12):
        raise ValueError("AU intensities and masks must have shapes [B, 12] and [B, 49, 12]")
    if au_valid.shape != (batch,) or spatial_valid.shape != (batch,):
        raise ValueError("AU and spatial validity must have shape [B]")
    if au_valid.dtype != torch.bool or spatial_valid.dtype != torch.bool:
        raise TypeError("AU and spatial validity must be boolean")

    device = tokens.device
    values = intensities.detach().to(device=device, dtype=torch.float32)
    spatial = masks.detach().to(device=device, dtype=torch.float32)
    if not torch.isfinite(values).all() or not torch.isfinite(spatial).all():
        raise ValueError("AU intensities and masks must be finite")
    if ((values < 0) | (values > 1)).any() or (spatial < 0).any():
        raise ValueError("AU intensities must be in [0, 1] and masks must be non-negative")

    native = tokens.float()
    mask_mass = spatial.sum(dim=1)
    region_features = torch.einsum("bic,bid->bcd", spatial, native)
    region_features = region_features / (mask_mass[..., None] + epsilon)
    valid = (
        au_valid.to(device)[:, None]
        & spatial_valid.to(device)[:, None]
        & (mask_mass >= tau_mask)
    )
    weights = values * valid.to(values.dtype)
    weight_sums = weights.sum(dim=0)
    centroids = torch.einsum("bc,bcd->cd", weights, region_features)
    centroids = centroids / (weight_sums[:, None] + epsilon)
    n_eff = weight_sums.square() / (weights.square().sum(dim=0) + epsilon)
    return region_features, centroids, n_eff, weight_sums


def compute_au_alignment(
    source_tokens,
    target_tokens,
    source_au,
    target_au,
    *,
    tau_mask,
    n_eff_min,
    q_c,
    epsilon=1.0e-8,
    epsilon_weight=1.0e-8,
):
    """Compute the unweighted local alignment loss and diagnostics."""
    if tau_mask < 0 or n_eff_min < 0 or epsilon <= 0 or epsilon_weight < 0:
        raise ValueError("Alignment thresholds must be non-negative and epsilon positive")
    q_c = torch.as_tensor(q_c, device=source_tokens.device, dtype=torch.float32)
    if q_c.shape != (12,) or not torch.isfinite(q_c).all() or (q_c < 0).any():
        raise ValueError("q_c must contain 12 finite non-negative values")

    source = _domain_statistics(
        source_tokens,
        source_au["au"],
        source_au["mask"],
        source_au["au_valid"],
        source_au.get("spatial_valid", source_au["landmark_valid"]),
        tau_mask,
        epsilon,
    )
    target = _domain_statistics(
        target_tokens,
        target_au["au"],
        target_au["mask"],
        target_au["au_valid"],
        target_au.get("spatial_valid", target_au["landmark_valid"]),
        tau_mask,
        epsilon,
    )
    source_regions, source_centroids, source_n_eff, source_sums = source
    target_regions, target_centroids, target_n_eff, target_sums = target
    valid_aus = (
        (source_n_eff >= n_eff_min)
        & (target_n_eff >= n_eff_min)
        & (source_sums > epsilon_weight)
        & (target_sums > epsilon_weight)
    )
    weights = q_c * valid_aus.to(q_c.dtype)
    distances = (source_centroids - target_centroids).square().sum(dim=-1)
    loss_components = weights * distances / (weights.sum() + epsilon)
    loss = loss_components.sum()
    return AUAlignmentResult(
        loss=loss,
        source_region_features=source_regions,
        target_region_features=target_regions,
        source_centroids=source_centroids,
        target_centroids=target_centroids,
        source_n_eff=source_n_eff,
        target_n_eff=target_n_eff,
        source_weight_sums=source_sums,
        target_weight_sums=target_sums,
        valid_aus=valid_aus,
        distances=distances,
        loss_components=loss_components,
    )
