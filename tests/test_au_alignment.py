import torch

from cs_aualign_vit.training.au_alignment import compute_au_alignment


def metadata(batch, *, values=None, masks=None, valid=None):
    return {
        "au": values if values is not None else torch.ones(batch, 12),
        "mask": masks if masks is not None else torch.ones(batch, 49, 12),
        "au_valid": valid if valid is not None else torch.ones(batch, dtype=torch.bool),
        "landmark_valid": torch.ones(batch, dtype=torch.bool),
    }


def compute(source_tokens, target_tokens, source_au, target_au, **changes):
    options = {
        "tau_mask": 0.5,
        "n_eff_min": 1.0,
        "q_c": torch.ones(12),
        "epsilon": 1.0e-8,
        "epsilon_weight": 1.0e-8,
    }
    options.update(changes)
    return compute_au_alignment(source_tokens, target_tokens, source_au, target_au, **options)


def test_row_major_region_pooling_and_intensity_weighted_centroids():
    source_tokens = torch.arange(49, dtype=torch.float32).reshape(1, 49, 1).requires_grad_()
    target_tokens = torch.stack(
        (torch.arange(49, dtype=torch.float32), torch.arange(49, dtype=torch.float32) + 10)
    ).reshape(2, 49, 1)
    source_masks = torch.zeros(1, 49, 12)
    target_masks = torch.zeros(2, 49, 12)
    source_masks[:, 17, :] = 1  # row 2, column 3 in row-major order
    target_masks[:, 17, :] = 1
    target_values = torch.ones(2, 12)
    target_values[:, 0] = torch.tensor([1.0, 3.0]) / 3.0

    result = compute(
        source_tokens,
        target_tokens,
        metadata(1, masks=source_masks),
        metadata(2, values=target_values, masks=target_masks),
    )

    assert result.source_region_features[0, 0, 0] == 17
    assert result.target_region_features[:, 0, 0].tolist() == [17, 27]
    assert torch.allclose(result.target_centroids[0, 0], torch.tensor(24.5))
    assert torch.allclose(result.target_n_eff[0], torch.tensor(1.6))
    assert torch.allclose(result.loss_components.sum(), result.loss)
    assert result.distances.shape == result.loss_components.shape == (12,)
    result.loss.backward()
    assert source_tokens.grad is not None and torch.count_nonzero(source_tokens.grad)


def test_invalid_regions_and_tiny_total_weight_do_not_count_as_evidence():
    tokens = torch.randn(2, 49, 3, requires_grad=True)
    values = torch.full((2, 12), 1.0e-12)
    masks = torch.ones(2, 49, 12)
    masks[:, :, 1] = 0
    result = compute(
        tokens,
        tokens.detach().clone().requires_grad_(),
        metadata(2, values=values, masks=masks),
        metadata(2, values=values, masks=masks),
        tau_mask=1.0,
        n_eff_min=1.0,
        epsilon_weight=1.0e-8,
    )

    assert not result.valid_aus.any()
    assert result.loss.isfinite() and result.loss == 0
    result.loss.backward()
    assert torch.count_nonzero(tokens.grad) == 0


def test_spatial_validity_excludes_rows_even_when_landmarks_are_valid():
    tokens = torch.randn(2, 49, 3, requires_grad=True)
    source = metadata(2)
    target = metadata(2)
    source["spatial_valid"] = torch.zeros(2, dtype=torch.bool)

    result = compute(tokens, tokens.detach().clone(), source, target)

    assert not result.valid_aus.any()
    assert result.loss == 0


def test_local_loss_supports_different_batch_sizes_and_is_pairing_invariant():
    source_tokens = torch.randn(2, 49, 4)
    target_tokens = torch.randn(3, 49, 4)
    source_au = metadata(2, values=torch.rand(2, 12))
    target_au = metadata(3, values=torch.rand(3, 12))

    expected = compute(source_tokens, target_tokens, source_au, target_au)
    permutation = torch.tensor([2, 0, 1])
    shuffled_au = {name: value[permutation] for name, value in target_au.items()}
    actual = compute(source_tokens, target_tokens[permutation], source_au, shuffled_au)

    assert expected.source_region_features.shape == (2, 12, 4)
    assert expected.target_region_features.shape == (3, 12, 4)
    assert torch.allclose(expected.loss, actual.loss)
    assert torch.allclose(expected.target_centroids, actual.target_centroids)
