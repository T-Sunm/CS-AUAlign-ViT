from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cs_aualign_vit.training import compute_prototype_losses

C = torch.log1p(torch.exp(torch.tensor(-2.0)))


def feature_bundle(source, target=None, target_to_source=None, source_to_target=None):
    return SimpleNamespace(
        source_features=source,
        target_features=source if target is None else target,
        target_to_source_features=source if target_to_source is None else target_to_source,
        source_to_target_features=source if source_to_target is None else source_to_target,
    )


def banks(source=None, target=None):
    source = torch.eye(2) if source is None else source
    target = source if target is None else target
    return SimpleNamespace(
        source_prototypes=nn.Parameter(source.clone()),
        target_prototypes=nn.Parameter(target.clone()),
    )


def losses(features, prototype_banks=None, source_labels=None, pseudo_labels=None, mask=None):
    batch = features.source_features.shape[0]
    return compute_prototype_losses(
        features,
        prototype_banks or banks(),
        torch.zeros(batch, dtype=torch.long) if source_labels is None else source_labels,
        torch.zeros(batch, dtype=torch.long) if pseudo_labels is None else pseudo_labels,
        torch.ones(batch, dtype=torch.bool) if mask is None else mask,
        num_classes=2,
    )


def test_source_loss_matches_hand_calculation():
    result = losses(feature_bundle(torch.tensor([[2.0, 0.0]], requires_grad=True)))
    assert torch.allclose(result.l_s, C)
    assert result.l_s is result.terms["s"]


def test_mask_excludes_value_gradient_and_invalid_untrusted_label():
    target = torch.tensor([[0.0, 2.0], [2.0, 0.0]], requires_grad=True)
    result = losses(
        feature_bundle(torch.zeros(2, 2, requires_grad=True), target=target),
        pseudo_labels=torch.tensor([1, 99]),
        mask=torch.tensor([True, False]),
    )
    result.l_t.backward()
    assert torch.allclose(result.l_t, C)
    assert torch.count_nonzero(target.grad[0])
    assert torch.count_nonzero(target.grad[1]) == 0
    assert result.num_trusted == 1


def test_cross_terms_use_the_specified_feature_bank_and_label():
    source_bank = torch.eye(2)
    target_bank = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    t2s = losses(
        feature_bundle(
            torch.zeros(1, 2),
            target=torch.tensor([[0.0, 2.0]]),
            target_to_source=torch.tensor([[2.0, 0.0]]),
        ),
        banks(source_bank, target_bank),
        source_labels=torch.tensor([0]),
        pseudo_labels=torch.tensor([0]),
    )
    assert torch.allclose(t2s.terms["t2s_cross"], C)
    assert torch.allclose(t2s.terms["t2s_native"], 2 + C)
    assert torch.allclose(t2s.l_t2s, 2 + 2 * C)

    s2t = losses(
        feature_bundle(
            torch.tensor([[2.0, 0.0]]),
            source_to_target=torch.tensor([[2.0, 0.0]]),
        ),
        banks(source_bank, target_bank),
        source_labels=torch.tensor([1]),
        pseudo_labels=torch.tensor([0]),
    )
    assert torch.allclose(s2t.terms["s2t_cross"], 2 + C)
    assert torch.allclose(s2t.terms["s2t_native"], C)
    assert torch.allclose(s2t.l_s2t, 2 + 2 * C)


def test_each_term_has_its_own_denominator():
    source = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    result = losses(
        feature_bundle(source, source_to_target=source),
        source_labels=torch.tensor([0, 0]),
        pseudo_labels=torch.tensor([0, 1]),
        mask=torch.tensor([True, False]),
    )
    assert torch.allclose(result.terms["s2t_cross"], C)
    assert torch.allclose(result.terms["s2t_native"], 1 + C)
    assert torch.allclose(result.l_s2t, 1 + 2 * C)


def test_empty_trusted_set_returns_graph_connected_zeros():
    tensors = [torch.randn(2, 2, requires_grad=True) for _ in range(4)]
    features = feature_bundle(tensors[0], tensors[1], tensors[2], tensors[3])
    prototype_banks = banks()
    result = losses(features, prototype_banks, mask=torch.zeros(2, dtype=torch.bool))
    assert result.num_trusted == 0
    assert all(result.terms[name].item() == 0 for name in ("t", "t2s_native", "s2t_cross"))
    (result.l_t + result.terms["t2s_native"] + result.terms["s2t_cross"]).backward()
    assert torch.equal(tensors[1].grad, torch.zeros_like(tensors[1]))
    assert torch.equal(tensors[3].grad, torch.zeros_like(tensors[3]))
    assert torch.equal(prototype_banks.source_prototypes.grad, torch.zeros_like(torch.eye(2)))
    assert torch.equal(prototype_banks.target_prototypes.grad, torch.zeros_like(torch.eye(2)))


def test_native_cross_term_routes_gradient_without_detach():
    source = torch.randn(1, 2, requires_grad=True)
    target = torch.randn(1, 2, requires_grad=True)
    target_to_source = torch.randn(1, 2, requires_grad=True)
    prototype_banks = banks()
    result = losses(
        feature_bundle(source, target=target, target_to_source=target_to_source), prototype_banks
    )
    result.terms["t2s_native"].backward()
    assert target.grad is not None and torch.count_nonzero(target.grad)
    assert prototype_banks.source_prototypes.grad is not None
    assert prototype_banks.target_prototypes.grad is None
    assert source.grad is None and target_to_source.grad is None

    source = torch.randn(1, 2, requires_grad=True)
    target = torch.randn(1, 2, requires_grad=True)
    source_to_target = torch.randn(1, 2, requires_grad=True)
    prototype_banks = banks()
    result = losses(
        feature_bundle(source, target=target, source_to_target=source_to_target), prototype_banks
    )
    result.terms["s2t_native"].backward()
    assert source.grad is not None and torch.count_nonzero(source.grad)
    assert prototype_banks.target_prototypes.grad is not None
    assert prototype_banks.source_prototypes.grad is None
    assert target.grad is None and source_to_target.grad is None


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ("shape", ValueError),
        ("source_label", ValueError),
        ("trusted_pseudo_label", ValueError),
        ("label_dtype", TypeError),
        ("mask_dtype", TypeError),
        ("non_finite", ValueError),
    ],
)
def test_invalid_inputs_fail_clearly(change, error):
    features = feature_bundle(torch.ones(2, 2))
    source_labels = torch.tensor([0, 1])
    pseudo_labels = torch.tensor([0, 1])
    mask = torch.tensor([True, True])
    if change == "shape":
        features.target_features = torch.ones(1, 2)
    elif change == "source_label":
        source_labels[0] = 2
    elif change == "trusted_pseudo_label":
        pseudo_labels[0] = 2
    elif change == "label_dtype":
        pseudo_labels = pseudo_labels.float()
    elif change == "mask_dtype":
        mask = mask.long()
    elif change == "non_finite":
        features.source_features[0, 0] = float("nan")
    with pytest.raises(error):
        losses(features, source_labels=source_labels, pseudo_labels=pseudo_labels, mask=mask)


def test_real_shapes_and_half_precision_produce_float32_losses_without_mutation():
    tensors = [torch.randn(4, 768, dtype=torch.float16, requires_grad=True) for _ in range(4)]
    originals = [value.detach().clone() for value in tensors]
    features = feature_bundle(*tensors)
    prototype_banks = SimpleNamespace(
        source_prototypes=nn.Parameter(torch.randn(7, 768, dtype=torch.float16)),
        target_prototypes=nn.Parameter(torch.randn(7, 768, dtype=torch.float16)),
    )
    result = compute_prototype_losses(
        features,
        prototype_banks,
        torch.tensor([0, 1, 2, 3]),
        torch.tensor([3, 2, 1, 0]),
        torch.tensor([True, False, True, False]),
    )
    assert all(value.ndim == 0 and value.dtype == torch.float32 for value in result.terms.values())
    assert all(value.requires_grad for value in (result.l_s, result.l_t, result.l_t2s, result.l_s2t))
    assert all(torch.equal(value, original) for value, original in zip(tensors, originals, strict=True))


def test_loss_computation_stays_float32_inside_autocast():
    features = feature_bundle(torch.randn(4, 8))
    prototype_banks = SimpleNamespace(
        source_prototypes=nn.Parameter(torch.randn(2, 8)),
        target_prototypes=nn.Parameter(torch.randn(2, 8)),
    )
    arguments = (
        features,
        prototype_banks,
        torch.tensor([0, 1, 0, 1]),
        torch.tensor([1, 0, 1, 0]),
        torch.tensor([True, True, False, True]),
    )
    expected = compute_prototype_losses(*arguments, num_classes=2)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = compute_prototype_losses(*arguments, num_classes=2)
    assert all(
        torch.equal(actual.terms[name], expected.terms[name]) for name in expected.terms
    )
