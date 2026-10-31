from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from cs_aualign_vit.training import (
    TargetSupervision,
    refresh_target_supervision,
    target_supervision_diagnostics,
)


class FeatureDataset(Dataset):
    def __init__(self, features, sample_ids):
        self.features = features
        self.sample_ids = sample_ids

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.sample_ids[index]


class FeatureModel(nn.Module):
    def forward_target_features(self, inputs):
        return inputs


class ParameterizedFeatureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))

    def forward_target_features(self, inputs):
        return inputs * self.scale


def refresh(features, prototypes, sample_ids=None, **kwargs):
    sample_ids = sample_ids or [f"target:{index}" for index in range(len(features))]
    loader = DataLoader(FeatureDataset(features, sample_ids), batch_size=2, shuffle=False)
    model = FeatureModel().train()
    result = refresh_target_supervision(
        model,
        SimpleNamespace(target_prototypes=nn.Parameter(prototypes.clone())),
        loader,
        sample_ids,
        epoch=kwargs.pop("epoch", 0),
        num_classes=len(prototypes),
        **kwargs,
    )
    assert model.training
    return result


def test_refresh_assigns_classes_trust_and_lookup_without_gradients():
    features = torch.tensor([[3.0, 0.0], [2.0, 0.1], [0.0, 3.0], [0.1, 2.0]])
    prototypes = torch.eye(2)
    result = refresh(features, prototypes, trust_threshold=0.8)

    assert result.pseudo_labels.tolist() == [0, 0, 1, 1]
    assert result.predicted_labels.tolist() == [0, 0, 1, 1]
    assert result.trusted_mask.tolist() == [True, True, True, True]
    labels, mask = result.lookup(["target:3", "target:0"])
    assert labels.tolist() == [1, 0] and mask.tolist() == [True, True]
    assert not result.pseudo_labels.requires_grad


def test_one_center_update_changes_an_edge_assignment():
    features = torch.tensor(
        [[-0.096, -0.995], [0.914, 0.406], [0.998, -0.070], [0.699, 0.715]]
    )
    prototypes = torch.tensor([[0.848, -0.530], [0.503, 0.864]])
    result = refresh(features, prototypes, trust_threshold=1.0)
    initial = torch.nn.functional.normalize(features) @ torch.nn.functional.normalize(prototypes).T
    assert initial.argmax(1).tolist() == [0, 1, 0, 1]
    assert result.pseudo_labels.tolist() == [0, 1, 1, 1]


def test_empty_and_cancelling_clusters_remain_finite(caplog):
    caplog.set_level("INFO")
    empty = refresh(torch.tensor([[1.0, 0.0]]), torch.eye(2))
    assert torch.isfinite(empty.confidence).all()
    assert "empty_clusters=1" in caplog.text

    caplog.clear()
    cancelling = refresh(
        torch.tensor([[1.0, 0.0], [-1.0, 0.0]]),
        torch.tensor([[1.0, 0.0]]),
    )
    assert torch.isfinite(cancelling.confidence).all()
    assert "near_zero_clusters=1" in caplog.text


def test_snapshot_round_trip_and_diagnostics():
    result = refresh(torch.eye(2), torch.eye(2), epoch=4)
    restored = TargetSupervision.from_state_dict(result.state_dict(), result.sample_ids)
    assert restored.epoch == 4
    assert torch.equal(restored.pseudo_labels, result.pseudo_labels)
    with pytest.raises(ValueError, match="sample_ids differ"):
        TargetSupervision.from_state_dict(result.state_dict(), list(reversed(result.sample_ids)))
    metrics = target_supervision_diagnostics(result, [0, 1])
    assert metrics == {"pseudo_label_accuracy": 1.0, "trusted_accuracy": None}


@pytest.mark.parametrize(
    ("features", "prototypes", "match"),
    [
        (torch.tensor([[float("nan"), 1.0]]), torch.eye(2), "finite"),
        (torch.tensor([[0.0, 0.0]]), torch.eye(2), "non-zero norm"),
        (torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 0.0], [0.0, 1.0]]), "non-zero norm"),
    ],
)
def test_refresh_rejects_invalid_vectors(features, prototypes, match):
    with pytest.raises(ValueError, match=match):
        refresh(features, prototypes)


def test_refresh_rejects_bad_scan_and_configuration():
    features = torch.eye(2)
    prototypes = torch.eye(2)
    loader = DataLoader(FeatureDataset(features, ["a", "b"]), batch_size=2)
    model = FeatureModel()
    banks = SimpleNamespace(target_prototypes=nn.Parameter(prototypes))
    with pytest.raises(ValueError, match="sample_ids differ"):
        refresh_target_supervision(model, banks, loader, ["b", "a"], epoch=0, num_classes=2)
    with pytest.raises(ValueError, match="positive integer"):
        refresh_target_supervision(
            model, banks, loader, ["a", "b"], epoch=0, num_classes=2, kmeans_iterations=0
        )
    with pytest.raises(ValueError, match="eps must be finite"):
        refresh_target_supervision(
            model, banks, loader, ["a", "b"], epoch=0, num_classes=2, eps=float("nan")
        )
    with pytest.raises(ValueError, match="trust_threshold must be finite"):
        refresh_target_supervision(
            model,
            banks,
            loader,
            ["a", "b"],
            epoch=0,
            num_classes=2,
            trust_threshold=float("nan"),
        )


def test_threshold_is_strict_and_empty_trusted_set_is_valid():
    result = refresh(torch.tensor([[1.0, 0.0]]), torch.eye(2), trust_threshold=0.8)
    threshold = float(result.confidence[0])
    equal = refresh(torch.tensor([[1.0, 0.0]]), torch.eye(2), trust_threshold=threshold)
    assert not equal.trusted_mask.any()
    metrics = target_supervision_diagnostics(equal, [0])
    assert metrics["trusted_accuracy"] is None


def test_refresh_preserves_parameters_gradients_and_model_mode():
    features = torch.eye(2)
    loader = DataLoader(FeatureDataset(features, ["a", "b"]), batch_size=2)
    model = ParameterizedFeatureModel().train()
    model.scale.grad = torch.tensor(3.0)
    target_prototypes = nn.Parameter(torch.eye(2))
    target_prototypes.grad = torch.full_like(target_prototypes, 4.0)
    model_before = model.scale.detach().clone()
    prototypes_before = target_prototypes.detach().clone()

    refresh_target_supervision(
        model,
        SimpleNamespace(target_prototypes=target_prototypes),
        loader,
        ["a", "b"],
        epoch=0,
        num_classes=2,
    )

    assert model.training
    assert torch.equal(model.scale, model_before)
    assert torch.equal(target_prototypes, prototypes_before)
    assert torch.equal(model.scale.grad, torch.tensor(3.0))
    assert torch.equal(target_prototypes.grad, torch.full_like(target_prototypes, 4.0))
