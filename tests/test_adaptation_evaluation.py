import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cs_aualign_vit.models import PrototypeBanks
from cs_aualign_vit.training import evaluate_target


class IdentityTargetModel(nn.Module):
    def forward_target_features(self, inputs):
        return inputs


def test_target_evaluation_reports_confusion_and_per_class_accuracy():
    loader = DataLoader(
        TensorDataset(
            torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, 0.0]]),
            torch.tensor([0, 1, 1]),
        ),
        batch_size=2,
    )

    result = evaluate_target(
        IdentityTargetModel(), PrototypeBanks(torch.eye(2)), loader, "cpu", num_classes=2
    )

    assert result["samples"] == 3
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["per_class_accuracy"] == [1.0, 0.5]
    assert result["confusion_matrix"] == [[1, 0], [1, 1]]
