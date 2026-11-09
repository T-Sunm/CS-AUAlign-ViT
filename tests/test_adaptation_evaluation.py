import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cs_aualign_vit.models import PrototypeBanks
from cs_aualign_vit.training import evaluate_target, metrics_from_confusion


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
    assert result["balanced_accuracy"] == pytest.approx(0.75)
    assert result["macro_f1"] == pytest.approx(2 / 3)
    assert result["weighted_f1"] == pytest.approx(2 / 3)
    assert result["per_class_precision"] == [0.5, 1.0]
    assert result["per_class_recall"] == [1.0, 0.5]
    assert result["per_class_f1"] == pytest.approx([2 / 3, 2 / 3])
    assert result["confusion_matrix"] == [[1, 0], [1, 1]]


def test_metrics_reproduce_e0_and_handle_zero_denominators():
    confusion = [
        [145, 45, 1, 59, 55, 24, 13],
        [11, 47, 0, 64, 148, 51, 17],
        [27, 2, 10, 18, 10, 8, 38],
        [5, 5, 3, 2451, 166, 37, 19],
        [16, 154, 0, 720, 1835, 190, 113],
        [35, 67, 4, 186, 255, 372, 17],
        [62, 15, 30, 111, 101, 27, 273],
    ]
    result = metrics_from_confusion(confusion)
    assert result["accuracy"] == pytest.approx(0.6366906474820144)
    assert result["balanced_accuracy"] == pytest.approx(0.4298, abs=1e-4)
    assert result["macro_f1"] == pytest.approx(0.4420, abs=1e-4)
    assert result["weighted_f1"] == pytest.approx(0.6227, abs=1e-4)

    zero_class = metrics_from_confusion([[1, 0], [0, 0]])
    assert zero_class["per_class_precision"][1] == 0
    assert zero_class["per_class_recall"][1] == 0
    assert zero_class["per_class_f1"][1] == 0
