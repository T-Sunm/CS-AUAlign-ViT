import pytest
import torch

from cs_aualign_vit.data.manifests import CLASSES
from cs_aualign_vit.models import PrototypeBanks
from cs_aualign_vit.training.source_pretraining import CHECKPOINT_VERSION


def source_checkpoint(tmp_path, **changes):
    checkpoint = {
        "schema_version": CHECKPOINT_VERSION,
        "class_mapping": list(CLASSES),
        "feature_dim": 4,
        "source_prototypes": torch.arange(28, dtype=torch.float32).reshape(7, 4),
    }
    checkpoint.update(changes)
    path = tmp_path / "source.pt"
    torch.save(checkpoint, path)
    return path


def test_fresh_initialization_validates_checkpoint_and_copies_independent_banks(tmp_path):
    path = source_checkpoint(tmp_path)
    banks = PrototypeBanks.from_source_checkpoint(path, feature_dim=4)
    assert banks.source_prototypes.shape == banks.target_prototypes.shape == (7, 4)
    assert torch.equal(banks.source_prototypes, banks.target_prototypes)
    assert banks.source_prototypes.data_ptr() != banks.target_prototypes.data_ptr()

    for changes in (
        {"class_mapping": None},
        {"class_mapping": list(reversed(CLASSES))},
        {"feature_dim": None},
        {"feature_dim": 5},
        {"source_prototypes": torch.empty(6, 4)},
        {"source_prototypes": None},
    ):
        with pytest.raises(ValueError):
            PrototypeBanks.from_source_checkpoint(source_checkpoint(tmp_path, **changes), feature_dim=4)


def test_logits_and_gradients_are_routed_to_the_selected_bank(tmp_path):
    banks = PrototypeBanks.from_source_checkpoint(source_checkpoint(tmp_path), feature_dim=4)
    features = torch.randn(3, 4)
    parameters = list(banks.parameters())
    assert len(parameters) == 2
    optimizer = torch.optim.SGD(banks.parameters(), lr=0.01)
    assert {id(p) for group in optimizer.param_groups for p in group["params"]} == {
        id(banks.source_prototypes), id(banks.target_prototypes)
    }

    source_logits = banks.source_logits(features)
    assert source_logits.shape == (3, 7)
    assert torch.equal(source_logits, features @ banks.source_prototypes.T)
    torch.nn.functional.cross_entropy(source_logits, torch.tensor([0, 1, 2])).backward()
    assert banks.source_prototypes.grad is not None
    assert banks.target_prototypes.grad is None

    optimizer.zero_grad(set_to_none=True)
    target_logits = banks.target_logits(features)
    assert target_logits.shape == (3, 7)
    assert torch.equal(target_logits, features @ banks.target_prototypes.T)
    torch.nn.functional.cross_entropy(target_logits, torch.tensor([0, 1, 2])).backward()
    assert banks.source_prototypes.grad is None
    assert banks.target_prototypes.grad is not None


def test_separate_banks_restore_without_copying_source_to_target(tmp_path):
    banks = PrototypeBanks.from_source_checkpoint(source_checkpoint(tmp_path), feature_dim=4)
    with torch.no_grad():
        banks.target_prototypes.add_(1)
    saved_state = {name: value.clone() for name, value in banks.state_dict().items()}

    resumed = PrototypeBanks(saved_state["source_prototypes"], saved_state["target_prototypes"])
    assert torch.equal(resumed.source_prototypes, saved_state["source_prototypes"])
    assert torch.equal(resumed.target_prototypes, saved_state["target_prototypes"])
    assert not torch.equal(resumed.source_prototypes, resumed.target_prototypes)
    assert resumed.source_prototypes.data_ptr() != resumed.target_prototypes.data_ptr()
