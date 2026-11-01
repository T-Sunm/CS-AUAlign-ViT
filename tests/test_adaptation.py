from dataclasses import replace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from cs_aualign_vit.models import BCSFeatures, PrototypeBanks
from cs_aualign_vit.training import AdaptationConfig, AdaptationTrainer, TargetSupervision


class TargetDataset(Dataset):
    def __init__(self, values):
        self.values = values
        self.sample_ids = [f"target:{index}" for index in range(len(values))]

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index], self.sample_ids[index]


class TinyBCS(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(2, 2, bias=False)

    def forward_features(self, source, target):
        source = self.projection(source)
        target = self.projection(target)
        return BCSFeatures(source, target, source + target, source - target)

    def forward_target_features(self, target):
        return self.projection(target)


class TinyAdversary(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, source, target):
        return self.scale * (source.mean() - target.mean()).square()


class FakeSTR:
    def __init__(self, source_loader, status=None):
        self.source_loader = source_loader
        self.transaction = None if status is None else {"status": status}
        self.created = self.runs = self.rebuilds = 0
        self.checkpoint_path = None

    def get_transaction(self, _round_id):
        return self.transaction

    def create_transaction(self, **values):
        self.created += 1
        self.checkpoint_path = values["checkpoint_path"]
        self.transaction = {"status": "in_progress", **values}
        return self.transaction

    def run(self, transaction, _model):
        self.runs += 1
        transaction["status"] = "completed"
        return transaction

    def rebuild_source_loader(self, _transaction):
        self.rebuilds += 1
        return self.source_loader


def make_supervision(sample_ids, epoch=0, trusted=True):
    size = len(sample_ids)
    return TargetSupervision(
        pseudo_labels=torch.zeros(size, dtype=torch.long),
        predicted_labels=torch.zeros(size, dtype=torch.long),
        confidence=torch.ones(size),
        trusted_mask=torch.full((size,), trusted, dtype=torch.bool),
        sample_ids=list(sample_ids),
        epoch=epoch,
    )


def build_trainer(tmp_path, *, config=None, str_service=None, refresh_fn=None):
    source = TensorDataset(torch.tensor([[2.0, 0.0], [1.0, 0.0]]), torch.tensor([0, 0]))
    target = TargetDataset(torch.tensor([[1.0, 0.0], [0.5, 0.0]]))
    source_loader = DataLoader(source, batch_size=1, shuffle=False, drop_last=True)
    target_loader = DataLoader(target, batch_size=1, shuffle=False, drop_last=True)
    feature_loader = DataLoader(target, batch_size=2, shuffle=False)
    model = TinyBCS()
    banks = PrototypeBanks(torch.eye(2))
    adversary = TinyAdversary()
    optimizer = torch.optim.SGD(
        [*model.parameters(), *banks.parameters(), *adversary.parameters()], lr=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    config = config or AdaptationConfig(max_epochs=2, kmeans_interval=1, str_epochs=(), num_classes=2)
    if refresh_fn is None:
        refresh_fn = lambda *_args, epoch, **_kwargs: make_supervision(target.sample_ids, epoch)
    trainer = AdaptationTrainer(
        model=model,
        prototype_banks=banks,
        adversarial_module=adversary,
        optimizer=optimizer,
        scheduler=scheduler,
        source_loader=source_loader,
        target_loader=target_loader,
        target_feature_loader=feature_loader,
        target_sample_ids=target.sample_ids,
        device="cpu",
        checkpoint_path=tmp_path / "adaptation.pt",
        config=config,
        str_service=str_service,
        refresh_fn=refresh_fn,
    )
    return trainer


def test_training_run_uses_target_epoch_refreshes_and_round_trips_checkpoint(tmp_path):
    refresh_epochs = []

    def refresh(*_args, epoch, **_kwargs):
        refresh_epochs.append(epoch)
        return make_supervision(["target:0", "target:1"], epoch, trusted=False)

    trainer = build_trainer(tmp_path, refresh_fn=refresh)
    initial = trainer.model.projection.weight.detach().clone()
    history = trainer.run()

    assert refresh_epochs == [0, 1]
    assert [row["steps"] for row in history] == [2, 2]
    assert trainer.global_step == 4
    assert not torch.equal(trainer.model.projection.weight, initial)
    assert trainer.checkpoint_path.exists()

    resumed = build_trainer(tmp_path, refresh_fn=refresh)
    resumed.load_checkpoint()
    assert resumed.completed_epoch == 2 and resumed.global_step == 4
    assert torch.equal(resumed.prototype_banks.target_prototypes, trainer.prototype_banks.target_prototypes)
    assert resumed.run() == []


def test_loss_is_assembled_once_with_baseline_weights(tmp_path):
    trainer = build_trainer(tmp_path)
    trainer.supervision = make_supervision(trainer.target_sample_ids)
    metrics = trainer.train_step(next(iter(trainer.source_loader)), next(iter(trainer.target_loader)))
    expected = metrics["loss_s"] + metrics["loss_t"] + metrics["loss_t2s"]
    expected += 0.5 * metrics["loss_s2t"] + 0.1 * metrics["loss_ad"]
    assert metrics["loss_total"] == pytest.approx(expected)
    assert trainer.model.projection.weight.grad is not None
    assert trainer.prototype_banks.source_prototypes.grad is not None
    assert trainer.prototype_banks.target_prototypes.grad is not None
    assert trainer.adversarial_module.scale.grad is not None


def test_empty_paired_loader_fails_with_actionable_context(tmp_path):
    trainer = build_trainer(tmp_path)
    empty = DataLoader(TensorDataset(torch.empty(0, 2), torch.empty(0, dtype=torch.long)))
    with pytest.raises(ValueError, match="source.*samples=0.*batch_size=1.*drop_last=False"):
        trainer._validate_loader(empty, "source")


def test_paired_loader_and_optimizer_configuration_are_validated(tmp_path):
    trainer = build_trainer(tmp_path)
    not_dropped = DataLoader(trainer.source_loader.dataset, batch_size=1, drop_last=False)
    with pytest.raises(ValueError, match="source.*drop_last=True"):
        trainer._validate_loader(not_dropped, "source")

    trainer.optimizer.param_groups[0]["params"].append(next(trainer.model.parameters()))
    with pytest.raises(ValueError, match="more than once"):
        trainer._validate_optimizer()


def test_str_absent_round_runs_once_and_completed_round_only_finalizes(tmp_path):
    config = AdaptationConfig(max_epochs=1, kmeans_interval=1, str_epochs=(1,), num_classes=2)
    seed = build_trainer(tmp_path, config=replace(config, str_epochs=()))
    service = FakeSTR(seed.source_loader)
    trainer = build_trainer(tmp_path, config=config, str_service=service)
    trainer.run()

    assert (service.created, service.runs, service.rebuilds) == (1, 1, 1)
    assert service.checkpoint_path.endswith("adaptation.epoch-1.boundary.pt")
    assert trainer.boundary_completed

    completed = FakeSTR(seed.source_loader, status="completed")
    pending = build_trainer(tmp_path, config=config, str_service=completed)
    pending.supervision = make_supervision(pending.target_sample_ids, epoch=1)
    pending.completed_epoch = 1
    pending.boundary_completed = False
    pending._finish_pending_boundary()
    assert (completed.created, completed.runs, completed.rebuilds) == (0, 0, 1)


@pytest.mark.parametrize("status", [None, "in_progress"])
def test_completed_boundary_rejects_inconsistent_live_str_state(tmp_path, status):
    config = AdaptationConfig(max_epochs=1, kmeans_interval=1, str_epochs=(1,), num_classes=2)
    source = build_trainer(tmp_path, config=replace(config, str_epochs=())).source_loader
    trainer = build_trainer(tmp_path, config=config, str_service=FakeSTR(source, status=status))
    trainer.supervision = make_supervision(trainer.target_sample_ids, epoch=1)
    trainer.completed_epoch = 1
    trainer.boundary_completed = True

    with pytest.raises(
        RuntimeError,
        match=f"epoch=1, expected_round_id=1, boundary=completed, transaction={'absent' if status is None else status}",
    ):
        trainer._finish_pending_boundary()
