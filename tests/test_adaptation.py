from dataclasses import replace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from cs_aualign_vit.models import AdversarialAlignment, BCSFeatures, PrototypeBanks
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

    def rebuild_source_loader(self, _transaction, *, source_transform=None):
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
    adversary = AdversarialAlignment(input_dim=2, hidden_dim=4, dropout=0.0)
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
    assert all(p.grad is not None for p in trainer.adversarial_module.parameters())
    for key in ("acc_s", "acc_t", "acc_balanced"):
        assert key in metrics


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


def test_adversarial_state_round_trips_and_optimizer_holds_discriminator(tmp_path):
    trainer = build_trainer(tmp_path)
    discriminator = {id(p) for p in trainer.adversarial_module.parameters()}
    assert discriminator <= {id(p) for g in trainer.optimizer.param_groups for p in g["params"]}
    trainer.run()
    resumed = build_trainer(tmp_path)
    resumed.load_checkpoint()
    for key, value in trainer.adversarial_module.state_dict().items():
        assert torch.equal(resumed.adversarial_module.state_dict()[key], value)


def test_adversarial_config_mismatch_fails_load(tmp_path):
    trainer = build_trainer(tmp_path)
    trainer.supervision = make_supervision(trainer.target_sample_ids)
    trainer.save_checkpoint()
    other = build_trainer(tmp_path)
    other.adversarial_module.hidden_dim = 8
    with pytest.raises(ValueError, match="adversarial configuration"):
        other.load_checkpoint()


# ---- real STR resampler on a small image fixture -------------------------------------
def build_real(tmp_path, *, config, target_colors=(1, 1, 1, 1), source_image_size=224):
    from str_helpers import TinyImageModel, make_banks, make_source, make_target

    from cs_aualign_vit.data.datasets import FaceDataset, build_loader, build_source_loader
    from cs_aualign_vit.models import AdversarialAlignment
    from cs_aualign_vit.training.self_training_resampling import SelfTrainingResampler

    source = make_source(tmp_path / "src", {0: 4, 1: 2})
    target = make_target(tmp_path / "tgt", list(target_colors))
    model, banks = TinyImageModel(), make_banks()
    adversary = AdversarialAlignment(input_dim=2, hidden_dim=4, dropout=0.0)
    optimizer = torch.optim.SGD(
        [*model.parameters(), *banks.parameters(), *adversary.parameters()], lr=0.001
    )
    feature_loader = build_loader(target, "target_features", batch_size=2)
    resampler = SelfTrainingResampler(
        source_manifest=source,
        target_feature_loader=feature_loader,
        prototype_banks=banks,
        output_dir=tmp_path / "run" / "str",
        global_seed=3,
        batch_size=2,
        device="cpu",
        num_classes=2,
    )
    trainer = AdaptationTrainer(
        model=model,
        prototype_banks=banks,
        adversarial_module=adversary,
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epochs),
        source_loader=build_source_loader(
            [source], batch_size=2, drop_last=True, seed=3, image_size=source_image_size
        ),
        target_loader=build_loader(target, "target_adaptation", batch_size=2, drop_last=True),
        target_feature_loader=feature_loader,
        target_sample_ids=[s["sample_id"] for s in FaceDataset(target, "target_features").samples],
        device="cpu",
        checkpoint_path=tmp_path / "run" / "adaptation.pt",
        config=config,
        str_service=resampler,
    )
    return trainer, resampler, source


def test_ten_boundaries_over_full_schedule_add_images_and_keep_isolation(tmp_path):
    config = AdaptationConfig(kmeans_interval=10, num_classes=2)
    trainer, resampler, source = build_real(tmp_path, config=config)
    source_bytes = source.read_bytes()
    history = trainer.run()

    assert len(history) == 256 and trainer.completed_epoch == 256
    transactions = [resampler.get_transaction(r) for r in range(1, 11)]
    assert all(t["status"] == "completed" for t in transactions)
    assert transactions[0]["created"] == 4 and all(t["created"] == 0 for t in transactions[1:])
    assert len(trainer.source_loader.dataset) == 6 + 4
    assert source.read_bytes() == source_bytes
    from cs_aualign_vit.data.manifests import read_jsonl

    rows = read_jsonl(resampler.manifest_path)
    assert {r["dataset"] for r in rows} == {"raf_db"} and len({r["target_id"] for r in rows}) == 4
    assert all("label" not in s for s in trainer.target_loader.dataset.samples)


def test_empty_round_does_not_stop_training(tmp_path):
    config = AdaptationConfig(max_epochs=3, kmeans_interval=1, str_epochs=(1,), num_classes=2)
    trainer, resampler, _ = build_real(tmp_path, config=config, target_colors=(0, 0, 0, 0))
    trainer.run()
    assert resampler.get_transaction(1)["created"] == 0
    assert trainer.completed_epoch == 3 and len(trainer.source_loader.dataset) == 6


def test_resume_after_completed_round_restores_str_images(tmp_path):
    config = AdaptationConfig(max_epochs=3, kmeans_interval=1, str_epochs=(1,), num_classes=2)
    trainer, *_ = build_real(tmp_path, config=config)
    trainer.run()
    assert len(trainer.source_loader.dataset) == 10

    resumed, resampler, _ = build_real(tmp_path, config=config)
    resumed.load_checkpoint()
    assert len(resumed.source_loader.dataset) == 6  # fresh loader, before restore
    assert resumed.run() == []
    assert len(resumed.source_loader.dataset) == 10
    assert resampler.get_transaction(1)["status"] == "completed"


def test_str_rebuild_keeps_source_training_transform(tmp_path):
    config = AdaptationConfig(max_epochs=2, kmeans_interval=1, str_epochs=(1,), num_classes=2)
    trainer, _, _ = build_real(tmp_path, config=config, source_image_size=32)
    original_transform = trainer.source_loader.dataset.datasets[0].transform

    trainer.run()

    assert all(
        dataset.transform is original_transform for dataset in trainer.source_loader.dataset.datasets
    )
    assert trainer.source_loader.dataset[0][0].shape == (3, 32, 32)

    resumed, _, _ = build_real(tmp_path, config=config, source_image_size=32)
    resumed.load_checkpoint()
    resumed.run()
    assert resumed.source_loader.dataset[0][0].shape == (3, 32, 32)


def test_snapshot_is_written_before_main_checkpoint(tmp_path, monkeypatch):
    config = AdaptationConfig(max_epochs=2, kmeans_interval=1, str_epochs=(2,), num_classes=2)
    trainer, *_ = build_real(tmp_path, config=config)
    original = trainer._atomic_save

    def crash_on_main(state, path):
        if state["completed_epoch"] == 2 and path == trainer.checkpoint_path:
            raise OSError("crash")
        original(state, path)

    monkeypatch.setattr(trainer, "_atomic_save", crash_on_main)
    with pytest.raises(OSError):
        trainer.run()
    assert trainer._boundary_checkpoint_path().exists()
    assert torch.load(trainer.checkpoint_path, weights_only=True)["completed_epoch"] == 1
