from dataclasses import replace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

import cs_aualign_vit.training.adaptation as adaptation_module
from cs_aualign_vit.data.au_guidance import compute_guidance
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

    def forward_features(self, source, target, guidance=None):
        self.last_guidance = guidance
        source = self.projection(source)
        target = self.projection(target)
        return BCSFeatures(source, target, source + target, source - target)

    def forward_target_features(self, target):
        return self.projection(target)


class AlignedTinyBCS(TinyBCS):
    def forward_features(self, source, target, guidance=None):
        features = super().forward_features(source, target, guidance)
        return BCSFeatures(
            features.source_features,
            features.target_features,
            features.target_to_source_features,
            features.source_to_target_features,
            features.source_features[:, None].expand(-1, 49, -1),
            features.target_features[:, None].expand(-1, 49, -1),
        )


class GuidedDataset(Dataset):
    def __init__(self, *, target=False):
        self.target = target
        self.sample_ids = [f"target:{index}" for index in range(2)]

    def __len__(self):
        return 2

    def __getitem__(self, index):
        metadata = {
            "au": torch.full((12,), 0.25 + 0.5 * index),
            "mask": torch.ones(49, 12),
            "au_valid": True,
            "landmark_valid": True,
        }
        identity = self.sample_ids[index] if self.target else 0
        return torch.tensor([1.0 + float(self.target), float(index)]), identity, metadata


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


def test_resume_keeps_target_supervision_on_cpu_for_cuda_training(tmp_path):
    trainer = build_trainer(tmp_path)
    trainer.supervision = make_supervision(trainer.target_sample_ids)
    trainer.save_checkpoint()

    resumed = build_trainer(tmp_path)
    resumed.device = torch.device("cuda:0")
    resumed.load_checkpoint()

    assert resumed.supervision.pseudo_labels.device.type == "cpu"
    assert resumed.supervision.trusted_mask.device.type == "cpu"


def test_resume_restores_training_loader_shuffle_order(tmp_path):
    trainer = build_trainer(tmp_path)
    source = TensorDataset(torch.arange(8), torch.arange(8))
    target = TargetDataset(torch.arange(8))
    trainer.source_loader = DataLoader(
        source,
        batch_size=2,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(11),
    )
    trainer.target_loader = DataLoader(
        target,
        batch_size=2,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(22),
    )
    list(trainer.source_loader)
    list(trainer.target_loader)
    trainer.supervision = make_supervision(trainer.target_sample_ids)
    trainer.save_checkpoint()
    expected_source = [labels.tolist() for _, labels in trainer.source_loader]
    expected_target = [list(sample_ids) for _, sample_ids in trainer.target_loader]

    resumed = build_trainer(tmp_path)
    resumed.source_loader = DataLoader(
        source,
        batch_size=2,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(11),
    )
    resumed.target_loader = DataLoader(
        target,
        batch_size=2,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(22),
    )
    resumed.load_checkpoint()

    assert [labels.tolist() for _, labels in resumed.source_loader] == expected_source
    assert [list(sample_ids) for _, sample_ids in resumed.target_loader] == expected_target


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


def test_dynamic_shuffled_builds_guidance_from_paired_metadata(tmp_path):
    source = GuidedDataset()
    target = GuidedDataset(target=True)
    source_loader = DataLoader(source, batch_size=2, drop_last=True)
    target_loader = DataLoader(target, batch_size=2, drop_last=True)
    model = TinyBCS()
    banks = PrototypeBanks(torch.eye(2))
    adversary = AdversarialAlignment(input_dim=2, hidden_dim=4, dropout=0.0)
    optimizer = torch.optim.SGD(
        [*model.parameters(), *banks.parameters(), *adversary.parameters()], lr=0.01
    )
    source_cache, target_cache = tmp_path / "source.csv", tmp_path / "target.csv"
    source_cache.write_text("source")
    target_cache.write_text("target")
    trainer = AdaptationTrainer(
        model=model,
        prototype_banks=banks,
        adversarial_module=adversary,
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1),
        source_loader=source_loader,
        target_loader=target_loader,
        target_feature_loader=DataLoader(TargetDataset(torch.ones(2, 2)), batch_size=2),
        target_sample_ids=target.sample_ids,
        device="cpu",
        checkpoint_path=tmp_path / "guided.pt",
        config=AdaptationConfig(
            max_epochs=1,
            kmeans_interval=1,
            str_epochs=(),
            num_classes=2,
            au_mode="dynamic_shuffled",
        ),
        source_au_cache=source_cache,
        target_au_cache=target_cache,
    )
    trainer.supervision = make_supervision(target.sample_ids)

    metrics = trainer.train_step(next(iter(source_loader)), next(iter(target_loader)))

    assert model.last_guidance.shape == (2, 49, 49)
    assert metrics["au_nonzero_pair_ratio"] == 1
    assert metrics["au_shuffled_ratio"] in {0, 0.5, 1}


def build_aligned_trainer(tmp_path, *, eta=0.1, mode="normal", au_mode="off"):
    source = GuidedDataset()
    target = GuidedDataset(target=True)
    source_loader = DataLoader(source, batch_size=2, drop_last=True)
    target_loader = DataLoader(target, batch_size=2, drop_last=True)
    model = AlignedTinyBCS()
    banks = PrototypeBanks(torch.eye(2))
    adversary = AdversarialAlignment(input_dim=2, hidden_dim=4, dropout=0.0)
    optimizer = torch.optim.SGD(
        [*model.parameters(), *banks.parameters(), *adversary.parameters()], lr=0.01
    )
    source_cache, target_cache = tmp_path / "source.csv", tmp_path / "target.csv"
    source_cache.write_text("source")
    target_cache.write_text("target")
    return AdaptationTrainer(
        model=model,
        prototype_banks=banks,
        adversarial_module=adversary,
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1),
        source_loader=source_loader,
        target_loader=target_loader,
        target_feature_loader=DataLoader(TargetDataset(torch.ones(2, 2)), batch_size=2),
        target_sample_ids=target.sample_ids,
        device="cpu",
        checkpoint_path=tmp_path / "aligned.pt",
        config=AdaptationConfig(
            max_epochs=1,
            kmeans_interval=1,
            str_epochs=(),
            num_classes=2,
            au_mode=au_mode,
            au_alignment_enabled=True,
            au_alignment_mode=mode,
            au_alignment_tau_mask=1.0,
            au_alignment_n_eff_min=1.0,
            au_alignment_q_c=(1.0,) * 12,
            au_alignment_eta=eta,
        ),
        source_au_cache=source_cache,
        target_au_cache=target_cache,
    )


def test_alignment_smoke_steps_checkpoint_and_resume(tmp_path):
    trainer = build_aligned_trainer(tmp_path)
    trainer.supervision = make_supervision(trainer.target_sample_ids)
    source_batch = next(iter(trainer.source_loader))
    target_batch = next(iter(trainer.target_loader))

    rows = [trainer.train_step(source_batch, target_batch) for _ in range(2)]
    trainer.save_checkpoint()

    resumed = build_aligned_trainer(tmp_path)
    resumed.load_checkpoint()
    assert resumed.global_step == 2
    rows.append(resumed.train_step(source_batch, target_batch))

    for metrics in rows:
        assert all(
            torch.isfinite(torch.tensor(value))
            for value in metrics.values()
            if isinstance(value, (int, float))
        )
        assert metrics["loss_local"] >= 0
        assert metrics["loss_local_weighted"] == pytest.approx(0.1 * metrics["loss_local"])
        assert metrics["au_alignment_valid_count"] == 12
        assert metrics["au_alignment_has_loss"] == 1
        assert all(metrics[f"au_alignment_source_n_eff_{index}"] >= 1 for index in range(12))
        assert all(metrics[f"au_alignment_target_n_eff_{index}"] >= 1 for index in range(12))
    assert resumed.model.last_guidance is None
    assert resumed.global_step == 3
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for module in (resumed.model, resumed.prototype_banks, resumed.adversarial_module)
        for parameter in module.parameters()
    )

    epoch_path = tmp_path / "epoch"
    epoch_path.mkdir()
    epoch_trainer = build_aligned_trainer(epoch_path)
    epoch_trainer.supervision = make_supervision(epoch_trainer.target_sample_ids)
    epoch_metrics = epoch_trainer.train_epoch()
    assert sum(
        epoch_metrics[f"au_alignment_contribution_{index}"] for index in range(12)
    ) == pytest.approx(1.0)

    mismatched = build_aligned_trainer(tmp_path, eta=0.2)
    with pytest.raises(ValueError, match="configuration differs"):
        mismatched.load_checkpoint()

    changed_cache = build_aligned_trainer(tmp_path)
    (tmp_path / "source.csv").write_text("changed")
    with pytest.raises(ValueError, match="AU caches differ"):
        changed_cache.load_checkpoint()


def test_alignment_shuffle_rng_round_trips_checkpoint(tmp_path):
    trainer = build_aligned_trainer(tmp_path, mode="shuffled")
    trainer.supervision = make_supervision(trainer.target_sample_ids)
    trainer.train_step(next(iter(trainer.source_loader)), next(iter(trainer.target_loader)))
    trainer.save_checkpoint()
    expected = torch.randperm(16, generator=trainer._alignment_shuffle_generator)

    resumed = build_aligned_trainer(tmp_path, mode="shuffled")
    resumed.load_checkpoint()
    actual = torch.randperm(16, generator=resumed._alignment_shuffle_generator)

    assert torch.equal(actual, expected)


def test_alignment_shuffle_does_not_change_attention_metadata(tmp_path, monkeypatch):
    observed = {}
    real_alignment = adaptation_module.compute_au_alignment

    def reverse_valid(values, valid, *, generator=None):
        indices = torch.nonzero(valid, as_tuple=False).flatten()
        shuffled = values.clone()
        shuffled[indices] = values[indices.flip(0)]
        return shuffled, len(indices)

    def capture_alignment(source_tokens, target_tokens, source_au, target_au, **options):
        observed["source"] = source_au["au"].clone()
        observed["target"] = target_au["au"].clone()
        return real_alignment(
            source_tokens, target_tokens, source_au, target_au, **options
        )

    monkeypatch.setattr(adaptation_module, "shuffle_valid_au", reverse_valid)
    monkeypatch.setattr(adaptation_module, "compute_au_alignment", capture_alignment)
    trainer = build_aligned_trainer(tmp_path, mode="shuffled", au_mode="dynamic")
    trainer.supervision = make_supervision(trainer.target_sample_ids)
    source_batch = next(iter(trainer.source_loader))
    target_batch = next(iter(trainer.target_loader))
    source_meta, target_meta = source_batch[2], target_batch[2]
    valid_pairs = torch.ones(len(source_batch[0]), dtype=torch.bool)
    expected_guidance = compute_guidance(
        target_meta["au"],
        source_meta["au"],
        target_meta["mask"],
        source_meta["mask"],
        valid_pairs,
    )

    trainer.train_step(source_batch, target_batch)

    assert torch.equal(trainer.model.last_guidance, expected_guidance)
    assert torch.equal(observed["source"], source_meta["au"].flip(0))
    assert torch.equal(observed["target"], target_meta["au"].flip(0))


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
    expected_generator_state = trainer.source_loader.generator.get_state()

    resumed, resampler, _ = build_real(tmp_path, config=config)
    resumed.load_checkpoint()
    assert len(resumed.source_loader.dataset) == 6  # fresh loader, before restore
    assert resumed.run() == []
    assert len(resumed.source_loader.dataset) == 10
    assert torch.equal(resumed.source_loader.generator.get_state(), expected_generator_state)
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

    resumed, resampler, _ = build_real(tmp_path, config=config)
    resumed.load_checkpoint()
    assert resumed.completed_epoch == 2
    assert resumed.boundary_completed is False
    assert resumed.run() == []
    assert resampler.get_transaction(1)["status"] == "completed"
