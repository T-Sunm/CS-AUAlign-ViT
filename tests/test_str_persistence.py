import random

import pytest
import torch
from str_helpers import TinyImageModel, save_snapshot
from test_str_selection import build, start

from cs_aualign_vit.data.manifests import read_jsonl
from cs_aualign_vit.data.manifests import write_jsonl as real_write_jsonl
from cs_aualign_vit.training import self_training_resampling as module


def test_crash_mid_round_resumes_without_duplicates_and_keeps_class(tmp_path, monkeypatch):
    resampler, model, banks = build(tmp_path, {0: 4, 1: 2}, [1, 1, 1])
    tx = start(resampler, model, banks, tmp_path)
    calls = []

    def flaky(path, rows):
        calls.append(1)
        if len(calls) == 2:
            raise OSError("crash")
        real_write_jsonl(path, rows)

    monkeypatch.setattr(module, "write_jsonl", flaky)
    with pytest.raises(OSError):
        resampler.run(tx, model)
    assert resampler.get_transaction(1)["status"] == "in_progress"
    assert len(read_jsonl(resampler.manifest_path)) == 1
    monkeypatch.setattr(module, "write_jsonl", real_write_jsonl)
    done = resampler.run(resampler.get_transaction(1), model)
    rows = read_jsonl(resampler.manifest_path)
    assert done["selected_class"] == 1 and done["status"] == "completed"
    assert len(rows) == len({r["sample_id"] for r in rows}) == 3
    assert len({r["target_id"] for r in rows}) == 3
    assert done["created"] == 3



@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_completed_str_resume_rejects_invalid_png(tmp_path, damage):
    resampler, model, banks = build(tmp_path, {0: 4, 1: 2}, [1, 1])
    resampler.run(start(resampler, model, banks, tmp_path), model)
    image = next(resampler.images_dir.glob("*.png"))
    image.unlink() if damage == "missing" else image.write_bytes(b"invalid png")

    with pytest.raises(ValueError, match="missing or corrupt"):
        resampler.rebuild_source_loader(resampler.get_transaction(1))


def test_completed_str_resume_rejects_missing_manifest(tmp_path):
    resampler, model, banks = build(tmp_path, {0: 4, 1: 2}, [1])
    resampler.run(start(resampler, model, banks, tmp_path), model)
    resampler.manifest_path.unlink()

    with pytest.raises(ValueError, match="class counts"):
        resampler.rebuild_source_loader(resampler.get_transaction(1))


@pytest.mark.parametrize(
    ("key", "changed"),
    [
        ("global_seed", 11),
        ("confidence_threshold", 0.9),
        ("num_classes", 3),
    ],
)
def test_str_resume_rejects_changed_config(tmp_path, key, changed):
    resampler, model, banks = build(tmp_path, {0: 4, 1: 2}, [1, 1])
    transaction = start(resampler, model, banks, tmp_path)
    config = {
        "global_seed": resampler.global_seed,
        "confidence_threshold": resampler.confidence_threshold,
        "num_classes": resampler.num_classes,
    }
    config[key] = changed
    resumed = module.SelfTrainingResampler(
        source_manifest=resampler.source_manifest,
        target_feature_loader=resampler.target_feature_loader,
        prototype_banks=banks,
        output_dir=resampler.output_dir,
        batch_size=resampler.batch_size,
        device="cpu",
        **config,
    )

    with pytest.raises(ValueError, match=key):
        resumed.get_transaction(1)
    with pytest.raises(ValueError, match=key):
        resumed.run(transaction, model)


def test_orphan_image_is_overwritten(tmp_path):
    resampler, model, banks = build(tmp_path, {0: 4, 1: 2}, [1])
    tx = start(resampler, model, banks, tmp_path)
    orphan = resampler.images_dir / f"{module.str_sample_id(1, 'expw_000')}.png"
    resampler.images_dir.mkdir(parents=True)
    orphan.write_bytes(b"garbage")
    resampler.run(tx, model)
    assert resampler.rebuild_source_loader(resampler.get_transaction(1)) is not None


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_missing_or_corrupt_recorded_image_fails(tmp_path, damage):
    resampler, model, banks = build(tmp_path, {0: 4, 1: 2}, [1, 1])
    resampler.run(start(resampler, model, banks, tmp_path), model)
    image = next(resampler.images_dir.glob("*.png"))
    image.unlink() if damage == "missing" else image.write_bytes(b"nope")
    tx2 = start(resampler, model, banks, tmp_path, round_id=2, epoch=50)
    with pytest.raises(ValueError, match="missing or corrupt"):
        resampler.run(tx2, model)


def test_uses_snapshot_not_live_model_and_leaves_live_state_untouched(tmp_path):
    resampler, model, banks = build(tmp_path, {0: 4, 1: 2}, [1, 1])
    tx = start(resampler, model, banks, tmp_path)  # snapshot predicts green -> class 1
    with torch.no_grad():
        model.projection.weight.mul_(-1)  # live model would predict the other class
        banks.target_prototypes.mul_(-1)
    model.train()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    py_state, torch_state = random.getstate(), torch.get_rng_state()
    done = resampler.run(tx, model)
    assert done["created"] == 2
    assert model.training
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    assert random.getstate() == py_state and torch.equal(torch.get_rng_state(), torch_state)


def test_snapshot_epoch_must_match(tmp_path):
    resampler, model, banks = build(tmp_path, {0: 4, 1: 2}, [1])
    tx = start(resampler, model, banks, tmp_path)
    save_snapshot(tx["checkpoint_path"], TinyImageModel(), banks, 24)
    with pytest.raises(ValueError, match="completed_epoch"):
        resampler.run(tx, model)


def test_rebuild_rejects_records_from_later_round(tmp_path):
    resampler, model, banks = build(tmp_path, {0: 4, 1: 2}, [1])
    resampler.run(start(resampler, model, banks, tmp_path), model)
    with pytest.raises(ValueError, match="later round"):
        resampler.rebuild_source_loader({"round_id": 0})
    loader = resampler.rebuild_source_loader(resampler.get_transaction(1))
    assert loader.drop_last and len(loader.dataset) == 7


def test_empty_str_manifest_does_not_break_loader(tmp_path):
    resampler, *_ = build(tmp_path, {0: 4, 1: 2}, [1])
    loader = resampler.rebuild_source_loader({"round_id": 1})
    assert len(loader.dataset) == 6
    resampler.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    resampler.manifest_path.write_text("")
    assert len(resampler.rebuild_source_loader({"round_id": 1}).dataset) == 6
