
import pytest
from str_helpers import (
    TinyImageModel,
    feature_loader,
    make_banks,
    make_source,
    make_target,
    save_snapshot,
)
from torch.utils.data import DataLoader

from cs_aualign_vit.data.manifests import read_jsonl
from cs_aualign_vit.training import self_training_resampling as module
from cs_aualign_vit.training.self_training_resampling import SelfTrainingResampler, str_sample_id


def build(tmp_path, counts, colors, bad_donors=(), bad_targets=(), **kwargs):
    source = make_source(tmp_path / "src", counts, bad_donors)
    target = make_target(tmp_path / "tgt", colors, bad_targets)
    banks = make_banks()
    resampler = SelfTrainingResampler(
        source_manifest=source,
        target_feature_loader=feature_loader(target),
        prototype_banks=banks,
        output_dir=tmp_path / "str",
        global_seed=7,
        batch_size=2,
        device="cpu",
        num_classes=2,
        **kwargs,
    )
    model = TinyImageModel()
    return resampler, model, banks


def start(resampler, model, banks, tmp_path, round_id=1, epoch=25):
    snapshot = tmp_path / f"snap{round_id}.pt"
    save_snapshot(snapshot, model, banks, epoch)
    return resampler.create_transaction(round_id=round_id, epoch=epoch, checkpoint_path=snapshot)



@pytest.mark.parametrize(
    "indices",
    [[0, 1], [0, 0, 2], [2, 1, 0]],
    ids=["missing", "duplicate", "out_of_order"],
)
def test_target_scan_rejects_missing_duplicate_or_reordered_samples(tmp_path, indices):
    resampler, model, banks = build(tmp_path, {0: 4, 1: 2}, [1, 1, 1])
    resampler.target_feature_loader = DataLoader(
        resampler.target_dataset, batch_size=2, sampler=indices
    )
    with pytest.raises(ValueError, match="Target feature scan"):
        resampler.run(start(resampler, model, banks, tmp_path), model)


def test_target_scan_rejects_drop_last(tmp_path):
    resampler, model, banks = build(tmp_path, {0: 4, 1: 2}, [1, 1, 1])
    resampler.target_feature_loader = DataLoader(
        resampler.target_dataset, batch_size=2, drop_last=True
    )
    with pytest.raises(ValueError, match="Target feature scan"):
        resampler.run(start(resampler, model, banks, tmp_path), model)


def test_minority_class_and_tie_break(tmp_path):
    resampler, model, banks = build(tmp_path, {0: 4, 1: 2}, [1, 1])
    assert start(resampler, model, banks, tmp_path)["selected_class"] == 1
    resampler, model, banks = build(tmp_path / "tie", {0: 3, 1: 3}, [1, 1])
    assert start(resampler, model, banks, tmp_path / "tie")["selected_class"] == 0


def test_confidence_gate_is_strict(tmp_path, monkeypatch):
    resampler, model, banks = build(tmp_path, {0: 4, 1: 2}, [1, 1, 1])
    tx = start(resampler, model, banks, tmp_path)
    monkeypatch.setattr(
        resampler,
        "_predict",
        lambda *_: {"expw_000": (1, 0.95), "expw_001": (1, 0.9500001), "expw_002": (0, 0.99)},
    )
    done = resampler.run(tx, model)
    assert done["gate_count"] == 1 and done["created"] == 1
    assert read_jsonl(resampler.manifest_path)[0]["target_id"] == "expw_001"


def test_class_fixed_in_round_and_counts_accumulate(tmp_path):
    resampler, model, banks = build(tmp_path, {0: 4, 1: 2}, [1, 1])
    tx = start(resampler, model, banks, tmp_path)
    done = resampler.run(tx, model)
    assert done["selected_class"] == 1 and done["counts_before"] == [4, 2]
    assert done["counts_after"] == [4, 4]
    assert resampler.get_transaction(1)["status"] == "completed"
    second = start(resampler, model, banks, tmp_path, round_id=2, epoch=50)
    assert second["counts_before"] == [4, 4]
    assert second["selected_class"] == 0  # tie at 4/4 -> smallest index


def test_target_used_once_and_donors_are_original_only(tmp_path):
    resampler, model, banks = build(tmp_path, {0: 4, 1: 2}, [1, 1, 1])
    resampler.run(start(resampler, model, banks, tmp_path), model)
    first = read_jsonl(resampler.manifest_path)
    source_ids = {f"raf1_{i}" for i in range(2)}
    assert {r["donor_id"] for r in first} <= source_ids and len(first) == 3
    # Make class 1 minority again by giving round 2 a fresh transaction for the same class.
    tx2 = start(resampler, model, banks, tmp_path, round_id=2, epoch=50)
    tx2["selected_class"] = 1
    done = resampler.run(tx2, model)
    assert done["created"] == 0
    rows = read_jsonl(resampler.manifest_path)
    assert len({r["target_id"] for r in rows}) == len(rows) == 3
    assert all(r["sample_id"] == str_sample_id(r["round_id"], r["target_id"]) for r in rows)


def test_donor_retry_skips_invalid_donor(tmp_path):
    resampler, model, banks = build(tmp_path, {0: 4, 1: 3}, [1] * 6, bad_donors={"raf1_0", "raf1_1"})
    done = resampler.run(start(resampler, model, banks, tmp_path), model)
    rows = read_jsonl(resampler.manifest_path)
    assert done["created"] == 6 and {r["donor_id"] for r in rows} == {"raf1_2"}


def test_no_valid_donor_is_counted(tmp_path):
    resampler, model, banks = build(
        tmp_path, {0: 4, 1: 2}, [1, 1], bad_donors={"raf1_0", "raf1_1"}
    )
    done = resampler.run(start(resampler, model, banks, tmp_path), model)
    assert done["created"] == 0 and done["rejects"]["no_valid_donor"] == 2


def test_bad_target_geometry_never_searches_donors(tmp_path, monkeypatch):
    resampler, model, banks = build(tmp_path, {0: 4, 1: 2}, [1, 1], bad_targets={"expw_000", "expw_001"})
    calls = []
    monkeypatch.setattr(module, "_donor_order", lambda *a: calls.append(a) or [])
    done = resampler.run(start(resampler, model, banks, tmp_path), model)
    assert calls == [] and done["rejects"]["invalid_landmark"] == 2 and done["created"] == 0


def test_donor_order_is_deterministic_and_seed_dependent():
    donors = list(range(20))
    a = module._donor_order(donors, 1, 1, "t")
    assert a == module._donor_order(donors, 1, 1, "t")
    assert a != module._donor_order(donors, 2, 1, "t") and sorted(a) == donors
