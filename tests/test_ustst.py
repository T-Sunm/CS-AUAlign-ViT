import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from str_helpers import make_source, make_target
from torch import nn
from torchvision.models import swin_t

from cs_aualign_vit.data.manifests import CLASSES, read_jsonl, write_jsonl
from cs_aualign_vit.models import USTSTFeatures, USTSTHeads, USTSTModel
from cs_aualign_vit.models.ustst import cast_block, window_attention
from cs_aualign_vit.training import USTSTConfig
from cs_aualign_vit.training.source_pretraining import CHECKPOINT_VERSION
from cs_aualign_vit.training.ustst_losses import (
    gaussian_mmd,
    hard_pseudo_label_loss,
    logit_distance,
    similarity_loss,
    soft_pseudo_label_loss,
    target_gates,
)


def load_script(name):
    path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_checkpoint(tmp_path):
    backbone = swin_t(weights=None)
    backbone.head = nn.Identity()
    path = tmp_path / "source.pt"
    torch.save(
        {
            "schema_version": CHECKPOINT_VERSION,
            "feature_dim": 768,
            "class_mapping": list(CLASSES),
            "backbone_state_dict": backbone.state_dict(),
            "source_prototypes": torch.randn(len(CLASSES), 768),
        },
        path,
    )
    return path, backbone


# -- model ---------------------------------------------------------------------------


def test_native_target_path_reproduces_pretrained_swin(tmp_path):
    path, backbone = source_checkpoint(tmp_path)
    model = USTSTModel.from_source_checkpoint(path).eval()
    backbone.eval()
    images = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        expected = backbone.norm(backbone.features(images)).mean(dim=(1, 2))
        actual = model.forward_target_features(images)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_window_attention_with_identity_index_matches_torchvision_w_and_sw_msa():
    backbone = swin_t(weights=None).eval()
    stage1 = backbone.features[1]
    assert stage1[0].attn.shift_size == [0, 0] and stage1[1].attn.shift_size == [3, 3]
    tokens = torch.randn(2, 56, 56, 96)
    index = torch.arange(2)
    with torch.no_grad():
        for block in stage1:
            normalized = block.norm1(tokens)
            assert torch.allclose(
                window_attention(block.attn, normalized, index),
                block.attn(normalized),
                atol=1e-5,
                rtol=1e-5,
            )
            assert torch.allclose(cast_block(block, tokens, index), block(tokens), atol=1e-5)


def test_window_cross_attention_stays_inside_regular_and_shifted_windows():
    backbone = swin_t(weights=None).eval()
    regular, shifted = backbone.features[1]
    query = torch.randn(1, 56, 56, 96)
    key_value = torch.randn(1, 56, 56, 96)
    changed = key_value.clone()
    changed[:, 5, 5] += 10.0
    index = torch.tensor([1, 1])  # both rows read keys/values from sample 1
    with torch.no_grad():
        for block, inside, outside in ((regular, (6, 6), (8, 8)), (shifted, (8, 8), (20, 20))):
            before = window_attention(block.attn, torch.cat((query, key_value)), index)[0]
            after = window_attention(block.attn, torch.cat((query, changed)), index)[0]
            delta = (before - after).abs().sum(dim=-1)
            assert delta[inside] > 0, "key/value change must reach queries in the same window"
            assert delta[outside] == 0, "W-MCA/SW-MCA must not leak across windows"
    # (8, 8) is outside the regular 7x7 window of (5, 5) but inside its shifted window.


def test_forward_streams_match_native_path_and_cross_stream_is_q_source_kv_weak(tmp_path):
    path, _ = source_checkpoint(tmp_path)
    model = USTSTModel.from_source_checkpoint(path).eval()
    source, weak, strong = (torch.randn(2, 3, 224, 224) for _ in range(3))
    with torch.no_grad():
        features = model(source, weak, strong)
        same = model(source, source, strong)
        other_strong = model(source, weak, torch.randn(2, 3, 224, 224))
        other_weak = model(source, torch.randn(2, 3, 224, 224), strong)

    assert isinstance(features, USTSTFeatures)
    for name in ("source_features", "target_weak_features", "target_strong_features", "cross_features"):
        assert getattr(features, name).shape == (2, 768)
    assert torch.allclose(features.source_features, model.forward_target_features(source), atol=1e-5)
    assert torch.allclose(features.target_weak_features, model.forward_target_features(weak), atol=1e-5)
    assert torch.allclose(features.target_strong_features, model.forward_target_features(strong), atol=1e-5)
    # Q_s with K/V of an identical image is plain self-attention at every CAST block.
    assert torch.allclose(same.cross_features, same.source_features, atol=1e-5)
    assert torch.allclose(other_strong.cross_features, features.cross_features, atol=1e-6)
    assert not torch.allclose(other_weak.cross_features, features.cross_features, atol=1e-3)
    assert not torch.allclose(features.cross_features, features.source_features, atol=1e-3)
    with pytest.raises(ValueError, match="same batch size"):
        model(source, weak[:1], strong)


def test_cross_start_block_limits_cross_attention_to_later_blocks(tmp_path):
    path, _ = source_checkpoint(tmp_path)
    full = USTSTModel.from_source_checkpoint(path).eval()
    last = USTSTModel.from_source_checkpoint(path, cross_start_block=11).eval()
    source, weak, strong = (torch.randn(1, 3, 224, 224) for _ in range(3))
    with torch.no_grad():
        full_features = full(source, weak, strong)
        last_features = last(source, weak, strong)
    assert torch.allclose(full_features.source_features, last_features.source_features, atol=1e-6)
    assert not torch.allclose(full_features.cross_features, last_features.cross_features, atol=1e-3)
    with pytest.raises(ValueError, match="cross_start_block"):
        USTSTModel.from_source_checkpoint(path, cross_start_block=12)


def test_cross_features_receive_gradient_from_both_domains(tmp_path):
    path, _ = source_checkpoint(tmp_path)
    model = USTSTModel.from_source_checkpoint(path).eval()  # no stochastic-depth drop
    source = torch.randn(1, 3, 224, 224, requires_grad=True)
    weak = torch.randn(1, 3, 224, 224, requires_grad=True)
    strong = torch.randn(1, 3, 224, 224, requires_grad=True)
    model(source, weak, strong).cross_features.square().sum().backward()
    assert torch.count_nonzero(source.grad) and torch.count_nonzero(weak.grad)
    assert strong.grad is None or not torch.count_nonzero(strong.grad)


def test_heads_start_from_source_prototypes_and_expose_bank_interface(tmp_path):
    path, _ = source_checkpoint(tmp_path)
    prototypes = torch.load(path, weights_only=True)["source_prototypes"]
    heads = USTSTHeads.from_source_checkpoint(path, feature_dim=768)
    features = torch.randn(3, 768)
    assert torch.allclose(heads.logits(features), features @ prototypes.T)
    assert torch.allclose(heads.cross_logits(features), heads.logits(features))
    assert heads.target_prototypes is heads.classifier is heads.source_prototypes
    assert heads.classifier.data_ptr() != heads.cross_classifier.data_ptr()
    assert {name for name, _ in heads.named_parameters()} == {"classifier", "cross_classifier"}


# -- losses --------------------------------------------------------------------------


def test_target_gates_select_confident_agreeing_and_ambiguous_samples():
    logits = torch.tensor(
        [
            [10.0, 0, 0, 0, 0, 0, 0],  # confident class 0, cluster agrees -> hard
            [0, 10.0, 0, 0, 0, 0, 0],  # confident class 1, cluster disagrees
            [0.1, 0, 0, 0, 0, 0, 0],  # flat -> soft
            [10.0, 9.9, 0, 0, 0, 0, 0],  # two-way split: entropy ~ log2/log7 < 0.5
        ]
    )
    clusters = torch.tensor([0, 2, 0, 1])
    gates = target_gates(logits, clusters, confidence_threshold=0.95, entropy_threshold=0.5)
    assert gates.hard_mask.tolist() == [True, False, False, False]
    assert gates.agree_mask.tolist() == [True, False, True, False]
    assert gates.soft_enabled
    assert gates.soft_mask.tolist() == [False, False, True, False]
    assert gates.normalized_entropy[2] > 0.99

    flat = target_gates(torch.zeros(2, 7), torch.zeros(2, dtype=torch.long))
    assert not flat.soft_enabled and not flat.soft_mask.any() and not flat.hard_mask.any()


def test_masked_pseudo_label_losses_average_over_the_whole_batch():
    logits = torch.randn(4, 7, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 3])
    mask = torch.tensor([True, False, True, False])
    expected = torch.nn.functional.cross_entropy(logits[mask], labels[mask], reduction="sum") / 4
    assert torch.allclose(hard_pseudo_label_loss(logits, labels, mask), expected)
    assert hard_pseudo_label_loss(logits, labels, torch.zeros(4, dtype=torch.bool)) == 0

    teacher = torch.randn(4, 7, requires_grad=True)
    loss = soft_pseudo_label_loss(logits, teacher, mask)
    loss.backward()
    assert teacher.grad is None
    assert logits.grad[~mask].abs().sum() == 0


def test_similarity_loss_uses_batch_hard_positive_and_negative():
    source = torch.tensor([[0.0, 0.0]])
    targets = torch.tensor([[1.0, 0.0], [3.0, 0.0], [2.0, 0.0], [0.5, 0.0]])
    labels = torch.tensor([0, 0, 1, 2])
    mask = torch.tensor([True, True, True, False])
    distance = logit_distance(source, targets)
    assert torch.allclose(distance, torch.tensor([[0.5, 4.5, 2.0, 0.125]]))
    # Selection by Eq. 3: hardest positive index 1 (4.5), hardest selected negative index 2
    # (2.0). Eq. 4 hinges squared L2 norms: ||A-P||^2 = 9, ||A-N||^2 = 4 (n = 2 classes).
    loss, anchors = similarity_loss(source, torch.tensor([0]), targets, labels, mask, margin=0.2)
    assert anchors == 1
    assert torch.isclose(loss, torch.tensor(9.0 - 4.0 + 0.2))

    none, count = similarity_loss(
        source, torch.tensor([0]), targets, labels, torch.zeros(4, dtype=torch.bool), margin=0.2
    )
    assert count == 0 and none == 0


def test_gaussian_mmd_is_zero_for_identical_batches_and_grows_with_shift():
    torch.manual_seed(0)
    features = torch.randn(16, 8)
    assert gaussian_mmd(features, features).abs() < 1e-6
    near = gaussian_mmd(features, features + 0.1)
    far = gaussian_mmd(features, features + 3.0)
    assert 0 < near < far


def test_config_rejects_au_and_invalid_thresholds():
    assert USTSTConfig().kmeans_interval == 1
    with pytest.raises(ValueError, match="AU"):
        USTSTConfig(au_mode="dynamic")
    with pytest.raises(ValueError, match="entropy_threshold"):
        USTSTConfig(entropy_threshold=1.5)
    with pytest.raises(ValueError, match="lambda_sim"):
        USTSTConfig(lambda_sim=-1)


def test_repo_config_validates():
    train_ustst = load_script("train_ustst")
    config = train_ustst.validate_config(
        yaml.safe_load((Path(__file__).parents[1] / "configs/training/ustst.yaml").read_text())
    )
    training = config["training"]
    USTSTConfig(
        max_epochs=training["max_epochs"],
        kmeans_interval=training["kmeans_interval"],
        str_epochs=tuple(training["str_epochs"]),
        str_threshold=config["str"]["confidence_threshold"],
        str_patch_scale_range=tuple(config["str"]["patch_scale_range"]),
        cross_start_block=config["model"]["cross_start_block"],
        **config["loss"],
    )
    assert len(training["str_epochs"]) == 10


# -- end to end ----------------------------------------------------------------------


class TinyUSTSTModel(nn.Module):
    """Pools images to RGB means; the cross stream mixes source and weak target."""

    feature_dim = 2

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 2, bias=False)
        self.mix = nn.Parameter(torch.tensor(0.5))
        with torch.no_grad():
            self.projection.weight.copy_(torch.tensor([[20.0, 0, 0], [0, 20.0, 0]]))

    def forward_target_features(self, images):
        return self.projection(images.mean(dim=(2, 3)))

    def forward_features(self, source, weak, strong):
        s, w, t = (self.forward_target_features(x) for x in (source, weak, strong))
        return USTSTFeatures(s, w, t, self.mix * s + (1 - self.mix) * w)


def prototypes():
    return torch.tensor(
        [[1, 0], [0, 1], [1, 1], [-1, 1], [1, -1], [-1, -1], [0.5, 1]], dtype=torch.float32
    )


def test_train_resume_str_and_evaluate(tmp_path, monkeypatch, capsys):
    train_ustst = load_script("train_ustst")
    evaluate_ustst = load_script("evaluate_ustst")
    monkeypatch.setattr(
        train_ustst.USTSTModel, "from_source_checkpoint", staticmethod(lambda _path, **_kw: TinyUSTSTModel())
    )
    monkeypatch.setattr(
        train_ustst.USTSTHeads,
        "from_source_checkpoint",
        staticmethod(lambda _path, *, feature_dim: USTSTHeads(prototypes())),
    )
    monkeypatch.setattr(
        evaluate_ustst.USTSTModel,
        "from_source_checkpoint",
        staticmethod(lambda _path, **_kw: TinyUSTSTModel()),
    )
    monkeypatch.setattr(
        evaluate_ustst.USTSTHeads,
        "from_source_checkpoint",
        staticmethod(lambda _path, *, feature_dim: USTSTHeads(prototypes())),
    )

    source_manifest = make_source(tmp_path / "source", {0: 2, 1: 2})
    target_manifest = make_target(tmp_path / "target", [0, 1, 0, 1, 0, 1, 0, 1])
    rows = read_jsonl(target_manifest)
    for row in rows[:2]:
        row["split"] = "validation"
    for row in rows[2:4]:
        row["split"] = "test"
    write_jsonl(target_manifest, rows)
    evaluation_manifest = make_target(tmp_path / "evaluation", [0, 1])
    evaluation_rows = read_jsonl(evaluation_manifest)
    for row in evaluation_rows:
        row["split"] = "test"
    write_jsonl(evaluation_manifest, evaluation_rows)
    preprocessing = {"image_size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    checkpoint_source = tmp_path / "source.pt"
    torch.save({"preprocessing_config": preprocessing}, checkpoint_source)
    config = yaml.safe_load((Path(__file__).parents[1] / "configs/training/ustst.yaml").read_text())
    config["training"].update(
        max_epochs=2, batch_size=2, num_workers=0, device="cpu", amp=False, str_epochs=[1]
    )
    config["adversarial"].update(hidden_dim=4, dropout=0.0)
    config_path = tmp_path / "ustst.yaml"
    config_path.write_text(yaml.safe_dump(config))
    output = tmp_path / "run"
    args = SimpleNamespace(
        source_manifest=source_manifest,
        target_manifest=target_manifest,
        source_checkpoint=checkpoint_source,
        config=config_path,
        output=output,
        resume=None,
        max_steps_per_epoch=None,
    )

    checkpoint = train_ustst.run(args)
    assert "Epoch 2/2 [" in capsys.readouterr().out
    lines = (output / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    for key in ("loss_s", "loss_t", "loss_c", "loss_sim", "loss_m", "loss_ad", "hard_ratio"):
        assert math.isfinite(first[key])
    state = torch.load(checkpoint, weights_only=True)
    assert state["completed_epoch"] == 2 and state["boundary_completed"]
    assert state["target_supervision"]["epoch"] == 1  # k-means refreshed after epoch 1
    assert state["config"]["str_threshold"] == 0.2
    assert tuple(state["config"]["str_patch_scale_range"]) == (0.3, 0.5)
    rounds = json.loads((output / "str" / "transactions" / "round-01.json").read_text())
    assert rounds["patch_scale_range"] == [0.3, 0.5] and rounds["status"] == "completed"
    assert (output / "str").is_dir() and (output / "best.pt").exists()
    test_results = json.loads((output / "test_results.json").read_text())
    assert test_results["final"]["completed_epoch"] == 2
    assert test_results["best"]["completed_epoch"] in (1, 2)
    assert test_results["final"]["samples"] == 2 and test_results["max_steps_per_epoch"] is None

    args.resume = checkpoint
    assert train_ustst.run(args) == checkpoint
    assert len((output / "metrics.jsonl").read_text().splitlines()) == 2

    # best.pt is a result, never a resume point (a resume would overwrite it).
    args.resume = output / "best.pt"
    with pytest.raises(ValueError, match="rolling adaptation.pt"):
        train_ustst.run(args)
    args.resume = checkpoint

    # A checkpoint saved before its epoch was validated gets validated on resume.
    (output / "metrics.jsonl").write_text(f"{lines[0]}\n")
    (output / "test_results.json").unlink()
    train_ustst.run(args)
    rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert [row["epoch"] for row in rows] == [1, 2]
    assert rows[1]["recovered_after_interrupt"] and "validation_accuracy" in rows[1]
    assert (output / "test_results.json").exists()

    # A row cut off while being appended is dropped and recovered from the checkpoint.
    (output / "metrics.jsonl").write_text(f'{lines[0]}\n{{"epoch":')
    train_ustst.run(args)
    rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert [row["epoch"] for row in rows] == [1, 2] and rows[1]["recovered_after_interrupt"]

    # Invalid JSON before the last row is corruption and is reported.
    good = (output / "metrics.jsonl").read_text()
    (output / "metrics.jsonl").write_text(f'{{"epoch":\n{good}')
    with pytest.raises(ValueError, match="line 1 is not valid JSON"):
        train_ustst.run(args)
    (output / "metrics.jsonl").write_text(good)

    # A fresh run into an existing run directory is refused before writing anything.
    before = {path: path.stat().st_mtime_ns for path in output.rglob("*")}
    args.resume = None
    with pytest.raises(FileExistsError, match="already holds run state"):
        train_ustst.run(args)
    assert {path: path.stat().st_mtime_ns for path in output.rglob("*")} == before
    args.resume = checkpoint

    # A smoke checkpoint cannot continue as a full run, nor the reverse.
    args.max_steps_per_epoch = 1
    with pytest.raises(ValueError, match="configuration differs"):
        train_ustst.run(args)
    args.max_steps_per_epoch = None

    result = evaluate_ustst.run(
        SimpleNamespace(
            manifest=evaluation_manifest,
            checkpoint=output / "best.pt",
            source_checkpoint=checkpoint_source,
            split="test",
            batch_size=2,
            num_workers=0,
            device="cpu",
            output=output / "evaluation.json",
        )
    )
    assert result["samples"] == 2 and len(result["confusion_matrix"]) == 7
    assert result["max_steps_per_epoch"] is None


def test_fresh_run_refuses_an_output_with_run_state(tmp_path):
    resolve_output = load_script("train_ustst").resolve_output
    output = tmp_path / "run"
    assert resolve_output(output, None) == output
    (output / "resolved_config.yaml").write_text("{}")  # crashed before any checkpoint
    assert resolve_output(output, None) == output
    for name in ("adaptation.pt", "adaptation.epoch-25.boundary.pt", "best.pt", "metrics.jsonl", "str"):
        state = output / name
        state.mkdir() if name == "str" else state.write_text("")
        with pytest.raises(FileExistsError, match=name.replace(".", r"\.")):
            resolve_output(output, None)
        state.rmdir() if name == "str" else state.unlink()
