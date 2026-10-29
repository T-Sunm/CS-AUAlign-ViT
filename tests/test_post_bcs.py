from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torchvision.models import swin_t

import cs_aualign_vit.models.post.bcs as bcs_module
from cs_aualign_vit.data.manifests import CLASSES
from cs_aualign_vit.models import POSTBCSModel
from cs_aualign_vit.models.post.bcs import BCSBlock
from cs_aualign_vit.training.source_pretraining import CHECKPOINT_VERSION


def source_checkpoint(tmp_path, **changes):
    backbone = swin_t(weights=None)
    backbone.head = nn.Identity()
    checkpoint = {
        "schema_version": CHECKPOINT_VERSION,
        "feature_dim": 768,
        "class_mapping": list(CLASSES),
        "backbone_state_dict": backbone.state_dict(),
    }
    checkpoint.update(changes)
    path = tmp_path / "source.pt"
    torch.save(checkpoint, path)
    return path, backbone


def pretrained_tokens_and_features(backbone, images):
    tokens = backbone.features(images)
    return tokens, backbone.norm(tokens).mean(dim=(1, 2))


def test_checkpoint_mapping_preserves_native_source_and_target_paths(tmp_path):
    path, backbone = source_checkpoint(tmp_path)
    model = POSTBCSModel.from_source_checkpoint(path).eval()
    backbone.eval()
    images = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        expected_tokens, expected_features = pretrained_tokens_and_features(backbone, images)
        source_tokens = model._native_tokens(images, "source")
        target_tokens = model._native_tokens(images, "target")
        source_features = model._pool(source_tokens, model.source_norm)
        target_features = model.forward_target_features(images)

    assert torch.allclose(source_tokens, expected_tokens, atol=1e-6, rtol=1e-5)
    assert torch.allclose(target_tokens, expected_tokens, atol=1e-6, rtol=1e-5)
    assert torch.allclose(source_features, expected_features, atol=1e-6, rtol=1e-5)
    assert torch.allclose(target_features, expected_features, atol=1e-6, rtol=1e-5)
    assert torch.equal(
        model.bcs_block1.source.attn.relative_position_index,
        backbone.features[7][1].attn.relative_position_index,
    )
    assert model.source_block0.norm1.weight.data_ptr() != model.target_block0.norm1.weight.data_ptr()
    assert model.source_norm.weight.data_ptr() != model.target_norm.weight.data_ptr()


def test_bcs_directions_keep_native_independence_and_cross_gradients():
    pretrained_block = swin_t(weights=None).features[7][1]
    block = BCSBlock(pretrained_block).eval()
    source = torch.randn(1, 7, 7, 768, requires_grad=True)
    target = torch.randn(1, 7, 7, 768, requires_grad=True)

    source_native, target_native, target_to_source, source_to_target = block(source, target)
    changed_target = target.detach().clone()
    changed_target[..., ::2] = -changed_target[..., ::2]
    changed = block(source.detach(), changed_target)

    assert source_native.shape == target_native.shape == target_to_source.shape == (1, 7, 7, 768)
    assert source_to_target.shape == (1, 7, 7, 768)
    assert torch.equal(source_native.detach(), changed[0])
    assert not torch.allclose(target_to_source.detach(), changed[2], atol=1e-6, rtol=1e-5)

    target_to_source.square().mean().backward()
    assert source.grad is not None and torch.count_nonzero(source.grad)
    assert target.grad is not None and torch.count_nonzero(target.grad)
    assert block.source.attn.qkv.weight.grad is not None
    assert block.target.attn.qkv.weight.grad is not None


def test_cross_attention_uses_exact_qkv_direction_and_destination_components():
    block = BCSBlock(swin_t(weights=None).features[7][1]).eval()
    source = torch.randn(1, 7, 7, 768)
    target = torch.randn(1, 7, 7, 768)
    with torch.no_grad():
        table = block.target.attn.relative_position_bias_table
        table.copy_(torch.linspace(-0.2, 0.2, table.numel()).reshape_as(table))
        block.target.attn.proj.weight.mul_(-1)
        block.target.attn.proj.bias.add_(0.25)
        block.target.norm2.weight.mul_(0.5)
        block.target.norm2.bias.add_(0.1)
        block.target.mlp[0].weight.mul_(-1)

        qs, ks, vs = block._qkv(block.source, source)
        qt, kt, vt = block._qkv(block.target, target)
        expected_target_to_source = block._finish_mlp(
            block.source,
            block._finish_attention(
                block.source,
                source,
                block._attention(block.source, qt, ks, vs),
            ),
        )
        expected_source_to_target = block._finish_mlp(
            block.target,
            block._finish_attention(
                block.target,
                target,
                block._attention(block.target, qs, kt, vt),
            ),
        )
        wrong_target_to_source = block._finish_mlp(
            block.target,
            block._finish_attention(
                block.target,
                source,
                block._attention(block.target, qs, kt, vt),
            ),
        )
        outputs = block(source, target)

    assert torch.allclose(outputs[2], expected_target_to_source)
    assert torch.allclose(outputs[3], expected_source_to_target)
    assert not torch.allclose(outputs[2], wrong_target_to_source)


def test_attention_finish_order_and_separate_drop_path_invocations(monkeypatch):
    events = []

    class Recorder(nn.Module):
        def __init__(self, name, offset):
            super().__init__()
            self.name = name
            self.offset = offset

        def forward(self, inputs):
            events.append(self.name)
            return inputs + self.offset

    def record_dropout(inputs, **_kwargs):
        events.append("dropout")
        return inputs * 2

    monkeypatch.setattr(bcs_module.F, "dropout", record_dropout)
    block = SimpleNamespace(
        attn=SimpleNamespace(proj=Recorder("linear", 1), dropout=0.5),
        stochastic_depth=Recorder("drop_path", 3),
        training=True,
    )
    residual = torch.zeros(1, 7, 7, 768)
    attention = torch.ones(1, 49, 768)

    first = BCSBlock._finish_attention(block, residual, attention)
    second = BCSBlock._finish_attention(block, residual, attention)

    assert events == ["linear", "dropout", "drop_path"] * 2
    assert torch.equal(first, torch.full_like(residual, 7))
    assert torch.equal(second, torch.full_like(residual, 7))


def test_forward_boundary_and_checkpoint_validation(tmp_path):
    path, _backbone = source_checkpoint(tmp_path)
    model = POSTBCSModel.from_source_checkpoint(path).eval()
    source = torch.randn(1, 3, 224, 224)
    target = torch.randn(1, 3, 224, 224)
    norm_outputs = {"source": [], "target": []}
    with torch.no_grad():
        model.target_norm.bias.add_(0.5)
    handles = (
        model.source_norm.register_forward_hook(
            lambda _module, _inputs, output: norm_outputs["source"].append(output.detach())
        ),
        model.target_norm.register_forward_hook(
            lambda _module, _inputs, output: norm_outputs["target"].append(output.detach())
        ),
    )

    with torch.no_grad():
        outputs = model.forward_features(source, target)
        target_only = model.forward_target_features(target)
    for handle in handles:
        handle.remove()

    assert outputs.source_features.shape == outputs.target_features.shape == (1, 768)
    assert outputs.target_to_source_features.shape == outputs.source_to_target_features.shape == (1, 768)
    assert torch.equal(outputs.source_features, norm_outputs["source"][0].mean(dim=(1, 2)))
    assert torch.equal(outputs.target_to_source_features, norm_outputs["source"][1].mean(dim=(1, 2)))
    assert torch.equal(outputs.target_features, norm_outputs["target"][0].mean(dim=(1, 2)))
    assert torch.equal(outputs.source_to_target_features, norm_outputs["target"][1].mean(dim=(1, 2)))
    assert torch.allclose(outputs.target_features, target_only, atol=1e-6, rtol=1e-5)
    with pytest.raises(ValueError, match="same batch size"):
        model.forward_features(source.repeat(2, 1, 1, 1), target)
    with pytest.raises(ValueError, match=r"\[B, 3, 224, 224\]"):
        model.forward_target_features(torch.randn(1, 3, 32, 32))

    bad_state = deepcopy(torch.load(path, weights_only=True)["backbone_state_dict"])
    bad_state.pop("features.7.1.attn.relative_position_index")
    bad_path, _ = source_checkpoint(tmp_path, backbone_state_dict=bad_state)
    with pytest.raises(ValueError, match="incompatible"):
        POSTBCSModel.from_source_checkpoint(bad_path)
