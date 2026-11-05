"""Stage-4 bidirectional cross-attention for POST adaptation."""

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from cs_aualign_vit.data.manifests import CLASSES
from cs_aualign_vit.training.source_pretraining import CHECKPOINT_VERSION


@dataclass(frozen=True)
class BCSFeatures:
    source_features: torch.Tensor
    target_features: torch.Tensor
    target_to_source_features: torch.Tensor
    source_to_target_features: torch.Tensor
    source_native_tokens: torch.Tensor | None = None
    target_native_tokens: torch.Tensor | None = None


class BCSBlock(nn.Module):
    """Run native and cross attention from the same pair of pretrained QKV tensors."""

    def __init__(self, pretrained_block, *, au_mode="off", g_scale=1.0):
        super().__init__()
        self.source = deepcopy(pretrained_block)
        self.target = deepcopy(pretrained_block)
        if self.source.attn.num_heads != 24:
            raise ValueError("POST BCS requires the 24-head Swin-T Stage 4 block")
        if au_mode not in {"off", "dynamic", "dynamic_shuffled"}:
            raise ValueError("au_mode must be off, dynamic, or dynamic_shuffled")
        if g_scale <= 0:
            raise ValueError("g_scale must be positive")
        self.au_mode = au_mode
        self.g_scale = float(g_scale)
        if au_mode == "off":
            self.register_parameter("lambda_target_to_source", None)
            self.register_parameter("lambda_source_to_target", None)
        else:
            self.lambda_target_to_source = nn.Parameter(torch.zeros(24))
            self.lambda_source_to_target = nn.Parameter(torch.zeros(24))
        self.last_guidance_metrics = {}

    @staticmethod
    def _qkv(block, inputs):
        normalized = block.norm1(inputs)
        batch, height, width, channels = normalized.shape
        heads = block.attn.num_heads
        head_dim = channels // heads
        qkv = block.attn.qkv(normalized.reshape(batch, height * width, channels))
        qkv = qkv.reshape(batch, height * width, 3, heads, head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4)
        return query, key, value

    @staticmethod
    def _attention(block, query, key, value, bias=None, metrics=None, valid_batches=None):
        logits = (query * query.shape[-1] ** -0.5) @ key.transpose(-2, -1)
        logits = logits + block.attn.get_relative_position_bias()
        if bias is None:
            weights = F.softmax(logits, dim=-1)
        else:
            baseline = F.softmax(logits, dim=-1)
            weights = F.softmax(logits + bias, dim=-1)
        if metrics is not None and bias is not None:
            change = 0.5 * (weights - baseline).abs().sum(dim=-1)
            newly_saturated = (baseline.amax(-1) < 0.95) & (weights.amax(-1) >= 0.95)
            if valid_batches is not None and valid_batches.any():
                change = change[valid_batches]
                newly_saturated = newly_saturated[valid_batches]
                measured_bias = bias[valid_batches]
                measured_logits = logits[valid_batches]
                metrics.update(
                    attention_change_p90=float(torch.quantile(change.detach().float(), 0.9)),
                    saturation_rate=float(newly_saturated.detach().float().mean()),
                    bias_logits_std_ratio=float(
                        measured_bias.detach().float().std()
                        / measured_logits.detach().float().std().clamp_min(1e-12)
                    ),
                )
        weights = F.dropout(weights, p=block.attn.attention_dropout, training=block.training)
        output = weights @ value
        batch, _heads, tokens, _head_dim = output.shape
        channels = query.shape[1] * query.shape[-1]
        return output.transpose(1, 2).reshape(batch, tokens, channels)

    @staticmethod
    def _finish_attention(block, residual, attention):
        batch, height, width, channels = residual.shape
        output = block.attn.proj(attention).reshape(batch, height, width, channels)
        output = F.dropout(output, p=block.attn.dropout, training=block.training)
        return residual + block.stochastic_depth(output)

    @staticmethod
    def _finish_mlp(block, inputs):
        return inputs + block.stochastic_depth(block.mlp(block.norm2(inputs)))

    def forward(self, source, target, guidance=None):
        if source.shape != target.shape:
            raise ValueError("Source and target token grids must have the same shape")
        if source.ndim != 4 or source.shape[1:] != (7, 7, 768):
            raise ValueError("BCS Block 1 expects token grids with shape [B, 7, 7, 768]")
        if self.au_mode == "off" and guidance is not None:
            raise ValueError("off mode does not accept AU guidance")
        if self.au_mode != "off":
            if guidance is None or guidance.shape != (source.shape[0], 49, 49):
                raise ValueError("AU guidance must have shape [B, 49, 49]")
            guidance = guidance.to(device=source.device, dtype=source.dtype)

        qs, ks, vs = self._qkv(self.source, source)
        qt, kt, vt = self._qkv(self.target, target)
        source_native = self._attention(self.source, qs, ks, vs)
        target_native = self._attention(self.target, qt, kt, vt)
        self.last_guidance_metrics = {}
        target_metrics, source_metrics = {}, {}
        target_bias = source_bias = None
        if guidance is not None:
            valid_batches = guidance.abs().sum(dim=(-2, -1)) > 0
            target_bias = (
                self.g_scale
                * self.lambda_target_to_source[None, :, None, None]
                * guidance[:, None]
            ).to(source.dtype)
            source_bias = (
                self.g_scale
                * self.lambda_source_to_target[None, :, None, None]
                * guidance.transpose(-2, -1)[:, None]
            ).to(source.dtype)
        target_to_source = self._attention(
            self.source,
            qt,
            ks,
            vs,
            target_bias,
            target_metrics if guidance is not None else None,
            valid_batches if guidance is not None else None,
        )
        source_to_target = self._attention(
            self.target,
            qs,
            kt,
            vt,
            source_bias,
            source_metrics if guidance is not None else None,
            valid_batches if guidance is not None else None,
        )
        self.last_guidance_metrics = {
            f"au_t2s_{name}": value for name, value in target_metrics.items()
        } | {f"au_s2t_{name}": value for name, value in source_metrics.items()}

        source_native = self._finish_attention(self.source, source, source_native)
        target_to_source = self._finish_attention(self.source, source, target_to_source)
        target_native = self._finish_attention(self.target, target, target_native)
        source_to_target = self._finish_attention(self.target, target, source_to_target)
        return tuple(
            self._finish_mlp(block, stream)
            for block, stream in (
                (self.source, source_native),
                (self.target, target_native),
                (self.source, target_to_source),
                (self.target, source_to_target),
            )
        )

    def forward_native(self, inputs, domain):
        block = self.source if domain == "source" else self.target
        query, key, value = self._qkv(block, inputs)
        attention = self._attention(block, query, key, value)
        return self._finish_mlp(block, self._finish_attention(block, inputs, attention))


class POSTBCSModel(nn.Module):
    feature_dim = 768

    def __init__(self, pretrained_backbone, *, au_mode="off", g_scale=1.0):
        super().__init__()
        if len(pretrained_backbone.features) != 8 or len(pretrained_backbone.features[7]) != 2:
            raise ValueError("Expected the torchvision Swin-T feature layout")
        self.shared_trunk = deepcopy(nn.Sequential(*pretrained_backbone.features[:7]))
        self.source_block0 = deepcopy(pretrained_backbone.features[7][0])
        self.target_block0 = deepcopy(pretrained_backbone.features[7][0])
        self.bcs_block1 = BCSBlock(
            pretrained_backbone.features[7][1], au_mode=au_mode, g_scale=g_scale
        )
        self.source_norm = deepcopy(pretrained_backbone.norm)
        self.target_norm = deepcopy(pretrained_backbone.norm)

    @classmethod
    def from_source_checkpoint(cls, path, *, au_mode="off", g_scale=1.0):
        from torchvision.models import swin_t

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("schema_version") != CHECKPOINT_VERSION:
            raise ValueError("Unsupported source checkpoint schema")
        if checkpoint.get("feature_dim") != cls.feature_dim:
            raise ValueError("Source checkpoint feature dimension must be 768")
        if checkpoint.get("class_mapping") != list(CLASSES):
            raise ValueError("Source checkpoint class mapping differs from the data pipeline")
        state = checkpoint.get("backbone_state_dict")
        if state is None:
            raise ValueError("Source checkpoint is missing backbone_state_dict")
        if not isinstance(state, dict):
            raise TypeError("Source checkpoint backbone_state_dict must be a dictionary")

        backbone = swin_t(weights=None)
        backbone.head = nn.Identity()
        try:
            backbone.load_state_dict(state, strict=True)
        except RuntimeError as error:
            raise ValueError("Source checkpoint is incompatible with torchvision Swin-T") from error
        return cls(backbone, au_mode=au_mode, g_scale=g_scale)

    @staticmethod
    def _validate_images(images, name):
        if images.ndim != 4 or images.shape[1:] != (3, 224, 224):
            raise ValueError(f"{name} must have shape [B, 3, 224, 224]")

    @staticmethod
    def _pool(tokens, norm):
        return norm(tokens).mean(dim=(1, 2))

    @staticmethod
    def _normalized_tokens(tokens, norm):
        normalized = norm(tokens)
        return normalized.reshape(normalized.shape[0], 49, normalized.shape[-1])

    def _native_tokens(self, images, domain):
        if domain not in {"source", "target"}:
            raise ValueError("domain must be source or target")
        self._validate_images(images, f"{domain} images")
        tokens = self.shared_trunk(images)
        block0 = self.source_block0 if domain == "source" else self.target_block0
        return self.bcs_block1.forward_native(block0(tokens), domain)

    def forward_target_features(self, target_images):
        return self._pool(self._native_tokens(target_images, "target"), self.target_norm)

    def forward_features(self, source_images, target_images, guidance=None):
        self._validate_images(source_images, "source images")
        self._validate_images(target_images, "target images")
        if source_images.shape[0] != target_images.shape[0]:
            raise ValueError("Source and target batches must have the same batch size")

        combined = self.shared_trunk(torch.cat((source_images, target_images)))
        source, target = combined.split((source_images.shape[0], target_images.shape[0]))
        source = self.source_block0(source)
        target = self.target_block0(target)
        source, target, target_to_source, source_to_target = self.bcs_block1(
            source, target, guidance
        )
        source_native_tokens = self._normalized_tokens(source, self.source_norm)
        target_native_tokens = self._normalized_tokens(target, self.target_norm)
        return BCSFeatures(
            source_features=source_native_tokens.mean(dim=1),
            target_features=target_native_tokens.mean(dim=1),
            target_to_source_features=self._pool(target_to_source, self.source_norm),
            source_to_target_features=self._pool(source_to_target, self.target_norm),
            source_native_tokens=source_native_tokens,
            target_native_tokens=target_native_tokens,
        )

    def forward(self, source_images, target_images, guidance=None):
        return self.forward_features(source_images, target_images, guidance)
