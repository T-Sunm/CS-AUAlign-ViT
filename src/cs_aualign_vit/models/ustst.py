"""USTST Cross-Swin-Transformer (CST) and its two linear expression heads.

USTST (Guo et al., MTAP 2023, Section 3.1, Figs. 2-4) feeds source and target images
through one weight-shared Swin-T whose blocks are Cross Attention Swin-Transformer
(CAST) blocks: the usual W-MSA/SW-MSA pairs become W-MCA/SW-MCA, where the source
queries attend to the target keys/values of the same (shifted) window,
``Attention(Q_s, K_t, V_t)`` (Eq. 2).

The repo realises this as four token streams through the shared backbone:

* ``F_S``, ``F_Tw`` and ``F_Ts`` run native window self-attention. They are exactly
  the pretrained Swin-T, which the native target path used for k-means, STR and
  evaluation relies on.
* ``F_c`` (the cross stream) starts from the source patch embedding. In every CAST
  block from ``cross_start_block`` on, its queries come from its own tokens and its
  keys/values from the weak-target stream at the same block, using the same window
  partition, cyclic shift and shift mask. Its residual and MLP follow the query
  stream. Patch merging and the final norm are shared with the native streams.
"""

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from cs_aualign_vit.data.manifests import CLASSES
from cs_aualign_vit.training.source_pretraining import CHECKPOINT_VERSION


@dataclass(frozen=True)
class USTSTFeatures:
    source_features: torch.Tensor
    target_weak_features: torch.Tensor
    target_strong_features: torch.Tensor
    cross_features: torch.Tensor


def _is_swin_stage(module):
    return isinstance(module, nn.Sequential) and hasattr(module[0], "attn")


def window_attention(attn, inputs, kv_index):
    """torchvision ``shifted_window_attention`` with keys/values gathered by ``kv_index``.

    ``inputs`` is the norm1 output ``[N, H, W, C]``. Sample ``i`` uses its own queries
    and the keys/values of sample ``kv_index[i]``; the identity index reproduces
    torchvision exactly. Both samples share one window partition, so the cyclic
    shift and the SW-MSA mask apply unchanged.
    """
    batch, height, width, channels = inputs.shape
    window = attn.window_size
    shift = list(attn.shift_size)
    heads = attn.num_heads
    pad_r = (window[1] - width % window[1]) % window[1]
    pad_b = (window[0] - height % window[0]) % window[0]
    x = F.pad(inputs, (0, 0, 0, pad_r, 0, pad_b))
    _, pad_h, pad_w, _ = x.shape
    if window[0] >= pad_h:
        shift[0] = 0
    if window[1] >= pad_w:
        shift[1] = 0
    if sum(shift) > 0:
        x = torch.roll(x, shifts=(-shift[0], -shift[1]), dims=(1, 2))
    rows, cols = pad_h // window[0], pad_w // window[1]
    windows = rows * cols
    tokens = window[0] * window[1]
    x = x.view(batch, rows, window[0], cols, window[1], channels)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(batch, windows, tokens, channels)

    qkv = F.linear(x, attn.qkv.weight, attn.qkv.bias)
    qkv = qkv.reshape(batch, windows, tokens, 3, heads, channels // heads)
    query = qkv[:, :, :, 0]
    key = qkv[kv_index, :, :, 1]
    value = qkv[kv_index, :, :, 2]
    # [N * nW, heads, T, head_dim]
    query, key, value = (
        t.permute(0, 1, 3, 2, 4).reshape(batch * windows, heads, tokens, -1)
        for t in (query, key, value)
    )
    logits = (query * (channels // heads) ** -0.5) @ key.transpose(-2, -1)
    logits = logits + attn.get_relative_position_bias()
    if sum(shift) > 0:
        mask = x.new_zeros((pad_h, pad_w))
        h_slices = ((0, -window[0]), (-window[0], -shift[0]), (-shift[0], None))
        w_slices = ((0, -window[1]), (-window[1], -shift[1]), (-shift[1], None))
        count = 0
        for h in h_slices:
            for w in w_slices:
                mask[h[0] : h[1], w[0] : w[1]] = count
                count += 1
        mask = mask.view(rows, window[0], cols, window[1]).permute(0, 2, 1, 3)
        mask = mask.reshape(windows, tokens)
        mask = mask.unsqueeze(1) - mask.unsqueeze(2)
        mask = mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)
        logits = logits.view(batch, windows, heads, tokens, tokens) + mask[None, :, None]
        logits = logits.view(-1, heads, tokens, tokens)
    weights = F.softmax(logits, dim=-1)
    weights = F.dropout(weights, p=attn.attention_dropout, training=attn.training)
    out = (weights @ value).transpose(1, 2).reshape(batch * windows, tokens, channels)
    out = F.linear(out, attn.proj.weight, attn.proj.bias)
    out = F.dropout(out, p=attn.dropout, training=attn.training)
    out = out.view(batch, rows, cols, window[0], window[1], channels)
    out = out.permute(0, 1, 3, 2, 4, 5).reshape(batch, pad_h, pad_w, channels)
    if sum(shift) > 0:
        out = torch.roll(out, shifts=(shift[0], shift[1]), dims=(1, 2))
    return out[:, :height, :width, :].contiguous()


def cast_block(block, tokens, kv_index):
    """One Swin block whose attention reads keys/values through ``kv_index``."""
    tokens = tokens + block.stochastic_depth(window_attention(block.attn, block.norm1(tokens), kv_index))
    return tokens + block.stochastic_depth(block.mlp(block.norm2(tokens)))


class USTSTModel(nn.Module):
    feature_dim = 768
    num_blocks = 12

    def __init__(self, pretrained_backbone, *, cross_start_block=0):
        super().__init__()
        if len(pretrained_backbone.features) != 8 or len(pretrained_backbone.features[7]) != 2:
            raise ValueError("Expected the torchvision Swin-T feature layout")
        blocks = sum(len(m) for m in pretrained_backbone.features if _is_swin_stage(m))
        if blocks != self.num_blocks:
            raise ValueError("Expected the 12-block Swin-T layout")
        if type(cross_start_block) is not int or not 0 <= cross_start_block < blocks:
            raise ValueError(f"cross_start_block must be an integer in [0, {blocks - 1}]")
        self.cross_start_block = cross_start_block
        self.features = deepcopy(pretrained_backbone.features)
        self.norm = deepcopy(pretrained_backbone.norm)

    @classmethod
    def from_source_checkpoint(cls, path, *, cross_start_block=0):
        from torchvision.models import swin_t

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("schema_version") != CHECKPOINT_VERSION:
            raise ValueError("Unsupported source checkpoint schema")
        if checkpoint.get("feature_dim") != cls.feature_dim:
            raise ValueError("Source checkpoint feature dimension must be 768")
        if checkpoint.get("class_mapping") != list(CLASSES):
            raise ValueError("Source checkpoint class mapping differs from the data pipeline")
        state = checkpoint.get("backbone_state_dict")
        if not isinstance(state, dict):
            raise TypeError("Source checkpoint is missing backbone_state_dict")
        backbone = swin_t(weights=None)
        backbone.head = nn.Identity()
        try:
            backbone.load_state_dict(state, strict=True)
        except RuntimeError as error:
            raise ValueError("Source checkpoint is incompatible with torchvision Swin-T") from error
        return cls(backbone, cross_start_block=cross_start_block)

    @staticmethod
    def _validate_images(images, name):
        if images.ndim != 4 or images.shape[1:] != (3, 224, 224):
            raise ValueError(f"{name} must have shape [B, 3, 224, 224]")

    def _pool(self, tokens):
        return self.norm(tokens).mean(dim=(1, 2))

    def forward_target_features(self, target_images):
        """Native self-attention path used for k-means, STR and evaluation."""
        self._validate_images(target_images, "target images")
        return self._pool(self.features(target_images))

    def forward_features(self, source_images, target_weak_images, target_strong_images):
        for images, name in (
            (source_images, "source images"),
            (target_weak_images, "weak target images"),
            (target_strong_images, "strong target images"),
        ):
            self._validate_images(images, name)
        batch = source_images.shape[0]
        if target_weak_images.shape[0] != batch or target_strong_images.shape[0] != batch:
            raise ValueError("Source and target batches must have the same batch size")

        device = source_images.device
        native_index = torch.arange(3 * batch, device=device)
        # Cross stream (last B rows) reads keys/values from the weak-target rows.
        cross_index = torch.cat((native_index, torch.arange(batch, 2 * batch, device=device)))
        tokens = torch.cat((source_images, target_weak_images, target_strong_images))
        block_id = 0
        for module in self.features:
            if not _is_swin_stage(module):
                tokens = module(tokens)
                continue
            for block in module:
                if block_id == self.cross_start_block:
                    tokens = torch.cat((tokens, tokens[:batch]))
                index = cross_index if block_id >= self.cross_start_block else native_index
                tokens = cast_block(block, tokens, index)
                block_id += 1
        source, weak, strong, cross = self._pool(tokens).split(batch)
        return USTSTFeatures(
            source_features=source,
            target_weak_features=weak,
            target_strong_features=strong,
            cross_features=cross,
        )

    def forward(self, source_images, target_weak_images, target_strong_images):
        return self.forward_features(source_images, target_weak_images, target_strong_images)


class USTSTHeads(nn.Module):
    """Expression classifier ``f`` and cross-feature classifier ``f_c``.

    Both are bias-free linear layers initialised from the source-pretrained
    prototypes, so epoch 0 reproduces the source model exactly. The attribute
    names mirror ``PrototypeBanks`` so k-means refresh, STR and evaluation reuse
    the POST implementations unchanged: ``f`` is the single classifier USTST
    applies to both domains.
    """

    def __init__(self, classifier, cross_classifier=None):
        super().__init__()
        if classifier.ndim != 2:
            raise ValueError("Classifier weights must have shape [K, m]")
        if cross_classifier is not None and cross_classifier.shape != classifier.shape:
            raise ValueError("Cross classifier shape differs from the classifier")
        self.classifier = nn.Parameter(classifier.detach().clone())
        self.cross_classifier = nn.Parameter(
            (classifier if cross_classifier is None else cross_classifier).detach().clone()
        )

    @classmethod
    def from_source_checkpoint(cls, path, *, feature_dim, class_mapping=CLASSES):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("schema_version") != CHECKPOINT_VERSION:
            raise ValueError("Unsupported source checkpoint schema")
        if checkpoint.get("class_mapping") != list(class_mapping):
            raise ValueError("Checkpoint class mapping differs from adaptation configuration")
        prototypes = checkpoint.get("source_prototypes")
        if not isinstance(prototypes, torch.Tensor) or prototypes.shape != (
            len(class_mapping),
            feature_dim,
        ):
            raise ValueError("Checkpoint source prototype shape differs from adaptation configuration")
        return cls(prototypes)

    @property
    def source_prototypes(self):
        return self.classifier

    @property
    def target_prototypes(self):
        return self.classifier

    def _logits(self, features, weight):
        if features.ndim != 2 or features.shape[1] != weight.shape[1]:
            raise ValueError(f"Features must have shape [B, {weight.shape[1]}]")
        return features @ weight.T

    def logits(self, features):
        return self._logits(features, self.classifier)

    def source_logits(self, features):
        return self.logits(features)

    def target_logits(self, features):
        return self.logits(features)

    def cross_logits(self, features):
        return self._logits(features, self.cross_classifier)
