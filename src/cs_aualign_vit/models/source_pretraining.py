"""Source-only Swin path and learnable expression prototypes."""

import torch
from torch import nn


class SourcePrototypeClassifier(nn.Module):
    def __init__(self, backbone, feature_dim, num_classes=7):
        super().__init__()
        if feature_dim <= 0 or num_classes <= 0:
            raise ValueError("feature_dim and num_classes must be positive")
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.source_prototypes = nn.Parameter(torch.empty(num_classes, feature_dim))
        nn.init.trunc_normal_(self.source_prototypes, std=0.02)

    def forward(self, images):
        features = self.backbone(images)
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Backbone must return [B, {self.feature_dim}], got {tuple(features.shape)}"
            )
        return features, features @ self.source_prototypes.T


def build_swin_t(weights="IMAGENET1K_V1", num_classes=7):
    """Build torchvision Swin-Tiny with its classification head removed."""
    from torchvision.models import Swin_T_Weights, swin_t

    if weights not in {None, "IMAGENET1K_V1"}:
        raise ValueError("weights must be IMAGENET1K_V1 or null")
    backbone = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1 if weights else None)
    feature_dim = backbone.head.in_features
    backbone.head = nn.Identity()
    return SourcePrototypeClassifier(backbone, feature_dim, num_classes)


def copy_target_prototypes(source_prototypes):
    """Copy values into an independently trainable target bank."""
    return nn.Parameter(source_prototypes.detach().clone())
