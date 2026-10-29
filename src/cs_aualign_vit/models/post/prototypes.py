"""Prototype classification heads used by POST adaptation."""

import torch
from torch import nn

from cs_aualign_vit.data.manifests import CLASSES
from cs_aualign_vit.models.source_pretraining import copy_target_prototypes
from cs_aualign_vit.training.source_pretraining import CHECKPOINT_VERSION


class PrototypeBanks(nn.Module):
    def __init__(self, source_prototypes, target_prototypes=None):
        super().__init__()
        if source_prototypes.ndim != 2:
            raise ValueError("Source prototypes must have shape [K, m]")
        if target_prototypes is not None and target_prototypes.shape != source_prototypes.shape:
            raise ValueError("Target prototype shape differs from source")
        self.source_prototypes = nn.Parameter(source_prototypes.detach().clone())
        self.target_prototypes = (
            copy_target_prototypes(self.source_prototypes)
            if target_prototypes is None
            else nn.Parameter(target_prototypes.detach().clone())
        )

    @classmethod
    def from_source_checkpoint(cls, path, *, feature_dim, class_mapping=CLASSES):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("schema_version") != CHECKPOINT_VERSION:
            raise ValueError("Unsupported source checkpoint schema")
        if checkpoint.get("class_mapping") != list(class_mapping):
            raise ValueError("Checkpoint class mapping differs from adaptation configuration")
        if checkpoint.get("feature_dim") != feature_dim:
            raise ValueError("Checkpoint feature dimension differs from adaptation configuration")
        prototypes = checkpoint.get("source_prototypes")
        if not isinstance(prototypes, torch.Tensor) or prototypes.shape != (
            len(class_mapping), feature_dim
        ):
            raise ValueError("Checkpoint source prototype shape differs from adaptation configuration")
        return cls(prototypes)

    def _logits(self, features, prototypes):
        if features.ndim != 2 or features.shape[1] != prototypes.shape[1]:
            raise ValueError(f"Features must have shape [B, {prototypes.shape[1]}]")
        return features @ prototypes.T

    def source_logits(self, features):
        return self._logits(features, self.source_prototypes)

    def target_logits(self, features):
        return self._logits(features, self.target_prototypes)
