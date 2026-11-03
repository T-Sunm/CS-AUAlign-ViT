"""Adversarial domain alignment (Task 4.3): GRL plus a domain discriminator."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_grl):
        ctx.lambda_grl = lambda_grl
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_grl * grad_output, None


def grad_reverse(x, lambda_grl=1.0):
    """Identity forward; multiplies the gradient by ``-lambda_grl`` backward."""
    return _GradReverse.apply(x, float(lambda_grl))


class DomainDiscriminator(nn.Module):
    """``m -> hidden -> 1`` MLP producing raw domain logits of shape ``[B]``."""

    def __init__(self, input_dim=768, hidden_dim=256, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        return self.net(features).squeeze(-1)


@torch.no_grad()
def domain_accuracy(z_source, z_target):
    """Accuracy at logit threshold 0 with source=1 and target=0."""
    acc_s = (z_source > 0).float().mean().item()
    acc_t = (z_target <= 0).float().mean().item()
    return {"acc_s": acc_s, "acc_t": acc_t, "acc_balanced": (acc_s + acc_t) / 2}


class AdversarialAlignment(nn.Module):
    """``L_ad`` on pooled source/target features ``[B, m]``; the trainer applies gamma."""

    def __init__(self, input_dim=768, hidden_dim=256, dropout=0.5, lambda_grl=1.0):
        super().__init__()
        if type(input_dim) is not int or input_dim < 1:
            raise ValueError("input_dim must be a positive integer")
        if type(hidden_dim) is not int or hidden_dim < 1:
            raise ValueError("hidden_dim must be a positive integer")
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not math.isfinite(lambda_grl) or lambda_grl < 0:
            raise ValueError("lambda_grl must be finite and non-negative")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = float(dropout)
        self.lambda_grl = float(lambda_grl)
        self.discriminator = DomainDiscriminator(input_dim, hidden_dim, dropout)
        self.last_metrics = {}

    def config(self):
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "lambda_grl": self.lambda_grl,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def _check(self, features, name):
        if features.ndim != 2 or features.shape[0] < 1 or features.shape[1] != self.input_dim:
            raise ValueError(f"{name} must have shape [B, {self.input_dim}] with B >= 1")

    def forward(self, source_features, target_features):
        self._check(source_features, "source_features")
        self._check(target_features, "target_features")
        z_source = self.discriminator(grad_reverse(source_features, self.lambda_grl))
        z_target = self.discriminator(grad_reverse(target_features, self.lambda_grl))
        loss = F.binary_cross_entropy_with_logits(
            z_source, torch.ones_like(z_source)
        ) + F.binary_cross_entropy_with_logits(z_target, torch.zeros_like(z_target))
        self.last_metrics = domain_accuracy(z_source.detach(), z_target.detach())
        return loss
