"""Masked keypoint modeling: reconstruction head + masked loss (WS-D D1)."""
from __future__ import annotations

import torch
import torch.nn as nn


class MKMHead(nn.Module):
    """Linear projection from encoder output back to per-frame keypoint space."""

    def __init__(self, d_model: int, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(d_model, input_dim)

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        return self.proj(memory)


def mkm_loss(pred: torch.Tensor, target: torch.Tensor,
             mask: torch.Tensor) -> torch.Tensor:
    """Masked-mean L1 reconstruction loss. `mask` True = position contributes."""
    if not mask.any():
        return pred.sum() * 0.0
    diff = (pred - target).abs()
    per_frame = diff.mean(dim=-1)          # (B, S)
    return per_frame[mask].mean()
