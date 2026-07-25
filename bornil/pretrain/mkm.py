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


from bornil.pretrain.masking import sample_span_mask


class MaskedSignEncoder(nn.Module):
    """Wraps a SignTransformer for masked-keypoint-modeling pretraining.

    Mask-token substitution happens at `model.project(src)` — in d_model
    space, before the TConv frontend (see SignTransformer.project's
    docstring for why). Only the encoder half of `model` is ever used here;
    `model`'s decoder/CTC head exist (build_model always constructs them)
    but receive no gradient during pretraining — only the extracted encoder
    submodules get saved (see bornil/pretrain/checkpoint.py, Task 5).
    """

    def __init__(self, model, mask_ratio: float = 0.6, min_span: int = 3,
                 max_span: int = 15):
        super().__init__()
        self.model = model
        self.mask_ratio = mask_ratio
        self.min_span = min_span
        self.max_span = max_span
        self.mask_token = nn.Parameter(torch.zeros(model.d_model))
        input_dim = model.input_proj.in_features
        self.head = MKMHead(model.d_model, input_dim)

    def forward(self, src: torch.Tensor, src_lengths: torch.Tensor,
                generator: torch.Generator = None):
        mask = sample_span_mask(src.size(1), src_lengths, self.mask_ratio,
                                self.min_span, self.max_span,
                                generator=generator).to(src.device)
        s = self.model.project(src)
        s = torch.where(mask.unsqueeze(-1), self.mask_token, s)
        s = self.model.apply_temporal(s)
        src_kpm = self.model.src_padding_mask(src_lengths, src.size(1))
        memory = self.model.transformer.encoder(s, src_key_padding_mask=src_kpm)
        pred = self.head(memory)
        return pred, mask

    def loss(self, src: torch.Tensor, src_lengths: torch.Tensor,
             generator: torch.Generator = None) -> torch.Tensor:
        pred, mask = self.forward(src, src_lengths, generator=generator)
        return mkm_loss(pred, src, mask)
