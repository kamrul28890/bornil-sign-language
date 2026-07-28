"""Pure-PyTorch Mamba-style selective-SSM encoder (WS-D D2, master plan §8.2).

The official `mamba-ssm` package ships custom CUDA kernels with no Windows
wheels; building it from source needs the CUDA Toolkit + MSVC, both absent
on this dev machine (verified during design). This module reimplements the
selective-scan recurrence in plain PyTorch: a Python loop over timesteps,
vectorized across batch/channel/state at each step. Simpler to get right
and test than a parallel associative scan; clips cap at 512 frames and this
is a research comparison, not a speed benchmark (see docs/logs/CHANGELOG.md
and the design doc's "Compute budget" for the actual measured cost).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def selective_scan(x: torch.Tensor, delta: torch.Tensor, A: torch.Tensor,
                   B: torch.Tensor, C: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """Sequential selective-SSM scan.

    x, delta: (B_batch, T, d_inner). A: (d_inner, d_state) (should be
    negative for a stable decay). B, C: (B_batch, T, d_state). D: (d_inner,).
    Returns y: (B_batch, T, d_inner).
    """
    B_batch, T, d_inner = x.shape
    d_state = A.shape[1]
    h = x.new_zeros(B_batch, d_inner, d_state)
    ys = []
    for t in range(T):
        dt = delta[:, t, :]
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))
        dBx = (
            dt.unsqueeze(-1)
            * B[:, t, :].unsqueeze(1)
            * x[:, t, :].unsqueeze(-1)
        )
        h = dA * h + dBx
        y_t = (
            (h * C[:, t, :].unsqueeze(1)).sum(-1)
            + D.unsqueeze(0) * x[:, t, :]
        )
        ys.append(y_t)
    return torch.stack(ys, dim=1)


class MambaBlock(nn.Module):
    """One gated selective-SSM block with an internal residual connection."""

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2,
                 conv_kernel: int = 4):
        super().__init__()
        d_inner = expand * d_model
        self.d_inner = d_inner
        self.d_state = d_state
        self.conv_kernel = conv_kernel
        self.norm = nn.RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_inner)
        self.conv = nn.Conv1d(
            d_inner,
            d_inner,
            kernel_size=conv_kernel,
            groups=d_inner,
            padding=conv_kernel - 1,
        )
        self.delta_proj = nn.Linear(d_inner, d_inner)
        self.B_proj = nn.Linear(d_inner, d_state)
        self.C_proj = nn.Linear(d_inner, d_state)
        A_init = (
            torch.arange(1, d_state + 1, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(d_inner, 1)
        )
        self.A_log = nn.Parameter(torch.log(A_init))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model)

    def forward(self, x: torch.Tensor,
                pad_mask: torch.Tensor = None) -> torch.Tensor:
        residual = x
        h = self.norm(x)
        xz = self.in_proj(h)
        x_in, z = xz.chunk(2, dim=-1)
        x_conv = (
            self.conv(x_in.transpose(1, 2))[..., :x_in.size(1)]
            .transpose(1, 2)
        )
        x_conv = torch.nn.functional.silu(x_conv)

        delta = torch.nn.functional.softplus(self.delta_proj(x_conv))
        if pad_mask is not None:
            delta = delta.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        Bt = self.B_proj(x_conv)
        Ct = self.C_proj(x_conv)
        A = -torch.exp(self.A_log)

        y = selective_scan(x_conv, delta, A, Bt, Ct, self.D)
        y = y * torch.nn.functional.silu(z)
        out = self.out_proj(y)
        return residual + out
