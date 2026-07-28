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
