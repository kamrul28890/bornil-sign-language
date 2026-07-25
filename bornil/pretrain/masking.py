"""Span masking for masked keypoint modeling (WS-D D1).

Masks contiguous temporal spans rather than i.i.d. frames — a masked span
can't be filled in by simple local interpolation the way an isolated masked
frame could, matching VideoMAE/SkeletonMAE practice. Never masks padded
positions (index >= that row's true length).
"""
from __future__ import annotations

import torch


def sample_span_mask(seq_len: int, lengths: torch.Tensor, ratio: float,
                     min_span: int = 3, max_span: int = 15,
                     generator: torch.Generator = None) -> torch.Tensor:
    """Bool mask (B, S), True = masked. Greedily samples spans per row until
    `ratio` of that row's valid (non-padded) length is masked, trimming the
    final span so it never crosses `lengths[b]`."""
    if not (0.0 < ratio < 1.0):
        raise ValueError(f"ratio must be in (0, 1), got {ratio}")
    if min_span < 1 or max_span < min_span:
        raise ValueError("require 1 <= min_span <= max_span")
    B = lengths.shape[0]
    mask = torch.zeros(B, seq_len, dtype=torch.bool)
    for b in range(B):
        valid = int(lengths[b].item())
        if valid <= 0:
            continue
        target = int(round(ratio * valid))
        if target <= 0:
            continue
        masked = 0
        guard = 0
        touching_rejections = 0
        while masked < target and guard < 10_000:
            guard += 1
            span = int(torch.randint(min_span, max_span + 1, (1,),
                                     generator=generator).item())
            span = min(span, valid)
            start = int(torch.randint(0, valid - span + 1, (1,),
                                      generator=generator).item())
            end = start + span
            touches_existing = (
                mask[b, start:end].any()
                or (start > 0 and mask[b, start - 1])
                or (end < valid and mask[b, end])
            )
            if touches_existing and touching_rejections < 100:
                touching_rejections += 1
                continue
            touching_rejections = 0
            newly = (~mask[b, start:end]).sum().item()
            if newly == 0:
                continue
            if masked + newly > target:
                # trim from the end of this span so we don't overshoot
                keep = target - masked
                mask[b, start:start + keep] = True
                masked += keep
                break
            mask[b, start:end] = True
            masked += newly
    return mask
