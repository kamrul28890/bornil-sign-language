"""Learning-rate schedules (WS-C training).

Noam-style warmup: the LR multiplier ramps linearly from ~0 to 1.0 over
`warmup_steps` optimizer steps, then decays as inverse square root. Plug into
torch.optim.lr_scheduler.LambdaLR and step it once per optimizer step.
"""
from __future__ import annotations

from typing import Callable


def noam_lambda(warmup_steps: int) -> Callable[[int], float]:
    """Return a step -> LR-multiplier function (peaks at 1.0 at warmup_steps)."""
    w = max(1, warmup_steps)

    def fn(step: int) -> float:
        s = step + 1                      # LambdaLR passes 0-indexed step
        return min(s / w, (w / s) ** 0.5)

    return fn
