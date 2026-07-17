"""Training checkpoint helpers for resumable long runs."""
from __future__ import annotations

import os

import torch


def save_training_state(path, epoch, model, optimizer, scheduler,
                        best_bleu, elapsed_min) -> None:
    """Persist full training state needed to resume after an epoch."""
    os.makedirs(os.path.dirname(os.fspath(path)), exist_ok=True)
    torch.save({
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "best_bleu": float(best_bleu),
        "elapsed_min": float(elapsed_min),
    }, path)


def load_training_state(path, model, optimizer, scheduler, device) -> dict:
    """Load a full training checkpoint and restore mutable training state."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    return {
        "epoch": int(ckpt["epoch"]),
        "best_bleu": float(ckpt["best_bleu"]),
        "elapsed_min": float(ckpt.get("elapsed_min", 0.0)),
    }
