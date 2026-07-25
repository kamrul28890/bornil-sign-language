"""Encoder-only checkpoints for masked-keypoint-modeling pretraining.

Only the transferable encoder submodules are saved/loaded — not the mask
token, MKM head, or the (untouched, randomly-initialized) decoder/CTC head.
"""
from __future__ import annotations

import os
from typing import Set

import torch

ENCODER_KEY_PREFIXES = ("input_proj.", "temporal_frontend.", "transformer.encoder.")


def encoder_state_dict(model) -> dict:
    return {k: v for k, v in model.state_dict().items()
            if k.startswith(ENCODER_KEY_PREFIXES)}


def save_encoder_checkpoint(path, model) -> None:
    os.makedirs(os.path.dirname(os.fspath(path)) or ".", exist_ok=True)
    torch.save(encoder_state_dict(model), path)


def load_encoder_checkpoint(model, path, device) -> Set[str]:
    sd = torch.load(path, map_location=device, weights_only=True)
    result = model.load_state_dict(sd, strict=False)
    loaded = set(sd.keys()) - set(result.unexpected_keys)
    if not loaded:
        raise ValueError(f"no matching encoder keys loaded from {path!r} "
                         f"(missing={result.missing_keys}, "
                         f"unexpected={result.unexpected_keys})")
    return loaded
