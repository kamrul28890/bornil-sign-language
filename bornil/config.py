"""Lightweight experiment configuration (plan §4.3).

A run is fully specified by one composed config dict. We support loading/saving
YAML, deep-merging a base config with experiment overrides, and computing a
stable, key-order-independent hash so every run can be tagged and reproduced.

Kept dependency-free (stdlib + PyYAML) on purpose: the dev machine has no Hydra,
and the merge/hash contract here is all the training loop needs. If we adopt
Hydra later, these helpers stay valid (Hydra resolves to a dict we can hash).
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import yaml

_HASH_LEN = 12


def load_config(path) -> Dict[str, Any]:
    """Load a YAML config file into a plain dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_config(cfg: Dict[str, Any], path) -> None:
    """Write a config dict to YAML (parents created as needed)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=True, allow_unicode=True)


def merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge `override` onto `base` without mutating either.

    Nested dicts are merged recursively; scalar/list values in `override` replace
    those in `base`; keys only in one side are kept.
    """
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = merge_config(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def config_hash(cfg: Dict[str, Any]) -> str:
    """Stable, key-order-independent short hex hash of a config.

    Canonicalizes via JSON with sorted keys so two logically equal configs hash
    identically regardless of insertion order.
    """
    canonical = json.dumps(cfg, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:_HASH_LEN]
