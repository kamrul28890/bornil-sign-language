"""Tests for the lightweight experiment-config system (plan §4.3).

Every run must be fully specified by one composed config whose resolved content
is hashed and stored with the run. Hydra isn't a dependency yet, so this is a
minimal yaml + deep-merge + stable-hash layer with the same guarantees.
"""
import pytest

from bornil import config as cfg


def test_load_save_roundtrip(tmp_path):
    data = {"model": {"name": "transformer", "layers": 6}, "seed": 42}
    p = tmp_path / "c.yaml"
    cfg.save_config(data, p)
    assert cfg.load_config(p) == data


def test_merge_overrides_nested_and_adds_keys():
    base = {"model": {"name": "bilstm", "hidden": 256}, "seed": 1}
    over = {"model": {"hidden": 512, "dropout": 0.1}, "lr": 3e-4}
    merged = cfg.merge_config(base, over)
    assert merged["model"] == {"name": "bilstm", "hidden": 512, "dropout": 0.1}
    assert merged["seed"] == 1
    assert merged["lr"] == 3e-4


def test_merge_does_not_mutate_inputs():
    base = {"a": {"b": 1}}
    over = {"a": {"c": 2}}
    cfg.merge_config(base, over)
    assert base == {"a": {"b": 1}}
    assert over == {"a": {"c": 2}}


def test_hash_is_deterministic():
    c = {"model": {"name": "x", "layers": 4}, "seed": 7}
    assert cfg.config_hash(c) == cfg.config_hash(c)


def test_hash_is_key_order_independent():
    a = {"seed": 7, "model": {"layers": 4, "name": "x"}}
    b = {"model": {"name": "x", "layers": 4}, "seed": 7}
    assert cfg.config_hash(a) == cfg.config_hash(b)


def test_hash_changes_when_value_changes():
    a = {"model": {"layers": 4}}
    b = {"model": {"layers": 5}}
    assert cfg.config_hash(a) != cfg.config_hash(b)


def test_hash_is_short_hex():
    h = cfg.config_hash({"a": 1})
    assert len(h) == 12
    int(h, 16)  # valid hex, raises if not
