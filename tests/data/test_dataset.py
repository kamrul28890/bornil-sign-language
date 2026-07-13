"""Tests for SignDataset + collate (WS-A), the training data contract.

Validated on a synthetic packed-feature array (no real dataset needed): the
dataset slices each clip to its true length, applies prepare_features, and pairs
it with subword targets; collate pads variable-length clips and targets.
"""
import numpy as np
import torch

from bornil.data.dataset import SignDataset, make_collate_fn


def _synthetic():
    rng = np.random.default_rng(0)
    N, MAXF, DIM = 4, 128, 258
    feats = rng.standard_normal((N, MAXF, DIM)).astype(np.float32)
    lengths = np.array([10, 20, 5, 128], dtype=np.int32)
    pack_idx = [0, 1, 2, 3]
    targets = [[2, 5, 6, 3], [2, 7, 3], [2, 8, 9, 10, 3], [2, 4, 3]]
    return feats, lengths, pack_idx, targets


def test_len():
    feats, lengths, pack_idx, targets = _synthetic()
    ds = SignDataset(feats, lengths, pack_idx, targets)
    assert len(ds) == 4


def test_getitem_shapes_and_types():
    feats, lengths, pack_idx, targets = _synthetic()
    ds = SignDataset(feats, lengths, pack_idx, targets)
    x, y = ds[1]
    assert x.shape == (20, 218)          # length[1]=20, upper-body dim 218
    assert x.dtype == torch.float32
    assert y.tolist() == [2, 7, 3]
    assert y.dtype == torch.long


def test_getitem_raw_dim_when_no_selection():
    feats, lengths, pack_idx, targets = _synthetic()
    ds = SignDataset(feats, lengths, pack_idx, targets, normalize=False, upper_body=False)
    x, _ = ds[0]
    assert x.shape == (10, 258)


def test_collate_pads_inputs_and_targets():
    feats, lengths, pack_idx, targets = _synthetic()
    ds = SignDataset(feats, lengths, pack_idx, targets)
    collate = make_collate_fn(pad_id=0)
    batch = [ds[2], ds[1]]               # lengths 5 and 20; targets len 5 and 3
    xs, ys, x_len, y_len = collate(batch)
    assert xs.shape == (2, 20, 218)      # padded to max clip length in batch
    assert ys.shape == (2, 5)            # padded to max target length in batch
    assert x_len.tolist() == [5, 20]
    assert y_len.tolist() == [5, 3]
    # padding is zeros for inputs, pad_id for targets
    assert torch.all(xs[0, 5:] == 0.0)
    assert ys[1, 3:].tolist() == [0, 0]
