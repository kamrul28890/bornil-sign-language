"""SignDataset + collate (WS-A): packed features -> model-ready batches.

Slices each clip from the packed feature array to its true frame count, applies
the load-time transform (`prepare_features`: normalize + upper-body select), and
pairs it with its subword-id target. `make_collate_fn` pads variable-length
clips (with zeros) and targets (with pad_id) per batch and returns the lengths
needed for masking / CTC / packing.

The feature array is anything indexable as `features[i, :T]` — an in-memory
ndarray in tests, or `np.load(..., mmap_mode="r")` over the packed memmap in
training (paged from disk, low RAM).
"""
from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from bornil.data.transforms import prepare_features

Batch = Tuple[torch.Tensor, torch.Tensor]


class SignDataset(Dataset):
    def __init__(self, features, lengths, pack_idx: Sequence[int],
                 targets: Sequence[Sequence[int]],
                 normalize: bool = True, upper_body: bool = True):
        if not (len(pack_idx) == len(targets)):
            raise ValueError("pack_idx and targets must have equal length")
        self.features = features
        self.lengths = lengths
        self.pack_idx = list(pack_idx)
        self.targets = [list(t) for t in targets]
        self.normalize = normalize
        self.upper_body = upper_body

    def __len__(self) -> int:
        return len(self.pack_idx)

    def item_lengths(self):
        """Per-item true frame count (for length-bucketed batching)."""
        return [int(self.lengths[int(p)]) for p in self.pack_idx]

    def __getitem__(self, i: int) -> Batch:
        idx = int(self.pack_idx[i])
        t = int(self.lengths[idx])
        raw = np.asarray(self.features[idx, :t])              # (T, 258)
        x = prepare_features(raw, normalize=self.normalize, upper_body=self.upper_body)
        x = torch.from_numpy(np.ascontiguousarray(x)).float()
        y = torch.tensor(self.targets[i], dtype=torch.long)
        return x, y


def make_collate_fn(pad_id: int) -> Callable[[List[Batch]], Tuple[torch.Tensor, ...]]:
    """Build a collate_fn that pads inputs with 0.0 and targets with `pad_id`."""
    def collate(batch: List[Batch]):
        xs, ys = zip(*batch)
        x_lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
        y_lengths = torch.tensor([y.shape[0] for y in ys], dtype=torch.long)
        xs_padded = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0.0)
        ys_padded = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=pad_id)
        return xs_padded, ys_padded, x_lengths, y_lengths
    return collate
