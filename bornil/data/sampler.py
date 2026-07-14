"""Length-bucketed batch sampler (WS-C training efficiency).

Keypoint clips vary a lot in length (median ~160 frames, up to the 512 ceiling).
A random batch that mixes short and long clips pads everything to the longest one
and wastes compute. This sampler groups clips of similar length into each batch
while staying stochastic across epochs:

  each epoch -> shuffle indices -> cut into "megabatches" -> sort each megabatch
  by length -> split into batches -> shuffle batch order.

The megabatch step preserves randomness (a clip lands in a different neighborhood
each epoch); sorting within it keeps each batch's lengths close.
"""
from __future__ import annotations

from typing import Iterator, List, Sequence

import numpy as np
from torch.utils.data import Sampler


class LengthBucketBatchSampler(Sampler):
    def __init__(self, lengths: Sequence[int], batch_size: int, shuffle: bool = True,
                 seed: int = 0, bucket_multiplier: int = 20, drop_last: bool = False):
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.bucket_multiplier = max(1, bucket_multiplier)
        self.drop_last = drop_last
        self.epoch = 0

    def __len__(self) -> int:
        n = len(self.lengths)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[List[int]]:
        n = len(self.lengths)
        idx = list(range(n))
        rng = np.random.default_rng(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(idx)
        self.epoch += 1

        mb = self.batch_size * self.bucket_multiplier
        batches: List[List[int]] = []
        for i in range(0, n, mb):
            chunk = idx[i:i + mb]
            chunk.sort(key=lambda j: self.lengths[j])
            for b in range(0, len(chunk), self.batch_size):
                batch = chunk[b:b + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch)

        if self.shuffle:
            for o in rng.permutation(len(batches)):
                yield batches[o]
        else:
            for batch in batches:
                yield batch
