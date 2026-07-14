"""Training and evaluation for the SLT baselines (WS-C).

`train_epoch` runs one teacher-forced epoch (shifted targets, causal + padding
masks). `evaluate` greedy-decodes each clip, detokenizes hypotheses and
references, and scores them with the locked metric protocol
(`bornil.benchmark.metrics`). Model-agnostic: any module exposing
`forward(src, tgt_in, src_key_padding_mask, tgt_key_padding_mask)`,
`src_padding_mask(lengths, S)`, and `generate(...)` works here.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from bornil.benchmark import metrics as M


def train_epoch(model, loader, optimizer, device: str = "cpu", pad_id: int = 0,
                clip_grad: float = 1.0, label_smoothing: float = 0.0,
                scheduler=None) -> float:
    """One teacher-forced epoch. Returns mean per-batch loss.

    `label_smoothing` softens the cross-entropy target (helps calibration);
    `scheduler` (a per-step LR scheduler) is stepped after each optimizer step.
    """
    model.to(device).train()
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=label_smoothing)
    total, nb = 0.0, 0
    for x, y, x_lengths, _ in loader:
        x, y = x.to(device), y.to(device)
        tgt_in, tgt_out = y[:, :-1], y[:, 1:]
        src_kpm = model.src_padding_mask(x_lengths.to(device), x.size(1))
        tgt_kpm = tgt_in == pad_id
        optimizer.zero_grad()
        logits = model(x, tgt_in, src_key_padding_mask=src_kpm, tgt_key_padding_mask=tgt_kpm)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        loss.backward()
        if clip_grad:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total += float(loss.item())
        nb += 1
    return total / max(nb, 1)


@torch.no_grad()
def evaluate(model, loader, tokenizer, device: str = "cpu", max_len: int = 128,
             beam_size: int = 1) -> Tuple[Dict[str, float], List[str], List[str]]:
    """Decode the loader (greedy if beam_size<=1, else beam search) and score
    with the locked metric protocol. Returns (metrics, hypotheses, references)."""
    model.to(device).eval()
    hyps: List[str] = []
    refs: List[str] = []
    for x, y, x_lengths, _ in loader:
        x = x.to(device)
        src_kpm = model.src_padding_mask(x_lengths.to(device), x.size(1))
        if beam_size > 1:
            gen = model.beam_search(x, bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id,
                                    beam_size=beam_size, max_len=max_len,
                                    src_key_padding_mask=src_kpm)
        else:
            gen = model.generate(x, bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id,
                                 max_len=max_len, src_key_padding_mask=src_kpm)
        for ids in gen:
            hyps.append(tokenizer.decode(ids))
        for row in y:
            refs.append(tokenizer.decode(row.tolist()))
    return M.corpus_metrics(hyps, refs), hyps, refs
