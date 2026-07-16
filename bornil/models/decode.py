"""Encoder-agnostic autoregressive decoding (greedy + beam).

Both take a step function that maps the current token prefix to next-token
logits; the encoder-specific memory and masks live in the caller's closure.
This lets every SLT model share one tested decode path instead of
re-implementing it.
"""
from __future__ import annotations

import torch

from bornil.models.transformer import has_consecutive_repeat


@torch.no_grad()
def greedy_decode(step_fn, bos_id, eos_id, batch_size, device,
                  max_len: int = 128, repetition_stop: int = 0):
    ys = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
    finished = [False] * batch_size
    outputs = [[] for _ in range(batch_size)]
    for _ in range(max_len):
        nxt = step_fn(ys).argmax(-1)
        ys = torch.cat([ys, nxt[:, None]], dim=1)
        for i in range(batch_size):
            if not finished[i]:
                tok = int(nxt[i])
                outputs[i].append(tok)
                if tok == eos_id or has_consecutive_repeat(outputs[i], repetition_stop):
                    finished[i] = True
        if all(finished):
            break
    return outputs


@torch.no_grad()
def beam_decode(step_fn_single, bos_id, eos_id, beam_size: int = 5,
                max_len: int = 128, length_penalty: float = 1.0,
                repetition_stop: int = 0):
    beams = [([bos_id], 0.0)]
    finished = []
    for _ in range(max_len):
        candidates = []
        for tokens, score in beams:
            if tokens[-1] == eos_id or has_consecutive_repeat(tokens[1:], repetition_stop):
                finished.append((tokens, score))
                continue
            ys = torch.tensor([tokens], dtype=torch.long)
            logp = torch.log_softmax(step_fn_single(ys), dim=-1)
            topv, topi = logp.topk(beam_size)
            for v, i in zip(topv.tolist(), topi.tolist()):
                candidates.append((tokens + [i], score + v))
        if not candidates:
            break
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:beam_size]
        if all(tok[-1] == eos_id for tok, _ in beams):
            finished.extend(beams)
            break
    finished = finished or beams
    best = max(finished, key=lambda x: x[1] / (len(x[0]) ** length_penalty))
    toks = best[0][1:]
    if eos_id in toks:
        toks = toks[:toks.index(eos_id) + 1]
    return toks
