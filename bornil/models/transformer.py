"""Transformer encoder-decoder SLT baseline (WS-C, T0.2).

Gloss-free "Sign Language Transformers"-style model: project per-frame keypoints
to d_model, add sinusoidal positional encoding, run a Transformer encoder over
the clip, and autoregressively decode Bangla subword tokens. Sized from the data
contract (input_dim from `feature_dim`, vocab from the tokenizer) — never a
hard-coded literal.

Kept small and dependency-light (pure torch) so it prototypes on the RTX 3070
and passes the overfit-tiny gate on CPU. Attention decoder here; a CTC head and
larger encoders are separate model modules.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def has_consecutive_repeat(tokens, n: int) -> bool:
    """True when the last token has repeated `n` times consecutively."""
    if n <= 0 or len(tokens) < n:
        return False
    tail = tokens[-n:]
    return all(tok == tail[0] for tok in tail)


def causal_mask(size: int) -> torch.Tensor:
    """Additive (-inf above diagonal) mask blocking attention to future tokens."""
    return torch.triu(torch.full((size, size), float("-inf")), diagonal=1)


class PositionalEncoding(nn.Module):
    """Standard fixed sinusoidal positional encoding (batch_first)."""

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))     # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class SignTransformer(nn.Module):
    def __init__(self, input_dim: int, vocab_size: int, d_model: int = 256,
                 nhead: int = 8, num_encoder_layers: int = 4,
                 num_decoder_layers: int = 4, dim_feedforward: int = 1024,
                 dropout: float = 0.1, pad_id: int = 0,
                 temporal_conv_layers: int = 0,
                 temporal_conv_kernel: int = 5,
                 use_ctc: bool = False):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.temporal_frontend = self._make_temporal_frontend(
            d_model, temporal_conv_layers, temporal_conv_kernel, dropout
        )
        self.tok_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True,
        )
        self.out = nn.Linear(d_model, vocab_size)
        self.use_ctc = use_ctc
        self.ctc_head = nn.Linear(d_model, vocab_size) if use_ctc else None

    @staticmethod
    def _make_temporal_frontend(d_model: int, layers: int, kernel: int,
                                dropout: float) -> nn.Module:
        """Optional local-motion frontend over projected frame features."""
        if layers <= 0:
            return nn.Identity()
        if kernel < 1 or kernel % 2 == 0:
            raise ValueError("temporal_conv_kernel must be a positive odd integer")
        blocks = []
        pad = kernel // 2
        for _ in range(layers):
            blocks.extend([
                nn.Conv1d(d_model, d_model, kernel_size=kernel, padding=pad),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        return nn.Sequential(*blocks)

    def _project_src(self, src: torch.Tensor) -> torch.Tensor:
        s = self.input_proj(src) * math.sqrt(self.d_model)
        if not isinstance(self.temporal_frontend, nn.Identity):
            s = self.temporal_frontend(s.transpose(1, 2)).transpose(1, 2)
        return self.pos(s)

    def src_padding_mask(self, src_lengths: torch.Tensor, seq_len: int) -> torch.Tensor:
        """(B, S) bool mask, True at padded frame positions."""
        ar = torch.arange(seq_len, device=src_lengths.device)[None, :]
        return ar >= src_lengths[:, None]

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor,
                src_key_padding_mask: torch.Tensor = None,
                tgt_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """src: (B, S, input_dim) keypoints; tgt_in: (B, L) subword ids.

        Returns logits (B, L, vocab_size). Padding masks are optional (bool,
        True == ignore); the causal mask on the target is always applied.
        """
        s = self._project_src(src)
        t = self.pos(self.tok_embed(tgt_in) * math.sqrt(self.d_model))
        tgt_mask = causal_mask(tgt_in.size(1)).to(src.device)
        h = self.transformer(
            s, t,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.out(h)

    def _decode_memory(self, memory: torch.Tensor, tgt_in: torch.Tensor,
                       src_key_padding_mask: torch.Tensor = None,
                       tgt_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        t = self.pos(self.tok_embed(tgt_in) * math.sqrt(self.d_model))
        h = self.transformer.decoder(
            t, memory,
            tgt_mask=causal_mask(tgt_in.size(1)).to(tgt_in.device),
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.out(h)

    def encode(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """Run the encoder over a clip -> memory (B, S, d_model)."""
        s = self._project_src(src)
        return self.transformer.encoder(s, src_key_padding_mask=src_key_padding_mask)

    def ctc_logits(self, src: torch.Tensor,
                   src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """Per-frame vocab logits from the encoder for a CTC loss/alignment.

        Returns (B, S, vocab_size). Requires use_ctc=True at construction.
        """
        if self.ctc_head is None:
            raise RuntimeError("SignTransformer built with use_ctc=False has no CTC head")
        memory = self.encode(src, src_key_padding_mask)
        return self.ctc_head(memory)

    def forward_with_ctc(self, src: torch.Tensor, tgt_in: torch.Tensor,
                         src_key_padding_mask: torch.Tensor = None,
                         tgt_key_padding_mask: torch.Tensor = None):
        """Return attention-decoder logits and CTC logits from one encoder pass."""
        if self.ctc_head is None:
            raise RuntimeError("SignTransformer built with use_ctc=False has no CTC head")
        memory = self.encode(src, src_key_padding_mask)
        logits = self._decode_memory(
            memory, tgt_in,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return logits, self.ctc_head(memory)

    @torch.no_grad()
    def generate(self, src: torch.Tensor, bos_id: int, eos_id: int,
                 max_len: int = 128, src_key_padding_mask: torch.Tensor = None,
                 repetition_stop: int = 0):
        """Greedy autoregressive decode. Returns a list (len B) of id lists,
        each the generated tokens after BOS, including the terminal EOS."""
        self.eval()
        memory = self.encode(src, src_key_padding_mask)
        b = src.size(0)
        ys = torch.full((b, 1), bos_id, dtype=torch.long, device=src.device)
        finished = [False] * b
        outputs = [[] for _ in range(b)]
        for _ in range(max_len):
            t = self.pos(self.tok_embed(ys) * math.sqrt(self.d_model))
            h = self.transformer.decoder(
                t, memory,
                tgt_mask=causal_mask(ys.size(1)).to(src.device),
                memory_key_padding_mask=src_key_padding_mask,
            )
            nxt = self.out(h[:, -1]).argmax(-1)            # (B,)
            ys = torch.cat([ys, nxt[:, None]], dim=1)
            for i in range(b):
                if not finished[i]:
                    tok = int(nxt[i])
                    outputs[i].append(tok)
                    if tok == eos_id or has_consecutive_repeat(outputs[i], repetition_stop):
                        finished[i] = True
            if all(finished):
                break
        return outputs

    @torch.no_grad()
    def beam_search(self, src: torch.Tensor, bos_id: int, eos_id: int,
                    beam_size: int = 5, max_len: int = 128,
                    length_penalty: float = 1.0, src_key_padding_mask: torch.Tensor = None,
                    repetition_stop: int = 0):
        """Beam-search decode. Returns a list (len B) of id lists (tokens after
        BOS, including terminal EOS). Processes one example at a time for clarity."""
        self.eval()
        results = []
        for b in range(src.size(0)):
            s = src[b:b + 1]
            mask = None if src_key_padding_mask is None else src_key_padding_mask[b:b + 1]
            memory = self.encode(s, mask)
            beams = [([bos_id], 0.0)]
            finished = []
            for _ in range(max_len):
                candidates = []
                for tokens, score in beams:
                    if tokens[-1] == eos_id or has_consecutive_repeat(tokens[1:], repetition_stop):
                        finished.append((tokens, score))
                        continue
                    ys = torch.tensor([tokens], dtype=torch.long, device=src.device)
                    t = self.pos(self.tok_embed(ys) * math.sqrt(self.d_model))
                    h = self.transformer.decoder(
                        t, memory, tgt_mask=causal_mask(len(tokens)).to(src.device),
                        memory_key_padding_mask=mask)
                    logp = torch.log_softmax(self.out(h[:, -1]), dim=-1)[0]
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
            toks = best[0][1:]                       # drop BOS
            if eos_id in toks:
                toks = toks[:toks.index(eos_id) + 1]
            results.append(toks)
        return results
