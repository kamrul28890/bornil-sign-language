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
                 dropout: float = 0.1, pad_id: int = 0):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
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
        s = self.pos(self.input_proj(src) * math.sqrt(self.d_model))
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

    def encode(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """Run the encoder over a clip -> memory (B, S, d_model)."""
        s = self.pos(self.input_proj(src) * math.sqrt(self.d_model))
        return self.transformer.encoder(s, src_key_padding_mask=src_key_padding_mask)

    @torch.no_grad()
    def generate(self, src: torch.Tensor, bos_id: int, eos_id: int,
                 max_len: int = 128, src_key_padding_mask: torch.Tensor = None):
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
                    if tok == eos_id:
                        finished[i] = True
            if all(finished):
                break
        return outputs
