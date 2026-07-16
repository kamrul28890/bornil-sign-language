"""BiLSTM encoder + Transformer decoder SLT baseline (WS-C, T0.1).

The classic sign-translation encoder: a bidirectional LSTM over projected
per-frame keypoints, whose outputs are consumed as memory by a standard
Transformer decoder. Reuses the shared decode module so greedy/beam behavior is
identical to the Transformer baseline.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from bornil.models.transformer import PositionalEncoding, causal_mask


class SignBiLSTM(nn.Module):
    def __init__(self, input_dim: int, vocab_size: int, d_model: int = 256,
                 enc_layers: int = 2, num_decoder_layers: int = 4,
                 dim_feedforward: int = 1024, nhead: int = 8,
                 dropout: float = 0.1, pad_id: int = 0):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.lstm = nn.LSTM(
            d_model, d_model // 2, num_layers=enc_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if enc_layers > 1 else 0.0,
        )
        self.tok_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_decoder_layers)
        self.out = nn.Linear(d_model, vocab_size)

    def src_padding_mask(self, src_lengths, seq_len):
        ar = torch.arange(seq_len, device=src_lengths.device)[None, :]
        return ar >= src_lengths[:, None]

    def encode(self, src, src_key_padding_mask=None):
        s = self.input_proj(src)
        mem, _ = self.lstm(s)
        return mem

    def _decode_memory(self, memory, tgt_in, src_key_padding_mask=None,
                       tgt_key_padding_mask=None):
        t = self.pos(self.tok_embed(tgt_in) * math.sqrt(self.d_model))
        h = self.decoder(
            t, memory, tgt_mask=causal_mask(tgt_in.size(1)).to(tgt_in.device),
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.out(h)

    def forward(self, src, tgt_in, src_key_padding_mask=None, tgt_key_padding_mask=None):
        memory = self.encode(src, src_key_padding_mask)
        return self._decode_memory(
            memory, tgt_in,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

    @torch.no_grad()
    def generate(self, src, bos_id, eos_id, max_len=128,
                 src_key_padding_mask=None, repetition_stop=0):
        from bornil.models.decode import greedy_decode

        self.eval()
        memory = self.encode(src, src_key_padding_mask)

        def step_fn(ys):
            ys = ys.to(src.device)
            return self._decode_memory(
                memory, ys, src_key_padding_mask=src_key_padding_mask)[:, -1]

        return greedy_decode(step_fn, bos_id, eos_id, src.size(0), src.device,
                             max_len=max_len, repetition_stop=repetition_stop)

    @torch.no_grad()
    def beam_search(self, src, bos_id, eos_id, beam_size=5, max_len=128,
                    length_penalty=1.0, src_key_padding_mask=None, repetition_stop=0):
        from bornil.models.decode import beam_decode

        self.eval()
        results = []
        for b in range(src.size(0)):
            mask = None if src_key_padding_mask is None else src_key_padding_mask[b:b + 1]
            memory = self.encode(src[b:b + 1], mask)

            def step_fn_single(ys, memory=memory, mask=mask):
                ys = ys.to(src.device)
                return self._decode_memory(
                    memory, ys, src_key_padding_mask=mask)[:, -1][0]

            results.append(beam_decode(
                step_fn_single, bos_id, eos_id, beam_size=beam_size,
                max_len=max_len, length_penalty=length_penalty,
                repetition_stop=repetition_stop,
            ))
        return results
