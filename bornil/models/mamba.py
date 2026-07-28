"""Pure-PyTorch Mamba-style selective-SSM encoder (WS-D D2, master plan §8.2).

The official `mamba-ssm` package ships custom CUDA kernels with no Windows
wheels; building it from source needs the CUDA Toolkit + MSVC, both absent
on this dev machine (verified during design). This module reimplements the
selective-scan recurrence in plain PyTorch: a Python loop over timesteps,
vectorized across batch/channel/state at each step. Simpler to get right
and test than a parallel associative scan; clips cap at 512 frames and this
is a research comparison, not a speed benchmark (see docs/logs/CHANGELOG.md
and the design doc's "Compute budget" for the actual measured cost).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from bornil.models.transformer import PositionalEncoding, causal_mask


def selective_scan(x: torch.Tensor, delta: torch.Tensor, A: torch.Tensor,
                   B: torch.Tensor, C: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """Sequential selective-SSM scan.

    x, delta: (B_batch, T, d_inner). A: (d_inner, d_state) (should be
    negative for a stable decay). B, C: (B_batch, T, d_state). D: (d_inner,).
    Returns y: (B_batch, T, d_inner).
    """
    B_batch, T, d_inner = x.shape
    d_state = A.shape[1]
    h = x.new_zeros(B_batch, d_inner, d_state)
    ys = []
    for t in range(T):
        dt = delta[:, t, :]
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))
        dBx = (
            dt.unsqueeze(-1)
            * B[:, t, :].unsqueeze(1)
            * x[:, t, :].unsqueeze(-1)
        )
        h = dA * h + dBx
        y_t = (
            (h * C[:, t, :].unsqueeze(1)).sum(-1)
            + D.unsqueeze(0) * x[:, t, :]
        )
        ys.append(y_t)
    return torch.stack(ys, dim=1)


class MambaBlock(nn.Module):
    """One gated selective-SSM block with an internal residual connection."""

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2,
                 conv_kernel: int = 4):
        super().__init__()
        d_inner = expand * d_model
        self.d_inner = d_inner
        self.d_state = d_state
        self.conv_kernel = conv_kernel
        self.norm = nn.RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_inner)
        self.conv = nn.Conv1d(
            d_inner,
            d_inner,
            kernel_size=conv_kernel,
            groups=d_inner,
            padding=conv_kernel - 1,
        )
        self.delta_proj = nn.Linear(d_inner, d_inner)
        self.B_proj = nn.Linear(d_inner, d_state)
        self.C_proj = nn.Linear(d_inner, d_state)
        A_init = (
            torch.arange(1, d_state + 1, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(d_inner, 1)
        )
        self.A_log = nn.Parameter(torch.log(A_init))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model)

    def forward(self, x: torch.Tensor,
                pad_mask: torch.Tensor = None) -> torch.Tensor:
        residual = x
        h = self.norm(x)
        xz = self.in_proj(h)
        x_in, z = xz.chunk(2, dim=-1)
        x_conv = (
            self.conv(x_in.transpose(1, 2))[..., :x_in.size(1)]
            .transpose(1, 2)
        )
        x_conv = torch.nn.functional.silu(x_conv)

        delta = torch.nn.functional.softplus(self.delta_proj(x_conv))
        if pad_mask is not None:
            delta = delta.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        Bt = self.B_proj(x_conv)
        Ct = self.C_proj(x_conv)
        A = -torch.exp(self.A_log)

        y = selective_scan(x_conv, delta, A, Bt, Ct, self.D)
        y = y * torch.nn.functional.silu(z)
        out = self.out_proj(y)
        return residual + out


class SignMamba(nn.Module):
    """Selective-SSM encoder + shared Transformer decoder (WS-D D2)."""

    def __init__(self, input_dim: int, vocab_size: int, d_model: int = 256,
                 nhead: int = 8, num_mamba_layers: int = 4,
                 d_state: int = 16, expand: int = 2, conv_kernel: int = 4,
                 num_decoder_layers: int = 4, dim_feedforward: int = 1024,
                 dropout: float = 0.1, pad_id: int = 0,
                 use_ctc: bool = False):
        super().__init__()
        if num_mamba_layers < 1:
            raise ValueError("num_mamba_layers must be >= 1")
        self.pad_id = pad_id
        self.d_model = d_model
        self.use_ctc = use_ctc

        self.input_proj = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList([
            MambaBlock(
                d_model,
                d_state=d_state,
                expand=expand,
                conv_kernel=conv_kernel,
            )
            for _ in range(num_mamba_layers)
        ])
        self.encoder_norm = nn.RMSNorm(d_model)

        self.tok_embed = nn.Embedding(
            vocab_size, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_decoder_layers)
        self.out = nn.Linear(d_model, vocab_size)
        self.ctc_head = nn.Linear(d_model, vocab_size) if use_ctc else None

    def src_padding_mask(self, src_lengths: torch.Tensor,
                         seq_len: int) -> torch.Tensor:
        ar = torch.arange(seq_len, device=src_lengths.device)[None, :]
        return ar >= src_lengths[:, None]

    def encode(self, src: torch.Tensor,
               src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        x = self.input_proj(src) * math.sqrt(self.d_model)
        for block in self.blocks:
            x = block(x, pad_mask=src_key_padding_mask)
        return self.encoder_norm(x)

    def _decode_memory(self, memory: torch.Tensor, tgt_in: torch.Tensor,
                       src_key_padding_mask: torch.Tensor = None,
                       tgt_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        t = self.pos(self.tok_embed(tgt_in) * math.sqrt(self.d_model))
        h = self.decoder(
            t, memory,
            tgt_mask=causal_mask(tgt_in.size(1)).to(tgt_in.device),
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.out(h)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor,
                src_key_padding_mask: torch.Tensor = None,
                tgt_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        memory = self.encode(src, src_key_padding_mask)
        return self._decode_memory(
            memory, tgt_in,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

    def ctc_logits(self, src: torch.Tensor,
                   src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        if self.ctc_head is None:
            raise RuntimeError(
                "SignMamba built with use_ctc=False has no CTC head")
        return self.ctc_head(self.encode(src, src_key_padding_mask))

    def forward_with_ctc(self, src: torch.Tensor, tgt_in: torch.Tensor,
                         src_key_padding_mask: torch.Tensor = None,
                         tgt_key_padding_mask: torch.Tensor = None):
        if self.ctc_head is None:
            raise RuntimeError(
                "SignMamba built with use_ctc=False has no CTC head")
        memory = self.encode(src, src_key_padding_mask)
        logits = self._decode_memory(
            memory, tgt_in,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return logits, self.ctc_head(memory)

    @torch.no_grad()
    def generate(self, src: torch.Tensor, bos_id: int, eos_id: int,
                 max_len: int = 128,
                 src_key_padding_mask: torch.Tensor = None,
                 repetition_stop: int = 0):
        from bornil.models.decode import greedy_decode

        self.eval()
        memory = self.encode(src, src_key_padding_mask)

        def step_fn(ys):
            ys = ys.to(src.device)
            return self._decode_memory(
                memory, ys, src_key_padding_mask=src_key_padding_mask)[:, -1]

        return greedy_decode(
            step_fn, bos_id, eos_id, src.size(0), src.device,
            max_len=max_len, repetition_stop=repetition_stop)

    @torch.no_grad()
    def beam_search(self, src: torch.Tensor, bos_id: int, eos_id: int,
                    beam_size: int = 5, max_len: int = 128,
                    length_penalty: float = 1.0,
                    src_key_padding_mask: torch.Tensor = None,
                    repetition_stop: int = 0):
        from bornil.models.decode import beam_decode

        self.eval()
        results = []
        for b in range(src.size(0)):
            s = src[b:b + 1]
            mask = (
                None
                if src_key_padding_mask is None
                else src_key_padding_mask[b:b + 1]
            )
            memory = self.encode(s, mask)

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
