"""ST-GCN skeleton encoder + Transformer decoder (WS-C, T1.1).

The kp-218 feature vector is converted to a homogeneous graph tensor by taking
xyz for 23 upper-body pose joints and both 21-joint hands, dropping the pose
visibility channel because hands do not have an equivalent channel. The graph
encoder returns one memory vector per frame, so it plugs into the same decoder,
generation, and optional CTC training contract as the other SLT encoders.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn

from bornil.models.transformer import PositionalEncoding, causal_mask

NUM_POSE_JOINTS = 23
NUM_HAND_JOINTS = 21
NUM_STGCN_JOINTS = NUM_POSE_JOINTS + 2 * NUM_HAND_JOINTS
KP218_DIM = 218


POSE_EDGES_0_22 = (
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15),
    (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16),
    (16, 18), (16, 20), (16, 22), (18, 20),
)

HAND_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)


def kp218_to_xyz65(src: torch.Tensor) -> torch.Tensor:
    """Convert kp-218 frames to `(B,T,65,3)` xyz graph features.

    Layout: pose 23x(x,y,z,visibility), left hand 21x(x,y,z), right hand
    21x(x,y,z). Visibility is pose-only, so ST-GCN drops it for a uniform
    channel count across all joints.
    """
    if src.size(-1) != KP218_DIM:
        raise ValueError(f"SignSTGCN expects kp-218 input, got {src.size(-1)}")
    pose = src[..., : NUM_POSE_JOINTS * 4].reshape(*src.shape[:-1], NUM_POSE_JOINTS, 4)
    left0 = NUM_POSE_JOINTS * 4
    left1 = left0 + NUM_HAND_JOINTS * 3
    left = src[..., left0:left1].reshape(*src.shape[:-1], NUM_HAND_JOINTS, 3)
    right = src[..., left1:].reshape(*src.shape[:-1], NUM_HAND_JOINTS, 3)
    return torch.cat([pose[..., :3], left, right], dim=-2)


def _add_undirected_edges(adj: torch.Tensor, edges: Iterable[tuple[int, int]]) -> None:
    for a, b in edges:
        adj[a, b] = 1.0
        adj[b, a] = 1.0


def build_stgcn_adjacency(normalize: bool = True) -> torch.Tensor:
    """Fixed MediaPipe upper-body+hands adjacency for the 65-joint tensor."""
    adj = torch.eye(NUM_STGCN_JOINTS, dtype=torch.float32)
    _add_undirected_edges(adj, POSE_EDGES_0_22)

    left_offset = NUM_POSE_JOINTS
    right_offset = NUM_POSE_JOINTS + NUM_HAND_JOINTS
    _add_undirected_edges(adj, ((a + left_offset, b + left_offset) for a, b in HAND_EDGES))
    _add_undirected_edges(adj, ((a + right_offset, b + right_offset) for a, b in HAND_EDGES))
    _add_undirected_edges(adj, ((15, left_offset), (16, right_offset)))

    if not normalize:
        return adj
    degree = adj.sum(dim=1).clamp_min(1.0)
    inv_sqrt = degree.pow(-0.5)
    return inv_sqrt[:, None] * adj * inv_sqrt[None, :]


class GraphConv(nn.Module):
    """One spatial graph convolution over joints for each frame."""

    def __init__(self, in_channels: int, out_channels: int, adj: torch.Tensor):
        super().__init__()
        self.register_buffer("adj", adj)
        self.proj = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        support = torch.einsum("vw,btwc->btvc", self.adj.to(dtype=x.dtype), x)
        return self.proj(support)


class STGCNBlock(nn.Module):
    """Spatial graph conv followed by temporal conv, with residual connection."""

    def __init__(self, in_channels: int, out_channels: int, adj: torch.Tensor,
                 temporal_kernel: int = 5, dropout: float = 0.1):
        super().__init__()
        if temporal_kernel < 1 or temporal_kernel % 2 == 0:
            raise ValueError("temporal_kernel must be a positive odd integer")
        self.gcn = GraphConv(in_channels, out_channels, adj)
        pad = temporal_kernel // 2
        self.temporal = nn.Conv2d(
            out_channels, out_channels, kernel_size=(temporal_kernel, 1),
            padding=(pad, 0),
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,V,C)
        res = x.permute(0, 3, 1, 2)
        res = self.residual(res)
        y = self.gcn(x).permute(0, 3, 1, 2)
        y = self.norm(self.temporal(y))
        y = self.drop(self.act(y + res))
        return y.permute(0, 2, 3, 1)


class SignSTGCN(nn.Module):
    def __init__(self, input_dim: int, vocab_size: int, d_model: int = 256,
                 nhead: int = 8, stgcn_channels: int = 128,
                 stgcn_layers: int = 3, temporal_kernel: int = 5,
                 num_decoder_layers: int = 4, dim_feedforward: int = 1024,
                 dropout: float = 0.1, pad_id: int = 0,
                 use_ctc: bool = False):
        super().__init__()
        if input_dim != KP218_DIM:
            raise ValueError(f"SignSTGCN expects input_dim=218, got {input_dim}")
        if stgcn_layers < 1:
            raise ValueError("stgcn_layers must be >= 1")
        self.pad_id = pad_id
        self.d_model = d_model
        self.use_ctc = use_ctc

        adj = build_stgcn_adjacency(normalize=True)
        channels = [3] + [int(stgcn_channels)] * int(stgcn_layers)
        self.blocks = nn.ModuleList([
            STGCNBlock(channels[i], channels[i + 1], adj, temporal_kernel, dropout)
            for i in range(stgcn_layers)
        ])
        self.memory_proj = nn.Linear(int(stgcn_channels), d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.pos = PositionalEncoding(d_model)

        self.tok_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_decoder_layers)
        self.out = nn.Linear(d_model, vocab_size)
        self.ctc_head = nn.Linear(d_model, vocab_size) if use_ctc else None

    def src_padding_mask(self, src_lengths: torch.Tensor, seq_len: int) -> torch.Tensor:
        ar = torch.arange(seq_len, device=src_lengths.device)[None, :]
        return ar >= src_lengths[:, None]

    def encode(self, src: torch.Tensor,
               src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        x = kp218_to_xyz65(src)
        if src_key_padding_mask is not None:
            x = x.masked_fill(src_key_padding_mask[:, :, None, None], 0.0)
        for block in self.blocks:
            x = block(x)
            if src_key_padding_mask is not None:
                x = x.masked_fill(src_key_padding_mask[:, :, None, None], 0.0)
        memory = self.memory_proj(x.mean(dim=2)) * math.sqrt(self.d_model)
        memory = self.memory_norm(self.pos(memory))
        if src_key_padding_mask is not None:
            memory = memory.masked_fill(src_key_padding_mask[:, :, None], 0.0)
        return memory

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
            raise RuntimeError("SignSTGCN built with use_ctc=False has no CTC head")
        return self.ctc_head(self.encode(src, src_key_padding_mask))

    def forward_with_ctc(self, src: torch.Tensor, tgt_in: torch.Tensor,
                         src_key_padding_mask: torch.Tensor = None,
                         tgt_key_padding_mask: torch.Tensor = None):
        if self.ctc_head is None:
            raise RuntimeError("SignSTGCN built with use_ctc=False has no CTC head")
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
            mask = None if src_key_padding_mask is None else src_key_padding_mask[b:b + 1]
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
