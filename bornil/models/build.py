"""Single source of truth for reconstructing a model from a run config.

Training and every offline script (decode ablations, input ablations, scoring)
must build the model the *same* way, or `load_state_dict` fails on a checkpoint
that was trained with a head the rebuilder forgot to construct. Keeping the
arch dispatch here instead of duplicated per-script is what prevents that.

    model = build_model(cfg, vocab_size=tok.vocab_size, pad_id=tok.pad_id).to(device)
"""
from __future__ import annotations

import torch.nn as nn

from bornil.data.transforms import feature_dim

SUPPORTED_ARCHS = ("transformer", "bilstm", "stgcn")


def build_model(cfg: dict, vocab_size: int, pad_id: int) -> nn.Module:
    """Construct the model described by `cfg` (a loaded run config.json)."""
    m = cfg["model"]
    arch = m.get("arch", "transformer")
    input_dim = feature_dim(cfg["data"]["upper_body"])
    use_ctc = bool(m.get("ctc_head", False))

    if arch == "bilstm":
        from bornil.models.bilstm import SignBiLSTM
        # SignBiLSTM has no CTC head; a config asking for one would silently
        # train a model whose checkpoint cannot be rebuilt here.
        if use_ctc:
            raise ValueError("arch='bilstm' does not support ctc_head=true")
        return SignBiLSTM(
            input_dim=input_dim, vocab_size=vocab_size,
            d_model=m["d_model"], enc_layers=int(m.get("enc_layers", 2)),
            num_decoder_layers=m["num_decoder_layers"],
            dim_feedforward=m["dim_feedforward"], nhead=m["nhead"],
            dropout=m["dropout"], pad_id=pad_id,
        )

    if arch == "stgcn":
        from bornil.models.stgcn import SignSTGCN
        return SignSTGCN(
            input_dim=input_dim, vocab_size=vocab_size,
            d_model=m["d_model"], nhead=m["nhead"],
            stgcn_channels=int(m.get("stgcn_channels", 64)),
            stgcn_layers=int(m.get("stgcn_layers", 3)),
            temporal_kernel=int(m.get("temporal_kernel", 5)),
            num_decoder_layers=m["num_decoder_layers"],
            dim_feedforward=m["dim_feedforward"], dropout=m["dropout"],
            pad_id=pad_id, use_ctc=use_ctc,
        )

    if arch == "transformer":
        from bornil.models.transformer import SignTransformer
        return SignTransformer(
            input_dim=input_dim, vocab_size=vocab_size,
            d_model=m["d_model"], nhead=m["nhead"],
            num_encoder_layers=m["num_encoder_layers"],
            num_decoder_layers=m["num_decoder_layers"],
            dim_feedforward=m["dim_feedforward"], dropout=m["dropout"],
            pad_id=pad_id,
            temporal_conv_layers=int(m.get("temporal_conv_layers", 0)),
            temporal_conv_kernel=int(m.get("temporal_conv_kernel", 5)),
            use_ctc=use_ctc,
        )

    raise ValueError(f"unknown model.arch {arch!r}; expected one of {SUPPORTED_ARCHS}")
