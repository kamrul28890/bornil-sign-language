"""build_model must reconstruct a checkpoint-compatible model from a run config.

The regression this guards: `scripts/eval_checkpoint_decode.py` rebuilt models
without passing `use_ctc`, so every CTC checkpoint failed to load with
"Unexpected key(s) in state_dict: ctc_head.weight".
"""
import pytest
import torch

from bornil.models.build import build_model

VOCAB, PAD = 50, 0


def cfg_for(arch, **model_overrides):
    model = {
        "arch": arch, "d_model": 32, "nhead": 4,
        "num_encoder_layers": 2, "num_decoder_layers": 2,
        "dim_feedforward": 64, "dropout": 0.1,
    }
    model.update(model_overrides)
    return {"model": model, "data": {"upper_body": True}}


def roundtrip(cfg):
    """Build twice and move one's weights into the other, as offline scripts do."""
    trained = build_model(cfg, vocab_size=VOCAB, pad_id=PAD)
    rebuilt = build_model(cfg, vocab_size=VOCAB, pad_id=PAD)
    rebuilt.load_state_dict(trained.state_dict())
    return rebuilt


@pytest.mark.parametrize("arch", ["transformer", "stgcn"])
def test_ctc_checkpoint_roundtrips(arch):
    """A ctc_head=true checkpoint must load into a config-rebuilt model."""
    model = roundtrip(cfg_for(arch, ctc_head=True))
    assert model.ctc_head is not None


@pytest.mark.parametrize("arch", ["transformer", "bilstm", "stgcn"])
def test_plain_checkpoint_roundtrips(arch):
    roundtrip(cfg_for(arch))


def test_transformer_without_ctc_has_no_head():
    assert build_model(cfg_for("transformer"), VOCAB, PAD).ctc_head is None


def test_ctc_state_dict_does_not_load_into_non_ctc_model():
    """The exact failure mode: heads must not silently mismatch."""
    with_ctc = build_model(cfg_for("transformer", ctc_head=True), VOCAB, PAD)
    without = build_model(cfg_for("transformer"), VOCAB, PAD)
    with pytest.raises(RuntimeError, match="ctc_head"):
        without.load_state_dict(with_ctc.state_dict())


def test_temporal_conv_layers_are_rebuilt():
    """TConv layers are config-driven and were another silent-drop candidate."""
    cfg = cfg_for("transformer", temporal_conv_layers=2, temporal_conv_kernel=5)
    roundtrip(cfg)


def test_bilstm_rejects_ctc_head():
    with pytest.raises(ValueError, match="does not support ctc_head"):
        build_model(cfg_for("bilstm", ctc_head=True), VOCAB, PAD)


def test_unknown_arch_raises():
    with pytest.raises(ValueError, match="unknown model.arch"):
        build_model(cfg_for("not_a_real_arch"), VOCAB, PAD)


def test_build_model_mamba():
    from bornil.models.mamba import SignMamba

    cfg = {
        "model": {
            "arch": "mamba",
            "d_model": 16,
            "nhead": 2,
            "num_mamba_layers": 2,
            "d_state": 4,
            "expand": 2,
            "conv_kernel": 4,
            "num_decoder_layers": 1,
            "dim_feedforward": 32,
            "dropout": 0.0,
            "ctc_head": True,
        },
        "data": {"upper_body": True},
    }
    model = build_model(cfg, vocab_size=10, pad_id=0)
    assert isinstance(model, SignMamba)
    assert model.use_ctc is True


def test_forward_shape_after_rebuild():
    """A rebuilt model must actually run, not just load weights."""
    model = roundtrip(cfg_for("transformer", ctc_head=True)).eval()
    x = torch.randn(2, 16, 218)
    y = torch.randint(1, VOCAB, (2, 5))
    with torch.no_grad():
        out = model(x, y)
    logits = out[0] if isinstance(out, tuple) else out
    assert logits.shape[:2] == (2, 5)
