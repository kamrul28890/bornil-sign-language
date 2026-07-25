import os

import torch

from bornil.models.transformer import SignTransformer
from bornil.pretrain.checkpoint import (encoder_state_dict, load_encoder_checkpoint,
                                        save_encoder_checkpoint)


def _tiny_model(seed):
    torch.manual_seed(seed)
    return SignTransformer(
        input_dim=12, vocab_size=10, d_model=32, nhead=4,
        num_encoder_layers=2, num_decoder_layers=1, dim_feedforward=64,
        dropout=0.0, pad_id=0, temporal_conv_layers=2, temporal_conv_kernel=5,
    )


def test_encoder_state_dict_excludes_decoder_and_heads():
    model = _tiny_model(0)
    sd = encoder_state_dict(model)
    assert any(k.startswith("input_proj.") for k in sd)
    assert any(k.startswith("transformer.encoder.") for k in sd)
    assert not any(k.startswith("transformer.decoder.") for k in sd)
    assert not any(k.startswith("tok_embed.") for k in sd)
    assert not any(k.startswith("out.") for k in sd)


def test_save_and_load_roundtrip_changes_weights(tmp_path):
    src_model = _tiny_model(0)
    dst_model = _tiny_model(1)   # different seed -> different weights
    path = os.path.join(tmp_path, "encoder_pretrain.pt")
    save_encoder_checkpoint(path, src_model)

    before = dst_model.input_proj.weight.clone()
    loaded_keys = load_encoder_checkpoint(dst_model, path, device="cpu")
    after = dst_model.input_proj.weight

    assert not torch.allclose(before, after)
    assert torch.allclose(after, src_model.input_proj.weight)
    assert any(k.startswith("input_proj.") for k in loaded_keys)


def test_load_raises_on_empty_checkpoint(tmp_path):
    path = os.path.join(tmp_path, "empty.pt")
    torch.save({}, path)
    model = _tiny_model(0)
    try:
        load_encoder_checkpoint(model, path, device="cpu")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_decoder_and_heads_unaffected_by_load(tmp_path):
    """This is the property scripts/train_sample.py's --init-encoder flag
    (Task 8) relies on: warm-starting the encoder must never touch the
    decoder or CTC head, which stay randomly initialized for finetuning."""
    src_model = _tiny_model(0)
    dst_model = _tiny_model(1)
    path = os.path.join(tmp_path, "encoder_pretrain.pt")
    save_encoder_checkpoint(path, src_model)

    decoder_before = [p.clone() for p in dst_model.transformer.decoder.parameters()]
    out_before = dst_model.out.weight.clone()
    load_encoder_checkpoint(dst_model, path, device="cpu")
    decoder_after = list(dst_model.transformer.decoder.parameters())

    assert all(torch.equal(b, a) for b, a in zip(decoder_before, decoder_after))
    assert torch.equal(out_before, dst_model.out.weight)
