"""Tests for SignMamba (WS-D D2). Mirrors tests/models/test_stgcn.py."""
import torch

from bornil.models.mamba import SignMamba


def _tiny_model(vocab=20, use_ctc=False):
    torch.manual_seed(0)
    return SignMamba(
        input_dim=12,
        vocab_size=vocab,
        d_model=32,
        nhead=4,
        num_mamba_layers=2,
        d_state=8,
        expand=2,
        conv_kernel=4,
        num_decoder_layers=2,
        dim_feedforward=64,
        dropout=0.0,
        pad_id=0,
        use_ctc=use_ctc,
    )


def test_forward_output_shape():
    model = _tiny_model()
    src = torch.randn(2, 15, 12)
    tgt_in = torch.randint(1, 20, (2, 6))
    logits = model(src, tgt_in)
    assert logits.shape == (2, 6, 20)


def test_encode_output_shape():
    model = _tiny_model()
    src = torch.randn(2, 15, 12)
    memory = model.encode(src)
    assert memory.shape == (2, 15, 32)


def test_ctc_logits_requires_use_ctc():
    model = _tiny_model(use_ctc=False)
    src = torch.randn(1, 10, 12)
    try:
        model.ctc_logits(src)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_ctc_logits_shape_when_enabled():
    model = _tiny_model(use_ctc=True)
    src = torch.randn(2, 10, 12)
    logits = model.ctc_logits(src)
    assert logits.shape == (2, 10, 20)


def test_generate_returns_token_lists():
    model = _tiny_model()
    src = torch.randn(2, 10, 12)
    out = model.generate(src, bos_id=1, eos_id=2, max_len=5)
    assert len(out) == 2
    assert all(isinstance(o, list) for o in out)


def test_overfit_tiny():
    """The full encoder-decoder path must memorize a two-example batch."""
    torch.manual_seed(0)
    model = _tiny_model(vocab=15)
    src = torch.randn(2, 12, 12)
    tgt = torch.randint(1, 15, (2, 7))
    tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss()
    first_loss = None
    for _ in range(150):
        opt.zero_grad()
        logits = model(src, tgt_in)
        loss = loss_fn(logits.reshape(-1, 15), tgt_out.reshape(-1))
        if first_loss is None:
            first_loss = loss.item()
        loss.backward()
        opt.step()
    assert loss.item() < first_loss * 0.1
