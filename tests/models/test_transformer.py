"""Tests for the Transformer seq2seq SLT baseline (WS-C, T0.2).

The critical test is `test_overfit_tiny`: the plan's overfit-tiny gate, automated.
If the model + loss + masking are wired correctly it can memorize a couple of
(keypoint-sequence -> subword-sequence) pairs and drive teacher-forced loss to
~0. If it can't, the architecture is broken — catch it here, before any GPU/cloud
run on real data.
"""
import torch

from bornil.models.transformer import SignTransformer, causal_mask


def _tiny_model(vocab=20):
    torch.manual_seed(0)
    return SignTransformer(
        input_dim=218, vocab_size=vocab, d_model=64, nhead=4,
        num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=128,
        dropout=0.0, pad_id=0,
    )


def test_forward_output_shape():
    model = _tiny_model()
    src = torch.randn(2, 15, 218)
    tgt_in = torch.randint(1, 20, (2, 6))
    logits = model(src, tgt_in)
    assert logits.shape == (2, 6, 20)


def test_causal_mask_is_upper_triangular():
    m = causal_mask(3)
    # position i may not attend to positions > i  => True (blocked) above diagonal
    assert torch.isinf(m[0, 1]) and torch.isinf(m[0, 2])
    assert m[2, 0].item() == 0.0 and m[2, 1].item() == 0.0


def test_overfit_tiny():
    vocab, pad = 20, 0
    model = _tiny_model(vocab)
    # Two fixed examples: random keypoint clips -> fixed short target sequences.
    torch.manual_seed(1)
    src = torch.randn(2, 12, 218)
    y = torch.tensor([[2, 5, 7, 9, 3], [2, 8, 6, 4, 3]])   # [BOS ... EOS]
    tgt_in, tgt_out = y[:, :-1], y[:, 1:]

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad)
    model.train()
    last = None
    for _ in range(300):
        opt.zero_grad()
        logits = model(src, tgt_in)
        loss = loss_fn(logits.reshape(-1, vocab), tgt_out.reshape(-1))
        loss.backward()
        opt.step()
        last = loss.item()

    assert last < 0.05, f"overfit-tiny failed to converge: loss={last}"
    # Teacher-forced argmax should reproduce the targets exactly.
    model.eval()
    with torch.no_grad():
        pred = model(src, tgt_in).argmax(-1)
    assert torch.equal(pred, tgt_out), f"pred {pred.tolist()} != {tgt_out.tolist()}"


def test_padding_mask_from_lengths():
    model = _tiny_model()
    mask = model.src_padding_mask(torch.tensor([3, 5]), seq_len=5)
    # True == padded position (to be ignored by attention)
    assert mask[0].tolist() == [False, False, False, True, True]
    assert mask[1].tolist() == [False, False, False, False, False]
