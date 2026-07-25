import torch

from bornil.models.transformer import SignTransformer
from bornil.pretrain.mkm import MaskedSignEncoder


def _tiny_model():
    torch.manual_seed(0)
    return SignTransformer(
        input_dim=12, vocab_size=10, d_model=32, nhead=4,
        num_encoder_layers=2, num_decoder_layers=1, dim_feedforward=64,
        dropout=0.0, pad_id=0, temporal_conv_layers=2, temporal_conv_kernel=5,
    )


def test_forward_output_shapes():
    enc = MaskedSignEncoder(_tiny_model(), mask_ratio=0.5, min_span=2, max_span=4)
    src = torch.randn(3, 20, 12)
    lengths = torch.tensor([20, 15, 20])
    pred, mask = enc(src, lengths, generator=torch.Generator().manual_seed(0))
    assert pred.shape == (3, 20, 12)
    assert mask.shape == (3, 20)
    assert not mask[1, 15:].any()   # padded region never masked


def test_loss_is_finite_and_positive_before_training():
    torch.manual_seed(0)
    enc = MaskedSignEncoder(_tiny_model(), mask_ratio=0.5, min_span=2, max_span=4)
    src = torch.randn(2, 20, 12)
    lengths = torch.tensor([20, 20])
    loss = enc.loss(src, lengths, generator=torch.Generator().manual_seed(0))
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_overfit_tiny():
    """Repeated tiny batch: masked-reconstruction loss must drop sharply.
    This is this repo's standing overfit-tiny gate (plan §4.6) — proves the
    masking -> encode -> reconstruct path is wired correctly before any real
    training run."""
    torch.manual_seed(0)
    enc = MaskedSignEncoder(_tiny_model(), mask_ratio=0.5, min_span=2, max_span=4)
    src = torch.randn(2, 20, 12)
    lengths = torch.tensor([20, 20])
    opt = torch.optim.Adam(enc.parameters(), lr=1e-2)
    first_loss = None
    for step in range(600):
        opt.zero_grad()
        loss = enc.loss(src, lengths)
        if first_loss is None:
            first_loss = loss.item()
        loss.backward()
        opt.step()
    assert loss.item() < first_loss * 0.1
