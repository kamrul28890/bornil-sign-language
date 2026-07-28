import torch

from bornil.models.mamba import MambaBlock


def test_output_shape():
    block = MambaBlock(d_model=32, d_state=8, expand=2, conv_kernel=4)
    x = torch.randn(3, 20, 32)
    out = block(x)
    assert out.shape == (3, 20, 32)


def test_padding_never_affects_real_positions():
    """Padded-tail values must not affect output at non-padded positions."""
    torch.manual_seed(0)
    block = MambaBlock(d_model=16, d_state=8, expand=2, conv_kernel=4)
    block.eval()

    x = torch.randn(1, 20, 16)
    real_len = 12
    x_a = x.clone()
    x_b = x.clone()
    x_b[:, real_len:] = torch.randn(1, 20 - real_len, 16)

    mask = torch.zeros(1, 20, dtype=torch.bool)
    mask[:, real_len:] = True

    out_a = block(x_a, pad_mask=mask)
    out_b = block(x_b, pad_mask=mask)

    assert torch.allclose(out_a[:, :real_len], out_b[:, :real_len], atol=1e-5)


def test_residual_present():
    """A fully zeroed branch must still return the residual input."""
    torch.manual_seed(1)
    block = MambaBlock(d_model=8, d_state=4, expand=2, conv_kernel=4)
    with torch.no_grad():
        for p in block.parameters():
            p.zero_()
    x = torch.randn(2, 10, 8)
    out = block(x)
    assert torch.allclose(out, x, atol=1e-5)
