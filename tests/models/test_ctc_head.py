import bornil.runtime  # noqa: F401
import torch

from bornil.models.transformer import SignTransformer


def test_ctc_logits_shape_when_enabled():
    m = SignTransformer(input_dim=218, vocab_size=50, d_model=32, nhead=4,
                        num_encoder_layers=2, num_decoder_layers=2,
                        dim_feedforward=64, use_ctc=True)
    src = torch.randn(3, 17, 218)
    logits = m.ctc_logits(src)
    assert logits.shape == (3, 17, 50)


def test_ctc_logits_raises_when_disabled():
    m = SignTransformer(input_dim=218, vocab_size=50, d_model=32, nhead=4,
                        num_encoder_layers=2, num_decoder_layers=2,
                        dim_feedforward=64, use_ctc=False)
    try:
        m.ctc_logits(torch.randn(1, 5, 218))
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
