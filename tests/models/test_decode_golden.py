import bornil.runtime  # noqa: F401
import torch

from bornil.models.transformer import SignTransformer


def _model():
    torch.manual_seed(0)
    return SignTransformer(input_dim=218, vocab_size=30, d_model=32, nhead=4,
                           num_encoder_layers=2, num_decoder_layers=2,
                           dim_feedforward=64)


def test_greedy_matches_reference_sequence():
    m = _model()
    src = torch.Generator().manual_seed(1)
    x = torch.randn(2, 12, 218, generator=src)
    out = m.generate(x, bos_id=1, eos_id=2, max_len=15)
    assert isinstance(out, list) and len(out) == 2
    out2 = m.generate(x, bos_id=1, eos_id=2, max_len=15)
    assert out == out2


def test_beam_matches_reference_sequence():
    m = _model()
    x = torch.randn(2, 12, 218, generator=torch.Generator().manual_seed(1))
    a = m.beam_search(x, bos_id=1, eos_id=2, beam_size=3, max_len=15)
    b = m.beam_search(x, bos_id=1, eos_id=2, beam_size=3, max_len=15)
    assert a == b and len(a) == 2
