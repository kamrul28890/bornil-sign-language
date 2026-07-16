import bornil.runtime  # noqa: F401
import torch
from torch.utils.data import DataLoader

from bornil.data.dataset import SignDataset, make_collate_fn
from bornil.models.bilstm import SignBiLSTM
from bornil.train.loop import train_epoch


def test_shapes():
    m = SignBiLSTM(input_dim=218, vocab_size=40, d_model=32, enc_layers=2,
                   num_decoder_layers=2, dim_feedforward=64)
    x = torch.randn(3, 11, 218)
    y = torch.randint(3, 40, (3, 6))
    logits = m(x, y, src_key_padding_mask=m.src_padding_mask(torch.tensor([11, 9, 7]), 11))
    assert logits.shape == (3, 6, 40)


def test_overfit_tiny():
    torch.manual_seed(0)
    N, T, D, V = 8, 16, 218, 40
    feats = torch.randn(N, T, D).numpy()
    targets = [[1, 5, 6, 7, 2] for _ in range(N)]
    ds = SignDataset(feats, [T] * N, list(range(N)), targets,
                     normalize=False, upper_body=False)
    loader = DataLoader(ds, batch_size=4, collate_fn=make_collate_fn(0))
    m = SignBiLSTM(input_dim=D, vocab_size=V, d_model=32, enc_layers=2,
                   num_decoder_layers=2, dim_feedforward=64)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    first = train_epoch(m, loader, opt, pad_id=0, eos_id=2)
    for _ in range(25):
        last = train_epoch(m, loader, opt, pad_id=0, eos_id=2)
    assert last < first * 0.5
