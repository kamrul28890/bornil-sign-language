import bornil.runtime  # noqa: F401
import torch
from torch.utils.data import DataLoader

from bornil.data.dataset import SignDataset, make_collate_fn
from bornil.models.transformer import SignTransformer
from bornil.train.loop import ctc_targets_from_padded, train_epoch


def test_ctc_targets_strip_bos_eos_pad():
    y = torch.tensor([[1, 5, 6, 2, 0], [1, 7, 2, 0, 0]])
    y_lengths = torch.tensor([4, 3])
    targets, tlens = ctc_targets_from_padded(y, y_lengths, bos_id=1, eos_id=2, pad_id=0)
    assert targets.tolist() == [5, 6, 7]
    assert tlens.tolist() == [2, 1]


def test_joint_ctc_training_step_runs_and_reduces_loss():
    torch.manual_seed(0)
    N, T, D, V = 8, 20, 218, 40
    feats = torch.randn(N, T, D).numpy()
    lengths = [T] * N
    targets = [[1, 5, 6, 7, 2] for _ in range(N)]
    ds = SignDataset(feats, lengths, list(range(N)), targets,
                     normalize=False, upper_body=False)
    loader = DataLoader(ds, batch_size=4, collate_fn=make_collate_fn(0))
    model = SignTransformer(input_dim=D, vocab_size=V, d_model=32, nhead=4,
                            num_encoder_layers=2, num_decoder_layers=2,
                            dim_feedforward=64, use_ctc=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    first = train_epoch(model, loader, opt, pad_id=0, eos_id=2, bos_id=1,
                        ctc_weight=0.3)
    for _ in range(15):
        last = train_epoch(model, loader, opt, pad_id=0, eos_id=2, bos_id=1,
                           ctc_weight=0.3)
    assert last < first
