import bornil.runtime  # noqa: F401
import torch
from torch.utils.data import DataLoader

from bornil.data.dataset import SignDataset, make_collate_fn
from bornil.models.stgcn import SignSTGCN, build_stgcn_adjacency, kp218_to_xyz65
from bornil.train.loop import train_epoch


def test_kp218_to_xyz65_drops_pose_visibility_and_keeps_hand_xyz():
    x = torch.arange(218, dtype=torch.float32).view(1, 1, 218)
    y = kp218_to_xyz65(x)

    assert y.shape == (1, 1, 65, 3)
    assert y[0, 0, 0].tolist() == [0.0, 1.0, 2.0]
    assert y[0, 0, 1].tolist() == [4.0, 5.0, 6.0]
    assert y[0, 0, 22].tolist() == [88.0, 89.0, 90.0]
    assert y[0, 0, 23].tolist() == [92.0, 93.0, 94.0]
    assert y[0, 0, 44].tolist() == [155.0, 156.0, 157.0]


def test_adjacency_has_expected_edges_and_self_loops():
    adj = build_stgcn_adjacency(normalize=False)

    assert adj.shape == (65, 65)
    assert torch.allclose(adj, adj.T)
    assert torch.all(torch.diag(adj) == 1)
    assert adj[11, 13] == 1 and adj[13, 15] == 1
    assert adj[23, 24] == 1 and adj[44, 45] == 1
    assert adj[15, 23] == 1 and adj[16, 44] == 1

    norm = build_stgcn_adjacency(normalize=True)
    assert norm.shape == (65, 65)
    assert torch.allclose(norm, norm.T, atol=1e-6)
    assert torch.isfinite(norm).all()


def test_stgcn_shapes_and_ctc_contract():
    m = SignSTGCN(
        input_dim=218, vocab_size=40, d_model=32, nhead=4,
        stgcn_channels=32, stgcn_layers=2, num_decoder_layers=2,
        dim_feedforward=64, use_ctc=True,
    )
    x = torch.randn(3, 11, 218)
    y = torch.randint(3, 40, (3, 6))
    mask = m.src_padding_mask(torch.tensor([11, 9, 7]), 11)

    logits = m(x, y, src_key_padding_mask=mask)
    ctc = m.ctc_logits(x, src_key_padding_mask=mask)
    dec, ctc2 = m.forward_with_ctc(x, y, src_key_padding_mask=mask)

    assert logits.shape == (3, 6, 40)
    assert ctc.shape == (3, 11, 40)
    assert dec.shape == logits.shape
    assert ctc2.shape == ctc.shape
    assert callable(m.generate)
    assert callable(m.beam_search)


def test_stgcn_tiny_ctc_training_loss_decreases():
    torch.manual_seed(0)
    N, T, D, V = 8, 20, 218, 40
    feats = torch.randn(N, T, D).numpy()
    targets = [[1, 5, 6, 7, 2] for _ in range(N)]
    ds = SignDataset(feats, [T] * N, list(range(N)), targets,
                     normalize=False, upper_body=False)
    loader = DataLoader(ds, batch_size=4, collate_fn=make_collate_fn(0))
    model = SignSTGCN(
        input_dim=D, vocab_size=V, d_model=32, nhead=4,
        stgcn_channels=32, stgcn_layers=1, num_decoder_layers=2,
        dim_feedforward=64, use_ctc=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    first = train_epoch(model, loader, opt, pad_id=0, eos_id=2, bos_id=1,
                        ctc_weight=0.3)
    for _ in range(12):
        last = train_epoch(model, loader, opt, pad_id=0, eos_id=2, bos_id=1,
                           ctc_weight=0.3)
    assert last < first
