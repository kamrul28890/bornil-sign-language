import torch

from bornil.pretrain.mkm import MKMHead, mkm_loss


def test_mkm_head_output_shape():
    head = MKMHead(d_model=16, input_dim=218)
    memory = torch.randn(2, 9, 16)
    out = head(memory)
    assert out.shape == (2, 9, 218)


def test_loss_is_zero_for_perfect_prediction_on_masked_positions():
    pred = torch.randn(2, 5, 3)
    target = pred.clone()
    mask = torch.zeros(2, 5, dtype=torch.bool)
    mask[0, 1:3] = True
    loss = mkm_loss(pred, target, mask)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_loss_ignores_unmasked_position_errors():
    pred = torch.zeros(1, 4, 2)
    target = torch.zeros(1, 4, 2)
    target[0, 3] = 100.0          # huge error at an UNMASKED position
    mask = torch.zeros(1, 4, dtype=torch.bool)
    mask[0, 0] = True             # only position 0 is masked, and pred==target there
    loss = mkm_loss(pred, target, mask)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_loss_reflects_masked_position_error():
    pred = torch.zeros(1, 4, 2)
    target = torch.zeros(1, 4, 2)
    target[0, 0] = torch.tensor([1.0, 1.0])   # error at the masked position
    mask = torch.zeros(1, 4, dtype=torch.bool)
    mask[0, 0] = True
    loss = mkm_loss(pred, target, mask)
    assert loss.item() > 0.0


def test_zero_masked_positions_returns_zero_tensor_not_nan():
    pred = torch.randn(2, 5, 3, requires_grad=True)
    target = torch.randn(2, 5, 3)
    mask = torch.zeros(2, 5, dtype=torch.bool)
    loss = mkm_loss(pred, target, mask)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    loss.backward()   # must not raise
