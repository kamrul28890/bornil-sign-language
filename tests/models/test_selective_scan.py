import math

import torch

from bornil.models.mamba import selective_scan


def _brute_force_reference(x, delta, A, B, C, D):
    """Independent, unvectorized reference (plain Python floats, a second
    code path from `selective_scan`) for a single batch element.

    x, delta: (T, d_inner). A: (d_inner, d_state). B, C: (T, d_state).
    D: (d_inner,). Returns a (T, d_inner) list of lists of floats.
    """
    T, d_inner = x.shape
    d_state = A.shape[1]
    h = [[0.0] * d_state for _ in range(d_inner)]
    y = []
    for t in range(T):
        y_t = []
        for c in range(d_inner):
            dt = delta[t, c].item()
            xc = x[t, c].item()
            y_c = D[c].item() * xc
            for n in range(d_state):
                dA = math.exp(dt * A[c, n].item())
                dBx = dt * B[t, n].item() * xc
                h[c][n] = dA * h[c][n] + dBx
                y_c += C[t, n].item() * h[c][n]
            y_t.append(y_c)
        y.append(y_t)
    return y


def test_matches_independent_brute_force_reference():
    torch.manual_seed(0)
    T, d_inner, d_state = 6, 4, 3
    x = torch.randn(1, T, d_inner)
    delta = torch.rand(1, T, d_inner) * 0.5 + 0.01
    A = -torch.rand(d_inner, d_state) - 0.1
    B = torch.randn(1, T, d_state)
    C = torch.randn(1, T, d_state)
    D = torch.randn(d_inner)

    y = selective_scan(x, delta, A, B, C, D)
    ref = _brute_force_reference(x[0], delta[0], A, B[0], C[0], D)

    assert y.shape == (1, T, d_inner)
    for t in range(T):
        for c in range(d_inner):
            assert abs(y[0, t, c].item() - ref[t][c]) < 1e-4


def test_zero_delta_freezes_state():
    torch.manual_seed(1)
    T, d_inner, d_state = 5, 2, 3
    x = torch.randn(1, T, d_inner)
    delta = torch.zeros(1, T, d_inner)
    A = -torch.rand(d_inner, d_state) - 0.1
    B = torch.randn(1, T, d_state)
    C = torch.randn(1, T, d_state)
    D = torch.zeros(d_inner)

    y = selective_scan(x, delta, A, B, C, D)
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-6)


def test_gradients_are_finite():
    torch.manual_seed(2)
    T, d_inner, d_state = 8, 4, 3
    x = torch.randn(2, T, d_inner, requires_grad=True)
    delta_raw = torch.randn(2, T, d_inner, requires_grad=True)
    delta = torch.nn.functional.softplus(delta_raw)
    A = (-torch.rand(d_inner, d_state) - 0.1).requires_grad_()
    B = torch.randn(2, T, d_state, requires_grad=True)
    C = torch.randn(2, T, d_state, requires_grad=True)
    D = torch.randn(d_inner, requires_grad=True)

    y = selective_scan(x, delta, A, B, C, D)
    y.sum().backward()

    for t in (x, delta_raw, A, B, C, D):
        assert t.grad is not None
        assert torch.isfinite(t.grad).all()
