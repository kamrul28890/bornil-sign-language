"""Tests for the warmup learning-rate schedule (WS-C training).

Transformers train badly with a flat LR — they need a short warmup (ramp up from
~0) then a decay. This is the Noam-style schedule: linear warmup to 1.0 at
`warmup_steps`, then inverse-sqrt decay.
"""
from bornil.train.schedule import noam_lambda


def test_ramps_up_during_warmup():
    f = noam_lambda(100)
    assert f(0) < f(50) < f(98)
    assert f(0) > 0


def test_peaks_at_one_at_warmup_end():
    f = noam_lambda(100)
    assert f(99) == 1.0            # step index 99 -> s=100 == warmup


def test_decays_after_warmup():
    f = noam_lambda(100)
    assert f(99) > f(200) > f(400)


def test_inverse_sqrt_value():
    f = noam_lambda(100)
    # s = 400 = 4*warmup -> sqrt(100/400) = 0.5
    assert abs(f(399) - 0.5) < 1e-9
