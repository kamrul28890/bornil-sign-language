import numpy as np

from bornil.align.agreement import boundary_f1, chance_boundary_f1, shift_null_f1


def test_perfect_match():
    r = boundary_f1([3, 10, 20], [3, 10, 20], tolerance=0)
    assert r["f1"] == 1.0 and r["precision"] == 1.0 and r["recall"] == 1.0


def test_tolerance_window_counts_near_misses():
    r = boundary_f1([4, 11], [3, 10], tolerance=2)
    assert r["f1"] == 1.0


def test_no_match():
    r = boundary_f1([0, 1], [50, 60], tolerance=2)
    assert r["f1"] == 0.0


def test_each_reference_matched_at_most_once():
    # two predictions crowding one reference -> recall 1/1, precision 1/2
    r = boundary_f1([10, 11], [10], tolerance=2)
    assert r["recall"] == 1.0
    assert r["precision"] == 0.5


def test_chance_baseline_is_between_zero_and_one():
    c = chance_boundary_f1(n_pred=5, n_ref=5, n_frames=100, tolerance=2, trials=50)
    assert 0.0 <= c <= 1.0


def test_empty_predictions_or_refs_give_zero_f1():
    assert boundary_f1([], [1, 2], tolerance=2)["f1"] == 0.0
    assert boundary_f1([1, 2], [], tolerance=2)["f1"] == 0.0
    assert boundary_f1([], [], tolerance=2)["f1"] == 0.0


def test_counts_are_reported():
    r = boundary_f1([1, 5, 9], [1, 5], tolerance=0)
    assert r["n_pred"] == 3 and r["n_ref"] == 2


def test_returns_plain_floats():
    r = boundary_f1([1], [1], tolerance=0)
    assert type(r["f1"]) is float and type(r["precision"]) is float


def test_greedy_matching_prefers_nearest():
    # ref 10 should be consumed by pred 10 (exact), leaving pred 8 unmatched
    r = boundary_f1([8, 10], [10], tolerance=2)
    assert r["recall"] == 1.0 and r["precision"] == 0.5


def test_more_boundaries_raise_chance():
    # Saturating the timeline with predictions must raise chance agreement.
    sparse = chance_boundary_f1(2, 2, 100, tolerance=2, trials=100, seed=0)
    dense = chance_boundary_f1(40, 40, 100, tolerance=2, trials=100, seed=0)
    assert dense > sparse


def test_shift_null_matches_count_and_range():
    rng = np.random.default_rng(0)
    pred = [5, 20, 35, 60]
    ref = [4, 22, 34, 61]
    c = shift_null_f1(pred, ref, n_frames=80, tolerance=2, trials=100, seed=1)
    assert 0.0 <= c <= 1.0


def test_shift_null_is_a_proper_null_below_real_f1():
    # Real alignment is perfect (F1 1.0); the shift null must sit far below it,
    # because a random circular offset destroys the phase relationship.
    pred = [5, 20, 35, 60, 75]
    ref = list(pred)
    assert boundary_f1(pred, ref, tolerance=1)["f1"] == 1.0
    null = shift_null_f1(pred, ref, n_frames=100, tolerance=1, trials=300, seed=0)
    assert null < 0.5


def test_shift_null_deterministic_given_seed():
    pred, ref = [3, 30, 70], [4, 31, 69]
    a = shift_null_f1(pred, ref, 100, tolerance=2, trials=100, seed=7)
    b = shift_null_f1(pred, ref, 100, tolerance=2, trials=100, seed=7)
    assert a == b
