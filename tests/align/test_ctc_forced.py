import bornil.runtime  # noqa: F401

import pytest
import torch

from bornil.align.ctc_forced import AlignmentError, align_clip, forced_align, path_to_spans


def _log_probs_favoring(seq, vocab=5, T=None, blank=0):
    """Build (T,V) log-probs that strongly favor emitting seq[t] at frame t."""
    T = T or len(seq)
    p = torch.full((T, vocab), 1e-6)
    for t, v in enumerate(seq):
        p[t, v] = 1.0
    return (p / p.sum(-1, keepdim=True)).log()


def test_recovers_obvious_alignment():
    # frames:      b  1  1  b  2  b     labels [1, 2]
    lp = _log_probs_favoring([0, 1, 1, 0, 2, 0])
    spans = align_clip(lp, [1, 2], blank=0)
    assert spans == [(0, 1, 2), (1, 4, 4)]


def test_path_length_equals_frame_count():
    lp = _log_probs_favoring([0, 1, 1, 0, 2, 0])
    path = forced_align(lp, [1, 2], blank=0)
    assert len(path) == 6


def test_spans_are_monotonic_and_non_overlapping():
    lp = _log_probs_favoring([0, 1, 1, 0, 2, 2, 0, 3, 0])
    spans = align_clip(lp, [1, 2, 3], blank=0)
    assert [s[0] for s in spans] == [0, 1, 2]
    for (_, s0, e0), (_, s1, _e1) in zip(spans, spans[1:]):
        assert s0 <= e0 < s1


def test_repeated_adjacent_labels_need_a_separating_blank():
    # labels [1, 1] require at least 3 frames: 1, b, 1
    lp = _log_probs_favoring([1, 0, 1])
    spans = align_clip(lp, [1, 1], blank=0)
    assert [s[0] for s in spans] == [0, 1]
    assert spans[0][2] < spans[1][1]


def test_infeasible_when_too_few_frames():
    lp = _log_probs_favoring([1, 2])          # 2 frames
    with pytest.raises(AlignmentError):
        forced_align(lp, [1, 2, 3, 4], blank=0)   # needs >= 4


def test_path_to_spans_skips_labels_that_got_no_frames():
    # A path that never visits label index 1 (extended state 3).
    path = [0, 1, 1, 2]      # blank, y1, y1, blank
    spans = path_to_spans(path, [7, 9])
    assert spans == [(0, 1, 2)]


# --- DP-exercising cases (plan amendment 3) -------------------------------
# The tests above use log-probs so peaked that a per-frame argmax already gives
# the answer -- they pass against a greedy implementation that ignores the CTC
# constraints. These force the constrained path to differ from greedy.


def test_forced_alignment_overrides_greedy_when_label_order_conflicts():
    """Greedy reads [2,1]; the labels say [1,2], so Viterbi must not follow it."""
    lp = _log_probs_favoring([2, 1])
    spans = align_clip(lp, [1, 2], blank=0)
    # Order is dictated by `labels`, never by the per-frame argmax.
    assert [s[0] for s in spans] == [0, 1]
    assert spans[0][1] <= spans[0][2] < spans[1][1]


def test_label_absent_from_argmax_is_still_aligned():
    """A label the encoder never argmaxes must still receive frames."""
    lp = _log_probs_favoring([1, 1, 1])
    spans = align_clip(lp, [1, 4], blank=0)   # 4 never wins any frame
    assert [s[0] for s in spans] == [0, 1]


def test_every_frame_is_assigned_and_states_are_monotonic():
    """The path must cover all frames and never move backwards."""
    torch.manual_seed(0)
    lp = torch.log_softmax(torch.randn(40, 12), dim=-1)
    labels = [3, 5, 7, 5, 2]
    path = forced_align(lp, labels, blank=0)
    assert len(path) == 40
    assert all(b >= a for a, b in zip(path, path[1:])), "path moved backwards"
    assert path[0] in (0, 1)
    assert path[-1] in (2 * len(labels), 2 * len(labels) - 1)


def test_all_labels_appear_when_frames_are_ample():
    torch.manual_seed(1)
    lp = torch.log_softmax(torch.randn(60, 12), dim=-1)
    labels = [3, 5, 7]
    spans = align_clip(lp, labels, blank=0)
    assert [s[0] for s in spans] == [0, 1, 2]


def test_viterbi_beats_a_suboptimal_valid_path():
    """The returned path must be the max-likelihood one, not merely feasible."""
    torch.manual_seed(2)
    lp = torch.log_softmax(torch.randn(25, 10), dim=-1)
    labels = [4, 6]
    path = forced_align(lp, labels, blank=0)
    ext = [0, 4, 0, 6, 0]
    score = sum(float(lp[t, ext[s]]) for t, s in enumerate(path))

    # Brute force every monotone path over the same extended states.
    import itertools
    best = float("-inf")
    T, S = 25, 5
    for cuts in itertools.combinations(range(1, T), S - 1):
        bounds = (0,) + cuts + (T,)
        cand = [s for s in range(S) for _ in range(bounds[s + 1] - bounds[s])]
        best = max(best, sum(float(lp[t, ext[s]]) for t, s in enumerate(cand)))
    assert score >= best - 1e-6


def test_exact_minimum_frame_budget_is_feasible():
    lp = _log_probs_favoring([1, 0, 1, 2])
    spans = align_clip(lp, [1, 1, 2], blank=0)   # needs exactly 4 frames
    assert [s[0] for s in spans] == [0, 1, 2]


def test_empty_labels_raises():
    with pytest.raises(AlignmentError):
        forced_align(_log_probs_favoring([0, 1]), [], blank=0)


def test_zero_frames_raises_alignment_error():
    with pytest.raises(AlignmentError):
        forced_align(torch.zeros((0, 5)).log(), [1], blank=0)


def test_rejects_non_2d_log_probs():
    with pytest.raises(ValueError):
        forced_align(torch.zeros((2, 3, 4)), [1], blank=0)


def test_non_zero_blank_id_is_honoured():
    # blank=9 here; id 0 is an ordinary label.
    lp = _log_probs_favoring([9, 0, 9, 3, 9], vocab=10)
    spans = align_clip(lp, [0, 3], blank=9)
    assert spans == [(0, 1, 1), (1, 3, 3)]
