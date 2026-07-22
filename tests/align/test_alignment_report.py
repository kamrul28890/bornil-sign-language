import bornil.runtime  # noqa: F401

import numpy as np
import pandas as pd

from scripts.alignment_report import (
    _agreement_record, pred_boundaries, summarize, _verdict,
)


def _clip_df(frames_confs):
    return pd.DataFrame([
        {"row": "r", "label_index": i, "token_id": 5, "token_str": "x",
         "start_frame": f, "end_frame": f, "n_frames": 100, "peak_conf": c}
        for i, (f, c) in enumerate(frames_confs)
    ])


def test_pred_boundaries_all_vs_anchors():
    df = _clip_df([(10, 0.9), (20, 0.01), (30, 0.6), (40, 0.0)])
    assert pred_boundaries(df) == [10, 20, 30, 40]
    assert pred_boundaries(df, anchor_min_conf=0.5) == [10, 30]


def test_pred_boundaries_sorted():
    df = _clip_df([(40, 0.9), (10, 0.9), (25, 0.9)])
    assert pred_boundaries(df) == [10, 25, 40]


def test_agreement_record_none_when_no_pred_or_ref():
    assert _agreement_record([], [1, 2], 100, 2, 10, 0) is None
    assert _agreement_record([1, 2], [], 100, 2, 10, 0) is None


def test_agreement_record_perfect_beats_shift():
    r = _agreement_record([5, 20, 40, 70], [5, 20, 40, 70], 100, 1, 100, 0)
    assert r["real"] == 1.0
    assert r["shift"] < r["real"]


def test_summarize_empty():
    s = summarize([])
    assert s["n_clips"] == 0 and s["real"] == 0.0


def test_summarize_ratios():
    recs = [{"real": 0.4, "chance": 0.2, "shift": 0.25},
            {"real": 0.6, "chance": 0.2, "shift": 0.15}]
    s = summarize(recs)
    assert s["n_clips"] == 2
    assert abs(s["real"] - 0.5) < 1e-9
    assert abs(s["ratio_shift"] - 0.5 / 0.2) < 1e-9


def test_verdict_at_chance_when_ratios_near_one():
    flat = {"real": 0.1, "chance": 0.1, "shift": 0.1, "ratio_shift": 1.0}
    v = _verdict(flat, flat, flat)
    assert v.startswith("AT CHANCE")


def test_verdict_above_chance_when_ratio_high():
    strong = {"real": 0.4, "chance": 0.15, "shift": 0.15, "ratio_shift": 2.6}
    weak = {"real": 0.1, "chance": 0.1, "shift": 0.1, "ratio_shift": 1.0}
    v = _verdict(strong, weak, weak)
    assert v.startswith("ABOVE CHANCE")
