import pandas as pd

from scripts.results_significance import build_report, load_predictions


def _write_run(tmp_path, name, refs, hyps):
    d = tmp_path / name
    d.mkdir()
    pd.DataFrame({"reference": refs, "hypothesis": hyps}).to_csv(
        d / "predictions.csv", index=False, encoding="utf-8")
    return str(d)


def test_load_predictions_roundtrip(tmp_path):
    d = _write_run(tmp_path, "r", ["ক খ গ"], ["ক খ ঘ"])
    hyps, refs = load_predictions(d)
    assert hyps == ["ক খ ঘ"] and refs == ["ক খ গ"]


def test_build_report_has_ci_and_pvalue_columns(tmp_path):
    refs = ["ক খ গ ঘ"] * 20
    a = _write_run(tmp_path, "base", refs, ["z z z z"] * 20)
    b = _write_run(tmp_path, "better", refs, ["ক খ গ ঘ"] * 20)
    md = build_report([("base", a), ("better", b)], baseline_label="base",
                      metrics=["chrf"], n=200, seed=0)
    assert "95% CI" in md
    assert "p vs base" in md
    assert "better" in md and "base" in md
