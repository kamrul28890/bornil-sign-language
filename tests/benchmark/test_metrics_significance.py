from bornil.benchmark.metrics import metric_ci, paired_bootstrap_pvalue


def test_metric_ci_brackets_point_estimate():
    hyps = ["আমি বাড়ি যাব", "সে স্কুলে যায়"] * 10
    refs = ["আমি বাড়ি যাব", "সে বাজারে যায়"] * 10
    point, lo, hi = metric_ci(hyps, refs, metric="chrf", n=200, seed=0)
    assert lo <= point <= hi
    assert hi >= lo


def test_paired_bootstrap_identical_systems_is_insignificant():
    hyps = ["ক খ গ", "a b c", "x y z", "p q r"] * 6
    refs = ["ক খ ঘ", "a b d", "x y w", "p q s"] * 6
    p = paired_bootstrap_pvalue(hyps, hyps, refs, metric="chrf", n=200, seed=0)
    assert p > 0.9


def test_paired_bootstrap_obvious_difference_is_significant():
    refs = ["ক খ গ ঘ"] * 40
    good = ["ক খ গ ঘ"] * 40
    bad = ["z z z z"] * 40
    p = paired_bootstrap_pvalue(bad, good, refs, metric="chrf", n=300, seed=0)
    assert p < 0.05
