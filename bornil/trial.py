"""Trial-run helpers shared by the sampling and orchestration scripts."""


def trial_dirname(percent: float) -> str:
    """Canonical output dir for a percentage sample: 2 -> 'trial_2pct',
    0.5 -> 'trial_0_5pct'. Keeps make_sample and run_trial in agreement."""
    s = f"{percent:g}".replace(".", "_")
    return f"trial_{s}pct"
