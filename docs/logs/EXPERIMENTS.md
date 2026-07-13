# Bornil — Experiment Log

> One row per training/eval run. This is the ground truth for every number that
> goes into the paper. Never report a result that isn't logged here with a config
> hash and a split name. Newest at top.

## Row template

```
### EXP-<id>  (YYYY-MM-DD)
- **Model:** <arch, size>            **Phase/Task:** <e.g. P5-T3>
- **Data/Split:** <bornil-random-v1 | bornil-signerindep-v1 | phoenix14t | ...>
- **Representation:** <kp-218 | kp-258 | kp+face | rgb-video | ...>
- **Config:** <path to frozen config + hash>   **Seed(s):** <...>
- **Compute:** <3070 / A100x1 / ...>  **Train time:** <h>
- **Metrics (test):** BLEU-1/2/3/4=... chrF=... chrF++=... ROUGE-L=... BERTScore=...
  - **95% CI (bootstrap):** BLEU-4 [lo, hi]
- **Checkpoint:** <path/URI>
- **Notes:** anything surprising; link failures to FAILURES.md
```

---

## Leaderboard snapshot (Bornil test, gloss-free SLT)

| Rank | Model | Split | Repr. | BLEU-4 | chrF++ | EXP-id |
|---|---|---|---|---|---|---|
| — | (fill as runs complete) | | | | | |

---

<!-- Log rows below, newest first -->
