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
| 1 | SignTransformer d256 (vanilla baseline) | random-v1 | kp-218 | 0.12 | 8.39 | data_full-4e112ae230b9 |

> **Baseline diagnosis (2026-07-14):** first full-dataset run. Outputs are fluent
> Bengali but semantically unrelated to the signs (819/1799 unique, none empty) —
> the decoder learned the language prior; visual grounding is weak. Expected for a
> vanilla, greedy-decoded, untuned model on 14k open-vocabulary pairs.
>
> **Trivial floors (random-v1, same split):** most-freq BLEU-4 0.01 / chrF++ 10.03;
> random 0.05 / 6.64; retrieval(1-NN) 0.07 / 9.11. The model (0.12 / 8.39) **beats
> all floors on BLEU-4 and ROUGE-L** (learning real signal) but chrF++ is *below*
> most-freq — chrF is gameable by emitting one frequent sentence, so read BLEU-4 /
> ROUGE-L as the discriminating metrics here. Next: warmup+scheduler, label
> smoothing, beam search; then the harder splits.

---

<!-- Log rows below, newest first -->
<!-- (smoke/GPU-path proof rows removed — pipeline validation only, not real results) -->

### EXP-trial_1pct-79233c0d47db  (2026-07-13)
- **Model:** SignTransformer d256 L4+4   **Phase/Task:** trial_1pct (WS-C T0.2)
- **Data/Split:** trial_1pct random-v1 (train 144 / test 18)
- **Representation:** kp-218   **Config hash:** 79233c0d47db   **Seed:** 42
- **Compute:** cuda   **Train time:** 0.3 min
- **Metrics (test):** BLEU-4=0.42 chrF=8.14 chrF++=6.43 ROUGE-L=0.027 (BLEU-1/2/3=6.0/1.4/0.7)
- **Checkpoint:** trial_1pct/runs/79233c0d47db/best.pt

### EXP-data_full-4e112ae230b9  (2026-07-14)
- **Model:** SignTransformer d256 L4+4   **Phase/Task:** data_full (WS-C T0.2)
- **Data/Split:** data_full random-v1 (train 14390 / test 1799)
- **Representation:** kp-218   **Config hash:** 4e112ae230b9   **Seed:** 42
- **Compute:** cuda   **Train time:** 35.4 min
- **Metrics (test):** BLEU-4=0.12 chrF=10.61 chrF++=8.39 ROUGE-L=0.034 (BLEU-1/2/3=8.6/1.6/0.4)
- **Checkpoint:** data_full/runs/4e112ae230b9/best.pt

### EXP-data_full-2db347d9db75  (2026-07-14)
- **Model:** SignTransformer d256 L4+4   **Phase/Task:** data_full (WS-C T0.2)
- **Data/Split:** data_full random-v1 (train 14390 / test 1799)
- **Representation:** kp-218   **Config hash:** 2db347d9db75   **Seed:** 42
- **Compute:** cuda   **Train time:** 119.2 min
- **Metrics (test):** BLEU-4=0.34 chrF=10.91 chrF++=8.79 ROUGE-L=0.049 (BLEU-1/2/3=10.2/2.8/0.9)
- **Checkpoint:** data_full/runs/2db347d9db75/best.pt

### EXP-data_full-c19d0ea08824  (2026-07-14)
- **Model:** SignTransformer d256 L4+4   **Phase/Task:** data_full (WS-C T0.2)
- **Data/Split:** data_full random-v1 (train 14390 / test 1799); source=frozen:splits/random-v1
- **Representation:** kp-218   **Config hash:** c19d0ea08824   **Seed:** 42
- **Compute:** cuda   **Train time:** 42.0 min
- **Decode:** greedy (beam_size=1)
- **Metrics (test):** BLEU-4=0.52 chrF=11.81 chrF++=9.65 ROUGE-L=0.055 (BLEU-1/2/3=11.7/3.3/1.1)
- **Checkpoint:** data_full/runs/c19d0ea08824/best.pt

### EXP-data_full-60320f4c6a60  (2026-07-15)
- **Model:** SignTransformer d256 L4+4   **Phase/Task:** data_full (WS-C T0.2)
- **Data/Split:** data_full unseen-sentence-v1 (train 14390 / test 1799); source=frozen:splits/unseen-sentence-v1
- **Representation:** kp-218   **Config hash:** 60320f4c6a60   **Seed:** 42
- **Compute:** cuda   **Train time:** 37.9 min
- **Decode:** greedy (beam_size=1)
- **Metrics (test):** BLEU-4=0.42 chrF=11.29 chrF++=9.19 ROUGE-L=0.048 (BLEU-1/2/3=11.1/3.0/1.0)
- **Checkpoint:** data_full/runs/60320f4c6a60/best.pt

### EXP-data_full-46a706570712  (2026-07-15)
- **Model:** SignTransformer d256 L4+4   **Phase/Task:** data_full (WS-C T0.2)
- **Data/Split:** data_full signerindep-v1 (train 14500 / test 1500); source=frozen:splits/signerindep-v1
- **Representation:** kp-218   **Config hash:** 46a706570712   **Seed:** 42
- **Compute:** cuda   **Train time:** 42.0 min
- **Decode:** greedy (beam_size=1)
- **Metrics (test):** BLEU-4=0.35 chrF=10.36 chrF++=8.38 ROUGE-L=0.048 (BLEU-1/2/3=8.8/2.4/0.8)
- **Checkpoint:** data_full/runs/46a706570712/best.pt

### EXP-data_full_tconv-c26953197929  (2026-07-15)
- **Model:** SignTransformer + TConv2k5 d256 L4+4   **Phase/Task:** data_full_tconv (WS-C T0.2)
- **Data/Split:** data_full_tconv random-v1 (train 14390 / test 1799); source=frozen:splits/random-v1
- **Representation:** kp-218   **Config hash:** c26953197929   **Seed:** 42
- **Compute:** cuda   **Train time:** 46.1 min
- **Decode:** greedy (beam_size=1)
- **Metrics (test):** BLEU-4=0.66 chrF=11.60 chrF++=9.48 ROUGE-L=0.054 (BLEU-1/2/3=11.8/3.4/1.3)
- **Checkpoint:** data_full/runs/c26953197929/best.pt

### EXP-data_full_tconv_len-a459d1148d0c  (2026-07-15)
- **Model:** SignTransformer + TConv2k5 d256 L4+4   **Phase/Task:** data_full_tconv_len (WS-C T0.2)
- **Data/Split:** data_full_tconv_len random-v1 (train 14390 / test 1799); source=frozen:splits/random-v1
- **Representation:** kp-218   **Config hash:** a459d1148d0c   **Seed:** 42
- **Compute:** cuda   **Train time:** 39.2 min
- **Decode:** greedy (beam_size=1)
- **Training extras:** early_eos_penalty=0.2
- **Metrics (test):** BLEU-4=0.11 chrF=8.99 chrF++=7.26 ROUGE-L=0.031 (BLEU-1/2/3=9.2/1.9/0.3)
- **Checkpoint:** data_full/runs/a459d1148d0c/best.pt

### EXP-data_full_tconv_lte12-bc38907bece6  (2026-07-16)
- **Model:** SignTransformer + TConv2k5 d256 L4+4   **Phase/Task:** data_full_tconv_lte12 (WS-C T0.2)
- **Data/Split:** data_full_tconv_lte12 random-v1 (train 12162 / test 1546); source=frozen:splits/random-v1|max_target_tokens<=12
- **Representation:** kp-218   **Config hash:** bc38907bece6   **Seed:** 42
- **Compute:** cuda   **Train time:** 38.4 min
- **Decode:** greedy (beam_size=1)
- **Training extras:** early_eos_penalty=0.0
- **Metrics (test):** BLEU-4=0.72 chrF=11.98 chrF++=9.95 ROUGE-L=0.055 (BLEU-1/2/3=15.6/4.3/1.4)
- **Checkpoint:** data_full/runs/bc38907bece6/best.pt

### EXP-data_full_tconv_lte12-58269a7b1003  (2026-07-16)
- **Model:** SignTransformer + TConv2k5 d256 L4+4   **Phase/Task:** data_full_tconv_lte12 (WS-C T0.2)
- **Data/Split:** data_full_tconv_lte12 unseen-sentence-v1 (train 12170 / test 1521); source=frozen:splits/unseen-sentence-v1|max_target_tokens<=12
- **Representation:** kp-218   **Config hash:** 58269a7b1003   **Seed:** 42
- **Compute:** cuda   **Train time:** 32.2 min
- **Decode:** greedy (beam_size=1)
- **Training extras:** early_eos_penalty=0.0
- **Metrics (test):** BLEU-4=0.60 chrF=11.24 chrF++=9.33 ROUGE-L=0.050 (BLEU-1/2/3=15.0/4.0/1.3)
- **Checkpoint:** data_full/runs/58269a7b1003/best.pt

### EXP-data_full_tconv_lte12-2ff734ae1aae  (2026-07-16)
- **Model:** SignTransformer + TConv2k5 d256 L4+4   **Phase/Task:** data_full_tconv_lte12 (WS-C T0.2)
- **Data/Split:** data_full_tconv_lte12 signerindep-v1 (train 12507 / test 1108); source=frozen:splits/signerindep-v1|max_target_tokens<=12
- **Representation:** kp-218   **Config hash:** 2ff734ae1aae   **Seed:** 42
- **Compute:** cuda   **Train time:** 32.2 min
- **Decode:** greedy (beam_size=1)
- **Training extras:** early_eos_penalty=0.0
- **Metrics (test):** BLEU-4=0.81 chrF=11.01 chrF++=9.22 ROUGE-L=0.055 (BLEU-1/2/3=15.4/4.0/1.5)
- **Checkpoint:** data_full/runs/2ff734ae1aae/best.pt
