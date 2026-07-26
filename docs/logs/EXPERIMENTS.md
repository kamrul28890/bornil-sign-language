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

> **Input ablation (2026-07-19, run `b0a590ea3317`):** supersedes the qualitative
> "visual grounding is weak" claim above with a measurement. Deranging only the
> video↔text pairing (`mismatch`) costs −4.71 chrF++ (10.91 → 6.20) and 13.7× BLEU-4,
> so the encoder **is** genuinely used. But shuffling frame order within each clip
> costs only −0.88 chrF++ (retains 92%), so the grounding is largely **static pose,
> not motion**. Gaussian-noise input still scores 9.08 chrF++ — near the retrieval
> floor (9.20) — so chrF++ margins of 1–2 over trivial floors are weak evidence;
> anchor claims to the `mismatch` control instead. Full analysis:
> `docs/logs/INPUT_ABLATION.md`.

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

### EXP-data_full_ctc-b0a590ea3317  (2026-07-16)
- **Model:** SignTransformer + TConv2k5 d256 L4+4   **Phase/Task:** data_full_ctc (WS-C T0.2)
- **Data/Split:** data_full_ctc random-v1 (train 14390 / test 1799); source=frozen:splits/random-v1
- **Representation:** kp-218   **Config hash:** b0a590ea3317   **Seed:** 42
- **Compute:** cuda   **Train time:** 44.8 min
- **Decode:** greedy (beam_size=1)
- **Training extras:** early_eos_penalty=0.0 ctc_weight=0.3
- **Decode extras:** decode_repetition_stop=0
- **Metrics (test):** BLEU-4=0.68 chrF=13.27 chrF++=10.91 ROUGE-L=0.067 (BLEU-1/2/3=13.3/4.3/1.5)
- **Checkpoint:** data_full/runs/b0a590ea3317/best.pt

### EXP-data_full_bilstm-07709687f916  (2026-07-16)
- **Model:** BiLSTM d256 L2+4   **Phase/Task:** data_full_bilstm (WS-C T0.1)
- **Data/Split:** data_full_bilstm random-v1 (train 14390 / test 1799); source=frozen:splits/random-v1
- **Representation:** kp-218   **Config hash:** 07709687f916   **Seed:** 42
- **Compute:** cuda   **Train time:** 28.5 min
- **Decode:** greedy (beam_size=1)
- **Training extras:** early_eos_penalty=0.0 ctc_weight=0.0
- **Decode extras:** decode_repetition_stop=0
- **Metrics (test):** BLEU-4=0.62 chrF=10.78 chrF++=8.99 ROUGE-L=0.064 (BLEU-1/2/3=12.5/4.0/1.5)
- **Checkpoint:** data_full/runs/07709687f916/best.pt

### EXP-data_full_ctc-40cffb9e62bb  (2026-07-16)
- **Model:** SignTransformer + TConv2k5 d256 L4+4   **Phase/Task:** data_full_ctc (WS-C T0.2)
- **Data/Split:** data_full_ctc signerindep-v1 (train 14500 / test 1500); source=frozen:splits/signerindep-v1
- **Representation:** kp-218   **Config hash:** 40cffb9e62bb   **Seed:** 42
- **Compute:** cuda   **Train time:** 47.5 min
- **Decode:** greedy (beam_size=1)
- **Training extras:** early_eos_penalty=0.0 ctc_weight=0.3
- **Decode extras:** decode_repetition_stop=0
- **Metrics (test):** BLEU-4=0.79 chrF=12.65 chrF++=10.36 ROUGE-L=0.072 (BLEU-1/2/3=10.6/4.0/1.7)
- **Checkpoint:** data_full/runs/40cffb9e62bb/best.pt

### EXP-data_full_ctc-a5a9060499a4  (2026-07-16)
- **Model:** SignTransformer + TConv2k5 d256 L4+4   **Phase/Task:** data_full_ctc (WS-C T0.2)
- **Data/Split:** data_full_ctc unseen-sentence-v1 (train 14390 / test 1799); source=frozen:splits/unseen-sentence-v1
- **Representation:** kp-218   **Config hash:** a5a9060499a4   **Seed:** 42
- **Compute:** cuda   **Train time:** 48.5 min
- **Decode:** greedy (beam_size=1)
- **Training extras:** early_eos_penalty=0.0 ctc_weight=0.3
- **Decode extras:** decode_repetition_stop=0
- **Metrics (test):** BLEU-4=0.89 chrF=13.24 chrF++=10.94 ROUGE-L=0.070 (BLEU-1/2/3=13.5/4.7/2.0)
- **Checkpoint:** data_full/runs/a5a9060499a4/best.pt

### EXP-trial_1pct-d9216f4420cd  (2026-07-16)
- **Model:** ST-GCN d256 L3+4   **Phase/Task:** trial_1pct (WS-C T1.1)
- **Data/Split:** trial_1pct random-v1 (train 144 / test 18); source=generated
- **Representation:** kp-218   **Config hash:** d9216f4420cd   **Seed:** 42
- **Compute:** cuda   **Train time:** 0.1 min
- **Decode:** greedy (beam_size=1)
- **Training extras:** early_eos_penalty=0.0 ctc_weight=0.3
- **Decode extras:** decode_repetition_stop=0
- **Metrics (test):** BLEU-4=0.00 chrF=0.00 chrF++=0.00 ROUGE-L=0.000 (BLEU-1/2/3=0.0/0.0/0.0)
- **Checkpoint:** trial_1pct/runs/d9216f4420cd/best.pt

### EXP-data_full_stgcn-2dfdcbdf4e2d  (2026-07-19)
- **Model:** ST-GCN d256 L3+4   **Phase/Task:** data_full_stgcn (WS-C T1.1)
- **Data/Split:** data_full_stgcn random-v1 (train 14390 / test 1799); source=frozen:splits/random-v1
- **Representation:** kp-218   **Config hash:** 2dfdcbdf4e2d   **Seed:** 42
- **Compute:** cuda   **Train time:** 2569.5 min
- **Decode:** greedy (beam_size=1)
- **Training extras:** early_eos_penalty=0.0 ctc_weight=0.3
- **Decode extras:** decode_repetition_stop=0
- **Metrics (test):** BLEU-4=0.36 chrF=10.25 chrF++=8.27 ROUGE-L=0.040 (BLEU-1/2/3=10.2/2.4/0.8)
- **Checkpoint:** data_full/runs/2dfdcbdf4e2d/best.pt
- **Notes:** **Negative result — do not tune further, do not run on the hard splits.**
  chrF++ 8.27 is *below* both the random-v1 retrieval floor (9.11) and the
  most-frequent-sentence floor (10.03); it beats the floors on BLEU-4 only. It is
  the only recent model to fall under the floors on chrF++, at 42.8 h vs 44.8 min
  for TConv+CTC (~57× the compute for a worse result). Mechanism, from
  `docs/logs/INPUT_ABLATION.md`: the exploited signal on this benchmark is largely
  static pose (frame-order shuffling costs the CTC model only −0.88 chrF++), and a
  spatio-temporal graph encoder's whole value proposition is modelling motion
  structure — so there is little for it to win on here. Closes WS-C T1.1.

### EXP-trial_2pct-b092ae873aef  (2026-07-19)
- **Model:** SignTransformer + TConv2k5 d256 L4+4   **Phase/Task:** trial_2pct (WS-C T0.2)
- **Data/Split:** trial_2pct random-v1 (train 1 / test 1); source=generated
- **Representation:** kp-218   **Config hash:** b092ae873aef   **Seed:** 42
- **Compute:** cuda   **Train time:** 0.0 min
- **Decode:** greedy (beam_size=1)
- **Training extras:** early_eos_penalty=0.0 ctc_weight=0.3
- **Decode extras:** decode_repetition_stop=0
- **Metrics (test):** BLEU-4=1.83 chrF=7.25 chrF++=3.62 ROUGE-L=0.000 (BLEU-1/2/3=1.8/1.8/1.8)
- **Checkpoint:** trial_2pct/runs/b092ae873aef/best.pt

### EXP-trial_2pct-2189811eb2fa  (2026-07-19)
- **Model:** ST-GCN d256 L3+4   **Phase/Task:** trial_2pct (WS-C T1.1)
- **Data/Split:** trial_2pct random-v1 (train 1 / test 1); source=generated
- **Representation:** kp-218   **Config hash:** 2189811eb2fa   **Seed:** 42
- **Compute:** cuda   **Train time:** 0.0 min
- **Decode:** greedy (beam_size=1)
- **Training extras:** early_eos_penalty=0.0 ctc_weight=0.3
- **Decode extras:** decode_repetition_stop=0
- **Metrics (test):** BLEU-4=0.14 chrF=0.26 chrF++=0.22 ROUGE-L=0.000 (BLEU-1/2/3=0.4/0.3/0.2)
- **Checkpoint:** trial_2pct/runs/2189811eb2fa/best.pt

### EXP-trial_2pct-745ef396baa4  (2026-07-19)
- **Model:** BiLSTM d256 L2+4   **Phase/Task:** trial_2pct (WS-C T0.1)
- **Data/Split:** trial_2pct random-v1 (train 1 / test 1); source=generated
- **Representation:** kp-218   **Config hash:** 745ef396baa4   **Seed:** 42
- **Compute:** cuda   **Train time:** 0.0 min
- **Decode:** greedy (beam_size=1)
- **Training extras:** early_eos_penalty=0.0 ctc_weight=0.0
- **Decode extras:** decode_repetition_stop=0
- **Metrics (test):** BLEU-4=0.20 chrF=1.53 chrF++=1.15 ROUGE-L=0.000 (BLEU-1/2/3=0.6/0.4/0.3)
- **Checkpoint:** trial_2pct/runs/745ef396baa4/best.pt

### EXP-data_full_pretrain_mkm-7d9e40ec1343  (2026-07-25)
- **Model:** MKM pretrain (encoder-only) d256 L4   **Phase/Task:** data_full_pretrain_mkm (WS-D D1)
- **Data/Split:** data_full_pretrain_mkm random-v1 train-only (train 14390); source=frozen:splits/random-v1
- **Config hash:** 7d9e40ec1343   **Seed:** 42
- **Compute:** cuda   **Train time:** 89.3 min
- **Pretrain extras:** mask_ratio=0.6 min_span=3 max_span=15
- **Recon loss:** first=0.1840 last=0.0801
- **Checkpoint:** data_full/runs/7d9e40ec1343/encoder_pretrain.pt

### EXP-data_full_ctc-5860b497604f  (2026-07-25)
- **Model:** SignTransformer + TConv2k5 d256 L4+4   **Phase/Task:** data_full_ctc (WS-C T0.2)
- **Data/Split:** data_full_ctc signerindep-v1 (train 14500 / test 1500); source=frozen:splits/signerindep-v1
- **Representation:** kp-218   **Config hash:** 5860b497604f   **Seed:** 42
- **Compute:** cuda   **Train time:** 49.5 min
- **Decode:** greedy (beam_size=1)
- **Training extras:** early_eos_penalty=0.0 ctc_weight=0.3 init_encoder=data_full/runs/7d9e40ec1343/encoder_pretrain.pt label_fraction=1.0
- **Decode extras:** decode_repetition_stop=0
- **Metrics (test):** BLEU-4=0.74 chrF=13.17 chrF++=10.72 ROUGE-L=0.060 (BLEU-1/2/3=10.5/3.4/1.5)
- **Checkpoint:** data_full/runs/5860b497604f/best.pt

### EXP-data_full_ctc-e848ac7a736e  (2026-07-25)
- **Model:** SignTransformer + TConv2k5 d256 L4+4   **Phase/Task:** data_full_ctc (WS-C T0.2)
- **Data/Split:** data_full_ctc random-v1 (train 1439 / test 1799); source=frozen:splits/random-v1|label_fraction=0.1
- **Representation:** kp-218   **Config hash:** e848ac7a736e   **Seed:** 42
- **Compute:** cuda   **Train time:** 5.6 min
- **Decode:** greedy (beam_size=1)
- **Training extras:** early_eos_penalty=0.0 ctc_weight=0.3 init_encoder=data_full/runs/7d9e40ec1343/encoder_pretrain.pt label_fraction=0.1
- **Decode extras:** decode_repetition_stop=0
- **Metrics (test):** BLEU-4=0.05 chrF=8.58 chrF++=6.74 ROUGE-L=0.020 (BLEU-1/2/3=8.8/0.8/0.1)
- **Checkpoint:** data_full/runs/e848ac7a736e/best.pt

### EXP-data_full_ctc-0319ebbea825  (2026-07-25)
- **Model:** SignTransformer + TConv2k5 d256 L4+4   **Phase/Task:** data_full_ctc (WS-C T0.2)
- **Data/Split:** data_full_ctc random-v1 (train 3598 / test 1799); source=frozen:splits/random-v1|label_fraction=0.25
- **Representation:** kp-218   **Config hash:** 0319ebbea825   **Seed:** 42
- **Compute:** cuda   **Train time:** 12.2 min
- **Decode:** greedy (beam_size=1)
- **Training extras:** early_eos_penalty=0.0 ctc_weight=0.3 init_encoder=data_full/runs/7d9e40ec1343/encoder_pretrain.pt label_fraction=0.25
- **Decode extras:** decode_repetition_stop=0
- **Metrics (test):** BLEU-4=0.14 chrF=5.51 chrF++=4.54 ROUGE-L=0.018 (BLEU-1/2/3=9.4/1.8/0.4)
- **Checkpoint:** data_full/runs/0319ebbea825/best.pt

### EXP-data_full_ctc-d42ff45dd94a  (2026-07-25)
- **Model:** SignTransformer + TConv2k5 d256 L4+4   **Phase/Task:** data_full_ctc (WS-C T0.2)
- **Data/Split:** data_full_ctc random-v1 (train 7195 / test 1799); source=frozen:splits/random-v1|label_fraction=0.5
- **Representation:** kp-218   **Config hash:** d42ff45dd94a   **Seed:** 42
- **Compute:** cuda   **Train time:** 22.8 min
- **Decode:** greedy (beam_size=1)
- **Training extras:** early_eos_penalty=0.0 ctc_weight=0.3 init_encoder=data_full/runs/7d9e40ec1343/encoder_pretrain.pt label_fraction=0.5
- **Decode extras:** decode_repetition_stop=0
- **Metrics (test):** BLEU-4=0.14 chrF=9.75 chrF++=7.67 ROUGE-L=0.025 (BLEU-1/2/3=7.3/1.0/0.3)
- **Checkpoint:** data_full/runs/d42ff45dd94a/best.pt

### EXP-data_full_ctc-bbc56ea042ad  (2026-07-25)
- **Model:** SignTransformer + TConv2k5 d256 L4+4   **Phase/Task:** data_full_ctc (WS-C T0.2)
- **Data/Split:** data_full_ctc random-v1 (train 14390 / test 1799); source=frozen:splits/random-v1|label_fraction=1.0
- **Representation:** kp-218   **Config hash:** bbc56ea042ad   **Seed:** 42
- **Compute:** cuda   **Train time:** 98.8 min
- **Decode:** greedy (beam_size=1)
- **Training extras:** early_eos_penalty=0.0 ctc_weight=0.3 init_encoder=data_full/runs/7d9e40ec1343/encoder_pretrain.pt label_fraction=1.0
- **Decode extras:** decode_repetition_stop=0
- **Metrics (test):** BLEU-4=0.46 chrF=12.01 chrF++=9.80 ROUGE-L=0.055 (BLEU-1/2/3=12.4/3.2/1.0)
- **Checkpoint:** data_full/runs/bbc56ea042ad/best.pt

### EXP-data_full_ctc-f2d3a83c9ba2  (2026-07-25)
- **Model:** SignTransformer + TConv2k5 d256 L4+4   **Phase/Task:** data_full_ctc (WS-C T0.2)
- **Data/Split:** data_full_ctc random-v1 (train 1439 / test 1799); source=frozen:splits/random-v1|label_fraction=0.1
- **Representation:** kp-218   **Config hash:** f2d3a83c9ba2   **Seed:** 42
- **Compute:** cuda   **Train time:** 8.5 min
- **Decode:** greedy (beam_size=1)
- **Training extras:** early_eos_penalty=0.0 ctc_weight=0.3 init_encoder=none label_fraction=0.1
- **Decode extras:** decode_repetition_stop=0
- **Metrics (test):** BLEU-4=0.11 chrF=9.23 chrF++=7.24 ROUGE-L=0.015 (BLEU-1/2/3=9.3/1.4/0.3)
- **Checkpoint:** data_full/runs/f2d3a83c9ba2/best.pt

### EXP-data_full_ctc-a2027b69137a  (2026-07-25)
- **Model:** SignTransformer + TConv2k5 d256 L4+4   **Phase/Task:** data_full_ctc (WS-C T0.2)
- **Data/Split:** data_full_ctc random-v1 (train 3598 / test 1799); source=frozen:splits/random-v1|label_fraction=0.25
- **Representation:** kp-218   **Config hash:** a2027b69137a   **Seed:** 42
- **Compute:** cuda   **Train time:** 13.4 min
- **Decode:** greedy (beam_size=1)
- **Training extras:** early_eos_penalty=0.0 ctc_weight=0.3 init_encoder=none label_fraction=0.25
- **Decode extras:** decode_repetition_stop=0
- **Metrics (test):** BLEU-4=0.14 chrF=11.24 chrF++=8.79 ROUGE-L=0.024 (BLEU-1/2/3=8.3/1.5/0.4)
- **Checkpoint:** data_full/runs/a2027b69137a/best.pt

### EXP-data_full_ctc-e519c8960e05  (2026-07-25)
- **Model:** SignTransformer + TConv2k5 d256 L4+4   **Phase/Task:** data_full_ctc (WS-C T0.2)
- **Data/Split:** data_full_ctc random-v1 (train 7195 / test 1799); source=frozen:splits/random-v1|label_fraction=0.5
- **Representation:** kp-218   **Config hash:** e519c8960e05   **Seed:** 42
- **Compute:** cuda   **Train time:** 25.4 min
- **Decode:** greedy (beam_size=1)
- **Training extras:** early_eos_penalty=0.0 ctc_weight=0.3 init_encoder=none label_fraction=0.5
- **Decode extras:** decode_repetition_stop=0
- **Metrics (test):** BLEU-4=0.22 chrF=11.12 chrF++=8.96 ROUGE-L=0.041 (BLEU-1/2/3=11.0/2.7/0.8)
- **Checkpoint:** data_full/runs/e519c8960e05/best.pt
