# Significance & Confidence Intervals

Bootstrap n=1000, seed=0. Paired p-values vs `vanilla`.

| Run | Metric | Point | 95% CI | p vs vanilla |
|---|---|---:|---|---:|
| vanilla | bleu4 | 0.520 | [0.217, 0.840] | - |
| vanilla | chrf++ | 9.646 | [9.342, 9.953] | - |
| tconv_ctc | bleu4 | 0.684 | [0.348, 0.971] | 0.462 |
| tconv_ctc | chrf++ | 10.913 | [10.589, 11.243] | 0.000 |
