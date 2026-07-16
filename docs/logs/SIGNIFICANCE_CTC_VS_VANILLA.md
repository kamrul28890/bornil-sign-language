# Significance & Confidence Intervals

Bootstrap n=200, seed=0. Paired p-values vs `vanilla`.

| Run | Metric | Point | 95% CI | p vs vanilla |
|---|---|---:|---|---:|
| vanilla | bleu4 | 0.520 | [0.217, 0.797] | - |
| vanilla | chrf++ | 9.646 | [9.334, 9.868] | - |
| tconv_ctc | bleu4 | 0.684 | [0.397, 0.958] | 0.390 |
| tconv_ctc | chrf++ | 10.913 | [10.636, 11.225] | 0.000 |
