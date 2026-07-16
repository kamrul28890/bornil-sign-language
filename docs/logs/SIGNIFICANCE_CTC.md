# Significance & Confidence Intervals

Bootstrap n=200, seed=0. Paired p-values vs `tconv`.

| Run | Metric | Point | 95% CI | p vs tconv |
|---|---|---:|---|---:|
| tconv | bleu4 | 0.655 | [0.377, 0.895] | - |
| tconv | chrf++ | 9.482 | [9.199, 9.752] | - |
| tconv_ctc | bleu4 | 0.684 | [0.397, 0.958] | 0.880 |
| tconv_ctc | chrf++ | 10.913 | [10.636, 11.225] | 0.000 |
