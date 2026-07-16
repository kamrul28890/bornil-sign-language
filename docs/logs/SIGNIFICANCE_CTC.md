# Significance & Confidence Intervals

Bootstrap n=1000, seed=0. Paired p-values vs `tconv`.

| Run | Metric | Point | 95% CI | p vs tconv |
|---|---|---:|---|---:|
| tconv | bleu4 | 0.655 | [0.373, 0.917] | - |
| tconv | chrf++ | 9.482 | [9.200, 9.765] | - |
| tconv_ctc | bleu4 | 0.684 | [0.348, 0.971] | 0.894 |
| tconv_ctc | chrf++ | 10.913 | [10.589, 11.243] | 0.000 |
