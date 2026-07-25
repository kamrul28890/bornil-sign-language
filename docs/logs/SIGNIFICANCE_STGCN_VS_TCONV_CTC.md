# Significance & Confidence Intervals

Bootstrap n=1000, seed=0. Paired p-values vs `tconv_ctc`.

| Run | Metric | Point | 95% CI | p vs tconv_ctc |
|---|---|---:|---|---:|
| tconv_ctc | bleu4 | 0.684 | [0.348, 0.971] | - |
| tconv_ctc | chrf++ | 10.913 | [10.589, 11.243] | - |
| stgcn | bleu4 | 0.356 | [0.160, 0.557] | 0.092 |
| stgcn | chrf++ | 8.265 | [8.041, 8.520] | 0.000 |
