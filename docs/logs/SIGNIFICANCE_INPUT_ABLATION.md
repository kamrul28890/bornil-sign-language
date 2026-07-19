# Significance & Confidence Intervals

Bootstrap n=1000, seed=0. Paired p-values vs `none`.

| Run | Metric | Point | 95% CI | p vs none |
|---|---|---:|---|---:|
| none | bleu4 | 0.684 | [0.348, 0.971] | - |
| none | chrf++ | 10.913 | [10.589, 11.243] | - |
| frame_shuffle | bleu4 | 0.074 | [0.046, 0.095] | 0.000 |
| frame_shuffle | chrf++ | 10.036 | [9.777, 10.302] | 0.000 |
| mismatch | bleu4 | 0.050 | [0.041, 0.056] | 0.000 |
| mismatch | chrf++ | 6.195 | [5.997, 6.388] | 0.000 |
