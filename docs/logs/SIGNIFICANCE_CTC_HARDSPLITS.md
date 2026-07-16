# Joint CTC Hard-Split Significance

All comparisons use two-sided paired bootstrap, n=1000, seed=0.

## signerindep-v1

| Run | Metric | Point | 95% CI | p vs vanilla_si |
|---|---|---:|---|---:|
| vanilla_si | bleu4 | 0.351 | [0.165, 0.539] | - |
| vanilla_si | chrf++ | 8.382 | [8.107, 8.678] | - |
| ctc_si | bleu4 | 0.792 | [0.520, 1.061] | 0.000 |
| ctc_si | chrf++ | 10.361 | [10.029, 10.700] | 0.000 |

## unseen-sentence-v1

| Run | Metric | Point | 95% CI | p vs vanilla_unseen |
|---|---|---:|---|---:|
| vanilla_unseen | bleu4 | 0.419 | [0.213, 0.598] | - |
| vanilla_unseen | chrf++ | 9.187 | [8.900, 9.468] | - |
| ctc_unseen | bleu4 | 0.889 | [0.565, 1.179] | 0.014 |
| ctc_unseen | chrf++ | 10.943 | [10.653, 11.250] | 0.000 |
