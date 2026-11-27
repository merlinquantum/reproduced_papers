### Data robustness (protocol paper, init seed 0, data seeds 101, 102, 103)

| Model | data seed 101 | data seed 102 | data seed 103 | mean |
|---|---:|---:|---:|---:|
| quantum (biased) L=1 | 4.653e-03 | 5.219e-03 | 5.248e-03 | 5.040e-03 |
| quantum (biased) L=2 | 5.196e-03 | 6.154e-03 | 6.042e-03 | 5.797e-03 |
| quantum (no bias, Fig. 10 CNOTs) L=1 | 6.454e-03 | 5.668e-03 | 6.944e-03 | 6.355e-03 |
| quantum (no bias, Fig. 10 CNOTs) L=2 | 7.850e-03 | 7.782e-03 | 8.264e-03 | 7.965e-03 |
| surrogate L=1 | 6.070e-03 | 6.321e-03 | 6.214e-03 | 6.202e-03 |
| surrogate L=2 | 1.866e-02 | 2.024e-02 | 1.734e-02 | 1.875e-02 |

| Biased | Other | Mean paired diff | 95% CI (paired t) | paired t p | Biased lower |
|---|---|---:|---|---:|---:|
| quantum (biased) L=1 | quantum (biased) L=2 | -7.57e-04 | [-1.25e-03, -2.65e-04] | 0.022 | 3/3 |
| quantum (biased) L=1 | quantum (no bias, Fig. 10 CNOTs) L=1 | -1.32e-03 | [-3.18e-03, 5.53e-04] | 0.094 | 3/3 |
| quantum (biased) L=1 | quantum (no bias, Fig. 10 CNOTs) L=2 | -2.93e-03 | [-3.74e-03, -2.11e-03] | 0.0041 | 3/3 |
| quantum (biased) L=1 | surrogate L=1 | -1.16e-03 | [-1.74e-03, -5.86e-04] | 0.013 | 3/3 |
| quantum (biased) L=1 | surrogate L=2 | -1.37e-02 | [-1.74e-02, -1.00e-02] | 0.0039 | 3/3 |
| quantum (biased) L=2 | quantum (biased) L=1 | 7.57e-04 | [2.65e-04, 1.25e-03] | 0.022 | 0/3 |
| quantum (biased) L=2 | quantum (no bias, Fig. 10 CNOTs) L=1 | -5.58e-04 | [-2.85e-03, 1.73e-03] | 0.4 | 2/3 |
| quantum (biased) L=2 | quantum (no bias, Fig. 10 CNOTs) L=2 | -2.17e-03 | [-3.45e-03, -8.89e-04] | 0.018 | 3/3 |
| quantum (biased) L=2 | surrogate L=1 | -4.05e-04 | [-1.42e-03, 6.06e-04] | 0.23 | 3/3 |
| quantum (biased) L=2 | surrogate L=2 | -1.30e-02 | [-1.66e-02, -9.31e-03] | 0.0043 | 3/3 |
