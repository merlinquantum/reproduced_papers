# e1_fig2a (COMPLETE SWEEP)

| Candidate | Params | Train MSE (mean ± std) | Dense MSE (mean ± std) | Median train MSE | n |
|---|---:|---:|---:|---:|---:|
| ff_n3 | 36 | 0.00982 ± 0.00894 | 0.00976 ± 0.0089 | 0.0064 | 10 |
| ff_n4 | 48 | 0.000476 ± 0.000365 | 0.000476 ± 0.000367 | 0.000395 | 10 |

| Paired comparison | Mean difference | First lower (seeds) | Wilcoxon p |
|---|---:|---:|---:|
| ff_n4_vs_ff_n3:train_mse | -0.00934 | 10/10 | 0.00195 |
| ff_n4_vs_ff_n3:dense_mse | -0.00928 | 10/10 | 0.00195 |
