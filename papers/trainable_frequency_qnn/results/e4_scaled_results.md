# e4_scaled (COMPLETE SWEEP)

| Candidate | Params | Train MSE (mean ± std) | Dense MSE (mean ± std) | Median train MSE | n |
|---|---:|---:|---:|---:|---:|
| ff | 36 | 0.113 ± 0.000393 | 0.112 ± 0.000384 | 0.113 | 10 |
| ffs | 37 | 0.125 ± 0.0223 | 0.124 ± 0.0218 | 0.114 | 10 |
| tf | 39 | 0.109 ± 0.00257 | 0.107 ± 0.00242 | 0.108 | 10 |

| Paired comparison | Mean difference | First lower (seeds) | Wilcoxon p |
|---|---:|---:|---:|
| ffs_vs_ff:train_mse | 0.0119 | 1/10 | 0.00391 |
| ffs_vs_ff:dense_mse | 0.0115 | 1/10 | 0.00586 |
| tf_vs_ff:train_mse | -0.00486 | 10/10 | 0.00195 |
| tf_vs_ff:dense_mse | -0.00485 | 10/10 | 0.00195 |
| tf_vs_ffs:train_mse | -0.0167 | 10/10 | 0.00195 |
| tf_vs_ffs:dense_mse | -0.0164 | 10/10 | 0.00195 |
