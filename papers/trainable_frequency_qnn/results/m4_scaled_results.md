# m4_scaled (COMPLETE SWEEP)

| Candidate | Params | Train MSE (mean ± std) | Dense MSE (mean ± std) | Median train MSE | n |
|---|---:|---:|---:|---:|---:|
| ff | 60 | 0.168 ± 0.00334 | 0.165 ± 0.00328 | 0.169 | 10 |
| ffs | 61 | 0.168 ± 0.00213 | 0.165 ± 0.00213 | 0.168 | 10 |
| tf | 63 | 0.162 ± 0.00407 | 0.16 ± 0.00385 | 0.162 | 10 |
| ff_init1.5 | 60 | 0.0648 ± 0.0114 | 0.0643 ± 0.0112 | 0.0628 | 10 |
| ffs_init1.5 | 61 | 0.065 ± 0.0114 | 0.0645 ± 0.0112 | 0.0627 | 10 |
| tf_init1.5 | 63 | 0.0654 ± 0.0125 | 0.0649 ± 0.0123 | 0.0628 | 10 |

| Paired comparison | Mean difference | First lower (seeds) | Wilcoxon p |
|---|---:|---:|---:|
| tf_vs_ff:train_mse | -0.00577 | 9/10 | 0.00586 |
| tf_vs_ff:dense_mse | -0.00522 | 9/10 | 0.00586 |
| ffs_init1.5_vs_ffs:train_mse | -0.103 | 10/10 | 0.00195 |
| ffs_init1.5_vs_ffs:dense_mse | -0.101 | 10/10 | 0.00195 |
| tf_init1.5_vs_tf:train_mse | -0.0965 | 10/10 | 0.00195 |
| tf_init1.5_vs_tf:dense_mse | -0.095 | 10/10 | 0.00195 |
