# m2_fig2b (COMPLETE SWEEP)

| Candidate | Params | Train MSE (mean ± std) | Dense MSE (mean ± std) | Median train MSE | n |
|---|---:|---:|---:|---:|---:|
| ff | 60 | 0.101 ± 0.00342 | 0.101 ± 0.00345 | 0.103 | 10 |
| tf | 63 | 0.0566 ± 0.00198 | 0.0565 ± 0.00192 | 0.0558 | 10 |
| ffs | 61 | 0.0673 ± 0.000144 | 0.0673 ± 0.000148 | 0.0674 | 10 |
| ff_unbunched | 60 | 0.0664 ± 0.0174 | 0.0664 ± 0.0176 | 0.0601 | 10 |
| tf_unbunched | 63 | 0.0374 ± 0.0201 | 0.0374 ± 0.0201 | 0.0475 | 10 |

| Paired comparison | Mean difference | First lower (seeds) | Wilcoxon p |
|---|---:|---:|---:|
| tf_vs_ff:train_mse | -0.0439 | 10/10 | 0.00195 |
| tf_vs_ff:dense_mse | -0.0445 | 10/10 | 0.00195 |
| tf_vs_ffs:train_mse | -0.0107 | 10/10 | 0.00195 |
| tf_vs_ffs:dense_mse | -0.0109 | 10/10 | 0.00195 |
| ffs_vs_ff:train_mse | -0.0332 | 10/10 | 0.00195 |
| ffs_vs_ff:dense_mse | -0.0337 | 10/10 | 0.00195 |
| tf_unbunched_vs_ff_unbunched:train_mse | -0.0289 | 10/10 | 0.00195 |
| tf_unbunched_vs_ff_unbunched:dense_mse | -0.029 | 10/10 | 0.00195 |
