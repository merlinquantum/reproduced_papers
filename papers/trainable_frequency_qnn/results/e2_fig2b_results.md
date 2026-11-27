# e2_fig2b (COMPLETE SWEEP)

| Candidate | Params | Train MSE (mean ± std) | Dense MSE (mean ± std) | Median train MSE | n |
|---|---:|---:|---:|---:|---:|
| ff | 36 | 0.0508 ± 0.000561 | 0.0506 ± 0.000608 | 0.0507 | 10 |
| tf | 39 | 0.00116 ± 0.00136 | 0.00113 ± 0.00133 | 0.000808 | 10 |
| ffs | 37 | 0.0537 ± 0.00785 | 0.0536 ± 0.00784 | 0.0489 | 10 |
| mlp | 40 | 0.15 ± 0.00749 | 0.15 ± 0.00749 | 0.148 | 10 |
| fourier | 10 | 0.00953 ± 5.07e-05 | 0.00957 ± 5.4e-05 | 0.00952 | 10 |
| ff_n4 | 48 | 0.0506 ± 9.28e-05 | 0.0503 ± 0.000102 | 0.0506 | 10 |
| tf_n4 | 52 | 0.00131 ± 0.00147 | 0.00128 ± 0.00143 | 0.000691 | 10 |

| Paired comparison | Mean difference | First lower (seeds) | Wilcoxon p |
|---|---:|---:|---:|
| tf_vs_ff:train_mse | -0.0497 | 10/10 | 0.00195 |
| tf_vs_ff:dense_mse | -0.0494 | 10/10 | 0.00195 |
| tf_vs_ffs:train_mse | -0.0525 | 10/10 | 0.00195 |
| tf_vs_ffs:dense_mse | -0.0525 | 10/10 | 0.00195 |
| ffs_vs_ff:train_mse | 0.00286 | 7/10 | 1 |
| ffs_vs_ff:dense_mse | 0.00305 | 7/10 | 1 |
| tf_vs_mlp:train_mse | -0.149 | 10/10 | 0.00195 |
| tf_vs_mlp:dense_mse | -0.149 | 10/10 | 0.00195 |
| tf_vs_fourier:train_mse | -0.00837 | 10/10 | 0.00195 |
| tf_vs_fourier:dense_mse | -0.00844 | 10/10 | 0.00195 |
| tf_n4_vs_ff_n4:train_mse | -0.0493 | 10/10 | 0.00195 |
| tf_n4_vs_ff_n4:dense_mse | -0.049 | 10/10 | 0.00195 |
