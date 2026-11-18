# e4b_scale_init (COMPLETE SWEEP)

| Candidate | Params | Train MSE (mean ± std) | Dense MSE (mean ± std) | Median train MSE | n |
|---|---:|---:|---:|---:|---:|
| ff_fixed1.5 | 36 | 6.23e-09 ± 1.9e-08 | 6.22e-09 ± 1.9e-08 | 7.15e-12 | 10 |
| ffs_init1.5 | 37 | 4.39e-06 ± 1.39e-05 | 4.41e-06 ± 1.39e-05 | 2.08e-11 | 10 |
| tf_init1.5 | 39 | 1.76e-06 ± 5.54e-06 | 1.77e-06 ± 5.56e-06 | 2.58e-11 | 10 |

| Paired comparison | Mean difference | First lower (seeds) | Wilcoxon p |
|---|---:|---:|---:|
| ffs_init1.5_vs_ff_fixed1.5:train_mse | 4.39e-06 | 1/10 | 0.00977 |
| ffs_init1.5_vs_ff_fixed1.5:dense_mse | 4.4e-06 | 1/10 | 0.00977 |
| tf_init1.5_vs_ff_fixed1.5:train_mse | 1.76e-06 | 3/10 | 0.105 |
| tf_init1.5_vs_ff_fixed1.5:dense_mse | 1.76e-06 | 3/10 | 0.105 |
