# e6_table1_with_controls (COMPLETE RUN COVERAGE (union of 2 sweeps))

**MAERM (%), denominator median(|y|) (primary, A10)**

| Candidate | u min | u max | u mean | v min | v max | v mean | p min | p max | p mean | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ff | 19.5 ± 0.683 | 20.7 ± 1.4 | 20.1 ± 0.932 | 56.8 ± 2.09 | 66.2 ± 2.59 | 61 ± 1.83 | 61.1 ± 4.03 | 92.9 ± 5.48 | 76.2 ± 3.46 | 10 |
| tf | 8.44 ± 1.27 | 10 ± 2.02 | 9.02 ± 1.42 | 27.4 ± 2.98 | 34.7 ± 4.37 | 30.4 ± 3.46 | 37 ± 3.06 | 58.8 ± 5.7 | 45.6 ± 3.61 | 10 |
| ffs | 10.6 ± 1.19 | 13 ± 2.8 | 11.2 ± 1.49 | 31.3 ± 1.94 | 40.7 ± 4.47 | 35.6 ± 2.9 | 38.2 ± 3.86 | 65.5 ± 13.5 | 49.9 ± 5.55 | 5 |
| mlp_lr1e-2 | 3.52 ± 0.258 | 4.89 ± 0.385 | 3.96 ± 0.257 | 14.3 ± 1.22 | 23 ± 2.37 | 17.1 ± 1.52 | 23 ± 4.01 | 39.9 ± 5.58 | 30.7 ± 4.02 | 5 |
| mlp_lr1e-3 | 4.66 ± 0.514 | 7.05 ± 0.91 | 5.4 ± 0.702 | 19.5 ± 2.14 | 38.6 ± 2.74 | 24.1 ± 2.21 | 32.6 ± 3.94 | 54.7 ± 4.96 | 43.8 ± 4.63 | 5 |

**MAERM (%), denominator |median(y)| (signed variant)**

| Candidate | u min | u max | u mean | v min | v max | v mean | p min | p max | p mean | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ff | 19.5 ± 0.683 | 20.7 ± 1.4 | 20.1 ± 0.932 | 175 ± 6.95 | 3.94e+03 ± 164 | 770 ± 24.8 | 61.4 ± 4.05 | 93 ± 5.31 | 76.7 ± 3.48 | 10 |
| tf | 8.44 ± 1.27 | 10 ± 2.02 | 9.02 ± 1.42 | 92.7 ± 9.1 | 1.9e+03 ± 330 | 369 ± 52.6 | 37.2 ± 2.96 | 59.7 ± 5.79 | 45.9 ± 3.64 | 10 |
| ffs | 10.6 ± 1.19 | 13 ± 2.8 | 11.2 ± 1.49 | 107 ± 7.72 | 2.39e+03 ± 307 | 445 ± 43.4 | 38.3 ± 3.87 | 66.5 ± 13.7 | 50.2 ± 5.59 | 5 |
| mlp_lr1e-2 | 3.52 ± 0.258 | 4.89 ± 0.385 | 3.96 ± 0.257 | 48.6 ± 3.24 | 1.24e+03 ± 166 | 219 ± 24.7 | 23 ± 4.01 | 40.2 ± 5.61 | 30.9 ± 4.04 | 5 |
| mlp_lr1e-3 | 4.66 ± 0.514 | 7.05 ± 0.91 | 5.4 ± 0.702 | 66 ± 9 | 1.65e+03 ± 150 | 297 ± 24 | 32.6 ± 4.01 | 55.1 ± 5.16 | 44.1 ± 4.66 | 5 |


| Paired comparison | Mean difference | First lower (seeds) | Wilcoxon p |
|---|---:|---:|---:|
| tf_vs_ffs:u_mean_maerm | -2.11 | 5/5 | 0.0625 |
| tf_vs_ffs:u_mean_maerm_signed | -2.11 | 5/5 | 0.0625 |
| tf_vs_ffs:v_mean_maerm | -5.44 | 5/5 | 0.0625 |
| tf_vs_ffs:v_mean_maerm_signed | -81.2 | 5/5 | 0.0625 |
| tf_vs_ffs:p_mean_maerm | -4.36 | 5/5 | 0.0625 |
| tf_vs_ffs:p_mean_maerm_signed | -4.39 | 5/5 | 0.0625 |
| tf_vs_ffs:final_loss | -0.00621 | 5/5 | 0.0625 |
| ffs_vs_ff:u_mean_maerm | -8.69 | 5/5 | 0.0625 |
| ffs_vs_ff:u_mean_maerm_signed | -8.69 | 5/5 | 0.0625 |
| ffs_vs_ff:v_mean_maerm | -25.2 | 5/5 | 0.0625 |
| ffs_vs_ff:v_mean_maerm_signed | -323 | 5/5 | 0.0625 |
| ffs_vs_ff:p_mean_maerm | -25.3 | 5/5 | 0.0625 |
| ffs_vs_ff:p_mean_maerm_signed | -25.4 | 5/5 | 0.0625 |
| ffs_vs_ff:final_loss | -0.0373 | 5/5 | 0.0625 |
| tf_vs_mlp_lr1e-2:u_mean_maerm | 5.13 | 0/5 | 0.0625 |
| tf_vs_mlp_lr1e-2:u_mean_maerm_signed | 5.13 | 0/5 | 0.0625 |
| tf_vs_mlp_lr1e-2:v_mean_maerm | 13 | 0/5 | 0.0625 |
| tf_vs_mlp_lr1e-2:v_mean_maerm_signed | 145 | 0/5 | 0.0625 |
| tf_vs_mlp_lr1e-2:p_mean_maerm | 14.9 | 0/5 | 0.0625 |
| tf_vs_mlp_lr1e-2:p_mean_maerm_signed | 15 | 0/5 | 0.0625 |
| tf_vs_mlp_lr1e-2:final_loss | 0.0129 | 0/5 | 0.0625 |
| tf_vs_mlp_lr1e-3:u_mean_maerm | 3.69 | 0/5 | 0.0625 |
| tf_vs_mlp_lr1e-3:u_mean_maerm_signed | 3.69 | 0/5 | 0.0625 |
| tf_vs_mlp_lr1e-3:v_mean_maerm | 6.02 | 0/5 | 0.0625 |
| tf_vs_mlp_lr1e-3:v_mean_maerm_signed | 67.1 | 0/5 | 0.0625 |
| tf_vs_mlp_lr1e-3:p_mean_maerm | 1.72 | 2/5 | 1 |
| tf_vs_mlp_lr1e-3:p_mean_maerm_signed | 1.74 | 2/5 | 1 |
| tf_vs_mlp_lr1e-3:final_loss | 0.00548 | 0/5 | 0.0625 |
