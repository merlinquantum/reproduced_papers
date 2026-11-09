# m6_table1_merlin (COMPLETE RUN COVERAGE (union of 3 sweeps))

**MAERM (%), denominator median(|y|) (primary, A10)**

| Candidate | u min | u max | u mean | v min | v max | v mean | p min | p max | p mean | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tf | 20.4 ± 7.4 | 23.1 ± 7.67 | 21.7 ± 7.44 | 89.1 ± 29 | 101 ± 28.2 | 94.9 ± 28.6 | 71.6 ± 13.5 | 93.4 ± 14.1 | 81.2 ± 14.3 | 10 |
| ff | 95.5 ± 0.185 | 99 ± 0.151 | 96.8 ± 0.0504 | 86.1 ± 0.486 | 94.5 ± 1.16 | 90.8 ± 0.595 | 108 ± 1.49 | 143 ± 3.11 | 125 ± 1.57 | 10 |
| ffs | 21.8 ± 1.91 | 23.4 ± 1.92 | 22.6 ± 1.85 | 108 ± 1.48 | 118 ± 0.785 | 113 ± 0.364 | 94.3 ± 6.41 | 117 ± 7.1 | 105 ± 6.49 | 5 |

**MAERM (%), denominator |median(y)| (signed variant)**

| Candidate | u min | u max | u mean | v min | v max | v mean | p min | p max | p mean | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tf | 20.4 ± 7.4 | 23.1 ± 7.67 | 21.7 ± 7.44 | 296 ± 88.8 | 5.72e+03 ± 1.79e+03 | 1.14e+03 ± 354 | 71.6 ± 13.5 | 93.8 ± 13.7 | 81.7 ± 14.4 | 10 |
| ff | 95.5 ± 0.185 | 99 ± 0.151 | 96.8 ± 0.0504 | 278 ± 1.44 | 5.44e+03 ± 25.7 | 1.1e+03 ± 5.71 | 108 ± 1.49 | 143 ± 2.93 | 126 ± 1.58 | 10 |
| ffs | 21.8 ± 1.91 | 23.4 ± 1.92 | 22.6 ± 1.85 | 356 ± 3.92 | 6.77e+03 ± 48 | 1.36e+03 ± 3.71 | 94.3 ± 6.41 | 117 ± 7.1 | 106 ± 6.53 | 5 |


| Paired comparison | Mean difference | First lower (seeds) | Wilcoxon p |
|---|---:|---:|---:|
| tf_vs_ff:u_mean_maerm | -75.1 | 10/10 | 0.00195 |
| tf_vs_ff:u_mean_maerm_signed | -75.1 | 10/10 | 0.00195 |
| tf_vs_ff:v_mean_maerm | 4.17 | 3/10 | 0.492 |
| tf_vs_ff:v_mean_maerm_signed | 38 | 3/10 | 0.492 |
| tf_vs_ff:p_mean_maerm | -43.6 | 10/10 | 0.00195 |
| tf_vs_ff:p_mean_maerm_signed | -43.9 | 10/10 | 0.00195 |
| tf_vs_ff:final_loss | -0.794 | 10/10 | 0.00195 |
| tf_vs_ffs:u_mean_maerm | 1.44 | 4/5 | 0.625 |
| tf_vs_ffs:u_mean_maerm_signed | 1.44 | 4/5 | 0.625 |
| tf_vs_ffs:v_mean_maerm | -19.9 | 5/5 | 0.0625 |
| tf_vs_ffs:v_mean_maerm_signed | -240 | 5/5 | 0.0625 |
| tf_vs_ffs:p_mean_maerm | -24.2 | 5/5 | 0.0625 |
| tf_vs_ffs:p_mean_maerm_signed | -24.4 | 5/5 | 0.0625 |
| tf_vs_ffs:final_loss | -0.0202 | 5/5 | 0.0625 |
| ffs_vs_ff:u_mean_maerm | -74.1 | 5/5 | 0.0625 |
| ffs_vs_ff:u_mean_maerm_signed | -74.1 | 5/5 | 0.0625 |
| ffs_vs_ff:v_mean_maerm | 22.5 | 0/5 | 0.0625 |
| ffs_vs_ff:v_mean_maerm_signed | 257 | 0/5 | 0.0625 |
| ffs_vs_ff:p_mean_maerm | -19.8 | 5/5 | 0.0625 |
| ffs_vs_ff:p_mean_maerm_signed | -19.9 | 5/5 | 0.0625 |
| ffs_vs_ff:final_loss | -0.767 | 5/5 | 0.0625 |
