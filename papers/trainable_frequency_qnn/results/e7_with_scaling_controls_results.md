# e7_with_scaling_controls (COMPLETE RUN COVERAGE (union of 3 sweeps))

**MAERM (%), denominator median(|y|) (primary, A10)**

| Candidate | u min | u max | u mean | v min | v max | v mean | p min | p max | p mean | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ff | 28.9 ± 4.34 | 34.9 ± 6.8 | 31.5 ± 5.38 | 65 ± 9.12 | 83.1 ± 10.7 | 73.2 ± 8.68 | 95.5 ± 11.5 | 154 ± 17.2 | 123 ± 13 | 10 |
| tf | 5.77 ± 0.936 | 8.9 ± 0.984 | 6.62 ± 0.909 | 17.8 ± 1.31 | 22.7 ± 1.28 | 20.1 ± 1.28 | 26.8 ± 2.58 | 51.8 ± 6.19 | 34.8 ± 2.6 | 10 |
| ff_minmax | 4.41 ± 0.207 | 6.38 ± 0.628 | 4.93 ± 0.273 | 19.7 ± 0.638 | 23.8 ± 1.26 | 21.2 ± 0.843 | 25.5 ± 2.9 | 40 ± 2.67 | 31.7 ± 1.89 | 5 |
| tf_minmax | 3.54 ± 0.378 | 6.75 ± 0.977 | 4.44 ± 0.486 | 16.2 ± 0.9 | 20.5 ± 0.895 | 18.2 ± 1.05 | 18.3 ± 0.896 | 35.8 ± 1.68 | 25 ± 1.97 | 5 |
| ffs | 5.14 ± 0.442 | 8.72 ± 1.21 | 6.09 ± 0.615 | 18.2 ± 1.62 | 23.4 ± 1.53 | 20.6 ± 1.34 | 25 ± 2.68 | 42.5 ± 4.27 | 32.4 ± 2.56 | 5 |

**MAERM (%), denominator |median(y)| (signed variant)**

| Candidate | u min | u max | u mean | v min | v max | v mean | p min | p max | p mean | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ff | 28.9 ± 4.34 | 34.9 ± 6.8 | 31.5 ± 5.38 | 212 ± 32.5 | 4.82e+03 ± 701 | 919 ± 111 | 96.1 ± 11.5 | 156 ± 17.9 | 124 ± 13.1 | 10 |
| tf | 5.77 ± 0.936 | 8.9 ± 0.984 | 6.62 ± 0.909 | 61.9 ± 3.66 | 1.29e+03 ± 81.4 | 247 ± 15.2 | 26.9 ± 2.6 | 52.6 ± 6.29 | 35 ± 2.61 | 10 |
| ff_minmax | 4.41 ± 0.207 | 6.38 ± 0.628 | 4.93 ± 0.273 | 64.7 ± 2.04 | 1.41e+03 ± 76 | 266 ± 12.2 | 25.7 ± 3.01 | 40.6 ± 2.67 | 31.9 ± 1.89 | 5 |
| tf_minmax | 3.54 ± 0.378 | 6.75 ± 0.977 | 4.44 ± 0.486 | 55.2 ± 4.41 | 1.2e+03 ± 69 | 226 ± 12.8 | 18.5 ± 0.846 | 36.3 ± 1.71 | 25.1 ± 1.98 | 5 |
| ffs | 5.14 ± 0.442 | 8.72 ± 1.21 | 6.09 ± 0.615 | 62.1 ± 3.44 | 1.32e+03 ± 104 | 252 ± 17.8 | 25.3 ± 2.79 | 43.1 ± 4.39 | 32.6 ± 2.57 | 5 |


| Paired comparison | Mean difference | First lower (seeds) | Wilcoxon p |
|---|---:|---:|---:|
| tf_vs_ffs:u_mean_maerm | 0.232 | 2/5 | 0.625 |
| tf_vs_ffs:u_mean_maerm_signed | 0.232 | 2/5 | 0.625 |
| tf_vs_ffs:v_mean_maerm | -0.47 | 3/5 | 0.812 |
| tf_vs_ffs:v_mean_maerm_signed | -2.85 | 3/5 | 0.812 |
| tf_vs_ffs:p_mean_maerm | 2.03 | 2/5 | 0.312 |
| tf_vs_ffs:p_mean_maerm_signed | 2.04 | 2/5 | 0.312 |
| tf_vs_ffs:final_loss | 0.000576 | 1/5 | 0.625 |
| tf_vs_ff_minmax:u_mean_maerm | 1.39 | 0/5 | 0.0625 |
| tf_vs_ff_minmax:u_mean_maerm_signed | 1.39 | 0/5 | 0.0625 |
| tf_vs_ff_minmax:v_mean_maerm | -1.09 | 4/5 | 0.125 |
| tf_vs_ff_minmax:v_mean_maerm_signed | -17.3 | 4/5 | 0.125 |
| tf_vs_ff_minmax:p_mean_maerm | 2.73 | 1/5 | 0.125 |
| tf_vs_ff_minmax:p_mean_maerm_signed | 2.76 | 1/5 | 0.125 |
| tf_vs_ff_minmax:final_loss | 0.000664 | 1/5 | 0.312 |
| tf_minmax_vs_ff_minmax:u_mean_maerm | -0.492 | 5/5 | 0.0625 |
| tf_minmax_vs_ff_minmax:u_mean_maerm_signed | -0.492 | 5/5 | 0.0625 |
| tf_minmax_vs_ff_minmax:v_mean_maerm | -3.02 | 5/5 | 0.0625 |
| tf_minmax_vs_ff_minmax:v_mean_maerm_signed | -40.5 | 5/5 | 0.0625 |
| tf_minmax_vs_ff_minmax:p_mean_maerm | -6.7 | 5/5 | 0.0625 |
| tf_minmax_vs_ff_minmax:p_mean_maerm_signed | -6.73 | 5/5 | 0.0625 |
| tf_minmax_vs_ff_minmax:final_loss | -0.00263 | 5/5 | 0.0625 |
| tf_minmax_vs_tf:u_mean_maerm | -1.88 | 5/5 | 0.0625 |
| tf_minmax_vs_tf:u_mean_maerm_signed | -1.88 | 5/5 | 0.0625 |
| tf_minmax_vs_tf:v_mean_maerm | -1.93 | 4/5 | 0.125 |
| tf_minmax_vs_tf:v_mean_maerm_signed | -23.2 | 4/5 | 0.125 |
| tf_minmax_vs_tf:p_mean_maerm | -9.43 | 5/5 | 0.0625 |
| tf_minmax_vs_tf:p_mean_maerm_signed | -9.49 | 5/5 | 0.0625 |
| tf_minmax_vs_tf:final_loss | -0.0033 | 5/5 | 0.0625 |
| ffs_vs_ff:u_mean_maerm | -25.3 | 5/5 | 0.0625 |
| ffs_vs_ff:u_mean_maerm_signed | -25.3 | 5/5 | 0.0625 |
| ffs_vs_ff:v_mean_maerm | -52.9 | 5/5 | 0.0625 |
| ffs_vs_ff:v_mean_maerm_signed | -674 | 5/5 | 0.0625 |
| ffs_vs_ff:p_mean_maerm | -91.7 | 5/5 | 0.0625 |
| ffs_vs_ff:p_mean_maerm_signed | -92.2 | 5/5 | 0.0625 |
| ffs_vs_ff:final_loss | -0.222 | 5/5 | 0.0625 |
| ff_minmax_vs_ff:u_mean_maerm | -26.4 | 5/5 | 0.0625 |
| ff_minmax_vs_ff:u_mean_maerm_signed | -26.4 | 5/5 | 0.0625 |
| ff_minmax_vs_ff:v_mean_maerm | -52.3 | 5/5 | 0.0625 |
| ff_minmax_vs_ff:v_mean_maerm_signed | -659 | 5/5 | 0.0625 |
| ff_minmax_vs_ff:p_mean_maerm | -92.4 | 5/5 | 0.0625 |
| ff_minmax_vs_ff:p_mean_maerm_signed | -92.9 | 5/5 | 0.0625 |
| ff_minmax_vs_ff:final_loss | -0.222 | 5/5 | 0.0625 |
