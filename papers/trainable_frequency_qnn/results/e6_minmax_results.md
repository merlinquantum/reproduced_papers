# e6_minmax (COMPLETE SWEEP)

**MAERM (%), denominator median(|y|) (primary, A10)**

| Candidate | u min | u max | u mean | v min | v max | v mean | p min | p max | p mean | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ff_minmax | 13.8 ± 0.196 | 15.7 ± 0.546 | 14.6 ± 0.311 | 78.7 ± 2.87 | 99.9 ± 2.54 | 87.7 ± 2.7 | 72.7 ± 2.74 | 102 ± 4.31 | 86.4 ± 2.91 | 5 |
| tf_minmax | 7.25 ± 1.41 | 7.9 ± 1.39 | 7.55 ± 1.45 | 26 ± 1.42 | 31.6 ± 1.83 | 28.5 ± 1.63 | 34 ± 1.36 | 51.2 ± 4.02 | 41.8 ± 1.84 | 5 |

**MAERM (%), denominator |median(y)| (signed variant)**

| Candidate | u min | u max | u mean | v min | v max | v mean | p min | p max | p mean | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ff_minmax | 13.8 ± 0.196 | 15.7 ± 0.546 | 14.6 ± 0.311 | 261 ± 9.21 | 5.83e+03 ± 166 | 1.11e+03 ± 32.2 | 72.7 ± 2.74 | 104 ± 4.36 | 86.9 ± 2.93 | 5 |
| tf_minmax | 7.25 ± 1.41 | 7.9 ± 1.39 | 7.55 ± 1.45 | 89.1 ± 5.16 | 1.68e+03 ± 86.7 | 337 ± 18.8 | 34.2 ± 1.3 | 51.9 ± 4.31 | 42.1 ± 1.86 | 5 |


| Paired comparison | Mean difference | First lower (seeds) | Wilcoxon p |
|---|---:|---:|---:|
| tf_minmax_vs_ff_minmax:u_mean_maerm | -7.01 | 5/5 | 0.0625 |
| tf_minmax_vs_ff_minmax:u_mean_maerm_signed | -7.01 | 5/5 | 0.0625 |
| tf_minmax_vs_ff_minmax:v_mean_maerm | -59.2 | 5/5 | 0.0625 |
| tf_minmax_vs_ff_minmax:v_mean_maerm_signed | -768 | 5/5 | 0.0625 |
| tf_minmax_vs_ff_minmax:p_mean_maerm | -44.6 | 5/5 | 0.0625 |
| tf_minmax_vs_ff_minmax:p_mean_maerm_signed | -44.9 | 5/5 | 0.0625 |
| tf_minmax_vs_ff_minmax:final_loss | -0.0707 | 5/5 | 0.0625 |
