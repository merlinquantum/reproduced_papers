# e7_appc_combined (COMPLETE RUN COVERAGE (union of 2 sweeps))

**MAERM (%), denominator median(|y|) (primary, A10)**

| Candidate | u min | u max | u mean | v min | v max | v mean | p min | p max | p mean | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ff | 28.9 ± 4.34 | 34.9 ± 6.8 | 31.5 ± 5.38 | 65 ± 9.12 | 83.1 ± 10.7 | 73.2 ± 8.68 | 95.5 ± 11.5 | 154 ± 17.2 | 123 ± 13 | 10 |
| tf | 5.77 ± 0.936 | 8.9 ± 0.984 | 6.62 ± 0.909 | 17.8 ± 1.31 | 22.7 ± 1.28 | 20.1 ± 1.28 | 26.8 ± 2.58 | 51.8 ± 6.19 | 34.8 ± 2.6 | 10 |

**MAERM (%), denominator |median(y)| (signed variant)**

| Candidate | u min | u max | u mean | v min | v max | v mean | p min | p max | p mean | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ff | 28.9 ± 4.34 | 34.9 ± 6.8 | 31.5 ± 5.38 | 212 ± 32.5 | 4.82e+03 ± 701 | 919 ± 111 | 96.1 ± 11.5 | 156 ± 17.9 | 124 ± 13.1 | 10 |
| tf | 5.77 ± 0.936 | 8.9 ± 0.984 | 6.62 ± 0.909 | 61.9 ± 3.66 | 1.29e+03 ± 81.4 | 247 ± 15.2 | 26.9 ± 2.6 | 52.6 ± 6.29 | 35 ± 2.61 | 10 |


| Paired comparison | Mean difference | First lower (seeds) | Wilcoxon p |
|---|---:|---:|---:|
| tf_vs_ff:u_mean_maerm | -24.9 | 10/10 | 0.00195 |
| tf_vs_ff:u_mean_maerm_signed | -24.9 | 10/10 | 0.00195 |
| tf_vs_ff:v_mean_maerm | -53.1 | 10/10 | 0.00195 |
| tf_vs_ff:v_mean_maerm_signed | -672 | 10/10 | 0.00195 |
| tf_vs_ff:p_mean_maerm | -88.6 | 10/10 | 0.00195 |
| tf_vs_ff:p_mean_maerm_signed | -89.1 | 10/10 | 0.00195 |
| tf_vs_ff:final_loss | -0.222 | 10/10 | 0.00195 |
