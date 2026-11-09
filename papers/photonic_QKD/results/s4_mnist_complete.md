# S4 MNIST: PQKD vs baselines (complete)

Label: partial RED-tier run (30 epochs, 3 seeds)

Coverage: 27/27 runs. Missing: none

## Per-arm results (final epoch, mean ± s.d. over seeds)

| arm | test acc % | val acc % | val CE | train acc % | Δ vs z=0 (pp) | trainable params | C_x |
|---|---|---|---|---|---|---|---|
| pqkd | 99.13 ± 0.04 (n=3) | 98.92 ± 0.16 (n=3) | 0.0337 ± 0.0019 (n=3) | 99.23 ± 0.04 (n=3) | 87.78 ± 0.04 (n=3) | 149439 ± 0 (n=3) | 1.502 ± 0.000 (n=3) |
| B1_kd_dense | 99.29 ± 0.08 (n=3) | 99.17 ± 0.04 (n=3) | 0.0302 ± 0.0007 (n=3) | 99.49 ± 0.07 (n=3) | — | 148626 ± 0 (n=3) | 1.510 ± 0.000 (n=3) |
| B0_zero | 10.81 ± 0.93 (n=3) | 11.14 ± 1.25 (n=3) | 2.3029 ± 0.0015 (n=3) | 10.75 ± 0.82 (n=3) | 0.00 ± 0.00 (n=3) | 149394 ± 0 (n=3) | 1.502 ± 0.000 (n=3) |
| B4_fixed_photonic | 99.14 ± 0.09 (n=3) | 98.90 ± 0.12 (n=3) | 0.0361 ± 0.0030 (n=3) | 99.20 ± 0.07 (n=3) | 87.79 ± 0.09 (n=3) | 149394 ± 0 (n=3) | 1.502 ± 0.000 (n=3) |
| B2_dictionary_trainable | 99.26 ± 0.06 (n=3) | 99.05 ± 0.06 (n=3) | 0.0311 ± 0.0019 (n=3) | 99.67 ± 0.03 (n=3) | — | 182418 ± 0 (n=3) | 1.230 ± 0.000 (n=3) |
| B3f_random_fixed | 99.18 ± 0.02 (n=3) | 98.95 ± 0.11 (n=3) | 0.0345 ± 0.0047 (n=3) | 99.36 ± 0.05 (n=3) | 87.83 ± 0.02 (n=3) | 149394 ± 0 (n=3) | 1.502 ± 0.000 (n=3) |
| B6_classical_histogram | 98.48 ± 0.09 (n=3) | 98.13 ± 0.12 (n=3) | 0.0655 ± 0.0014 (n=3) | 97.83 ± 0.51 (n=3) | 88.48 ± 1.16 (n=3) | 149394 ± 0 (n=3) | 1.502 ± 0.000 (n=3) |
| B3r_random_resampled | 98.15 ± 0.35 (n=3) | 97.78 ± 0.41 (n=3) | 0.0814 ± 0.0148 (n=3) | 96.81 ± 0.55 (n=3) | 86.80 ± 0.35 (n=3) | 149394 ± 0 (n=3) | 1.502 ± 0.000 (n=3) |
| B5_surrogate | 98.82 ± 0.09 (n=3) | 98.44 ± 0.16 (n=3) | 0.0502 ± 0.0053 (n=3) | 98.29 ± 0.37 (n=3) | 88.54 ± 0.92 (n=3) | 149439 ± 0 (n=3) | 1.502 ± 0.000 (n=3) |

## Paired per-seed differences vs pqkd (arm − pqkd)

Rule: no winner declared; arm differs from pqkd only if |mean paired difference| > its s.d. across seeds

| arm | seeds | Δ test acc (pp) | exceeds s.d.? | Δ val acc (pp) | Δ val CE |
|---|---|---|---|---|---|
| B1_kd_dense | [0, 1, 2] | 0.16 ± 0.05 (n=3) | yes | 0.25 ± 0.12 (n=3) | -0.0035 ± 0.0018 (n=3) |
| B0_zero | [0, 1, 2] | -88.32 ± 0.91 (n=3) | yes | -87.78 ± 1.09 (n=3) | 2.2692 ± 0.0014 (n=3) |
| B4_fixed_photonic | [0, 1, 2] | 0.01 ± 0.09 (n=3) | no | -0.02 ± 0.10 (n=3) | 0.0024 ± 0.0018 (n=3) |
| B2_dictionary_trainable | [0, 1, 2] | 0.13 ± 0.02 (n=3) | yes | 0.13 ± 0.22 (n=3) | -0.0026 ± 0.0034 (n=3) |
| B3f_random_fixed | [0, 1, 2] | 0.05 ± 0.04 (n=3) | yes | 0.03 ± 0.18 (n=3) | 0.0008 ± 0.0060 (n=3) |
| B6_classical_histogram | [0, 1, 2] | -0.65 ± 0.08 (n=3) | yes | -0.79 ± 0.05 (n=3) | 0.0318 ± 0.0020 (n=3) |
| B3r_random_resampled | [0, 1, 2] | -0.98 ± 0.39 (n=3) | yes | -1.14 ± 0.40 (n=3) | 0.0477 ± 0.0150 (n=3) |
| B5_surrogate | [0, 1, 2] | -0.31 ± 0.06 (n=3) | yes | -0.48 ± 0.31 (n=3) | 0.0166 ± 0.0071 (n=3) |

## Table 1 (MNIST) comparison

| quantity | paper | reproduced |
|---|---|---|
| teacher train acc % | 100.00 ± 0.00 (n=5) | 99.61 ± 0.03 (n=3) |
| teacher val acc % | 99.07 ± 0.13 (n=5) | 99.03 ± 0.09 (n=3) |
| teacher val CE | 0.0310 ± 0.0020 (n=5) | 0.0340 ± 0.0024 (n=3) |
| PQKD train acc % | 99.71 ± 0.02 (n=5) | 99.23 ± 0.04 (n=3) |
| PQKD val acc % | 99.09 ± 0.09 (n=5) | 98.92 ± 0.16 (n=3) |
| PQKD val CE | 0.0300 ± 0.0010 (n=5) | 0.0337 ± 0.0019 (n=3) |
| photonic Δ (J first − final) | 5.94 ± 0.02 (n=5) | 0.61 ± 0.10 (n=3) |

- Teacher val/train are final-epoch (as in the paper's Table 1); teacher test is the best-val checkpoint.
- Student metrics are final-epoch (paper reports final epoch).
- The paper tunes theta on the validation split (SPSA) and reports validation; test is the primary metric here.
