# S6 MNIST, 4 photons: photonic regimes (complete)

Label: Phase 4 reduced-scope validation (MNIST, 4 photons, 30 epochs, 3 seeds)

Coverage: 24/24 runs. Missing: none

## Per-arm results (final epoch, mean ± s.d. over seeds)

| arm | test acc % | fresh-feature test acc % | val acc % | val CE | Δ vs z=0 (pp) | ‖θ‖ | train h/run |
|---|---|---|---|---|---|---|---|
| P0_pqkd_theta0 | 99.19 ± 0.09 (n=3) | 99.20 ± 0.08 (n=3) | 98.93 ± 0.19 (n=3) | 0.0330 ± 0.0016 (n=3) | 87.84 ± 0.09 (n=3) | 0.93 ± 0.77 (n=3) | 1.54 ± 0.21 (n=3) |
| F1_mixing_s200_ema | 98.78 ± 0.49 (n=3) | 44.87 ± 47.11 (n=3) | 98.55 ± 0.35 (n=3) | 0.0461 ± 0.0180 (n=3) | 87.85 ± 0.97 (n=3) | 13.48 ± 1.23 (n=3) | 1.58 ± 0.17 (n=3) |
| N1_native_grad_random | 99.15 ± 0.08 (n=3) | 99.15 ± 0.08 (n=3) | 98.89 ± 0.05 (n=3) | 0.0341 ± 0.0048 (n=3) | 87.80 ± 0.08 (n=3) | 13.18 ± 0.58 (n=3) | 1.73 ± 0.39 (n=3) |
| F2_mixing_s200_noema | 97.80 ± 2.19 (n=3) | 97.38 ± 2.96 (n=3) | 97.48 ± 2.22 (n=3) | 0.0968 ± 0.0939 (n=3) | 86.45 ± 2.19 (n=3) | 13.06 ± 0.87 (n=3) | 1.76 ± 0.62 (n=3) |
| N3_surrogate_grad | 99.10 ± 0.02 (n=3) | 99.10 ± 0.02 (n=3) | 98.94 ± 0.06 (n=3) | 0.0363 ± 0.0028 (n=3) | 88.29 ± 0.91 (n=3) | 12.16 ± 1.04 (n=3) | 1.65 ± 0.59 (n=3) |
| F3_mixing_s25_ema | 98.58 ± 0.63 (n=3) | 12.46 ± 2.90 (n=3) | 98.39 ± 0.60 (n=3) | 0.0567 ± 0.0188 (n=3) | 87.23 ± 0.63 (n=3) | 18.36 ± 1.86 (n=3) | 1.70 ± 0.41 (n=3) |
| N2_native_grad_zero | 99.15 ± 0.08 (n=3) | 99.15 ± 0.08 (n=3) | 98.80 ± 0.10 (n=3) | 0.0371 ± 0.0066 (n=3) | 87.80 ± 0.08 (n=3) | 0.00 ± 0.00 (n=3) | 1.66 ± 0.29 (n=3) |
| F4_mixing_s25_noema | 95.26 ± 1.60 (n=3) | 47.92 ± 42.67 (n=3) | 94.97 ± 1.32 (n=3) | 0.2153 ± 0.0761 (n=3) | 83.91 ± 1.60 (n=3) | 17.54 ± 5.11 (n=3) | 1.68 ± 0.29 (n=3) |
| S4 pqkd | 99.13 ± 0.04 (n=3) | 99.12 ± 0.06 (n=3) | 98.92 ± 0.16 (n=3) | 0.0337 ± 0.0019 (n=3) | 87.78 ± 0.04 (n=3) | 1.66 ± 0.53 (n=3) | 1.45 ± 0.18 (n=3) |
| S4 B3f_random_fixed | 99.18 ± 0.02 (n=3) | 99.18 ± 0.02 (n=3) | 98.95 ± 0.11 (n=3) | 0.0345 ± 0.0047 (n=3) | 87.83 ± 0.02 (n=3) | — | 1.13 ± 0.03 (n=3) |
| S4 B1_kd_dense | 99.29 ± 0.08 (n=3) | — | 99.17 ± 0.04 (n=3) | 0.0302 ± 0.0007 (n=3) | — | — | 0.94 ± 0.11 (n=3) |

## Paired per-seed differences (arm − reference)

Rule: no winner declared; arm differs from a reference only if |mean paired difference| > its s.d. across seeds

| arm | reference | seeds | Δ test acc (pp) | exceeds s.d.? | signs (+/−/0) | Δ val acc (pp) | Δ val CE |
|---|---|---|---|---|---|---|---|
| F1_mixing_s200_ema | P0_pqkd_theta0 | [0, 1, 2] | -0.41 ± 0.56 (n=3) | no | 1/2/0 | -0.37 ± 0.53 (n=3) | 0.0131 ± 0.0196 (n=3) |
| F2_mixing_s200_noema | P0_pqkd_theta0 | [0, 1, 2] | -1.40 ± 2.24 (n=3) | no | 1/2/0 | -1.45 ± 2.24 (n=3) | 0.0639 ± 0.0938 (n=3) |
| F3_mixing_s25_ema | P0_pqkd_theta0 | [0, 1, 2] | -0.61 ± 0.54 (n=3) | yes | 0/3/0 | -0.54 ± 0.44 (n=3) | 0.0237 ± 0.0174 (n=3) |
| F4_mixing_s25_noema | P0_pqkd_theta0 | [0, 1, 2] | -3.93 ± 1.53 (n=3) | yes | 0/3/0 | -3.96 ± 1.29 (n=3) | 0.1823 ± 0.0763 (n=3) |
| N1_native_grad_random | P0_pqkd_theta0 | [0, 1, 2] | -0.04 ± 0.04 (n=3) | no | 0/3/0 | -0.03 ± 0.23 (n=3) | 0.0011 ± 0.0061 (n=3) |
| N2_native_grad_zero | P0_pqkd_theta0 | [0, 1, 2] | -0.04 ± 0.08 (n=3) | no | 1/2/0 | -0.13 ± 0.12 (n=3) | 0.0041 ± 0.0071 (n=3) |
| N3_surrogate_grad | P0_pqkd_theta0 | [0, 1, 2] | -0.09 ± 0.11 (n=3) | no | 1/2/0 | 0.01 ± 0.17 (n=3) | 0.0033 ± 0.0041 (n=3) |
| N1_native_grad_random | N3_surrogate_grad | [0, 1, 2] | 0.05 ± 0.10 (n=3) | no | 2/1/0 | -0.05 ± 0.08 (n=3) | -0.0022 ± 0.0020 (n=3) |
| N2_native_grad_zero | N3_surrogate_grad | [0, 1, 2] | 0.05 ± 0.09 (n=3) | no | 2/1/0 | -0.14 ± 0.06 (n=3) | 0.0008 ± 0.0045 (n=3) |
| N1_native_grad_random | N2_native_grad_zero | [0, 1, 2] | 0.00 ± 0.11 (n=3) | no | 1/1/1 | 0.09 ± 0.13 (n=3) | -0.0029 ± 0.0038 (n=3) |
| F1_mixing_s200_ema | F2_mixing_s200_noema | [0, 1, 2] | 0.98 ± 2.31 (n=3) | no | 1/2/0 | 1.07 ± 2.31 (n=3) | -0.0507 ± 0.0990 (n=3) |
| F3_mixing_s25_ema | F4_mixing_s25_noema | [0, 1, 2] | 3.32 ± 1.15 (n=3) | yes | 3/0/0 | 3.42 ± 1.02 (n=3) | -0.1586 ± 0.0708 (n=3) |
| F1_mixing_s200_ema | F3_mixing_s25_ema | [0, 1, 2] | 0.20 ± 1.04 (n=3) | no | 1/2/0 | 0.17 ± 0.88 (n=3) | -0.0106 ± 0.0349 (n=3) |
| P0_pqkd_theta0 | B3f_random_fixed (S4) | [0, 1, 2] | 0.02 ± 0.08 (n=3) | no | 2/1/0 | -0.03 ± 0.27 (n=3) | -0.0015 ± 0.0062 (n=3) |
| P0_pqkd_theta0 | B1_kd_dense (S4) | [0, 1, 2] | -0.09 ± 0.13 (n=3) | no | 1/2/0 | -0.24 ± 0.16 (n=3) | 0.0028 ± 0.0019 (n=3) |
| N1_native_grad_random | B3f_random_fixed (S4) | [0, 1, 2] | -0.02 ± 0.07 (n=3) | no | 1/2/0 | -0.06 ± 0.07 (n=3) | -0.0004 ± 0.0018 (n=3) |
| N1_native_grad_random | B1_kd_dense (S4) | [0, 1, 2] | -0.13 ± 0.09 (n=3) | yes | 0/3/0 | -0.27 ± 0.07 (n=3) | 0.0039 ± 0.0042 (n=3) |
| N3_surrogate_grad | B3f_random_fixed (S4) | [0, 1, 2] | -0.08 ± 0.04 (n=3) | yes | 0/3/0 | -0.01 ± 0.10 (n=3) | 0.0018 ± 0.0025 (n=3) |
| N3_surrogate_grad | B1_kd_dense (S4) | [0, 1, 2] | -0.19 ± 0.09 (n=3) | yes | 0/3/0 | -0.23 ± 0.02 (n=3) | 0.0061 ± 0.0022 (n=3) |
| P0_pqkd_theta0 | pqkd (S4) | [0, 1, 2] | 0.06 ± 0.08 (n=3) | no | 2/1/0 | 0.01 ± 0.13 (n=3) | -0.0007 ± 0.0009 (n=3) |

- Student metrics are final-epoch; test accuracy is the primary metric.
- test_accuracy_fresh_feature re-evaluates with one new feature draw (shots without EMA for F arms, exact for N arms).
- theta_norm is the L2 norm of the conditioning parameters (theta for photonic arms, phi for N3).
- S4 arms (pqkd, B3f, B1) come from the 8-photon Phase 3 sweep; B3f and B1 involve no photons.
- training_hours sums per-epoch seconds with 3 runs in parallel x 4 CPU threads; pauses are excluded.
