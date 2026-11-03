### E2 quantum, s = 1 (published figure), 200 steps

Run `outdir/run_20261028-100128`, 200 steps, seeds [0, 1, 2, 3, 4, 42]; success = final MSE within 0.0001 of B1.

| Case | Model | Params | Final MSE mean ± std | Min | Max | B1 MSE | Success |
|---|---|---:|---:|---:|---:|---:|---:|
| fig4s1_serial_r1 | quantum | 6 | 4.070e-02 ± 2.9e-03 | 3.71e-02 | 4.37e-02 | 3.574e-02 | 0/6 |
| fig4s1_serial_r3 | quantum | 12 | 3.743e-02 ± 1.1e-02 | 3.04e-02 | 5.71e-02 | 1.908e-02 | 0/6 |
| fig4s1_serial_r5 | quantum | 18 | 2.474e-02 ± 1.5e-02 | 2.56e-03 | 4.37e-02 | 1.912e-32 | 0/6 |
| fig4s1_parallel_r1 | quantum | 18 | 4.586e-02 ± 9.9e-03 | 3.72e-02 | 6.06e-02 | 3.574e-02 | 0/6 |
| fig4s1_parallel_r3 | quantum | 54 | 3.433e-02 ± 1.2e-02 | 2.11e-02 | 5.40e-02 | 1.908e-02 | 0/6 |
| fig4s1_parallel_r5 | quantum | 90 | 3.314e-03 ± 3.2e-03 | 7.10e-05 | 7.38e-03 | 1.912e-32 | 2/6 |

| Layout | Model | r → r' | Seeds with lower MSE at r' |
|---|---|---|---:|
| serial | quantum | 1 → 3 | 4/6 |
| serial | quantum | 3 → 5 | 4/6 |
| parallel | quantum | 1 → 3 | 5/6 |
| parallel | quantum | 3 → 5 | 6/6 |

### E2 quantum, s = 2 (caption), 200 steps

Run `outdir/run_20261028-100100`, 200 steps, seeds [0, 1, 2, 3, 4, 42]; success = final MSE within 0.0001 of B1.

| Case | Model | Params | Final MSE mean ± std | Min | Max | B1 MSE | Success |
|---|---|---:|---:|---:|---:|---:|---:|
| fig4_serial_r1 | quantum | 6 | 3.882e-02 ± 3.0e-03 | 3.66e-02 | 4.47e-02 | 3.610e-02 | 0/6 |
| fig4_serial_r3 | quantum | 12 | 3.706e-02 ± 1.0e-02 | 2.57e-02 | 5.68e-02 | 1.923e-02 | 0/6 |
| fig4_serial_r5 | quantum | 18 | 2.880e-02 ± 2.9e-02 | 4.18e-03 | 6.89e-02 | 2.247e-32 | 0/6 |
| fig4_parallel_r1 | quantum | 18 | 4.188e-02 ± 3.9e-03 | 3.70e-02 | 4.69e-02 | 3.610e-02 | 0/6 |
| fig4_parallel_r3 | quantum | 54 | 3.401e-02 ± 1.1e-02 | 2.46e-02 | 5.23e-02 | 1.923e-02 | 0/6 |
| fig4_parallel_r5 | quantum | 90 | 1.729e-03 ± 1.2e-03 | 7.45e-04 | 3.50e-03 | 2.247e-32 | 0/6 |

| Layout | Model | r → r' | Seeds with lower MSE at r' |
|---|---|---|---:|
| serial | quantum | 1 → 3 | 5/6 |
| serial | quantum | 3 → 5 | 4/6 |
| parallel | quantum | 1 → 3 | 5/6 |
| parallel | quantum | 3 → 5 | 6/6 |

### E2 B2 classical, s = 2, 200 steps

Run `outdir/run_20261028-100159`, 200 steps, seeds [0, 1, 2, 3, 4, 42]; success = final MSE within 0.0001 of B1.

| Case | Model | Params | Final MSE mean ± std | Min | Max | B1 MSE | Success |
|---|---|---:|---:|---:|---:|---:|---:|
| fig4_serial_r1 | fourier_classical | 3 | 3.753e-02 ± 1.3e-03 | 3.63e-02 | 3.88e-02 | 3.610e-02 | 0/6 |
| fig4_serial_r3 | fourier_classical | 7 | 2.087e-02 ± 7.7e-04 | 1.96e-02 | 2.16e-02 | 1.923e-02 | 0/6 |
| fig4_serial_r5 | fourier_classical | 11 | 2.126e-07 ± 1.4e-07 | 8.11e-08 | 4.33e-07 | 2.247e-32 | 6/6 |
| fig4_parallel_r1 | fourier_classical | 3 | 3.753e-02 ± 1.3e-03 | 3.63e-02 | 3.88e-02 | 3.610e-02 | 0/6 |
| fig4_parallel_r3 | fourier_classical | 7 | 2.087e-02 ± 7.7e-04 | 1.96e-02 | 2.16e-02 | 1.923e-02 | 0/6 |
| fig4_parallel_r5 | fourier_classical | 11 | 2.126e-07 ± 1.4e-07 | 8.11e-08 | 4.33e-07 | 2.247e-32 | 6/6 |

| Layout | Model | r → r' | Seeds with lower MSE at r' |
|---|---|---|---:|
| serial | fourier_classical | 1 → 3 | 6/6 |
| serial | fourier_classical | 3 → 5 | 6/6 |
| parallel | fourier_classical | 1 → 3 | 6/6 |
| parallel | fourier_classical | 3 → 5 | 6/6 |

### E2 quantum, s = 1, 1000 steps (sensitivity, non-paper step budget)

Run `outdir/run_20261028-100525`, 1000 steps, seeds [0, 1, 2, 3, 4, 42]; success = final MSE within 0.0001 of B1.

| Case | Model | Params | Final MSE mean ± std | Min | Max | B1 MSE | Success |
|---|---|---:|---:|---:|---:|---:|---:|
| fig4s1_serial_r1 | quantum | 6 | 4.122e-02 ± 5.4e-03 | 3.61e-02 | 5.15e-02 | 3.574e-02 | 0/6 |
| fig4s1_serial_r3 | quantum | 12 | 2.909e-02 ± 5.7e-03 | 2.35e-02 | 3.79e-02 | 1.908e-02 | 0/6 |
| fig4s1_serial_r5 | quantum | 18 | 3.389e-02 ± 5.5e-02 | 1.00e-06 | 1.36e-01 | 1.912e-32 | 3/6 |
| fig4s1_parallel_r1 | quantum | 18 | 4.175e-02 ± 3.9e-03 | 3.72e-02 | 4.86e-02 | 3.574e-02 | 0/6 |
| fig4s1_parallel_r3 | quantum | 54 | 3.227e-02 ± 1.6e-02 | 2.26e-02 | 6.37e-02 | 1.908e-02 | 0/6 |
| fig4s1_parallel_r5 | quantum | 90 | 5.666e-04 ± 9.6e-04 | 1.00e-06 | 2.48e-03 | 1.912e-32 | 2/6 |

| Layout | Model | r → r' | Seeds with lower MSE at r' |
|---|---|---|---:|
| serial | quantum | 1 → 3 | 6/6 |
| serial | quantum | 3 → 5 | 4/6 |
| parallel | quantum | 1 → 3 | 5/6 |
| parallel | quantum | 3 → 5 | 6/6 |

### E2 quantum, s = 2, 1000 steps (sensitivity, non-paper step budget)

Run `outdir/run_20261028-100643`, 1000 steps, seeds [0, 1, 2, 3, 4, 42]; success = final MSE within 0.0001 of B1.

| Case | Model | Params | Final MSE mean ± std | Min | Max | B1 MSE | Success |
|---|---|---:|---:|---:|---:|---:|---:|
| fig4_serial_r1 | quantum | 6 | 4.097e-02 ± 5.5e-03 | 3.70e-02 | 5.06e-02 | 3.610e-02 | 0/6 |
| fig4_serial_r3 | quantum | 12 | 4.290e-02 ± 1.1e-02 | 3.11e-02 | 6.09e-02 | 1.923e-02 | 0/6 |
| fig4_serial_r5 | quantum | 18 | 1.399e-02 ± 2.1e-02 | 8.11e-08 | 5.63e-02 | 2.247e-32 | 1/6 |
| fig4_parallel_r1 | quantum | 18 | 4.732e-02 ± 9.4e-03 | 3.90e-02 | 6.28e-02 | 3.610e-02 | 0/6 |
| fig4_parallel_r3 | quantum | 54 | 2.684e-02 ± 5.4e-03 | 2.06e-02 | 3.56e-02 | 1.923e-02 | 0/6 |
| fig4_parallel_r5 | quantum | 90 | 7.607e-04 ± 1.4e-03 | 2.38e-07 | 3.60e-03 | 2.247e-32 | 3/6 |

| Layout | Model | r → r' | Seeds with lower MSE at r' |
|---|---|---|---:|
| serial | quantum | 1 → 3 | 3/6 |
| serial | quantum | 3 → 5 | 6/6 |
| parallel | quantum | 1 → 3 | 6/6 |
| parallel | quantum | 3 → 5 | 6/6 |

### E2 B2 classical, s = 2, 1000 steps (sensitivity, non-paper step budget)

Run `outdir/run_20261028-100802`, 1000 steps, seeds [0, 1, 2, 3, 4, 42]; success = final MSE within 0.0001 of B1.

| Case | Model | Params | Final MSE mean ± std | Min | Max | B1 MSE | Success |
|---|---|---:|---:|---:|---:|---:|---:|
| fig4_serial_r1 | fourier_classical | 3 | 4.069e-02 ± 5.2e-03 | 3.69e-02 | 5.09e-02 | 3.610e-02 | 0/6 |
| fig4_serial_r3 | fourier_classical | 7 | 2.428e-02 ± 2.6e-03 | 2.07e-02 | 2.84e-02 | 1.923e-02 | 0/6 |
| fig4_serial_r5 | fourier_classical | 11 | 6.977e-33 ± 8.0e-33 | 1.75e-33 | 2.24e-32 | 2.247e-32 | 6/6 |
| fig4_parallel_r1 | fourier_classical | 3 | 4.069e-02 ± 5.2e-03 | 3.69e-02 | 5.09e-02 | 3.610e-02 | 0/6 |
| fig4_parallel_r3 | fourier_classical | 7 | 2.428e-02 ± 2.6e-03 | 2.07e-02 | 2.84e-02 | 1.923e-02 | 0/6 |
| fig4_parallel_r5 | fourier_classical | 11 | 6.977e-33 ± 8.0e-33 | 1.75e-33 | 2.24e-32 | 2.247e-32 | 6/6 |

| Layout | Model | r → r' | Seeds with lower MSE at r' |
|---|---|---|---:|
| serial | fourier_classical | 1 → 3 | 6/6 |
| serial | fourier_classical | 3 → 5 | 6/6 |
| parallel | fourier_classical | 1 → 3 | 6/6 |
| parallel | fourier_classical | 3 → 5 | 6/6 |
