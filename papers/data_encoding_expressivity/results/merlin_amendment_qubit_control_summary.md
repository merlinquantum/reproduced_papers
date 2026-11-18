### Qubit fixed vs trainable observable, P0 paper protocol

Run `outdir/run_20261028-155555`, 200 steps, seeds [0, 1, 2, 3, 4, 42]; success = final MSE within 0.0001 of B1.

| Case | Model | Params | Final MSE mean ± std | Min | Max | B1 MSE | Success |
|---|---|---:|---:|---:|---:|---:|---:|
| fig4s1_serial_r5_z0 | quantum | 18 | 2.474e-02 ± 1.5e-02 | 2.56e-03 | 4.37e-02 | 1.912e-32 | 0/6 |
| fig4s1_serial_r5_trainable_linear | quantum | 21 | 5.703e-03 ± 3.0e-03 | 1.89e-03 | 9.79e-03 | 1.912e-32 | 0/6 |
| fig4s1_parallel_r5_z0 | quantum | 90 | 3.314e-03 ± 3.2e-03 | 7.10e-05 | 7.38e-03 | 1.912e-32 | 2/6 |
| fig4s1_parallel_r5_trainable_linear | quantum | 123 | 2.129e-03 ± 2.5e-03 | 1.73e-04 | 5.86e-03 | 1.912e-32 | 0/6 |

### Qubit fixed vs trainable observable, P1 (non-paper protocol)

Run `outdir/run_20261028-155615`, 1000 steps, seeds [0, 1, 2, 3, 4, 42]; success = final MSE within 0.0001 of B1.

| Case | Model | Params | Final MSE mean ± std | Min | Max | B1 MSE | Success |
|---|---|---:|---:|---:|---:|---:|---:|
| fig4s1_serial_r5_z0 | quantum | 18 | 1.387e-03 ± 2.0e-03 | 2.07e-05 | 5.27e-03 | 1.912e-32 | 2/6 |
| fig4s1_serial_r5_trainable_linear | quantum | 21 | 4.507e-04 ± 3.1e-04 | 3.11e-06 | 8.80e-04 | 1.912e-32 | 1/6 |
| fig4s1_parallel_r5_z0 | quantum | 90 | 4.103e-04 ± 9.8e-04 | 4.79e-18 | 2.42e-03 | 1.912e-32 | 5/6 |
| fig4s1_parallel_r5_trainable_linear | quantum | 123 | 1.524e-11 ± 1.8e-11 | 5.26e-23 | 4.28e-11 | 1.912e-32 | 6/6 |

### Qubit fixed vs trainable observable, P2 (non-paper protocol)

Run `outdir/run_20261028-155808`, 1000 steps, seeds [0, 1, 2, 3, 4, 42]; success = final MSE within 0.0001 of B1.

| Case | Model | Params | Final MSE mean ± std | Min | Max | B1 MSE | Success |
|---|---|---:|---:|---:|---:|---:|---:|
| fig4s1_serial_r5_z0 | quantum | 18 | 1.225e-03 ± 3.0e-03 | 7.36e-31 | 7.25e-03 | 1.912e-32 | 5/6 |
| fig4s1_serial_r5_trainable_linear | quantum | 21 | 1.729e-07 ± 2.6e-07 | 7.88e-32 | 6.26e-07 | 1.912e-32 | 6/6 |
| fig4s1_parallel_r5_z0 | quantum | 90 | 7.383e-07 ± 1.5e-06 | 2.91e-31 | 3.72e-06 | 1.912e-32 | 6/6 |
| fig4s1_parallel_r5_trainable_linear | quantum | 123 | 1.265e-05 ± 2.3e-05 | 1.91e-07 | 5.87e-05 | 1.912e-32 | 6/6 |

### Qubit fixed vs trainable observable, P3 (non-paper protocol)

Run `outdir/run_20261028-155958`, 1000 steps, seeds [0, 1, 2, 3, 4, 42]; success = final MSE within 0.0001 of B1.

| Case | Model | Params | Final MSE mean ± std | Min | Max | B1 MSE | Success |
|---|---|---:|---:|---:|---:|---:|---:|
| fig4s1_serial_r5_z0 | quantum | 18 | 5.228e-03 ± 4.3e-03 | 9.10e-14 | 1.07e-02 | 1.912e-32 | 1/6 |
| fig4s1_serial_r5_trainable_linear | quantum | 21 | 5.996e-06 ± 1.4e-05 | 3.65e-31 | 3.55e-05 | 1.912e-32 | 6/6 |
| fig4s1_parallel_r5_z0 | quantum | 90 | 7.323e-32 ± 2.9e-32 | 3.84e-32 | 1.10e-31 | 1.912e-32 | 6/6 |
| fig4s1_parallel_r5_trainable_linear | quantum | 123 | 6.207e-08 ± 1.1e-07 | 3.47e-19 | 2.58e-07 | 1.912e-32 | 6/6 |
