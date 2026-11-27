### Photonic improved, P0 paper protocol (mini-batch 25, lr 0.3, 200 steps)

Run `outdir/run_20261028-154813`, 200 steps, seeds [0, 1, 2, 3, 4, 42]; success = final MSE within 0.0001 of B1.

| Case | Model | Params | Final MSE mean ± std | Min | Max | B1 MSE | Success |
|---|---|---:|---:|---:|---:|---:|---:|
| fig4_n2_L3_phase4_model | photonic | 24 | 1.126e-02 ± 6.3e-03 | 4.39e-03 | 1.93e-02 | 2.003e-32 | 0/6 |
| fig4_n2_L3_screens_fixed_readout | photonic | 30 | 3.356e-02 ± 2.5e-02 | 4.78e-03 | 6.43e-02 | 2.003e-32 | 0/6 |
| fig4_n2_L3_screens_trainable_readout | photonic | 37 | 1.500e-02 ± 1.3e-02 | 1.00e-04 | 3.54e-02 | 2.003e-32 | 0/6 |
| fig4_n5_L1_screens_trainable_readout | photonic | 128 | 4.098e-02 ± 7.1e-03 | 3.30e-02 | 5.05e-02 | 1.912e-32 | 0/6 |
| fig4_n2_multiscale_1_3_screens_trainable_readout | photonic | 38 | 5.091e-03 ± 8.7e-03 | 6.21e-04 | 2.28e-02 | 2.003e-32 | 0/6 |
| fig4_single_photon_m6_theorem_setting | photonic | 72 | 5.101e-03 ± 7.0e-03 | 2.22e-05 | 1.43e-02 | 1.912e-32 | 1/6 |

### Photonic improved, P1 (full batch, lr 0.3, 1000 steps; non-paper protocol)

Run `outdir/run_20261028-154841`, 1000 steps, seeds [0, 1, 2, 3, 4, 42]; success = final MSE within 0.0001 of B1.

| Case | Model | Params | Final MSE mean ± std | Min | Max | B1 MSE | Success |
|---|---|---:|---:|---:|---:|---:|---:|
| fig4_n2_L3_phase4_model | photonic | 24 | 2.771e-03 ± 1.0e-03 | 1.80e-03 | 4.28e-03 | 2.003e-32 | 0/6 |
| fig4_n2_L3_screens_fixed_readout | photonic | 30 | 3.370e-03 ± 7.3e-04 | 2.06e-03 | 4.15e-03 | 2.003e-32 | 0/6 |
| fig4_n2_L3_screens_trainable_readout | photonic | 37 | 3.837e-02 ± 9.3e-02 | 1.02e-14 | 2.28e-01 | 2.003e-32 | 3/6 |
| fig4_n5_L1_screens_trainable_readout | photonic | 128 | 1.260e-02 ± 1.3e-02 | 1.89e-04 | 2.68e-02 | 1.912e-32 | 0/6 |
| fig4_n2_multiscale_1_3_screens_trainable_readout | photonic | 38 | 9.689e-05 ± 2.1e-04 | 7.13e-12 | 5.18e-04 | 2.003e-32 | 5/6 |
| fig4_single_photon_m6_theorem_setting | photonic | 72 | 1.066e-05 ± 2.5e-05 | 5.91e-10 | 6.23e-05 | 1.912e-32 | 6/6 |

### Photonic improved, P2 (full batch, lr 0.05, 1000 steps; non-paper protocol)

Run `outdir/run_20261028-155113`, 1000 steps, seeds [0, 1, 2, 3, 4, 42]; success = final MSE within 0.0001 of B1.

| Case | Model | Params | Final MSE mean ± std | Min | Max | B1 MSE | Success |
|---|---|---:|---:|---:|---:|---:|---:|
| fig4_n2_L3_phase4_model | photonic | 24 | 3.351e-03 ± 6.0e-04 | 2.60e-03 | 4.15e-03 | 2.003e-32 | 0/6 |
| fig4_n2_L3_screens_fixed_readout | photonic | 30 | 2.508e-03 ± 1.0e-03 | 1.59e-03 | 3.86e-03 | 2.003e-32 | 0/6 |
| fig4_n2_L3_screens_trainable_readout | photonic | 37 | 1.404e-05 ± 2.0e-05 | 1.49e-07 | 5.05e-05 | 2.003e-32 | 6/6 |
| fig4_n5_L1_screens_trainable_readout | photonic | 128 | 1.863e-04 ± 3.3e-04 | 7.17e-07 | 8.37e-04 | 1.912e-32 | 4/6 |
| fig4_n2_multiscale_1_3_screens_trainable_readout | photonic | 38 | 4.792e-06 ± 1.1e-05 | 9.28e-10 | 2.78e-05 | 2.003e-32 | 6/6 |
| fig4_single_photon_m6_theorem_setting | photonic | 72 | 1.309e-05 ± 1.5e-05 | 1.07e-16 | 3.31e-05 | 1.912e-32 | 6/6 |

### Photonic improved, P3 (full batch, lr 0.01, 1000 steps; non-paper protocol)

Run `outdir/run_20261028-155331`, 1000 steps, seeds [0, 1, 2, 3, 4, 42]; success = final MSE within 0.0001 of B1.

| Case | Model | Params | Final MSE mean ± std | Min | Max | B1 MSE | Success |
|---|---|---:|---:|---:|---:|---:|---:|
| fig4_n2_L3_phase4_model | photonic | 24 | 3.780e-03 ± 1.3e-03 | 2.26e-03 | 6.08e-03 | 2.003e-32 | 0/6 |
| fig4_n2_L3_screens_fixed_readout | photonic | 30 | 2.911e-03 ± 1.6e-03 | 1.63e-03 | 5.57e-03 | 2.003e-32 | 0/6 |
| fig4_n2_L3_screens_trainable_readout | photonic | 37 | 5.857e-06 ± 8.8e-06 | 5.92e-08 | 1.96e-05 | 2.003e-32 | 6/6 |
| fig4_n5_L1_screens_trainable_readout | photonic | 128 | 6.587e-06 ± 1.5e-05 | 4.09e-10 | 3.74e-05 | 1.912e-32 | 6/6 |
| fig4_n2_multiscale_1_3_screens_trainable_readout | photonic | 38 | 1.287e-07 ± 3.0e-07 | 1.74e-31 | 7.34e-07 | 2.003e-32 | 6/6 |
| fig4_single_photon_m6_theorem_setting | photonic | 72 | 3.643e-06 ± 8.9e-06 | 1.19e-31 | 2.19e-05 | 1.912e-32 | 6/6 |
