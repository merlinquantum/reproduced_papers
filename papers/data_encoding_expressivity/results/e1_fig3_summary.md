### E1 quantum (paper protocol, 200 steps)

Run `outdir/run_20261028-100040`, 200 steps, seeds [0, 1, 2, 3, 4, 42]; success = final MSE within 0.0001 of B1.

| Case | Model | Params | Final MSE mean ± std | Min | Max | B1 MSE | Success |
|---|---|---:|---:|---:|---:|---:|---:|
| fig3_row_a_degree1 | quantum | 6 | 4.236e-09 ± 6.8e-09 | 2.27e-11 | 1.79e-08 | 2.105e-32 | 6/6 |
| fig3_row_b_mismatch | quantum | 6 | 1.031e-01 ± 1.3e-02 | 9.42e-02 | 1.28e-01 | 8.971e-02 | 0/6 |
| fig3_row_c_degree2 | quantum | 6 | 9.877e-02 ± 4.8e-03 | 9.02e-02 | 1.03e-01 | 8.971e-02 | 0/6 |

### E1 B2 classical (paper protocol, 200 steps)

Run `outdir/run_20261028-100052`, 200 steps, seeds [0, 1, 2, 3, 4, 42]; success = final MSE within 0.0001 of B1.

| Case | Model | Params | Final MSE mean ± std | Min | Max | B1 MSE | Success |
|---|---|---:|---:|---:|---:|---:|---:|
| fig3_row_a_degree1 | fourier_classical | 3 | 1.357e-08 ± 1.2e-08 | 2.02e-09 | 3.30e-08 | 2.105e-32 | 6/6 |
| fig3_row_b_mismatch | fourier_classical | 3 | 9.499e-02 ± 6.0e-03 | 8.98e-02 | 1.06e-01 | 8.971e-02 | 0/6 |
| fig3_row_c_degree2 | fourier_classical | 3 | 9.233e-02 ± 1.7e-03 | 9.07e-02 | 9.46e-02 | 8.971e-02 | 0/6 |
