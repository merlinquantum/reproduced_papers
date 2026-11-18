### E2/E1c MerLin adaptation, photonic, paper protocol (200 steps)

Run `outdir/run_20261028-110320`, 200 steps, seeds [0, 1, 2, 3, 4, 42]; success = final MSE within 0.0001 of B1.

| Case | Model | Params | Final MSE mean ± std | Min | Max | B1 MSE | Success |
|---|---|---:|---:|---:|---:|---:|---:|
| fig4_photonic_n1_L1_ablation | photonic | 4 | 4.454e-02 ± 5.7e-03 | 3.81e-02 | 5.22e-02 | 3.574e-02 | 0/6 |
| fig4_photonic_n3_L1 | photonic | 24 | 4.273e-02 ± 1.4e-02 | 2.63e-02 | 5.79e-02 | 1.908e-02 | 0/6 |
| fig4_photonic_n5_L1 | photonic | 60 | 3.712e-02 ± 4.5e-03 | 3.19e-02 | 4.35e-02 | 1.912e-32 | 0/6 |
| fig4_photonic_n2_L1 | photonic | 12 | 3.625e-02 ± 3.8e-03 | 3.18e-02 | 4.11e-02 | 2.764e-02 | 0/6 |
| fig4_photonic_n2_L2 | photonic | 18 | 2.733e-02 ± 1.1e-02 | 1.62e-02 | 4.28e-02 | 9.911e-03 | 0/6 |
| fig4_photonic_n2_L3 | photonic | 24 | 1.126e-02 ± 6.3e-03 | 4.39e-03 | 1.93e-02 | 2.003e-32 | 0/6 |
| fig3c_photonic_n1_L1_ablation | photonic | 4 | 9.544e-02 ± 9.7e-03 | 9.06e-02 | 1.15e-01 | 8.971e-02 | 0/6 |
| fig3c_photonic_n2_L1 | photonic | 12 | 3.814e-02 ± 4.0e-03 | 3.37e-02 | 4.43e-02 | 5.632e-32 | 0/6 |

| Family | Model | degree r → r' | Seeds with lower MSE at r' |
|---|---|---|---:|
| photonic/photon_number | photonic | 1 → 3 | 4/6 |
| photonic/photon_number | photonic | 3 → 5 | 3/6 |
| photonic/reuploading | photonic | 2 → 4 | 4/6 |
| photonic/reuploading | photonic | 4 → 6 | 5/6 |
| photonic/photon_number_fig3c | photonic | 1 → 2 | 6/6 |

### E2/E1c MerLin adaptation, photonic, 1000 steps (sensitivity, non-paper step budget)

Run `outdir/run_20261028-110624`, 1000 steps, seeds [0, 1, 2, 3, 4, 42]; success = final MSE within 0.0001 of B1.

| Case | Model | Params | Final MSE mean ± std | Min | Max | B1 MSE | Success |
|---|---|---:|---:|---:|---:|---:|---:|
| fig4_photonic_n1_L1_ablation | photonic | 4 | 4.093e-02 ± 4.4e-03 | 3.68e-02 | 4.79e-02 | 3.574e-02 | 0/6 |
| fig4_photonic_n3_L1 | photonic | 24 | 3.656e-02 ± 4.6e-03 | 3.05e-02 | 4.17e-02 | 1.908e-02 | 0/6 |
| fig4_photonic_n5_L1 | photonic | 60 | 3.632e-02 ± 4.3e-03 | 2.99e-02 | 4.33e-02 | 1.912e-32 | 0/6 |
| fig4_photonic_n2_L1 | photonic | 12 | 3.981e-02 ± 9.2e-03 | 3.17e-02 | 5.50e-02 | 2.764e-02 | 0/6 |
| fig4_photonic_n2_L2 | photonic | 18 | 2.752e-02 ± 9.6e-03 | 1.37e-02 | 3.65e-02 | 9.911e-03 | 0/6 |
| fig4_photonic_n2_L3 | photonic | 24 | 1.562e-02 ± 1.3e-02 | 4.95e-03 | 3.41e-02 | 2.003e-32 | 0/6 |
| fig3c_photonic_n1_L1_ablation | photonic | 4 | 1.020e-01 ± 5.6e-03 | 9.24e-02 | 1.07e-01 | 8.971e-02 | 0/6 |
| fig3c_photonic_n2_L1 | photonic | 12 | 3.724e-02 ± 7.9e-03 | 3.20e-02 | 5.32e-02 | 5.632e-32 | 0/6 |

| Family | Model | degree r → r' | Seeds with lower MSE at r' |
|---|---|---|---:|
| photonic/photon_number | photonic | 1 → 3 | 4/6 |
| photonic/photon_number | photonic | 3 → 5 | 3/6 |
| photonic/reuploading | photonic | 2 → 4 | 6/6 |
| photonic/reuploading | photonic | 4 → 6 | 5/6 |
| photonic/photon_number_fig3c | photonic | 1 → 2 | 6/6 |
