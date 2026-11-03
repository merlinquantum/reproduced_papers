### Photonic universality test (single photon, theorem setting), P0 paper protocol

Run `outdir/run_20261028-160148`, 200 steps, seeds [0, 1, 2, 3, 4, 42]; success = final MSE within 0.0001 of B1.

| Case | Model | Params | Final MSE mean ± std | Min | Max | B1 MSE | Success |
|---|---|---:|---:|---:|---:|---:|---:|
| square_single_photon_m2_K1 | photonic | 8 | 2.087e-01 ± 2.6e-02 | 1.88e-01 | 2.61e-01 | 1.768e-01 | 0/6 |
| square_single_photon_m3_K2 | photonic | 18 | 2.063e-01 ± 3.1e-02 | 1.84e-01 | 2.66e-01 | 1.768e-01 | 0/6 |
| square_single_photon_m4_K3 | photonic | 32 | 1.269e-01 ± 2.2e-02 | 1.09e-01 | 1.67e-01 | 8.671e-02 | 0/6 |
| square_single_photon_m5_K4 | photonic | 50 | 1.511e-01 ± 1.8e-02 | 1.21e-01 | 1.67e-01 | 8.671e-02 | 0/6 |
| square_single_photon_m6_K5 | photonic | 72 | 1.056e-01 ± 2.0e-02 | 7.19e-02 | 1.25e-01 | 5.428e-02 | 0/6 |
| square_single_photon_m7_K6 | photonic | 98 | 1.352e-01 ± 3.5e-02 | 9.09e-02 | 1.75e-01 | 5.428e-02 | 0/6 |
| square_single_photon_m8_K7 | photonic | 128 | 1.783e-01 ± 1.1e-01 | 7.84e-02 | 3.93e-01 | 3.774e-02 | 0/6 |

| Family | Model | degree r → r' | Seeds with lower MSE at r' |
|---|---|---|---:|
| photonic/theorem_single_photon | photonic | 1 → 2 | 3/6 |
| photonic/theorem_single_photon | photonic | 2 → 3 | 6/6 |
| photonic/theorem_single_photon | photonic | 3 → 4 | 2/6 |
| photonic/theorem_single_photon | photonic | 4 → 5 | 5/6 |
| photonic/theorem_single_photon | photonic | 5 → 6 | 1/6 |
| photonic/theorem_single_photon | photonic | 6 → 7 | 2/6 |

### Photonic universality test (single photon, theorem setting), P3 (full batch, lr 0.01, 1000 steps)

Run `outdir/run_20261028-160227`, 1000 steps, seeds [0, 1, 2, 3, 4, 42]; success = final MSE within 0.0001 of B1.

| Case | Model | Params | Final MSE mean ± std | Min | Max | B1 MSE | Success |
|---|---|---:|---:|---:|---:|---:|---:|
| square_single_photon_m2_K1 | photonic | 8 | 1.768e-01 ± 1.0e-09 | 1.77e-01 | 1.77e-01 | 1.768e-01 | 6/6 |
| square_single_photon_m3_K2 | photonic | 18 | 1.768e-01 ± 1.2e-08 | 1.77e-01 | 1.77e-01 | 1.768e-01 | 6/6 |
| square_single_photon_m4_K3 | photonic | 32 | 8.671e-02 ± 5.2e-11 | 8.67e-02 | 8.67e-02 | 8.671e-02 | 6/6 |
| square_single_photon_m5_K4 | photonic | 50 | 8.671e-02 ± 7.1e-08 | 8.67e-02 | 8.67e-02 | 8.671e-02 | 6/6 |
| square_single_photon_m6_K5 | photonic | 72 | 5.428e-02 ± 1.4e-06 | 5.43e-02 | 5.43e-02 | 5.428e-02 | 6/6 |
| square_single_photon_m7_K6 | photonic | 98 | 5.429e-02 ± 4.1e-06 | 5.43e-02 | 5.43e-02 | 5.428e-02 | 6/6 |
| square_single_photon_m8_K7 | photonic | 128 | 3.774e-02 ± 1.1e-10 | 3.77e-02 | 3.77e-02 | 3.774e-02 | 6/6 |

| Family | Model | degree r → r' | Seeds with lower MSE at r' |
|---|---|---|---:|
| photonic/theorem_single_photon | photonic | 1 → 2 | 2/6 |
| photonic/theorem_single_photon | photonic | 2 → 3 | 6/6 |
| photonic/theorem_single_photon | photonic | 3 → 4 | 1/6 |
| photonic/theorem_single_photon | photonic | 4 → 5 | 6/6 |
| photonic/theorem_single_photon | photonic | 5 → 6 | 1/6 |
| photonic/theorem_single_photon | photonic | 6 → 7 | 6/6 |
