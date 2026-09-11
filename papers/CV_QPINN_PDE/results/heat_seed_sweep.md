# Heat-equation matched-effort multi-seed sweep

5 seeds per architecture, all under the smoke configuration shared by
`configs/heat_smoke.json` (QPINN, 2+2 layers, cutoff 10, 60 + 200 epochs)
and `configs/heat_pinn.json` (classical FFN PINN, 42 trainable parameters
matched to the paper's heat-equation baseline target, 300 + 1000 epochs).
Both use the same consistency-loss training scheme, the same RK45
reference, and the same Sobol-sampled collocation set.

## CV-QPINN (5 seeds)

| Seed | RMSE | MAE | L∞ | Wall (s) |
|---:|---:|---:|---:|---:|
| 7    | 2.026e-02 | 1.500e-02 | 1.045e-01 | 1647 |
| 42   | 1.030e-02 | 7.502e-03 | 4.998e-02 | 1643 |
| 123  | 9.007e-03 | 7.045e-03 | 3.711e-02 | 1976 |
| 256  | 1.310e-02 | 8.978e-03 | 5.391e-02 | 1975 |
| 1024 | 8.674e-03 | 5.949e-03 | 4.247e-02 |  185 |
| **mean ± std** | **1.227e-02 ± 4.8e-03** | **8.86e-03 ± 3.4e-03** | **5.74e-02 ± 2.6e-02** | 1485 |

## Classical PINN baseline (5 seeds)

| Seed | RMSE | MAE | L∞ | Wall (s) |
|---:|---:|---:|---:|---:|
| 7    | 9.483e-03 | 7.626e-03 | 3.006e-02 | 46 |
| 42   | 8.931e-03 | 7.325e-03 | 2.784e-02 | 35 |
| 123  | 8.671e-03 | 7.006e-03 | 2.837e-02 | 42 |
| 256  | 6.755e-03 | 5.241e-03 | 2.889e-02 | 39 |
| 1024 | 9.860e-03 | 7.941e-03 | 3.668e-02 | 30 |
| **mean ± std** | **8.740e-03 ± 1.2e-03** | **7.03e-03 ± 1.0e-03** | **3.04e-02 ± 3.4e-03** | 38 |

## Verdict

- **Mean ratio (QPINN / PINN):** 1.40 — classical PINN wins by 40% on average.
- **Variance ratio (QPINN / PINN):** ~4 — classical PINN is ~4x more stable across seeds.
- **Statistical separation:** QPINN mean sits **2.9 PINN-standard-deviations** above the PINN mean.

The paper's Table IV claim ("the quantum network slightly outperforms the
classical counterpart, RMSE 1.24e-2 vs 2.09e-2") is **not reproduced**
under a matched-effort baseline. Two contributing factors:

1. The paper's classical PINN RMSE 2.09e-2 is ~2.4x worse than our
   matched-parameter, matched-loss classical PINN. The paper likely
   trained its classical PINN without the consistency-loss enhancement
   and/or without IC pre-training.
2. The paper's reported QPINN RMSE 1.24e-2 reproduces well in our hands
   (1.23e-2 ± 4.8e-3, 5 seeds) — so the QPINN reproduction is faithful,
   but the *comparative* claim against a fair classical baseline does
   not hold.

Wall-clock cost: QPINN sweep ~25 min/seed (parallel batches of 2);
classical PINN sweep ~40 s/seed.
