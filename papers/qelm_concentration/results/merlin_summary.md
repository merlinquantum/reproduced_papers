# Phase 4 MerLin summary (partial validation; simulated, ideal photonics)

## Landsat (F1 macro, 45 evaluations)

| Readout | photonic 16/4 | photonic 24/6 |
|---|---|---|
| w1_click_scaled | 0.735 ± 0.040 | 0.755 ± 0.029 |
| w1_click_unit | 0.735 ± 0.041 | 0.754 ± 0.028 |
| w1_et_scaled | 0.736 ± 0.041 | 0.755 ± 0.029 |
| w1_et_unit | 0.735 ± 0.041 | 0.754 ± 0.028 |
| w1_etcut_scaled | 0.736 ± 0.041 | 0.755 ± 0.029 |
| w1_etcut_unit | 0.735 ± 0.041 | 0.754 ± 0.028 |
| w2_click_scaled | 0.750 ± 0.041 | 0.780 ± 0.032 |
| w2_click_unit | 0.748 ± 0.039 | 0.776 ± 0.030 |
| w2_et_scaled | 0.749 ± 0.039 | 0.777 ± 0.032 |
| w2_et_unit | 0.748 ± 0.039 | 0.776 ± 0.033 |
| w2_etcut_scaled | 0.749 ± 0.039 | 0.777 ± 0.032 |
| w2_etcut_unit | 0.748 ± 0.039 | 0.776 ± 0.033 |

- 16/4: best w2_et_scaled 0.749 vs qubit w2 ET 0.795 (difference -0.047): **preserved**.
  - w2_et_scaled - w2_click_scaled: -0.0016 ± 0.0007 (n = 45, 10/21)
  - w1_et_scaled - w1_click_scaled: +0.0005 ± 0.0006 (n = 45, 8/5)
  - w2_etcut_scaled - w2_et_scaled: +0.0000 ± 0.0000 (n = 45, 0/0)
  - w2_click_scaled - w2_click_unit: +0.0018 ± 0.0008 (n = 45, 23/9)
  - w2_et_scaled - w2_et_unit: +0.0005 ± 0.0007 (n = 45, 12/8)
  - w2_click_unit - w1_click_unit: +0.0135 ± 0.0025 (n = 45, 33/12)
  - shots curve (100/1e3/1e4/1e5): photonic 0.652 / 0.719 / 0.736 / 0.737; qubit 0.239 / 0.475 / 0.674 / 0.766
  - eigentasks retained by the cutoff (w2_etcut_scaled): 0.01: 24.0, 0.03: 24.0, 0.1: 24.0, 0.3: 24.0, 1.0: 24.0, 3.0: 24.0, 10.0: 24.0, 30.0: 24.0, 100.0: 24.0
- 24/6: best w2_et_scaled 0.777 vs qubit w2 ET 0.810 (difference -0.033): **preserved**.
  - w2_et_scaled - w2_click_scaled: -0.0033 ± 0.0008 (n = 45, 7/21)
  - w1_et_scaled - w1_click_scaled: -0.0001 ± 0.0006 (n = 45, 5/5)
  - w2_etcut_scaled - w2_et_scaled: +0.0000 ± 0.0000 (n = 45, 0/0)
  - w2_click_scaled - w2_click_unit: +0.0043 ± 0.0011 (n = 45, 25/10)
  - w2_et_scaled - w2_et_unit: +0.0009 ± 0.0013 (n = 45, 17/13)
  - w2_click_unit - w1_click_unit: +0.0220 ± 0.0026 (n = 45, 42/3)
  - shots curve (100/1e3/1e4/1e5): photonic 0.694 / 0.739 / 0.752 / 0.754; qubit 0.277 / 0.527 / 0.717 / 0.789
  - eigentasks retained by the cutoff (w2_etcut_scaled): 0.01: 34.7, 0.03: 36.0, 0.1: 36.0, 0.3: 36.0, 1.0: 36.0, 3.0: 36.0, 10.0: 36.0, 30.0: 36.0, 100.0: 36.0

## NARMA (cumulative test R^2 over n = 1..L)

- 16/4 (L = 8): photonic 1e5 shots 7.00, exact 7.18, Gaussian 3.50; qubit 1e5 shots 7.58; degree-2 ridge 7.80; raw ridge 6.77; ESN 7.96. Relative difference -7.7%, R^2 at L 0.83 → 0.25: **preserved**.
- 24/6 (L = 12): photonic 1e5 shots 9.87, exact 9.96, Gaussian 4.02; qubit 1e5 shots 10.86; degree-2 ridge 11.32; raw ridge 9.92; ESN 11.53. Relative difference -9.1%, R^2 at L 0.63 → 0.17: **preserved**.

## Concentration

- Ordering at 24 modes, 16 layers (mapped > balanced, Haar), per seed and weight: {'seed 0|w1': True, 'seed 0|w2': True, 'seed 1|w1': True, 'seed 1|w2': True, 'seed 2|w1': True, 'seed 2|w2': True}.
- Haar variability vs modes (w1): 1.19e-03, 8.44e-04, 5.66e-04, 3.75e-04; decreasing: True.
- **preserved**.
