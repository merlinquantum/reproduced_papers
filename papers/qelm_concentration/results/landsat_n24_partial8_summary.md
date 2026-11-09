# Landsat summary (n24_partial8, scope: partial)

Sources: run_20261026-094113

## QELM readouts (F1 macro, mean ± std over seeds)

| Readout | F1 macro | Accuracy | Features | Retained ET | Paper F1 (124 q) |
|---|---|---|---|---|---|
| w1_et_scaled | 0.721 ± 0.023 | 0.800 ± 0.019 | 72 | 72.000 ± 0.000 | 0.812 |
| w1_et_unit | 0.719 ± 0.022 | 0.800 ± 0.017 | 72 | 72.000 ± 0.000 | 0.783 |
| w1_etcut_scaled | 0.721 ± 0.021 | 0.800 ± 0.019 | 71 | 71.083 ± 0.640 |  |
| w1_etcut_unit | 0.716 ± 0.025 | 0.798 ± 0.020 | 71 | 71.083 ± 0.640 | 0.789 |
| w1_pauli_scaled | 0.714 ± 0.023 | 0.798 ± 0.020 | 72 |  | 0.800 |
| w1_pauli_unit | 0.711 ± 0.021 | 0.795 ± 0.017 | 72 |  | 0.790 |
| w1_x_scaled | 0.672 ± 0.025 | 0.777 ± 0.018 | 24 |  | 0.782 |
| w1_x_unit | 0.676 ± 0.030 | 0.778 ± 0.020 | 24 |  | 0.783 |
| w2_et_scaled | 0.718 ± 0.024 | 0.799 ± 0.020 | 180 | 180.000 ± 0.000 | 0.801 |
| w2_et_unit | 0.738 ± 0.023 | 0.799 ± 0.020 | 180 | 180.000 ± 0.000 | 0.749 |
| w2_etcut_scaled | 0.708 ± 0.024 | 0.796 ± 0.020 | 70 | 70.000 ± 2.708 | 0.830 |
| w2_etcut_unit | 0.720 ± 0.025 | 0.798 ± 0.021 | 70 | 70.000 ± 2.708 | 0.792 |
| w2_pauli_scaled | 0.724 ± 0.025 | 0.802 ± 0.020 | 180 |  | 0.789 |
| w2_pauli_unit | 0.748 ± 0.025 | 0.810 ± 0.023 | 180 |  | 0.750 |
| w2_x_scaled | 0.699 ± 0.029 | 0.787 ± 0.020 | 36 |  | 0.789 |
| w2_x_unit | 0.697 ± 0.027 | 0.786 ± 0.019 | 36 |  | 0.750 |

## Paired F1-macro differences (same split, reservoir and shot draw)

| A - B | mean | std | stderr | n | A > B | A < B |
|---|---|---|---|---|---|---|
| w1_pauli_scaled - w1_x_scaled | +0.0415 | 0.0250 | 0.0052 | 24 | 23 | 1 |
| w1_et_scaled - w1_pauli_scaled | +0.0066 | 0.0114 | 0.0024 | 24 | 16 | 7 |
| w2_et_scaled - w2_pauli_scaled | -0.0060 | 0.0214 | 0.0045 | 24 | 8 | 16 |
| w2_etcut_scaled - w2_et_scaled | -0.0095 | 0.0229 | 0.0048 | 24 | 9 | 15 |
| w2_etcut_scaled - w1_et_scaled | -0.0122 | 0.0266 | 0.0055 | 24 | 9 | 15 |
| w1_pauli_scaled - w1_pauli_unit | +0.0031 | 0.0134 | 0.0028 | 24 | 13 | 9 |
| w2_pauli_scaled - w2_pauli_unit | -0.0244 | 0.0220 | 0.0046 | 24 | 4 | 19 |
| w2_et_scaled - w2_et_unit | -0.0202 | 0.0214 | 0.0045 | 24 | 5 | 19 |
| w2_etcut_scaled - w2_etcut_unit | -0.0117 | 0.0217 | 0.0045 | 24 | 8 | 16 |
| w2_pauli_unit - w1_pauli_unit | +0.0374 | 0.0297 | 0.0062 | 24 | 21 | 3 |

## shots_sweep: w1_pauli_scaled

| value | F1 macro | extra |
|---|---|---|
| 100 | 0.233 ± 0.025 |  |
| 1000 | 0.456 ± 0.026 |  |
| 10000 | 0.632 ± 0.030 |  |
| 100000 | 0.713 ± 0.024 |  |

## train_size_sweep: w1_pauli_scaled

| value | F1 macro | extra |
|---|---|---|
| 10 | 0.432 ± 0.116 |  |
| 25 | 0.597 ± 0.062 |  |
| 50 | 0.651 ± 0.052 |  |
| 100 | 0.676 ± 0.044 |  |
| 200 | 0.703 ± 0.036 |  |
| 400 | 0.717 ± 0.033 |  |
| 688 | 0.723 ± 0.029 |  |

## cutoff_sweep: w2_etcut_scaled

| value | F1 macro | extra |
|---|---|---|
| 0.01 | 0.064 ± 0.000 | retained 0.0/180, REC 0.0/85.4, empty 1.00 |
| 0.03 | 0.602 ± 0.041 | retained 10.3/180, REC 10.1/85.4, empty 0.00 |
| 0.1 | 0.670 ± 0.030 | retained 21.4/180, REC 20.5/85.4, empty 0.00 |
| 0.3 | 0.697 ± 0.026 | retained 38.8/180, REC 35.2/85.4, empty 0.00 |
| 1.0 | 0.708 ± 0.024 | retained 70.0/180, REC 55.1/85.4, empty 0.00 |
| 3.0 | 0.717 ± 0.021 | retained 128.0/180, REC 75.2/85.4, empty 0.00 |
| 10.0 | 0.718 ± 0.024 | retained 180.0/180, REC 85.4/85.4, empty 0.00 |
| 30.0 | 0.718 ± 0.024 | retained 180.0/180, REC 85.4/85.4, empty 0.00 |
| 100.0 | 0.718 ± 0.024 | retained 180.0/180, REC 85.4/85.4, empty 0.00 |

## cutoff_sweep: w2_etcut_unit

| value | F1 macro | extra |
|---|---|---|
| 0.01 | 0.064 ± 0.000 | retained 0.0/180, REC 0.0/85.4, empty 1.00 |
| 0.03 | 0.603 ± 0.042 | retained 10.3/180, REC 10.1/85.4, empty 0.00 |
| 0.1 | 0.671 ± 0.028 | retained 21.4/180, REC 20.5/85.4, empty 0.00 |
| 0.3 | 0.699 ± 0.026 | retained 38.8/180, REC 35.2/85.4, empty 0.00 |
| 1.0 | 0.720 ± 0.025 | retained 70.0/180, REC 55.1/85.4, empty 0.00 |
| 3.0 | 0.735 ± 0.023 | retained 128.0/180, REC 75.2/85.4, empty 0.00 |
| 10.0 | 0.738 ± 0.023 | retained 180.0/180, REC 85.4/85.4, empty 0.00 |
| 30.0 | 0.738 ± 0.023 | retained 180.0/180, REC 85.4/85.4, empty 0.00 |
| 100.0 | 0.738 ± 0.023 | retained 180.0/180, REC 85.4/85.4, empty 0.00 |

