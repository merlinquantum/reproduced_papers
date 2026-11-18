# Landsat summary (n16, scope: partial)

Sources: run_20261023-173640, run_20261023-172941

## QELM readouts (F1 macro, mean ± std over seeds)

| Readout | F1 macro | Accuracy | Features | Retained ET | Paper F1 (124 q) |
|---|---|---|---|---|---|
| w1_et_scaled | 0.719 ± 0.040 | 0.800 ± 0.030 | 48 | 48.000 ± 0.000 | 0.812 |
| w1_et_unit | 0.719 ± 0.041 | 0.799 ± 0.031 | 48 | 48.000 ± 0.000 | 0.783 |
| w1_etcut_scaled | 0.719 ± 0.040 | 0.800 ± 0.029 | 48 | 47.644 ± 0.564 |  |
| w1_etcut_unit | 0.718 ± 0.041 | 0.798 ± 0.031 | 48 | 47.644 ± 0.564 | 0.789 |
| w1_pauli_scaled | 0.715 ± 0.039 | 0.799 ± 0.029 | 48 |  | 0.800 |
| w1_pauli_unit | 0.719 ± 0.039 | 0.800 ± 0.029 | 48 |  | 0.790 |
| w1_x_scaled | 0.629 ± 0.038 | 0.740 ± 0.028 | 16 |  | 0.782 |
| w1_x_unit | 0.633 ± 0.038 | 0.741 ± 0.029 | 16 |  | 0.783 |
| w2_et_scaled | 0.720 ± 0.038 | 0.803 ± 0.028 | 120 | 120.000 ± 0.000 | 0.801 |
| w2_et_unit | 0.736 ± 0.036 | 0.805 ± 0.028 | 120 | 120.000 ± 0.000 | 0.749 |
| w2_etcut_scaled | 0.706 ± 0.036 | 0.797 ± 0.027 | 45 | 45.000 ± 1.592 | 0.830 |
| w2_etcut_unit | 0.714 ± 0.037 | 0.798 ± 0.029 | 45 | 45.000 ± 1.592 | 0.792 |
| w2_pauli_scaled | 0.720 ± 0.039 | 0.803 ± 0.027 | 120 |  | 0.789 |
| w2_pauli_unit | 0.739 ± 0.034 | 0.810 ± 0.026 | 120 |  | 0.750 |
| w2_x_scaled | 0.656 ± 0.042 | 0.758 ± 0.032 | 24 |  | 0.789 |
| w2_x_unit | 0.653 ± 0.040 | 0.755 ± 0.031 | 24 |  | 0.750 |

## Paired F1-macro differences (same split, reservoir and shot draw)

| A - B | mean | std | stderr | n | A > B | A < B |
|---|---|---|---|---|---|---|
| w1_pauli_scaled - w1_x_scaled | +0.0867 | 0.0354 | 0.0053 | 45 | 45 | 0 |
| w1_et_scaled - w1_pauli_scaled | +0.0037 | 0.0114 | 0.0017 | 45 | 19 | 12 |
| w2_et_scaled - w2_pauli_scaled | -0.0000 | 0.0261 | 0.0039 | 45 | 21 | 24 |
| w2_etcut_scaled - w2_et_scaled | -0.0135 | 0.0205 | 0.0031 | 45 | 11 | 34 |
| w2_etcut_scaled - w1_et_scaled | -0.0130 | 0.0251 | 0.0038 | 45 | 15 | 30 |
| w1_pauli_scaled - w1_pauli_unit | -0.0039 | 0.0108 | 0.0016 | 45 | 15 | 24 |
| w2_pauli_scaled - w2_pauli_unit | -0.0197 | 0.0208 | 0.0031 | 45 | 7 | 35 |
| w2_et_scaled - w2_et_unit | -0.0160 | 0.0226 | 0.0034 | 45 | 10 | 35 |
| w2_etcut_scaled - w2_etcut_unit | -0.0083 | 0.0211 | 0.0032 | 45 | 14 | 28 |
| w2_pauli_unit - w1_pauli_unit | +0.0200 | 0.0325 | 0.0049 | 45 | 33 | 12 |

## shots_sweep: w1_pauli_scaled

| value | F1 macro | extra |
|---|---|---|
| 100 | 0.194 ± 0.025 |  |
| 1000 | 0.370 ± 0.035 |  |
| 10000 | 0.573 ± 0.034 |  |
| 100000 | 0.713 ± 0.032 |  |

## train_size_sweep: w1_pauli_scaled

| value | F1 macro | extra |
|---|---|---|
| 10 | 0.393 ± 0.123 |  |
| 25 | 0.584 ± 0.072 |  |
| 50 | 0.643 ± 0.062 |  |
| 100 | 0.669 ± 0.050 |  |
| 200 | 0.697 ± 0.046 |  |
| 400 | 0.710 ± 0.042 |  |
| 688 | 0.717 ± 0.040 |  |

## cutoff_sweep: w2_etcut_scaled

| value | F1 macro | extra |
|---|---|---|
| 0.01 | 0.064 ± 0.000 | retained 0.0/120, REC 0.0/56.2, empty 1.00 |
| 0.03 | 0.519 ± 0.036 | retained 5.2/120, REC 5.1/56.2, empty 0.00 |
| 0.1 | 0.619 ± 0.038 | retained 13.0/120, REC 12.4/56.2, empty 0.00 |
| 0.3 | 0.684 ± 0.037 | retained 24.5/120, REC 22.2/56.2, empty 0.00 |
| 1.0 | 0.706 ± 0.036 | retained 45.0/120, REC 35.0/56.2, empty 0.00 |
| 3.0 | 0.720 ± 0.035 | retained 86.1/120, REC 49.5/56.2, empty 0.00 |
| 10.0 | 0.720 ± 0.038 | retained 120.0/120, REC 56.2/56.2, empty 0.00 |
| 30.0 | 0.720 ± 0.038 | retained 120.0/120, REC 56.2/56.2, empty 0.00 |
| 100.0 | 0.720 ± 0.038 | retained 120.0/120, REC 56.2/56.2, empty 0.00 |

## cutoff_sweep: w2_etcut_unit

| value | F1 macro | extra |
|---|---|---|
| 0.01 | 0.064 ± 0.000 | retained 0.0/120, REC 0.0/56.2, empty 1.00 |
| 0.03 | 0.519 ± 0.037 | retained 5.2/120, REC 5.1/56.2, empty 0.00 |
| 0.1 | 0.620 ± 0.040 | retained 13.0/120, REC 12.4/56.2, empty 0.00 |
| 0.3 | 0.688 ± 0.036 | retained 24.5/120, REC 22.2/56.2, empty 0.00 |
| 1.0 | 0.714 ± 0.037 | retained 45.0/120, REC 35.0/56.2, empty 0.00 |
| 3.0 | 0.731 ± 0.037 | retained 86.1/120, REC 49.5/56.2, empty 0.00 |
| 10.0 | 0.736 ± 0.036 | retained 120.0/120, REC 56.2/56.2, empty 0.00 |
| 30.0 | 0.736 ± 0.036 | retained 120.0/120, REC 56.2/56.2, empty 0.00 |
| 100.0 | 0.736 ± 0.036 | retained 120.0/120, REC 56.2/56.2, empty 0.00 |

## Classical baselines (F1 macro, mean ± std over splits)

| Model | F1 macro | Accuracy | Paper F1 |
|---|---|---|---|
| adaboost (Table A2) | 0.711 ± 0.052 | 0.750 ± 0.040 | 0.699 |
| decision_tree (Table A2) | 0.768 ± 0.030 | 0.797 ± 0.029 | 0.776 |
| knn (Table A2) | 0.810 ± 0.032 | 0.837 ± 0.030 | 0.834 |
| mlp (Table A2) | 0.835 ± 0.024 | 0.855 ± 0.026 | 0.830 |
| naive_bayes (Table A2) | 0.751 ± 0.029 | 0.769 ± 0.027 | 0.782 |
| qda (Table A2) | 0.727 ± 0.024 | 0.799 ± 0.021 | 0.716 |
| random_forest (Table A2) | 0.856 ± 0.026 | 0.876 ± 0.022 | 0.840 |
| ridge (Table A2) | 0.588 ± 0.028 | 0.748 ± 0.021 | 0.608 |
| svm_rbf (Table A2) | 0.842 ± 0.032 | 0.869 ± 0.029 | 0.809 |
| ridge_degree2 | 0.749 ± 0.017 | 0.823 ± 0.016 | |
| elm fourier|n_features=120|noise_std=0.0 | 0.820 ± 0.024 | 0.855 ± 0.022 | |
| elm fourier|n_features=120|noise_std=0.01 | 0.816 ± 0.023 | 0.852 ± 0.020 | |
| elm fourier|n_features=180|noise_std=0.0 | 0.831 ± 0.025 | 0.863 ± 0.022 | |
| elm fourier|n_features=180|noise_std=0.01 | 0.831 ± 0.024 | 0.863 ± 0.022 | |
| elm fourier|n_features=48|noise_std=0.0 | 0.787 ± 0.013 | 0.838 ± 0.012 | |
| elm fourier|n_features=48|noise_std=0.01 | 0.787 ± 0.013 | 0.838 ± 0.013 | |
| elm fourier|n_features=72|noise_std=0.0 | 0.803 ± 0.030 | 0.848 ± 0.021 | |
| elm fourier|n_features=72|noise_std=0.01 | 0.801 ± 0.028 | 0.847 ± 0.020 | |
| elm tanh|n_features=120|noise_std=0.0 | 0.793 ± 0.015 | 0.842 ± 0.014 | |
| elm tanh|n_features=120|noise_std=0.01 | 0.781 ± 0.016 | 0.836 ± 0.012 | |
| elm tanh|n_features=180|noise_std=0.0 | 0.800 ± 0.020 | 0.847 ± 0.019 | |
| elm tanh|n_features=180|noise_std=0.01 | 0.782 ± 0.015 | 0.838 ± 0.014 | |
| elm tanh|n_features=48|noise_std=0.0 | 0.749 ± 0.023 | 0.821 ± 0.020 | |
| elm tanh|n_features=48|noise_std=0.01 | 0.741 ± 0.019 | 0.818 ± 0.018 | |
| elm tanh|n_features=72|noise_std=0.0 | 0.775 ± 0.022 | 0.836 ± 0.021 | |
| elm tanh|n_features=72|noise_std=0.01 | 0.764 ± 0.017 | 0.831 ± 0.019 | |
| tuned knn | 0.804 ± 0.031 | 0.834 ± 0.028 | |
| tuned random_forest | 0.862 ± 0.030 | 0.881 ± 0.025 | |
| tuned svm_rbf | 0.843 ± 0.027 | 0.866 ± 0.029 | |

