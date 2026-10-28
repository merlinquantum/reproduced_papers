# Landsat summary (n24, scope: partial)

Sources: run_20261026-094113, run_20261026-084131, run_20261023-172941

## QELM readouts (F1 macro, mean ± std over seeds)

| Readout | F1 macro | Accuracy | Features | Retained ET | Paper F1 (124 q) |
|---|---|---|---|---|---|
| w1_et_scaled | 0.737 ± 0.030 | 0.814 ± 0.024 | 72 | 72.000 ± 0.000 | 0.812 |
| w1_et_unit | 0.736 ± 0.028 | 0.814 ± 0.022 | 72 | 72.000 ± 0.000 | 0.783 |
| w1_etcut_scaled | 0.738 ± 0.030 | 0.815 ± 0.024 | 71 | 71.022 ± 0.614 |  |
| w1_etcut_unit | 0.733 ± 0.031 | 0.813 ± 0.024 | 71 | 71.022 ± 0.614 | 0.789 |
| w1_pauli_scaled | 0.734 ± 0.031 | 0.815 ± 0.025 | 72 |  | 0.800 |
| w1_pauli_unit | 0.731 ± 0.032 | 0.812 ± 0.024 | 72 |  | 0.790 |
| w1_x_scaled | 0.690 ± 0.032 | 0.792 ± 0.025 | 24 |  | 0.782 |
| w1_x_unit | 0.691 ± 0.033 | 0.791 ± 0.025 | 24 |  | 0.783 |
| w2_et_scaled | 0.741 ± 0.033 | 0.817 ± 0.026 | 180 | 180.000 ± 0.000 | 0.801 |
| w2_et_unit | 0.754 ± 0.029 | 0.813 ± 0.024 | 180 | 180.000 ± 0.000 | 0.749 |
| w2_etcut_scaled | 0.731 ± 0.033 | 0.813 ± 0.025 | 70 | 70.111 ± 2.652 | 0.830 |
| w2_etcut_unit | 0.738 ± 0.032 | 0.813 ± 0.025 | 70 | 70.111 ± 2.652 | 0.792 |
| w2_pauli_scaled | 0.739 ± 0.030 | 0.816 ± 0.023 | 180 |  | 0.789 |
| w2_pauli_unit | 0.757 ± 0.025 | 0.818 ± 0.022 | 180 |  | 0.750 |
| w2_x_scaled | 0.713 ± 0.033 | 0.801 ± 0.025 | 36 |  | 0.789 |
| w2_x_unit | 0.713 ± 0.033 | 0.800 ± 0.026 | 36 |  | 0.750 |

## Paired F1-macro differences (same split, reservoir and shot draw)

| A - B | mean | std | stderr | n | A > B | A < B |
|---|---|---|---|---|---|---|
| w1_pauli_scaled - w1_x_scaled | +0.0434 | 0.0265 | 0.0040 | 45 | 42 | 3 |
| w1_et_scaled - w1_pauli_scaled | +0.0035 | 0.0127 | 0.0019 | 45 | 26 | 17 |
| w2_et_scaled - w2_pauli_scaled | +0.0015 | 0.0203 | 0.0031 | 45 | 25 | 20 |
| w2_etcut_scaled - w2_et_scaled | -0.0100 | 0.0201 | 0.0030 | 45 | 14 | 30 |
| w2_etcut_scaled - w1_et_scaled | -0.0064 | 0.0242 | 0.0036 | 45 | 21 | 24 |
| w1_pauli_scaled - w1_pauli_unit | +0.0029 | 0.0130 | 0.0020 | 45 | 23 | 18 |
| w2_pauli_scaled - w2_pauli_unit | -0.0175 | 0.0199 | 0.0030 | 45 | 9 | 35 |
| w2_et_scaled - w2_et_unit | -0.0129 | 0.0259 | 0.0039 | 45 | 14 | 31 |
| w2_etcut_scaled - w2_etcut_unit | -0.0077 | 0.0211 | 0.0032 | 45 | 17 | 27 |
| w2_pauli_unit - w1_pauli_unit | +0.0260 | 0.0322 | 0.0049 | 45 | 36 | 9 |

## shots_sweep: w1_pauli_scaled

| value | F1 macro | extra |
|---|---|---|
| 100 | 0.240 ± 0.028 |  |
| 1000 | 0.466 ± 0.027 |  |
| 10000 | 0.642 ± 0.032 |  |
| 100000 | 0.739 ± 0.035 |  |

## train_size_sweep: w1_pauli_scaled

| value | F1 macro | extra |
|---|---|---|
| 10 | 0.442 ± 0.115 |  |
| 25 | 0.615 ± 0.069 |  |
| 50 | 0.668 ± 0.059 |  |
| 100 | 0.693 ± 0.048 |  |
| 200 | 0.720 ± 0.040 |  |
| 400 | 0.734 ± 0.036 |  |
| 688 | 0.740 ± 0.034 |  |

## cutoff_sweep: w2_etcut_scaled

| value | F1 macro | extra |
|---|---|---|
| 0.01 | 0.064 ± 0.000 | retained 0.0/180, REC 0.0/85.6, empty 1.00 |
| 0.03 | 0.619 ± 0.041 | retained 10.0/180, REC 9.8/85.6, empty 0.00 |
| 0.1 | 0.693 ± 0.037 | retained 21.5/180, REC 20.5/85.6, empty 0.00 |
| 0.3 | 0.718 ± 0.034 | retained 39.3/180, REC 35.7/85.6, empty 0.00 |
| 1.0 | 0.731 ± 0.033 | retained 70.1/180, REC 55.2/85.6, empty 0.00 |
| 3.0 | 0.740 ± 0.032 | retained 128.4/180, REC 75.5/85.6, empty 0.00 |
| 10.0 | 0.741 ± 0.033 | retained 180.0/180, REC 85.6/85.6, empty 0.00 |
| 30.0 | 0.741 ± 0.033 | retained 180.0/180, REC 85.6/85.6, empty 0.00 |
| 100.0 | 0.741 ± 0.033 | retained 180.0/180, REC 85.6/85.6, empty 0.00 |

## cutoff_sweep: w2_etcut_unit

| value | F1 macro | extra |
|---|---|---|
| 0.01 | 0.064 ± 0.000 | retained 0.0/180, REC 0.0/85.6, empty 1.00 |
| 0.03 | 0.619 ± 0.041 | retained 10.0/180, REC 9.8/85.6, empty 0.00 |
| 0.1 | 0.690 ± 0.035 | retained 21.5/180, REC 20.5/85.6, empty 0.00 |
| 0.3 | 0.720 ± 0.034 | retained 39.3/180, REC 35.7/85.6, empty 0.00 |
| 1.0 | 0.738 ± 0.032 | retained 70.1/180, REC 55.2/85.6, empty 0.00 |
| 3.0 | 0.754 ± 0.030 | retained 128.4/180, REC 75.5/85.6, empty 0.00 |
| 10.0 | 0.754 ± 0.029 | retained 180.0/180, REC 85.6/85.6, empty 0.00 |
| 30.0 | 0.754 ± 0.029 | retained 180.0/180, REC 85.6/85.6, empty 0.00 |
| 100.0 | 0.754 ± 0.029 | retained 180.0/180, REC 85.6/85.6, empty 0.00 |

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

