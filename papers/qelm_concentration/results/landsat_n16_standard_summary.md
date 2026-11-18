# Landsat summary (n16_standard, scope: partial)

Sources: run_20261026-155536, run_20261023-172941

## QELM readouts (F1 macro, mean ± std over seeds)

| Readout | F1 macro | Accuracy | Features | Retained ET | Paper F1 (124 q) |
|---|---|---|---|---|---|
| w1_et_scaled | 0.776 ± 0.031 | 0.821 ± 0.024 | 48 | 48.000 ± 0.000 | 0.812 |
| w1_et_unit | 0.777 ± 0.031 | 0.821 ± 0.023 | 48 | 48.000 ± 0.000 | 0.783 |
| w1_etcut_scaled | 0.776 ± 0.031 | 0.821 ± 0.024 | 48 | 48.000 ± 0.000 |  |
| w1_etcut_unit | 0.777 ± 0.031 | 0.821 ± 0.023 | 48 | 48.000 ± 0.000 | 0.789 |
| w1_pauli_scaled | 0.775 ± 0.030 | 0.821 ± 0.022 | 48 |  | 0.800 |
| w1_pauli_unit | 0.776 ± 0.030 | 0.821 ± 0.022 | 48 |  | 0.790 |
| w1_x_scaled | 0.661 ± 0.044 | 0.746 ± 0.029 | 16 |  | 0.782 |
| w1_x_unit | 0.653 ± 0.046 | 0.743 ± 0.028 | 16 |  | 0.783 |
| w2_et_scaled | 0.791 ± 0.032 | 0.832 ± 0.024 | 120 | 120.000 ± 0.000 | 0.801 |
| w2_et_unit | 0.795 ± 0.033 | 0.833 ± 0.027 | 120 | 120.000 ± 0.000 | 0.749 |
| w2_etcut_scaled | 0.787 ± 0.032 | 0.830 ± 0.024 | 85 | 85.489 ± 3.563 | 0.830 |
| w2_etcut_unit | 0.787 ± 0.032 | 0.828 ± 0.025 | 85 | 85.489 ± 3.563 | 0.792 |
| w2_pauli_scaled | 0.791 ± 0.035 | 0.832 ± 0.026 | 120 |  | 0.789 |
| w2_pauli_unit | 0.795 ± 0.032 | 0.834 ± 0.025 | 120 |  | 0.750 |
| w2_x_scaled | 0.710 ± 0.037 | 0.770 ± 0.029 | 24 |  | 0.789 |
| w2_x_unit | 0.712 ± 0.038 | 0.771 ± 0.030 | 24 |  | 0.750 |

## Paired F1-macro differences (same split, reservoir and shot draw)

| A - B | mean | std | stderr | n | A > B | A < B |
|---|---|---|---|---|---|---|
| w1_pauli_scaled - w1_x_scaled | +0.1140 | 0.0434 | 0.0065 | 45 | 45 | 0 |
| w1_et_scaled - w1_pauli_scaled | +0.0010 | 0.0102 | 0.0015 | 45 | 21 | 15 |
| w2_et_scaled - w2_pauli_scaled | +0.0005 | 0.0168 | 0.0025 | 45 | 19 | 25 |
| w2_etcut_scaled - w2_et_scaled | -0.0044 | 0.0111 | 0.0017 | 45 | 16 | 24 |
| w2_etcut_scaled - w1_et_scaled | +0.0110 | 0.0208 | 0.0031 | 45 | 32 | 13 |
| w1_pauli_scaled - w1_pauli_unit | -0.0014 | 0.0090 | 0.0014 | 45 | 17 | 23 |
| w2_pauli_scaled - w2_pauli_unit | -0.0039 | 0.0153 | 0.0023 | 45 | 17 | 27 |
| w2_et_scaled - w2_et_unit | -0.0041 | 0.0218 | 0.0033 | 45 | 21 | 24 |
| w2_etcut_scaled - w2_etcut_unit | +0.0000 | 0.0187 | 0.0028 | 45 | 25 | 20 |
| w2_pauli_unit - w1_pauli_unit | +0.0184 | 0.0222 | 0.0033 | 45 | 37 | 8 |

## shots_sweep: w1_pauli_scaled

| value | F1 macro | extra |
|---|---|---|
| 100 | 0.239 ± 0.026 |  |
| 1000 | 0.475 ± 0.028 |  |
| 10000 | 0.674 ± 0.037 |  |
| 100000 | 0.766 ± 0.035 |  |

## train_size_sweep: w1_pauli_scaled

| value | F1 macro | extra |
|---|---|---|
| 10 | 0.402 ± 0.121 |  |
| 25 | 0.600 ± 0.079 |  |
| 50 | 0.667 ± 0.066 |  |
| 100 | 0.709 ± 0.052 |  |
| 200 | 0.742 ± 0.042 |  |
| 400 | 0.759 ± 0.037 |  |
| 688 | 0.767 ± 0.034 |  |

## cutoff_sweep: w2_etcut_scaled

| value | F1 macro | extra |
|---|---|---|
| 0.01 | 0.238 ± 0.160 | retained 1.9/120, REC 1.9/80.6, empty 0.33 |
| 0.03 | 0.653 ± 0.046 | retained 13.2/120, REC 13.0/80.6, empty 0.00 |
| 0.1 | 0.734 ± 0.036 | retained 25.9/120, REC 25.0/80.6, empty 0.00 |
| 0.3 | 0.771 ± 0.035 | retained 46.1/120, REC 41.9/80.6, empty 0.00 |
| 1.0 | 0.787 ± 0.032 | retained 85.5/120, REC 67.0/80.6, empty 0.00 |
| 3.0 | 0.791 ± 0.031 | retained 119.5/120, REC 80.5/80.6, empty 0.00 |
| 10.0 | 0.791 ± 0.032 | retained 120.0/120, REC 80.6/80.6, empty 0.00 |
| 30.0 | 0.791 ± 0.032 | retained 120.0/120, REC 80.6/80.6, empty 0.00 |
| 100.0 | 0.791 ± 0.032 | retained 120.0/120, REC 80.6/80.6, empty 0.00 |

## cutoff_sweep: w2_etcut_unit

| value | F1 macro | extra |
|---|---|---|
| 0.01 | 0.238 ± 0.160 | retained 1.9/120, REC 1.9/80.6, empty 0.33 |
| 0.03 | 0.655 ± 0.046 | retained 13.2/120, REC 13.0/80.6, empty 0.00 |
| 0.1 | 0.735 ± 0.035 | retained 25.9/120, REC 25.0/80.6, empty 0.00 |
| 0.3 | 0.774 ± 0.038 | retained 46.1/120, REC 41.9/80.6, empty 0.00 |
| 1.0 | 0.787 ± 0.032 | retained 85.5/120, REC 67.0/80.6, empty 0.00 |
| 3.0 | 0.795 ± 0.033 | retained 119.5/120, REC 80.5/80.6, empty 0.00 |
| 10.0 | 0.795 ± 0.033 | retained 120.0/120, REC 80.6/80.6, empty 0.00 |
| 30.0 | 0.795 ± 0.033 | retained 120.0/120, REC 80.6/80.6, empty 0.00 |
| 100.0 | 0.795 ± 0.033 | retained 120.0/120, REC 80.6/80.6, empty 0.00 |

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

