# Landsat summary (n24_standard, scope: partial)

Sources: run_20261026-171518, run_20261023-172941

## QELM readouts (F1 macro, mean ± std over seeds)

| Readout | F1 macro | Accuracy | Features | Retained ET | Paper F1 (124 q) |
|---|---|---|---|---|---|
| w1_et_scaled | 0.788 ± 0.029 | 0.830 ± 0.022 | 72 | 72.000 ± 0.000 | 0.812 |
| w1_et_unit | 0.792 ± 0.029 | 0.832 ± 0.023 | 72 | 72.000 ± 0.000 | 0.783 |
| w1_etcut_scaled | 0.788 ± 0.029 | 0.830 ± 0.022 | 72 | 72.000 ± 0.000 |  |
| w1_etcut_unit | 0.792 ± 0.029 | 0.832 ± 0.023 | 72 | 72.000 ± 0.000 | 0.789 |
| w1_pauli_scaled | 0.785 ± 0.029 | 0.830 ± 0.022 | 72 |  | 0.800 |
| w1_pauli_unit | 0.790 ± 0.029 | 0.831 ± 0.023 | 72 |  | 0.790 |
| w1_x_scaled | 0.725 ± 0.033 | 0.792 ± 0.024 | 24 |  | 0.782 |
| w1_x_unit | 0.721 ± 0.035 | 0.791 ± 0.024 | 24 |  | 0.783 |
| w2_et_scaled | 0.804 ± 0.031 | 0.842 ± 0.025 | 180 | 180.000 ± 0.000 | 0.801 |
| w2_et_unit | 0.810 ± 0.031 | 0.844 ± 0.025 | 180 | 180.000 ± 0.000 | 0.749 |
| w2_etcut_scaled | 0.802 ± 0.034 | 0.840 ± 0.027 | 128 | 127.956 ± 5.232 | 0.830 |
| w2_etcut_unit | 0.807 ± 0.031 | 0.842 ± 0.026 | 128 | 127.956 ± 5.232 | 0.792 |
| w2_pauli_scaled | 0.807 ± 0.030 | 0.844 ± 0.024 | 180 |  | 0.789 |
| w2_pauli_unit | 0.810 ± 0.028 | 0.844 ± 0.023 | 180 |  | 0.750 |
| w2_x_scaled | 0.754 ± 0.032 | 0.809 ± 0.023 | 36 |  | 0.789 |
| w2_x_unit | 0.755 ± 0.031 | 0.809 ± 0.023 | 36 |  | 0.750 |

## Paired F1-macro differences (same split, reservoir and shot draw)

| A - B | mean | std | stderr | n | A > B | A < B |
|---|---|---|---|---|---|---|
| w1_pauli_scaled - w1_x_scaled | +0.0602 | 0.0250 | 0.0038 | 45 | 45 | 0 |
| w1_et_scaled - w1_pauli_scaled | +0.0029 | 0.0092 | 0.0014 | 45 | 24 | 15 |
| w2_et_scaled - w2_pauli_scaled | -0.0021 | 0.0150 | 0.0023 | 45 | 20 | 25 |
| w2_etcut_scaled - w2_et_scaled | -0.0026 | 0.0104 | 0.0016 | 45 | 15 | 26 |
| w2_etcut_scaled - w1_et_scaled | +0.0135 | 0.0182 | 0.0027 | 45 | 32 | 13 |
| w1_pauli_scaled - w1_pauli_unit | -0.0045 | 0.0103 | 0.0015 | 45 | 14 | 27 |
| w2_pauli_scaled - w2_pauli_unit | -0.0032 | 0.0120 | 0.0018 | 45 | 15 | 25 |
| w2_et_scaled - w2_et_unit | -0.0056 | 0.0170 | 0.0026 | 45 | 16 | 29 |
| w2_etcut_scaled - w2_etcut_unit | -0.0046 | 0.0169 | 0.0026 | 45 | 22 | 23 |
| w2_pauli_unit - w1_pauli_unit | +0.0198 | 0.0221 | 0.0033 | 45 | 37 | 8 |

## shots_sweep: w1_pauli_scaled

| value | F1 macro | extra |
|---|---|---|
| 100 | 0.277 ± 0.035 |  |
| 1000 | 0.527 ± 0.037 |  |
| 10000 | 0.717 ± 0.030 |  |
| 100000 | 0.789 ± 0.033 |  |

## train_size_sweep: w1_pauli_scaled

| value | F1 macro | extra |
|---|---|---|
| 10 | 0.430 ± 0.115 |  |
| 25 | 0.617 ± 0.074 |  |
| 50 | 0.681 ± 0.064 |  |
| 100 | 0.722 ± 0.050 |  |
| 200 | 0.756 ± 0.039 |  |
| 400 | 0.776 ± 0.036 |  |
| 688 | 0.783 ± 0.035 |  |

## cutoff_sweep: w2_etcut_scaled

| value | F1 macro | extra |
|---|---|---|
| 0.01 | 0.484 ± 0.088 | retained 3.1/180, REC 3.1/121.0, empty 0.00 |
| 0.03 | 0.662 ± 0.037 | retained 16.8/180, REC 16.5/121.0, empty 0.00 |
| 0.1 | 0.761 ± 0.037 | retained 41.2/180, REC 39.5/121.0, empty 0.00 |
| 0.3 | 0.787 ± 0.031 | retained 73.5/180, REC 66.6/121.0, empty 0.00 |
| 1.0 | 0.802 ± 0.034 | retained 128.0/180, REC 101.1/121.0, empty 0.00 |
| 3.0 | 0.804 ± 0.031 | retained 178.7/180, REC 120.7/121.0, empty 0.00 |
| 10.0 | 0.804 ± 0.031 | retained 180.0/180, REC 121.0/121.0, empty 0.00 |
| 30.0 | 0.804 ± 0.031 | retained 180.0/180, REC 121.0/121.0, empty 0.00 |
| 100.0 | 0.804 ± 0.031 | retained 180.0/180, REC 121.0/121.0, empty 0.00 |

## cutoff_sweep: w2_etcut_unit

| value | F1 macro | extra |
|---|---|---|
| 0.01 | 0.484 ± 0.087 | retained 3.1/180, REC 3.1/121.0, empty 0.00 |
| 0.03 | 0.670 ± 0.034 | retained 16.8/180, REC 16.5/121.0, empty 0.00 |
| 0.1 | 0.766 ± 0.032 | retained 41.2/180, REC 39.5/121.0, empty 0.00 |
| 0.3 | 0.788 ± 0.030 | retained 73.5/180, REC 66.6/121.0, empty 0.00 |
| 1.0 | 0.807 ± 0.031 | retained 128.0/180, REC 101.1/121.0, empty 0.00 |
| 3.0 | 0.810 ± 0.030 | retained 178.7/180, REC 120.7/121.0, empty 0.00 |
| 10.0 | 0.810 ± 0.031 | retained 180.0/180, REC 121.0/121.0, empty 0.00 |
| 30.0 | 0.810 ± 0.031 | retained 180.0/180, REC 121.0/121.0, empty 0.00 |
| 100.0 | 0.810 ± 0.031 | retained 180.0/180, REC 121.0/121.0, empty 0.00 |

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

