# Landsat summary (classical, scope: partial)

Sources: run_20261023-172941

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

