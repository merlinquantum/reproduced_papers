# Sweep results

Full candidate tables of the included sweeps, recomputed from every run's `metrics.json` and checked against each sweep's `summary.json`. Metric: stratified 5-fold CV accuracy (higher is better); aggregate: mean over seeds 0-4 of the 5-fold mean; variability: std over seeds (ddof = 1); the mean fold std is the paper's ±. Two methods are equivalent when |difference| ≤ 2 σ_pooled. No candidate is selected across a sweep.

## `table1_original`

Table 1, paper protocol, with the exact-kernel and majority-class references. Sweep folder `sweep_table1_original_20261020-173338`; COMPLETE SWEEP, 30/30 runs.

| Candidate | Method | Model | Runs | Accuracy | Std over seeds | Mean fold std | Balanced accuracy |
|---|---|---|---:|---:|---:|---:|---:|
| accuracy_framingham_original | classical_linear | kernel_pca_2_components | 5/5 | 0.6675 | 0.0035 | 0.0166 | 0.6013 |
| accuracy_framingham_original | classical_linear | kernel_pca_3_components | 5/5 | 0.6676 | 0.0062 | 0.0106 | 0.5912 |
| accuracy_framingham_original | classical_linear | kernel_pca_4_components | 5/5 | 0.6761 | 0.0037 | 0.0121 | 0.6050 |
| accuracy_framingham_original | classical_linear | kernel_pca_5_components | 5/5 | 0.6785 | 0.0061 | 0.0106 | 0.6054 |
| accuracy_framingham_original | classical_linear | kernel_pca_6_components | 5/5 | 0.6811 | 0.0041 | 0.0122 | 0.6161 |
| accuracy_framingham_original | classical_linear | kernel_pca_7_components | 5/5 | 0.6806 | 0.0034 | 0.0134 | 0.6166 |
| accuracy_framingham_original | classical_linear | kernel_pca_best_on_test_folds | 5/5 | 0.6824 | 0.0037 | 0.0117 | 0.6153 |
| accuracy_framingham_original | classical_linear | kernel_svm | 5/5 | 0.6894 | 0.0028 | 0.0137 | 0.6373 |
| accuracy_framingham_original | quantum_centralised | kernel_pca_2_components | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| accuracy_framingham_original | quantum_centralised | kernel_pca_3_components | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| accuracy_framingham_original | quantum_centralised | kernel_pca_4_components | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| accuracy_framingham_original | quantum_centralised | kernel_pca_5_components | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| accuracy_framingham_original | quantum_centralised | kernel_pca_6_components | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| accuracy_framingham_original | quantum_centralised | kernel_pca_7_components | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| accuracy_framingham_original | quantum_centralised | kernel_pca_best_on_test_folds | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| accuracy_framingham_original | quantum_centralised | kernel_svm | 5/5 | 0.6335 | 0.0093 | 0.0136 | 0.5889 |
| accuracy_framingham_original | quantum_distributed | kernel_pca_2_components | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| accuracy_framingham_original | quantum_distributed | kernel_pca_3_components | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| accuracy_framingham_original | quantum_distributed | kernel_pca_4_components | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| accuracy_framingham_original | quantum_distributed | kernel_pca_5_components | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| accuracy_framingham_original | quantum_distributed | kernel_pca_6_components | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| accuracy_framingham_original | quantum_distributed | kernel_pca_7_components | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| accuracy_framingham_original | quantum_distributed | kernel_pca_best_on_test_folds | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| accuracy_framingham_original | quantum_distributed | kernel_svm | 5/5 | 0.6393 | 0.0101 | 0.0098 | 0.5949 |
| accuracy_framingham_reference | exact_cosine | kernel_pca_2_components | 5/5 | 0.6458 | 0.0068 | 0.0143 | 0.6057 |
| accuracy_framingham_reference | exact_cosine | kernel_pca_3_components | 5/5 | 0.6455 | 0.0051 | 0.0141 | 0.6072 |
| accuracy_framingham_reference | exact_cosine | kernel_pca_4_components | 5/5 | 0.6711 | 0.0041 | 0.0129 | 0.6357 |
| accuracy_framingham_reference | exact_cosine | kernel_pca_5_components | 5/5 | 0.6691 | 0.0057 | 0.0137 | 0.6330 |
| accuracy_framingham_reference | exact_cosine | kernel_pca_6_components | 5/5 | 0.6792 | 0.0033 | 0.0119 | 0.6454 |
| accuracy_framingham_reference | exact_cosine | kernel_pca_7_components | 5/5 | 0.6786 | 0.0041 | 0.0099 | 0.6458 |
| accuracy_framingham_reference | exact_cosine | kernel_pca_best_on_test_folds | 5/5 | 0.6805 | 0.0030 | 0.0103 | 0.6470 |
| accuracy_framingham_reference | exact_cosine | kernel_svm | 5/5 | 0.6887 | 0.0018 | 0.0162 | 0.6530 |
| accuracy_framingham_reference | exact_squared_overlap | kernel_pca_2_components | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| accuracy_framingham_reference | exact_squared_overlap | kernel_pca_3_components | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| accuracy_framingham_reference | exact_squared_overlap | kernel_pca_4_components | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| accuracy_framingham_reference | exact_squared_overlap | kernel_pca_5_components | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| accuracy_framingham_reference | exact_squared_overlap | kernel_pca_6_components | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| accuracy_framingham_reference | exact_squared_overlap | kernel_pca_7_components | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| accuracy_framingham_reference | exact_squared_overlap | kernel_pca_best_on_test_folds | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| accuracy_framingham_reference | exact_squared_overlap | kernel_svm | 5/5 | 0.6881 | 0.0053 | 0.0124 | 0.6204 |
| accuracy_framingham_reference | majority_class | kernel_svm | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| accuracy_parkinsons_original | classical_linear | kernel_pca_2_components | 5/5 | 0.7564 | 0.0195 | 0.0695 | 0.7109 |
| accuracy_parkinsons_original | classical_linear | kernel_pca_3_components | 5/5 | 0.7373 | 0.0202 | 0.0631 | 0.6936 |
| accuracy_parkinsons_original | classical_linear | kernel_pca_4_components | 5/5 | 0.7767 | 0.0134 | 0.0916 | 0.7431 |
| accuracy_parkinsons_original | classical_linear | kernel_pca_5_components | 5/5 | 0.8055 | 0.0059 | 0.0817 | 0.7759 |
| accuracy_parkinsons_original | classical_linear | kernel_pca_6_components | 5/5 | 0.8034 | 0.0068 | 0.0832 | 0.7713 |
| accuracy_parkinsons_original | classical_linear | kernel_pca_7_components | 5/5 | 0.8075 | 0.0091 | 0.0799 | 0.7794 |
| accuracy_parkinsons_original | classical_linear | kernel_pca_best_on_test_folds | 5/5 | 0.8118 | 0.0046 | 0.0767 | 0.7833 |
| accuracy_parkinsons_original | classical_linear | kernel_svm | 5/5 | 0.8054 | 0.0126 | 0.0625 | 0.7893 |
| accuracy_parkinsons_original | quantum_centralised | kernel_pca_2_components | 5/5 | 0.6276 | 0.0000 | 0.0049 | 0.5000 |
| accuracy_parkinsons_original | quantum_centralised | kernel_pca_3_components | 5/5 | 0.6276 | 0.0000 | 0.0049 | 0.5000 |
| accuracy_parkinsons_original | quantum_centralised | kernel_pca_4_components | 5/5 | 0.6342 | 0.0356 | 0.0363 | 0.5413 |
| accuracy_parkinsons_original | quantum_centralised | kernel_pca_5_components | 5/5 | 0.6691 | 0.0584 | 0.0389 | 0.6000 |
| accuracy_parkinsons_original | quantum_centralised | kernel_pca_6_components | 5/5 | 0.6756 | 0.0564 | 0.0432 | 0.6140 |
| accuracy_parkinsons_original | quantum_centralised | kernel_pca_7_components | 5/5 | 0.7013 | 0.0490 | 0.0559 | 0.6407 |
| accuracy_parkinsons_original | quantum_centralised | kernel_pca_best_on_test_folds | 5/5 | 0.7033 | 0.0447 | 0.0423 | 0.6347 |
| accuracy_parkinsons_original | quantum_centralised | kernel_svm | 5/5 | 0.7725 | 0.0213 | 0.0714 | 0.7405 |
| accuracy_parkinsons_original | quantum_distributed | kernel_pca_2_components | 5/5 | 0.6276 | 0.0000 | 0.0049 | 0.5000 |
| accuracy_parkinsons_original | quantum_distributed | kernel_pca_3_components | 5/5 | 0.6276 | 0.0000 | 0.0049 | 0.5000 |
| accuracy_parkinsons_original | quantum_distributed | kernel_pca_4_components | 5/5 | 0.6362 | 0.0340 | 0.0319 | 0.5418 |
| accuracy_parkinsons_original | quantum_distributed | kernel_pca_5_components | 5/5 | 0.6607 | 0.0650 | 0.0342 | 0.5875 |
| accuracy_parkinsons_original | quantum_distributed | kernel_pca_6_components | 5/5 | 0.6808 | 0.0574 | 0.0400 | 0.6188 |
| accuracy_parkinsons_original | quantum_distributed | kernel_pca_7_components | 5/5 | 0.6960 | 0.0545 | 0.0517 | 0.6348 |
| accuracy_parkinsons_original | quantum_distributed | kernel_pca_best_on_test_folds | 5/5 | 0.7002 | 0.0467 | 0.0487 | 0.6305 |
| accuracy_parkinsons_original | quantum_distributed | kernel_svm | 5/5 | 0.7557 | 0.0347 | 0.0760 | 0.7229 |
| accuracy_parkinsons_reference | exact_cosine | kernel_pca_2_components | 5/5 | 0.7296 | 0.0143 | 0.0875 | 0.7213 |
| accuracy_parkinsons_reference | exact_cosine | kernel_pca_3_components | 5/5 | 0.7084 | 0.0144 | 0.0894 | 0.6918 |
| accuracy_parkinsons_reference | exact_cosine | kernel_pca_4_components | 5/5 | 0.7555 | 0.0106 | 0.0716 | 0.7432 |
| accuracy_parkinsons_reference | exact_cosine | kernel_pca_5_components | 5/5 | 0.7661 | 0.0161 | 0.0610 | 0.7562 |
| accuracy_parkinsons_reference | exact_cosine | kernel_pca_6_components | 5/5 | 0.7683 | 0.0157 | 0.0661 | 0.7551 |
| accuracy_parkinsons_reference | exact_cosine | kernel_pca_7_components | 5/5 | 0.7705 | 0.0056 | 0.0624 | 0.7557 |
| accuracy_parkinsons_reference | exact_cosine | kernel_pca_best_on_test_folds | 5/5 | 0.7747 | 0.0114 | 0.0638 | 0.7603 |
| accuracy_parkinsons_reference | exact_cosine | kernel_svm | 5/5 | 0.7915 | 0.0204 | 0.0669 | 0.7713 |
| accuracy_parkinsons_reference | exact_squared_overlap | kernel_pca_2_components | 5/5 | 0.6276 | 0.0000 | 0.0049 | 0.5000 |
| accuracy_parkinsons_reference | exact_squared_overlap | kernel_pca_3_components | 5/5 | 0.6276 | 0.0000 | 0.0049 | 0.5000 |
| accuracy_parkinsons_reference | exact_squared_overlap | kernel_pca_4_components | 5/5 | 0.6342 | 0.0302 | 0.0407 | 0.5401 |
| accuracy_parkinsons_reference | exact_squared_overlap | kernel_pca_5_components | 5/5 | 0.6596 | 0.0586 | 0.0316 | 0.5861 |
| accuracy_parkinsons_reference | exact_squared_overlap | kernel_pca_6_components | 5/5 | 0.6692 | 0.0471 | 0.0398 | 0.6054 |
| accuracy_parkinsons_reference | exact_squared_overlap | kernel_pca_7_components | 5/5 | 0.6949 | 0.0596 | 0.0568 | 0.6333 |
| accuracy_parkinsons_reference | exact_squared_overlap | kernel_pca_best_on_test_folds | 5/5 | 0.7012 | 0.0468 | 0.0499 | 0.6313 |
| accuracy_parkinsons_reference | exact_squared_overlap | kernel_svm | 5/5 | 0.7726 | 0.0161 | 0.0797 | 0.7423 |
| accuracy_parkinsons_reference | majority_class | kernel_svm | 5/5 | 0.6276 | 0.0000 | 0.0049 | 0.5000 |
| accuracy_wine_original | classical_linear | kernel_svm | 5/5 | 0.9860 | 0.0000 | 0.0172 | 0.9884 |
| accuracy_wine_original | quantum_centralised | kernel_svm | 5/5 | 0.9067 | 0.0190 | 0.0328 | 0.9105 |
| accuracy_wine_original | quantum_distributed | kernel_svm | 5/5 | 0.8957 | 0.0090 | 0.0370 | 0.9015 |
| accuracy_wine_reference | exact_cosine | kernel_svm | 5/5 | 0.9857 | 0.0000 | 0.0175 | 0.9865 |
| accuracy_wine_reference | exact_squared_overlap | kernel_svm | 5/5 | 0.9012 | 0.0000 | 0.0419 | 0.9064 |
| accuracy_wine_reference | majority_class | kernel_svm | 5/5 | 0.4012 | 0.0000 | 0.0103 | 0.3333 |

## `noise_original`

Fig. 3, the authors' noise model with noiseless state preparation. Sweep folder `sweep_noise_original_20261020-173524`; COMPLETE SWEEP, 10/10 runs.

| Candidate | Method | Model | Runs | Accuracy | Std over seeds | Mean fold std | Balanced accuracy |
|---|---|---|---:|---:|---:|---:|---:|
| noise_framingham_original | classical_linear | kernel_pca_2_components | 5/5 | 0.6675 | 0.0035 | 0.0166 | 0.6013 |
| noise_framingham_original | classical_linear | kernel_pca_3_components | 5/5 | 0.6676 | 0.0062 | 0.0106 | 0.5912 |
| noise_framingham_original | classical_linear | kernel_pca_4_components | 5/5 | 0.6761 | 0.0037 | 0.0121 | 0.6050 |
| noise_framingham_original | classical_linear | kernel_pca_5_components | 5/5 | 0.6785 | 0.0061 | 0.0106 | 0.6054 |
| noise_framingham_original | classical_linear | kernel_pca_6_components | 5/5 | 0.6811 | 0.0041 | 0.0122 | 0.6161 |
| noise_framingham_original | classical_linear | kernel_pca_7_components | 5/5 | 0.6806 | 0.0034 | 0.0134 | 0.6166 |
| noise_framingham_original | classical_linear | kernel_pca_best_on_test_folds | 5/5 | 0.6824 | 0.0037 | 0.0117 | 0.6153 |
| noise_framingham_original | classical_linear | kernel_svm | 5/5 | 0.6894 | 0.0028 | 0.0137 | 0.6373 |
| noise_framingham_original | quantum_distributed_noise_level_0 | kernel_pca_2_components | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| noise_framingham_original | quantum_distributed_noise_level_0 | kernel_pca_3_components | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| noise_framingham_original | quantum_distributed_noise_level_0 | kernel_pca_4_components | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| noise_framingham_original | quantum_distributed_noise_level_0 | kernel_pca_5_components | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| noise_framingham_original | quantum_distributed_noise_level_0 | kernel_pca_6_components | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| noise_framingham_original | quantum_distributed_noise_level_0 | kernel_pca_7_components | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| noise_framingham_original | quantum_distributed_noise_level_0 | kernel_pca_best_on_test_folds | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| noise_framingham_original | quantum_distributed_noise_level_0 | kernel_svm | 5/5 | 0.6335 | 0.0093 | 0.0136 | 0.5889 |
| noise_framingham_original | quantum_distributed_noise_level_1 | kernel_pca_2_components | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| noise_framingham_original | quantum_distributed_noise_level_1 | kernel_pca_3_components | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| noise_framingham_original | quantum_distributed_noise_level_1 | kernel_pca_4_components | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| noise_framingham_original | quantum_distributed_noise_level_1 | kernel_pca_5_components | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| noise_framingham_original | quantum_distributed_noise_level_1 | kernel_pca_6_components | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| noise_framingham_original | quantum_distributed_noise_level_1 | kernel_pca_7_components | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| noise_framingham_original | quantum_distributed_noise_level_1 | kernel_pca_best_on_test_folds | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| noise_framingham_original | quantum_distributed_noise_level_1 | kernel_svm | 5/5 | 0.6382 | 0.0054 | 0.0151 | 0.5935 |
| noise_framingham_original | quantum_distributed_noise_level_2 | kernel_pca_2_components | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| noise_framingham_original | quantum_distributed_noise_level_2 | kernel_pca_3_components | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| noise_framingham_original | quantum_distributed_noise_level_2 | kernel_pca_4_components | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| noise_framingham_original | quantum_distributed_noise_level_2 | kernel_pca_5_components | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| noise_framingham_original | quantum_distributed_noise_level_2 | kernel_pca_6_components | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| noise_framingham_original | quantum_distributed_noise_level_2 | kernel_pca_7_components | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| noise_framingham_original | quantum_distributed_noise_level_2 | kernel_pca_best_on_test_folds | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| noise_framingham_original | quantum_distributed_noise_level_2 | kernel_svm | 5/5 | 0.6228 | 0.0114 | 0.0149 | 0.5763 |
| noise_parkinsons_original | classical_linear | kernel_pca_2_components | 5/5 | 0.7564 | 0.0195 | 0.0695 | 0.7109 |
| noise_parkinsons_original | classical_linear | kernel_pca_3_components | 5/5 | 0.7373 | 0.0202 | 0.0631 | 0.6936 |
| noise_parkinsons_original | classical_linear | kernel_pca_4_components | 5/5 | 0.7767 | 0.0134 | 0.0916 | 0.7431 |
| noise_parkinsons_original | classical_linear | kernel_pca_5_components | 5/5 | 0.8055 | 0.0059 | 0.0817 | 0.7759 |
| noise_parkinsons_original | classical_linear | kernel_pca_6_components | 5/5 | 0.8034 | 0.0068 | 0.0832 | 0.7713 |
| noise_parkinsons_original | classical_linear | kernel_pca_7_components | 5/5 | 0.8075 | 0.0091 | 0.0799 | 0.7794 |
| noise_parkinsons_original | classical_linear | kernel_pca_best_on_test_folds | 5/5 | 0.8118 | 0.0046 | 0.0767 | 0.7833 |
| noise_parkinsons_original | classical_linear | kernel_svm | 5/5 | 0.8054 | 0.0126 | 0.0625 | 0.7893 |
| noise_parkinsons_original | quantum_distributed_noise_level_0 | kernel_pca_2_components | 5/5 | 0.6276 | 0.0000 | 0.0049 | 0.5000 |
| noise_parkinsons_original | quantum_distributed_noise_level_0 | kernel_pca_3_components | 5/5 | 0.6276 | 0.0000 | 0.0049 | 0.5000 |
| noise_parkinsons_original | quantum_distributed_noise_level_0 | kernel_pca_4_components | 5/5 | 0.6342 | 0.0356 | 0.0363 | 0.5413 |
| noise_parkinsons_original | quantum_distributed_noise_level_0 | kernel_pca_5_components | 5/5 | 0.6691 | 0.0584 | 0.0389 | 0.6000 |
| noise_parkinsons_original | quantum_distributed_noise_level_0 | kernel_pca_6_components | 5/5 | 0.6756 | 0.0564 | 0.0432 | 0.6140 |
| noise_parkinsons_original | quantum_distributed_noise_level_0 | kernel_pca_7_components | 5/5 | 0.7013 | 0.0490 | 0.0559 | 0.6407 |
| noise_parkinsons_original | quantum_distributed_noise_level_0 | kernel_pca_best_on_test_folds | 5/5 | 0.7033 | 0.0447 | 0.0423 | 0.6347 |
| noise_parkinsons_original | quantum_distributed_noise_level_0 | kernel_svm | 5/5 | 0.7725 | 0.0213 | 0.0714 | 0.7405 |
| noise_parkinsons_original | quantum_distributed_noise_level_1 | kernel_pca_2_components | 5/5 | 0.6276 | 0.0000 | 0.0049 | 0.5000 |
| noise_parkinsons_original | quantum_distributed_noise_level_1 | kernel_pca_3_components | 5/5 | 0.6276 | 0.0000 | 0.0049 | 0.5000 |
| noise_parkinsons_original | quantum_distributed_noise_level_1 | kernel_pca_4_components | 5/5 | 0.6268 | 0.0203 | 0.0352 | 0.5260 |
| noise_parkinsons_original | quantum_distributed_noise_level_1 | kernel_pca_5_components | 5/5 | 0.6597 | 0.0524 | 0.0377 | 0.5815 |
| noise_parkinsons_original | quantum_distributed_noise_level_1 | kernel_pca_6_components | 5/5 | 0.6736 | 0.0440 | 0.0372 | 0.6059 |
| noise_parkinsons_original | quantum_distributed_noise_level_1 | kernel_pca_7_components | 5/5 | 0.6864 | 0.0475 | 0.0429 | 0.6219 |
| noise_parkinsons_original | quantum_distributed_noise_level_1 | kernel_pca_best_on_test_folds | 5/5 | 0.6971 | 0.0406 | 0.0477 | 0.6350 |
| noise_parkinsons_original | quantum_distributed_noise_level_1 | kernel_svm | 5/5 | 0.7641 | 0.0252 | 0.0756 | 0.7280 |
| noise_parkinsons_original | quantum_distributed_noise_level_2 | kernel_pca_2_components | 5/5 | 0.6276 | 0.0000 | 0.0049 | 0.5000 |
| noise_parkinsons_original | quantum_distributed_noise_level_2 | kernel_pca_3_components | 5/5 | 0.6276 | 0.0000 | 0.0049 | 0.5000 |
| noise_parkinsons_original | quantum_distributed_noise_level_2 | kernel_pca_4_components | 5/5 | 0.6213 | 0.0086 | 0.0149 | 0.4950 |
| noise_parkinsons_original | quantum_distributed_noise_level_2 | kernel_pca_5_components | 5/5 | 0.6150 | 0.0088 | 0.0242 | 0.4965 |
| noise_parkinsons_original | quantum_distributed_noise_level_2 | kernel_pca_6_components | 5/5 | 0.6140 | 0.0110 | 0.0246 | 0.4956 |
| noise_parkinsons_original | quantum_distributed_noise_level_2 | kernel_pca_7_components | 5/5 | 0.6235 | 0.0182 | 0.0262 | 0.5101 |
| noise_parkinsons_original | quantum_distributed_noise_level_2 | kernel_pca_best_on_test_folds | 5/5 | 0.6319 | 0.0096 | 0.0116 | 0.5127 |
| noise_parkinsons_original | quantum_distributed_noise_level_2 | kernel_svm | 5/5 | 0.7066 | 0.0171 | 0.0520 | 0.6397 |

## `noise_decomposed_initialize`

Fig. 3 diagnostic (designed after the gap was seen): noisy, decomposed state preparation. Sweep folder `sweep_noise_decomposed_initialize_20261021-110925`; COMPLETE SWEEP, 10/10 runs.

| Candidate | Method | Model | Runs | Accuracy | Std over seeds | Mean fold std | Balanced accuracy |
|---|---|---|---:|---:|---:|---:|---:|
| noise_framingham_decomposed_initialize | quantum_distributed_noise_level_1_noisy_initialize | kernel_pca_2_components | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| noise_framingham_decomposed_initialize | quantum_distributed_noise_level_1_noisy_initialize | kernel_pca_3_components | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| noise_framingham_decomposed_initialize | quantum_distributed_noise_level_1_noisy_initialize | kernel_pca_4_components | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| noise_framingham_decomposed_initialize | quantum_distributed_noise_level_1_noisy_initialize | kernel_pca_5_components | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| noise_framingham_decomposed_initialize | quantum_distributed_noise_level_1_noisy_initialize | kernel_pca_6_components | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| noise_framingham_decomposed_initialize | quantum_distributed_noise_level_1_noisy_initialize | kernel_pca_7_components | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| noise_framingham_decomposed_initialize | quantum_distributed_noise_level_1_noisy_initialize | kernel_pca_best_on_test_folds | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| noise_framingham_decomposed_initialize | quantum_distributed_noise_level_1_noisy_initialize | kernel_svm | 5/5 | 0.6349 | 0.0045 | 0.0127 | 0.5901 |
| noise_framingham_decomposed_initialize | quantum_distributed_noise_level_2_noisy_initialize | kernel_pca_2_components | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| noise_framingham_decomposed_initialize | quantum_distributed_noise_level_2_noisy_initialize | kernel_pca_3_components | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| noise_framingham_decomposed_initialize | quantum_distributed_noise_level_2_noisy_initialize | kernel_pca_4_components | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| noise_framingham_decomposed_initialize | quantum_distributed_noise_level_2_noisy_initialize | kernel_pca_5_components | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| noise_framingham_decomposed_initialize | quantum_distributed_noise_level_2_noisy_initialize | kernel_pca_6_components | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| noise_framingham_decomposed_initialize | quantum_distributed_noise_level_2_noisy_initialize | kernel_pca_7_components | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| noise_framingham_decomposed_initialize | quantum_distributed_noise_level_2_noisy_initialize | kernel_pca_best_on_test_folds | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| noise_framingham_decomposed_initialize | quantum_distributed_noise_level_2_noisy_initialize | kernel_svm | 5/5 | 0.6138 | 0.0076 | 0.0140 | 0.5642 |
| noise_parkinsons_decomposed_initialize | quantum_distributed_noise_level_1_noisy_initialize | kernel_pca_2_components | 5/5 | 0.6276 | 0.0000 | 0.0049 | 0.5000 |
| noise_parkinsons_decomposed_initialize | quantum_distributed_noise_level_1_noisy_initialize | kernel_pca_3_components | 5/5 | 0.6276 | 0.0000 | 0.0049 | 0.5000 |
| noise_parkinsons_decomposed_initialize | quantum_distributed_noise_level_1_noisy_initialize | kernel_pca_4_components | 5/5 | 0.6289 | 0.0262 | 0.0348 | 0.5295 |
| noise_parkinsons_decomposed_initialize | quantum_distributed_noise_level_1_noisy_initialize | kernel_pca_5_components | 5/5 | 0.6512 | 0.0567 | 0.0411 | 0.5705 |
| noise_parkinsons_decomposed_initialize | quantum_distributed_noise_level_1_noisy_initialize | kernel_pca_6_components | 5/5 | 0.6640 | 0.0566 | 0.0418 | 0.5942 |
| noise_parkinsons_decomposed_initialize | quantum_distributed_noise_level_1_noisy_initialize | kernel_pca_7_components | 5/5 | 0.6789 | 0.0566 | 0.0488 | 0.6101 |
| noise_parkinsons_decomposed_initialize | quantum_distributed_noise_level_1_noisy_initialize | kernel_pca_best_on_test_folds | 5/5 | 0.6896 | 0.0420 | 0.0480 | 0.6162 |
| noise_parkinsons_decomposed_initialize | quantum_distributed_noise_level_1_noisy_initialize | kernel_svm | 5/5 | 0.7481 | 0.0336 | 0.0755 | 0.7112 |
| noise_parkinsons_decomposed_initialize | quantum_distributed_noise_level_2_noisy_initialize | kernel_pca_2_components | 5/5 | 0.6276 | 0.0000 | 0.0049 | 0.5000 |
| noise_parkinsons_decomposed_initialize | quantum_distributed_noise_level_2_noisy_initialize | kernel_pca_3_components | 5/5 | 0.6276 | 0.0000 | 0.0049 | 0.5000 |
| noise_parkinsons_decomposed_initialize | quantum_distributed_noise_level_2_noisy_initialize | kernel_pca_4_components | 5/5 | 0.6276 | 0.0000 | 0.0049 | 0.5000 |
| noise_parkinsons_decomposed_initialize | quantum_distributed_noise_level_2_noisy_initialize | kernel_pca_5_components | 5/5 | 0.6276 | 0.0000 | 0.0049 | 0.5000 |
| noise_parkinsons_decomposed_initialize | quantum_distributed_noise_level_2_noisy_initialize | kernel_pca_6_components | 5/5 | 0.6276 | 0.0000 | 0.0049 | 0.5000 |
| noise_parkinsons_decomposed_initialize | quantum_distributed_noise_level_2_noisy_initialize | kernel_pca_7_components | 5/5 | 0.6276 | 0.0000 | 0.0049 | 0.5000 |
| noise_parkinsons_decomposed_initialize | quantum_distributed_noise_level_2_noisy_initialize | kernel_pca_best_on_test_folds | 5/5 | 0.6276 | 0.0000 | 0.0049 | 0.5000 |
| noise_parkinsons_decomposed_initialize | quantum_distributed_noise_level_2_noisy_initialize | kernel_svm | 5/5 | 0.6436 | 0.0189 | 0.0220 | 0.5285 |

## `shots_original`

Fig. 4, Digits, 128 to 1,024 shots. Sweep folder `sweep_shots_original_20261020-182145`; COMPLETE SWEEP, 5/5 runs.

| Candidate | Method | Model | Runs | Accuracy | Std over seeds | Mean fold std | Balanced accuracy |
|---|---|---|---:|---:|---:|---:|---:|
| shots_digits_original | classical_linear | kernel_svm | 5/5 | 0.9200 | 0.0000 | 0.0245 | 0.9200 |
| shots_digits_original | exact_squared_overlap | kernel_svm | 5/5 | 0.8200 | 0.0000 | 0.1364 | 0.8200 |
| shots_digits_original | quantum_distributed_1024_shots | kernel_svm | 5/5 | 0.7880 | 0.0084 | 0.1398 | 0.7880 |
| shots_digits_original | quantum_distributed_128_shots | kernel_svm | 5/5 | 0.7600 | 0.0510 | 0.1240 | 0.7600 |
| shots_digits_original | quantum_distributed_256_shots | kernel_svm | 5/5 | 0.7780 | 0.0130 | 0.1148 | 0.7780 |
| shots_digits_original | quantum_distributed_512_shots | kernel_svm | 5/5 | 0.7880 | 0.0259 | 0.1237 | 0.7880 |

## `table1_corrected`

Corrected protocol: paper columns and tuned RBF / polynomial SVMs (Framingham C at most 10). Sweep folder `sweep_table1_corrected_20261021-111645`; COMPLETE SWEEP, 30/30 runs.

| Candidate | Method | Model | Runs | Accuracy | Std over seeds | Mean fold std | Balanced accuracy |
|---|---|---|---:|---:|---:|---:|---:|
| accuracy_framingham_classical | tuned_poly | kernel_pca_nested_cv | 5/5 | 0.8442 | 0.0005 | 0.0081 | 0.5171 |
| accuracy_framingham_classical | tuned_poly | kernel_svm | 5/5 | 0.8362 | 0.0028 | 0.0090 | 0.5354 |
| accuracy_framingham_classical | tuned_rbf | kernel_pca_nested_cv | 5/5 | 0.8401 | 0.0023 | 0.0098 | 0.5090 |
| accuracy_framingham_classical | tuned_rbf | kernel_svm | 5/5 | 0.8395 | 0.0014 | 0.0108 | 0.5121 |
| accuracy_framingham_corrected | classical_linear | kernel_pca_nested_cv | 5/5 | 0.8328 | 0.0017 | 0.0125 | 0.5473 |
| accuracy_framingham_corrected | classical_linear | kernel_svm | 5/5 | 0.7913 | 0.0028 | 0.0165 | 0.6293 |
| accuracy_framingham_corrected | exact_cosine | kernel_pca_nested_cv | 5/5 | 0.8473 | 0.0000 | 0.0008 | 0.5000 |
| accuracy_framingham_corrected | exact_cosine | kernel_svm | 5/5 | 0.8472 | 0.0001 | 0.0008 | 0.5001 |
| accuracy_framingham_corrected | exact_squared_overlap | kernel_pca_nested_cv | 5/5 | 0.8473 | 0.0000 | 0.0008 | 0.5000 |
| accuracy_framingham_corrected | exact_squared_overlap | kernel_svm | 5/5 | 0.8472 | 0.0001 | 0.0008 | 0.5000 |
| accuracy_framingham_corrected | majority_class | kernel_svm | 5/5 | 0.8473 | 0.0000 | 0.0008 | 0.5000 |
| accuracy_framingham_corrected | quantum_centralised | kernel_pca_nested_cv | 5/5 | 0.8473 | 0.0000 | 0.0008 | 0.5000 |
| accuracy_framingham_corrected | quantum_centralised | kernel_svm | 5/5 | 0.8466 | 0.0012 | 0.0014 | 0.5007 |
| accuracy_framingham_corrected | quantum_distributed | kernel_pca_nested_cv | 5/5 | 0.8473 | 0.0000 | 0.0008 | 0.5000 |
| accuracy_framingham_corrected | quantum_distributed | kernel_svm | 5/5 | 0.8463 | 0.0013 | 0.0020 | 0.5007 |
| accuracy_parkinsons_classical | tuned_poly | kernel_pca_nested_cv | 5/5 | 0.8115 | 0.0058 | 0.0538 | 0.7077 |
| accuracy_parkinsons_classical | tuned_poly | kernel_svm | 5/5 | 0.8977 | 0.0079 | 0.0387 | 0.8873 |
| accuracy_parkinsons_classical | tuned_rbf | kernel_pca_nested_cv | 5/5 | 0.8089 | 0.0123 | 0.0626 | 0.6978 |
| accuracy_parkinsons_classical | tuned_rbf | kernel_svm | 5/5 | 0.9271 | 0.0035 | 0.0178 | 0.9128 |
| accuracy_parkinsons_corrected | classical_linear | kernel_pca_nested_cv | 5/5 | 0.8089 | 0.0106 | 0.0688 | 0.7448 |
| accuracy_parkinsons_corrected | classical_linear | kernel_svm | 5/5 | 0.8189 | 0.0095 | 0.0504 | 0.7510 |
| accuracy_parkinsons_corrected | exact_cosine | kernel_pca_nested_cv | 5/5 | 0.7448 | 0.0124 | 0.0384 | 0.6356 |
| accuracy_parkinsons_corrected | exact_cosine | kernel_svm | 5/5 | 0.8010 | 0.0163 | 0.0627 | 0.7888 |
| accuracy_parkinsons_corrected | exact_squared_overlap | kernel_pca_nested_cv | 5/5 | 0.7451 | 0.0282 | 0.0562 | 0.5721 |
| accuracy_parkinsons_corrected | exact_squared_overlap | kernel_svm | 5/5 | 0.7747 | 0.0186 | 0.0622 | 0.6813 |
| accuracy_parkinsons_corrected | majority_class | kernel_svm | 5/5 | 0.7565 | 0.0000 | 0.0148 | 0.5000 |
| accuracy_parkinsons_corrected | quantum_centralised | kernel_pca_nested_cv | 5/5 | 0.7654 | 0.0134 | 0.0598 | 0.5821 |
| accuracy_parkinsons_corrected | quantum_centralised | kernel_svm | 5/5 | 0.7833 | 0.0200 | 0.0455 | 0.6216 |
| accuracy_parkinsons_corrected | quantum_distributed | kernel_pca_nested_cv | 5/5 | 0.7628 | 0.0269 | 0.0587 | 0.5909 |
| accuracy_parkinsons_corrected | quantum_distributed | kernel_svm | 5/5 | 0.7821 | 0.0339 | 0.0339 | 0.6229 |
| accuracy_wine_classical | tuned_poly | kernel_svm | 5/5 | 0.9791 | 0.0000 | 0.0171 | 0.9828 |
| accuracy_wine_classical | tuned_rbf | kernel_svm | 5/5 | 0.9791 | 0.0000 | 0.0171 | 0.9828 |
| accuracy_wine_corrected | classical_linear | kernel_svm | 5/5 | 0.9791 | 0.0000 | 0.0171 | 0.9828 |
| accuracy_wine_corrected | exact_cosine | kernel_svm | 5/5 | 0.9645 | 0.0000 | 0.0226 | 0.9675 |
| accuracy_wine_corrected | exact_squared_overlap | kernel_svm | 5/5 | 0.8872 | 0.0000 | 0.0272 | 0.8917 |
| accuracy_wine_corrected | majority_class | kernel_svm | 5/5 | 0.4012 | 0.0000 | 0.0103 | 0.3333 |
| accuracy_wine_corrected | quantum_centralised | kernel_svm | 5/5 | 0.8914 | 0.0146 | 0.0250 | 0.8963 |
| accuracy_wine_corrected | quantum_distributed | kernel_svm | 5/5 | 0.8899 | 0.0118 | 0.0298 | 0.8954 |

## `merlin_original`

MerLin photonic swap test (adaptation), Table 1 and Fig. 4 settings, paper protocol. Sweep folder `sweep_merlin_original_20261021-115717`; COMPLETE SWEEP, 20/20 runs.

| Candidate | Method | Model | Runs | Accuracy | Std over seeds | Mean fold std | Balanced accuracy |
|---|---|---|---:|---:|---:|---:|---:|
| accuracy_framingham_merlin | merlin_photonic_swap_test | kernel_pca_2_components | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| accuracy_framingham_merlin | merlin_photonic_swap_test | kernel_pca_3_components | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| accuracy_framingham_merlin | merlin_photonic_swap_test | kernel_pca_4_components | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| accuracy_framingham_merlin | merlin_photonic_swap_test | kernel_pca_5_components | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| accuracy_framingham_merlin | merlin_photonic_swap_test | kernel_pca_6_components | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| accuracy_framingham_merlin | merlin_photonic_swap_test | kernel_pca_7_components | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| accuracy_framingham_merlin | merlin_photonic_swap_test | kernel_pca_best_on_test_folds | 5/5 | 0.6249 | 0.0000 | 0.0002 | 0.5000 |
| accuracy_framingham_merlin | merlin_photonic_swap_test | kernel_svm | 5/5 | 0.6322 | 0.0111 | 0.0148 | 0.5882 |
| accuracy_parkinsons_merlin | merlin_photonic_swap_test | kernel_pca_2_components | 5/5 | 0.6276 | 0.0000 | 0.0049 | 0.5000 |
| accuracy_parkinsons_merlin | merlin_photonic_swap_test | kernel_pca_3_components | 5/5 | 0.6276 | 0.0000 | 0.0049 | 0.5000 |
| accuracy_parkinsons_merlin | merlin_photonic_swap_test | kernel_pca_4_components | 5/5 | 0.6331 | 0.0306 | 0.0370 | 0.5398 |
| accuracy_parkinsons_merlin | merlin_photonic_swap_test | kernel_pca_5_components | 5/5 | 0.6628 | 0.0633 | 0.0357 | 0.5903 |
| accuracy_parkinsons_merlin | merlin_photonic_swap_test | kernel_pca_6_components | 5/5 | 0.6778 | 0.0593 | 0.0479 | 0.6168 |
| accuracy_parkinsons_merlin | merlin_photonic_swap_test | kernel_pca_7_components | 5/5 | 0.7011 | 0.0468 | 0.0507 | 0.6423 |
| accuracy_parkinsons_merlin | merlin_photonic_swap_test | kernel_pca_best_on_test_folds | 5/5 | 0.7065 | 0.0447 | 0.0497 | 0.6425 |
| accuracy_parkinsons_merlin | merlin_photonic_swap_test | kernel_svm | 5/5 | 0.7639 | 0.0191 | 0.0849 | 0.7308 |
| accuracy_wine_merlin | merlin_photonic_swap_test | kernel_svm | 5/5 | 0.9085 | 0.0134 | 0.0387 | 0.9141 |
| shots_digits_merlin | merlin_photonic_swap_test_1024_shots | kernel_svm | 5/5 | 0.8200 | 0.0187 | 0.1036 | 0.8200 |
| shots_digits_merlin | merlin_photonic_swap_test_128_shots | kernel_svm | 5/5 | 0.7280 | 0.0545 | 0.1050 | 0.7280 |
| shots_digits_merlin | merlin_photonic_swap_test_256_shots | kernel_svm | 5/5 | 0.7660 | 0.0344 | 0.1176 | 0.7660 |
| shots_digits_merlin | merlin_photonic_swap_test_512_shots | kernel_svm | 5/5 | 0.8000 | 0.0187 | 0.1279 | 0.8000 |

## `fair_comparison`

Fair comparison: tuned RBF / polynomial SVMs on the planned grids and MerLin, corrected protocol. Sweep folder `sweep_fair_comparison_20261021-134946`; COMPLETE SWEEP, 25/25 runs.

| Candidate | Method | Model | Runs | Accuracy | Std over seeds | Mean fold std | Balanced accuracy |
|---|---|---|---:|---:|---:|---:|---:|
| accuracy_framingham_fair_classical | tuned_poly | kernel_pca_nested_cv | 5/5 | 0.8442 | 0.0005 | 0.0081 | 0.5171 |
| accuracy_framingham_fair_classical | tuned_poly | kernel_svm | 5/5 | 0.8362 | 0.0028 | 0.0090 | 0.5354 |
| accuracy_framingham_fair_classical | tuned_rbf | kernel_pca_nested_cv | 5/5 | 0.8406 | 0.0030 | 0.0101 | 0.5107 |
| accuracy_framingham_fair_classical | tuned_rbf | kernel_svm | 5/5 | 0.8395 | 0.0013 | 0.0108 | 0.5119 |
| accuracy_parkinsons_fair_classical | tuned_poly | kernel_pca_nested_cv | 5/5 | 0.8115 | 0.0058 | 0.0538 | 0.7077 |
| accuracy_parkinsons_fair_classical | tuned_poly | kernel_svm | 5/5 | 0.8977 | 0.0079 | 0.0387 | 0.8873 |
| accuracy_parkinsons_fair_classical | tuned_rbf | kernel_pca_nested_cv | 5/5 | 0.8089 | 0.0123 | 0.0626 | 0.6978 |
| accuracy_parkinsons_fair_classical | tuned_rbf | kernel_svm | 5/5 | 0.9271 | 0.0035 | 0.0178 | 0.9128 |
| accuracy_parkinsons_merlin_corrected | merlin_photonic_swap_test | kernel_pca_nested_cv | 5/5 | 0.7745 | 0.0106 | 0.0625 | 0.5841 |
| accuracy_parkinsons_merlin_corrected | merlin_photonic_swap_test | kernel_svm | 5/5 | 0.7654 | 0.0098 | 0.0510 | 0.6174 |
| accuracy_wine_fair_classical | tuned_poly | kernel_svm | 5/5 | 0.9791 | 0.0000 | 0.0171 | 0.9828 |
| accuracy_wine_fair_classical | tuned_rbf | kernel_svm | 5/5 | 0.9791 | 0.0000 | 0.0171 | 0.9828 |
| accuracy_wine_merlin_corrected | merlin_photonic_swap_test | kernel_svm | 5/5 | 0.8969 | 0.0191 | 0.0321 | 0.9013 |
