# Evaluation protocols: `paper` and `corrected`

This note details the "Protocols" deviation of the README. Line numbers refer to the authors' repository at commit `aed2cf9`.

## The authors' evaluation

The authors evaluate each kernel by stratified 5-fold cross-validation on a fixed 80 % subset of each dataset. Four steps of their pipeline use the test folds before testing:

1. **Oversampling before the split.** `RandomOverSampler(sampling_strategy=0.6)` duplicates random minority-class samples until there are 6 minority samples per 10 majority ones. This helps training on imbalanced data, but the authors apply it to the whole dataset before the split and the cross-validation (`Heart_Dataset_Experiments/tester.py:18-20` and `heart.py:29-31`; `Parkinsons_Dataset_Experiments/tester.py:20-23` and `parkinons.py:27-29`). A sample and its copy can then fall one in a training fold, one in the test fold, and the test folds have 62.5 % majority samples instead of the real share.
2. **Scaler fitted on the whole subset**, test folds included (`Heart_Dataset_Experiments/tester.py:21-22`, `heart.py:32-33`; `Wine_Dataset_Experiments/wine.py:24-25`). Its effect is probably small: the mean and standard deviation over 5 folds differ little from those over 4.
3. **Kernel PCA fitted on all samples** before the cross-validation (`library.py:266-273`, `compute_pca_metrics_and_svm`).
4. **Component count chosen on the test folds.** The code reports the cross-validation accuracy for 2 to 7 components (`Heart_Dataset_Experiments/tester.py:66-67`), and Table 1 gives the best of them: in the authors' Parkinson's log (`Parkinsons_Dataset_Experiments/park_dataset_kernel_1055216.txt`), the distributed kernel PCA value 0.7660 is the 6-component score, the highest of the six.

## Our two protocols

- **`paper`** reproduces these four steps, to compare with Table 1: `lib/data.py:222` (oversampling of the whole dataset), `lib/data.py:245` (scaler on the subset), `lib/evaluation.py::paper_kernel_svm` and `paper_kernel_pca` (kernel PCA on all samples, best component count on the test folds, line 130).
- **`corrected`** fits everything on the training folds only (`lib/evaluation.py::corrected_cv`): scaler (line 226), oversampling of training rows only (lines 234-238), C chosen by an inner 5-fold cross-validation (line 221), kernel PCA and its component count chosen the same way (`_corrected_kernel_pca_fold`). It is needed to tune the classical SVMs fairly. For Framingham the C grid stops at 10, for runtime (`configs/accuracy_framingham_corrected.json`).

## Both protocols lead to the same conclusion

Balanced accuracy of the kernel SVM, mean over seeds 0–4 (`results/sweep_results.md`):

| Dataset | `paper`: classical linear / distributed quantum | `corrected`: classical linear / distributed quantum |
|---|---|---|
| Wine | 0.988 / 0.902 | 0.983 / 0.895 |
| Parkinson's | 0.789 / 0.723 | 0.751 / 0.623 |
| Framingham | 0.637 / 0.595 | 0.629 / 0.501 |

The classical kernel is higher in both protocols. Plain accuracy can mislead on Framingham: under `corrected` the test folds keep the real class balance (about 85 % majority), and the quantum kernel reaches 0.846 accuracy by predicting the majority class almost always, with a balanced accuracy of 0.501. This depends on how C is chosen: the inner cross-validation selects C by accuracy, so on Framingham every method picks the smallest C of the grid (0.01); with the paper's C = 1, the quantum kernel's balanced accuracy is 0.595. Under `paper`, its kernel PCA score (0.6249) is the majority share of the oversampled data.
