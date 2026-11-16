Corrected protocol (oversampling and selection inside training folds), stratified 5-fold CV, mean ± std over seeds 0-4. Accuracy is the primary metric; balanced accuracy is needed for Framingham (majority rate 0.847). Quantum values come from an exact simulation, MerLin values from MerLin simulation; nothing ran on hardware. All rows use the same samples, folds, preprocessing, C grid and inner cross-validation.

| Dataset | Model | System | Role | Runs | Accuracy | Balanced accuracy | Resources per outer fold | Median s per seed |
|---|---|---|---|---|---:|---:|---|---:|
| wine | kernel SVM | Distributed quantum kernel (swap test, 1,024 shots) | paper method | 5/5 | 0.8899 ± 0.0118 | 0.8954 ± 0.0098 | 9,747 kernel entries × 1,024 shots per outer fold; 17 qubits | 0.1 |
| wine | kernel SVM | Centralised quantum kernel (swap test, 1,024 shots) | paper method | 5/5 | 0.8914 ± 0.0146 | 0.8963 ± 0.0143 | 9,747 kernel entries × 1,024 shots per outer fold; 9 qubits | 0.1 |
| wine | kernel SVM | MerLin photonic swap test (1,024 shots) | adaptation | 5/5 | 0.8969 ± 0.0191 | 0.9013 ± 0.0179 | 9,747 kernel entries × 1,024 shots per outer fold; 4 photons, 12 modes | 3.1 |
| wine | kernel SVM | Exact squared-overlap kernel (infinite shots) | reference | 5/5 | 0.8872 ± 0.0000 | 0.8917 ± 0.0000 | 9,747 closed-form kernel entries per outer fold | 0.1 |
| wine | kernel SVM | Exact cosine kernel | reference | 5/5 | 0.9645 ± 0.0000 | 0.9675 ± 0.0000 | 9,747 closed-form kernel entries per outer fold | 0.1 |
| wine | kernel SVM | Linear SVM (paper's classical column), C tuned | paper baseline | 5/5 | 0.9791 ± 0.0000 | 0.9828 ± 0.0000 | 9,747 closed-form kernel entries per outer fold | 0.1 |
| wine | kernel SVM | Tuned RBF SVM (fair baseline) | fair baseline | 5/5 | 0.9791 ± 0.0000 | 0.9828 ± 0.0000 | 9,747 closed-form kernel entries per outer fold | 0.5 |
| wine | kernel SVM | Tuned polynomial SVM (fair baseline) | fair baseline | 5/5 | 0.9791 ± 0.0000 | 0.9828 ± 0.0000 | 9,747 closed-form kernel entries per outer fold | 0.4 |
| wine | kernel SVM | Majority class | floor | 5/5 | 0.4012 ± 0.0000 | 0.3333 ± 0.0000 | none | 0.0 |
| parkinsons | kernel SVM | Distributed quantum kernel (swap test, 1,024 shots) | paper method | 5/5 | 0.7821 ± 0.0339 | 0.6229 ± 0.0339 | 11,750 kernel entries × 1,024 shots per outer fold; 21 qubits | 1.0 |
| parkinsons | kernel SVM | Centralised quantum kernel (swap test, 1,024 shots) | paper method | 5/5 | 0.7833 ± 0.0200 | 0.6216 ± 0.0375 | 11,750 kernel entries × 1,024 shots per outer fold; 11 qubits | 1.0 |
| parkinsons | kernel SVM | MerLin photonic swap test (1,024 shots) | adaptation | 5/5 | 0.7654 ± 0.0098 | 0.6174 ± 0.0311 | 11,750 kernel entries × 1,024 shots per outer fold; 4 photons, 16 modes | 25.5 |
| parkinsons | kernel SVM | Exact squared-overlap kernel (infinite shots) | reference | 5/5 | 0.7747 ± 0.0186 | 0.6813 ± 0.0201 | 11,750 closed-form kernel entries per outer fold | 1.0 |
| parkinsons | kernel SVM | Exact cosine kernel | reference | 5/5 | 0.8010 ± 0.0163 | 0.7888 ± 0.0166 | 11,750 closed-form kernel entries per outer fold | 1.0 |
| parkinsons | kernel SVM | Linear SVM (paper's classical column), C tuned | paper baseline | 5/5 | 0.8189 ± 0.0095 | 0.7510 ± 0.0162 | 11,750 closed-form kernel entries per outer fold | 2.2 |
| parkinsons | kernel SVM | Tuned RBF SVM (fair baseline) | fair baseline | 5/5 | 0.9271 ± 0.0035 | 0.9128 ± 0.0078 | 11,750 closed-form kernel entries per outer fold | 1.3 |
| parkinsons | kernel SVM | Tuned polynomial SVM (fair baseline) | fair baseline | 5/5 | 0.8977 ± 0.0079 | 0.8873 ± 0.0069 | 11,750 closed-form kernel entries per outer fold | 1.3 |
| parkinsons | kernel SVM | Majority class | floor | 5/5 | 0.7565 ± 0.0000 | 0.5000 ± 0.0000 | none | 0.0 |
| parkinsons | kernel PCA | Distributed quantum kernel (swap test, 1,024 shots) | paper method | 5/5 | 0.7628 ± 0.0269 | 0.5909 ± 0.0388 | 11,750 kernel entries × 1,024 shots per outer fold; 21 qubits | 1.0 |
| parkinsons | kernel PCA | Centralised quantum kernel (swap test, 1,024 shots) | paper method | 5/5 | 0.7654 ± 0.0134 | 0.5821 ± 0.0212 | 11,750 kernel entries × 1,024 shots per outer fold; 11 qubits | 1.0 |
| parkinsons | kernel PCA | MerLin photonic swap test (1,024 shots) | adaptation | 5/5 | 0.7745 ± 0.0106 | 0.5841 ± 0.0169 | 11,750 kernel entries × 1,024 shots per outer fold; 4 photons, 16 modes | 25.5 |
| parkinsons | kernel PCA | Exact squared-overlap kernel (infinite shots) | reference | 5/5 | 0.7451 ± 0.0282 | 0.5721 ± 0.0376 | 11,750 closed-form kernel entries per outer fold | 1.0 |
| parkinsons | kernel PCA | Exact cosine kernel | reference | 5/5 | 0.7448 ± 0.0124 | 0.6356 ± 0.0246 | 11,750 closed-form kernel entries per outer fold | 1.0 |
| parkinsons | kernel PCA | Linear SVM (paper's classical column), C tuned | paper baseline | 5/5 | 0.8089 ± 0.0106 | 0.7448 ± 0.0124 | 11,750 closed-form kernel entries per outer fold | 2.2 |
| parkinsons | kernel PCA | Tuned RBF SVM (fair baseline) | fair baseline | 5/5 | 0.8089 ± 0.0123 | 0.6978 ± 0.0258 | 11,750 closed-form kernel entries per outer fold | 1.3 |
| parkinsons | kernel PCA | Tuned polynomial SVM (fair baseline) | fair baseline | 5/5 | 0.8115 ± 0.0058 | 0.7077 ± 0.0145 | 11,750 closed-form kernel entries per outer fold | 1.3 |
| parkinsons | kernel PCA | Majority class | floor | 5/5 | 0.7565 ± 0.0000 | 0.5000 ± 0.0000 | none | 0.0 |
| framingham | kernel SVM | Distributed quantum kernel (swap test, 1,024 shots) | paper method | 5/5 | 0.8463 ± 0.0013 | 0.5007 ± 0.0010 | 4,318,200 kernel entries × 1,024 shots per outer fold; 17 qubits | 54.7 |
| framingham | kernel SVM | Centralised quantum kernel (swap test, 1,024 shots) | paper method | 5/5 | 0.8466 ± 0.0012 | 0.5007 ± 0.0010 | 4,318,200 kernel entries × 1,024 shots per outer fold; 9 qubits | 54.4 |
| framingham | kernel SVM | MerLin photonic swap test (1,024 shots) | adaptation | not run: MerLin corrected protocol on Framingham omitted (plan); kernel equals the gate-model kernel to 2.6e-8 | — | — | — | — |
| framingham | kernel SVM | Exact squared-overlap kernel (infinite shots) | reference | 5/5 | 0.8472 ± 0.0001 | 0.5000 ± 0.0001 | 4,318,200 closed-form kernel entries per outer fold | 56.0 |
| framingham | kernel SVM | Exact cosine kernel | reference | 5/5 | 0.8472 ± 0.0001 | 0.5001 ± 0.0004 | 4,318,200 closed-form kernel entries per outer fold | 63.4 |
| framingham | kernel SVM | Linear SVM (paper's classical column), C tuned | paper baseline | 5/5 | 0.7913 ± 0.0028 | 0.6293 ± 0.0011 | 4,318,200 closed-form kernel entries per outer fold | 135.6 |
| framingham | kernel SVM | Tuned RBF SVM (fair baseline) | fair baseline | 5/5 | 0.8395 ± 0.0013 | 0.5119 ± 0.0043 | 4,318,200 closed-form kernel entries per outer fold | 94.6 |
| framingham | kernel SVM | Tuned polynomial SVM (fair baseline) | fair baseline | 5/5 | 0.8362 ± 0.0028 | 0.5354 ± 0.0040 | 4,318,200 closed-form kernel entries per outer fold | 132.5 |
| framingham | kernel SVM | Majority class | floor | 5/5 | 0.8473 ± 0.0000 | 0.5000 ± 0.0000 | none | 0.0 |
| framingham | kernel PCA | Distributed quantum kernel (swap test, 1,024 shots) | paper method | 5/5 | 0.8473 ± 0.0000 | 0.5000 ± 0.0000 | 4,318,200 kernel entries × 1,024 shots per outer fold; 17 qubits | 54.7 |
| framingham | kernel PCA | Centralised quantum kernel (swap test, 1,024 shots) | paper method | 5/5 | 0.8473 ± 0.0000 | 0.5000 ± 0.0000 | 4,318,200 kernel entries × 1,024 shots per outer fold; 9 qubits | 54.4 |
| framingham | kernel PCA | MerLin photonic swap test (1,024 shots) | adaptation | not run: MerLin corrected protocol on Framingham omitted (plan); kernel equals the gate-model kernel to 2.6e-8 | — | — | — | — |
| framingham | kernel PCA | Exact squared-overlap kernel (infinite shots) | reference | 5/5 | 0.8473 ± 0.0000 | 0.5000 ± 0.0000 | 4,318,200 closed-form kernel entries per outer fold | 56.0 |
| framingham | kernel PCA | Exact cosine kernel | reference | 5/5 | 0.8473 ± 0.0000 | 0.5000 ± 0.0000 | 4,318,200 closed-form kernel entries per outer fold | 63.4 |
| framingham | kernel PCA | Linear SVM (paper's classical column), C tuned | paper baseline | 5/5 | 0.8328 ± 0.0017 | 0.5473 ± 0.0062 | 4,318,200 closed-form kernel entries per outer fold | 135.6 |
| framingham | kernel PCA | Tuned RBF SVM (fair baseline) | fair baseline | 5/5 | 0.8406 ± 0.0030 | 0.5107 ± 0.0044 | 4,318,200 closed-form kernel entries per outer fold | 94.6 |
| framingham | kernel PCA | Tuned polynomial SVM (fair baseline) | fair baseline | 5/5 | 0.8442 ± 0.0005 | 0.5171 ± 0.0023 | 4,318,200 closed-form kernel entries per outer fold | 132.5 |
| framingham | kernel PCA | Majority class | floor | 5/5 | 0.8473 ± 0.0000 | 0.5000 ± 0.0000 | none | 0.0 |

Pairwise decisions (rule: equivalent when |delta| <= 2 sigma_pooled (seed std, ddof = 1)):

| Dataset | Model | First | Second | Metric | Difference | 2 σ pooled | Verdict |
|---|---|---|---|---|---:|---:|---|
| wine | kernel SVM | quantum_distributed | quantum_centralised | accuracy | -0.0015 | 0.0265 | equivalent |
| wine | kernel SVM | quantum_distributed | quantum_centralised | balanced_accuracy | -0.0009 | 0.0246 | equivalent |
| wine | kernel SVM | quantum_distributed | tuned_rbf | accuracy | -0.0892 | 0.0167 | second higher |
| wine | kernel SVM | quantum_distributed | tuned_rbf | balanced_accuracy | -0.0874 | 0.0139 | second higher |
| wine | kernel SVM | quantum_distributed | tuned_poly | accuracy | -0.0892 | 0.0167 | second higher |
| wine | kernel SVM | quantum_distributed | tuned_poly | balanced_accuracy | -0.0874 | 0.0139 | second higher |
| wine | kernel SVM | quantum_distributed | classical_linear | accuracy | -0.0892 | 0.0167 | second higher |
| wine | kernel SVM | quantum_distributed | classical_linear | balanced_accuracy | -0.0874 | 0.0139 | second higher |
| wine | kernel SVM | quantum_distributed | exact_cosine | accuracy | -0.0746 | 0.0167 | second higher |
| wine | kernel SVM | quantum_distributed | exact_cosine | balanced_accuracy | -0.0721 | 0.0139 | second higher |
| wine | kernel SVM | quantum_distributed | majority_class | accuracy | +0.4887 | 0.0167 | first higher |
| wine | kernel SVM | quantum_distributed | majority_class | balanced_accuracy | +0.5621 | 0.0139 | first higher |
| wine | kernel SVM | merlin | tuned_rbf | accuracy | -0.0821 | 0.0270 | second higher |
| wine | kernel SVM | merlin | tuned_rbf | balanced_accuracy | -0.0816 | 0.0253 | second higher |
| wine | kernel SVM | merlin | quantum_distributed | accuracy | +0.0070 | 0.0318 | equivalent |
| wine | kernel SVM | merlin | quantum_distributed | balanced_accuracy | +0.0058 | 0.0288 | equivalent |
| parkinsons | kernel SVM | quantum_distributed | quantum_centralised | accuracy | -0.0013 | 0.0557 | equivalent |
| parkinsons | kernel SVM | quantum_distributed | quantum_centralised | balanced_accuracy | +0.0013 | 0.0716 | equivalent |
| parkinsons | kernel SVM | quantum_distributed | tuned_rbf | accuracy | -0.1450 | 0.0483 | second higher |
| parkinsons | kernel SVM | quantum_distributed | tuned_rbf | balanced_accuracy | -0.2899 | 0.0492 | second higher |
| parkinsons | kernel SVM | quantum_distributed | tuned_poly | accuracy | -0.1157 | 0.0493 | second higher |
| parkinsons | kernel SVM | quantum_distributed | tuned_poly | balanced_accuracy | -0.2645 | 0.0490 | second higher |
| parkinsons | kernel SVM | quantum_distributed | classical_linear | accuracy | -0.0369 | 0.0499 | equivalent |
| parkinsons | kernel SVM | quantum_distributed | classical_linear | balanced_accuracy | -0.1281 | 0.0532 | second higher |
| parkinsons | kernel SVM | quantum_distributed | exact_cosine | accuracy | -0.0190 | 0.0533 | equivalent |
| parkinsons | kernel SVM | quantum_distributed | exact_cosine | balanced_accuracy | -0.1659 | 0.0534 | second higher |
| parkinsons | kernel SVM | quantum_distributed | majority_class | accuracy | +0.0256 | 0.0480 | equivalent |
| parkinsons | kernel SVM | quantum_distributed | majority_class | balanced_accuracy | +0.1229 | 0.0480 | first higher |
| parkinsons | kernel SVM | merlin | tuned_rbf | accuracy | -0.1616 | 0.0147 | second higher |
| parkinsons | kernel SVM | merlin | tuned_rbf | balanced_accuracy | -0.2954 | 0.0453 | second higher |
| parkinsons | kernel SVM | merlin | quantum_distributed | accuracy | -0.0166 | 0.0500 | equivalent |
| parkinsons | kernel SVM | merlin | quantum_distributed | balanced_accuracy | -0.0055 | 0.0651 | equivalent |
| parkinsons | kernel PCA | quantum_distributed | quantum_centralised | accuracy | -0.0027 | 0.0425 | equivalent |
| parkinsons | kernel PCA | quantum_distributed | quantum_centralised | balanced_accuracy | +0.0088 | 0.0625 | equivalent |
| parkinsons | kernel PCA | quantum_distributed | tuned_rbf | accuracy | -0.0461 | 0.0419 | second higher |
| parkinsons | kernel PCA | quantum_distributed | tuned_rbf | balanced_accuracy | -0.1069 | 0.0659 | second higher |
| parkinsons | kernel PCA | quantum_distributed | tuned_poly | accuracy | -0.0487 | 0.0389 | second higher |
| parkinsons | kernel PCA | quantum_distributed | tuned_poly | balanced_accuracy | -0.1168 | 0.0586 | second higher |
| parkinsons | kernel PCA | quantum_distributed | classical_linear | accuracy | -0.0461 | 0.0409 | second higher |
| parkinsons | kernel PCA | quantum_distributed | classical_linear | balanced_accuracy | -0.1539 | 0.0576 | second higher |
| parkinsons | kernel PCA | quantum_distributed | exact_cosine | accuracy | +0.0179 | 0.0419 | equivalent |
| parkinsons | kernel PCA | quantum_distributed | exact_cosine | balanced_accuracy | -0.0447 | 0.0650 | equivalent |
| parkinsons | kernel PCA | quantum_distributed | majority_class | accuracy | +0.0063 | 0.0380 | equivalent |
| parkinsons | kernel PCA | quantum_distributed | majority_class | balanced_accuracy | +0.0909 | 0.0549 | first higher |
| parkinsons | kernel PCA | merlin | tuned_rbf | accuracy | -0.0344 | 0.0230 | second higher |
| parkinsons | kernel PCA | merlin | tuned_rbf | balanced_accuracy | -0.1137 | 0.0436 | second higher |
| parkinsons | kernel PCA | merlin | quantum_distributed | accuracy | +0.0117 | 0.0409 | equivalent |
| parkinsons | kernel PCA | merlin | quantum_distributed | balanced_accuracy | -0.0069 | 0.0599 | equivalent |
| framingham | kernel SVM | quantum_distributed | quantum_centralised | accuracy | -0.0003 | 0.0024 | equivalent |
| framingham | kernel SVM | quantum_distributed | quantum_centralised | balanced_accuracy | +0.0000 | 0.0020 | equivalent |
| framingham | kernel SVM | quantum_distributed | tuned_rbf | accuracy | +0.0069 | 0.0026 | first higher |
| framingham | kernel SVM | quantum_distributed | tuned_rbf | balanced_accuracy | -0.0112 | 0.0062 | second higher |
| framingham | kernel SVM | quantum_distributed | tuned_poly | accuracy | +0.0101 | 0.0043 | first higher |
| framingham | kernel SVM | quantum_distributed | tuned_poly | balanced_accuracy | -0.0347 | 0.0058 | second higher |
| framingham | kernel SVM | quantum_distributed | classical_linear | accuracy | +0.0551 | 0.0043 | first higher |
| framingham | kernel SVM | quantum_distributed | classical_linear | balanced_accuracy | -0.1286 | 0.0021 | second higher |
| framingham | kernel SVM | quantum_distributed | exact_cosine | accuracy | -0.0009 | 0.0018 | equivalent |
| framingham | kernel SVM | quantum_distributed | exact_cosine | balanced_accuracy | +0.0006 | 0.0016 | equivalent |
| framingham | kernel SVM | quantum_distributed | majority_class | accuracy | -0.0009 | 0.0018 | equivalent |
| framingham | kernel SVM | quantum_distributed | majority_class | balanced_accuracy | +0.0007 | 0.0014 | equivalent |
| framingham | kernel PCA | quantum_distributed | quantum_centralised | accuracy | +0.0000 | 0.0000 | equivalent |
| framingham | kernel PCA | quantum_distributed | quantum_centralised | balanced_accuracy | +0.0000 | 0.0000 | equivalent |
| framingham | kernel PCA | quantum_distributed | tuned_rbf | accuracy | +0.0067 | 0.0043 | first higher |
| framingham | kernel PCA | quantum_distributed | tuned_rbf | balanced_accuracy | -0.0107 | 0.0062 | second higher |
| framingham | kernel PCA | quantum_distributed | tuned_poly | accuracy | +0.0031 | 0.0008 | first higher |
| framingham | kernel PCA | quantum_distributed | tuned_poly | balanced_accuracy | -0.0171 | 0.0032 | second higher |
| framingham | kernel PCA | quantum_distributed | classical_linear | accuracy | +0.0145 | 0.0023 | first higher |
| framingham | kernel PCA | quantum_distributed | classical_linear | balanced_accuracy | -0.0473 | 0.0088 | second higher |
| framingham | kernel PCA | quantum_distributed | exact_cosine | accuracy | +0.0000 | 0.0000 | equivalent |
| framingham | kernel PCA | quantum_distributed | exact_cosine | balanced_accuracy | +0.0000 | 0.0000 | equivalent |
| framingham | kernel PCA | quantum_distributed | majority_class | accuracy | +0.0000 | 0.0000 | equivalent |
| framingham | kernel PCA | quantum_distributed | majority_class | balanced_accuracy | +0.0000 | 0.0000 | equivalent |

Paper protocol (the paper's own comparison, whose protocol leaks; see docs/protocols.md):

| Dataset | Model | Linear SVM | Distributed quantum | MerLin | Quantum − linear | 2 σ pooled | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| wine | kernel SVM | 0.9860 ± 0.0000 | 0.8957 ± 0.0090 | 0.9085 ± 0.0134 | -0.0902 | 0.0128 | linear SVM higher |
| parkinsons | kernel SVM | 0.8054 ± 0.0126 | 0.7557 ± 0.0347 | 0.7639 ± 0.0191 | -0.0498 | 0.0522 | equivalent |
| parkinsons | kernel PCA | 0.8118 ± 0.0046 | 0.7002 ± 0.0467 | 0.7065 ± 0.0447 | -0.1116 | 0.0663 | linear SVM higher |
| framingham | kernel SVM | 0.6894 ± 0.0028 | 0.6393 ± 0.0101 | 0.6322 ± 0.0111 | -0.0502 | 0.0148 | linear SVM higher |
| framingham | kernel PCA | 0.6824 ± 0.0037 | 0.6249 ± 0.0000 | 0.6249 ± 0.0000 | -0.0575 | 0.0053 | linear SVM higher |
