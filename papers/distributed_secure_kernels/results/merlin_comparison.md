Metric: stratified 5-fold CV accuracy, mean ± std over seeds 0-4, paper protocol, 1,024 shots unless stated. Gate model: our distributed quantum column (noiseless). MerLin: photonic swap test (adaptation). Preserved: |difference| ≤ 2 σ_pooled.

| Setting | Gate model | MerLin adaptation | Difference | 2 σ pooled | Preserved |
|---|---:|---:|---:|---:|---|
| Wine, kernel SVM | 0.8957 ± 0.0090 | 0.9085 ± 0.0134 | +0.0128 | 0.0228 | yes |
| Parkinson's, kernel SVM | 0.7557 ± 0.0347 | 0.7639 ± 0.0191 | +0.0083 | 0.0560 | yes |
| Parkinson's, kernel PCA | 0.7002 ± 0.0467 | 0.7065 ± 0.0447 | +0.0063 | 0.0914 | yes |
| Framingham, kernel SVM | 0.6393 ± 0.0101 | 0.6322 ± 0.0111 | -0.0070 | 0.0212 | yes |
| Framingham, kernel PCA | 0.6249 ± 0.0000 | 0.6249 ± 0.0000 | +0.0000 | 0.0000 | yes |
| Digits, 128 shots | 0.7600 ± 0.0510 | 0.7280 ± 0.0545 | -0.0320 | 0.1055 | yes |
| Digits, 256 shots | 0.7780 ± 0.0130 | 0.7660 ± 0.0344 | -0.0120 | 0.0520 | yes |
| Digits, 512 shots | 0.7880 ± 0.0259 | 0.8000 ± 0.0187 | +0.0120 | 0.0452 | yes |
| Digits, 1024 shots | 0.7880 ± 0.0084 | 0.8200 ± 0.0187 | +0.0320 | 0.0290 | no |
