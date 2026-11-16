Metric: stratified 5-fold CV accuracy. Ours: mean ± std over seeds 0-4. Matched: |difference| ≤ the paper's fold std, or ≤ 0.02 when the paper gives none.

| Experiment | Setting | Paper | Ours (5 seeds) | Difference | Matched |
|---|---|---:|---:|---:|---|
| Table 1 | Wine, kernel SVM, centralised classical | 0.9860 ± 0.0172 | 0.9860 ± 0.0000 | -0.0000 | yes |
| Table 1 | Wine, kernel SVM, centralised quantum | 0.8805 | 0.9067 ± 0.0190 | +0.0262 | no |
| Table 1 | Wine, kernel SVM, distributed quantum | 0.8874 ± 0.0259 | 0.8957 ± 0.0090 | +0.0083 | yes |
| Table 1 | Parkinson's, kernel SVM, centralised classical | 0.8196 ± 0.0644 | 0.8054 ± 0.0126 | -0.0142 | yes |
| Table 1 | Parkinson's, kernel SVM, centralised quantum | 0.7875 | 0.7725 ± 0.0213 | -0.0150 | yes |
| Table 1 | Parkinson's, kernel SVM, distributed quantum | 0.7983 ± 0.0798 | 0.7557 ± 0.0347 | -0.0426 | yes |
| Table 1 | Parkinson's, kernel PCA, centralised classical | 0.7872 ± 0.0716 | 0.8118 ± 0.0046 | +0.0246 | yes |
| Table 1 | Parkinson's, kernel PCA, centralised quantum | 0.7451 | 0.7033 ± 0.0447 | -0.0418 | no |
| Table 1 | Parkinson's, kernel PCA, distributed quantum | 0.7660 ± 0.0744 | 0.7002 ± 0.0467 | -0.0658 | yes |
| Table 1 | Framingham, kernel SVM, centralised classical | 0.6788 ± 0.0108 | 0.6894 ± 0.0028 | +0.0106 | yes |
| Table 1 | Framingham, kernel SVM, centralised quantum | 0.6308 | 0.6335 ± 0.0093 | +0.0027 | yes |
| Table 1 | Framingham, kernel SVM, distributed quantum | 0.6340 ± 0.0143 | 0.6393 ± 0.0101 | +0.0053 | yes |
| Table 1 | Framingham, kernel PCA, centralised classical | 0.6788 ± 0.0095 | 0.6824 ± 0.0037 | +0.0036 | yes |
| Table 1 | Framingham, kernel PCA, centralised quantum | 0.6249 | 0.6249 ± 0.0000 | +0.0000 | yes |
| Table 1 | Framingham, kernel PCA, distributed quantum | 0.6422 ± 0.0092 | 0.6249 ± 0.0000 | -0.0173 | no |
| Fig. 3 | Framingham, kernel SVM, 0.1 % noise | 0.5807 ± 0.0098 | 0.6382 ± 0.0054 | +0.0575 | no |
| Fig. 3 | Framingham, kernel SVM, 1 % noise | 0.5814 ± 0.0072 | 0.6228 ± 0.0114 | +0.0414 | no |
| Fig. 3 | Parkinson's, kernel SVM, 0.1 % noise | 0.7873 ± 0.0655 | 0.7641 ± 0.0252 | -0.0232 | yes |
| Fig. 3 | Parkinson's, kernel SVM, 1 % noise | 0.6330 ± 0.0087 | 0.7066 ± 0.0171 | +0.0736 | no |
| Fig. 3 | Parkinson's, 6-component kernel PCA, 0.1 % noise | 0.7445 ± 0.0410 | 0.6736 ± 0.0440 | -0.0709 | no |
| Fig. 3 | Parkinson's, 6-component kernel PCA, 1 % noise | 0.6276 ± 0.0049 | 0.6140 ± 0.0110 | -0.0136 | no |
| Fig. 4 | Digits, 128 shots | 0.7100 | 0.7600 ± 0.0510 | +0.0500 | no |
| Fig. 4 | Digits, 256 shots | 0.7100 | 0.7780 ± 0.0130 | +0.0680 | no |
| Fig. 4 | Digits, 512 shots | 0.7400 | 0.7880 ± 0.0259 | +0.0480 | no |
| Fig. 4 | Digits, 1,024 shots | 0.8300 | 0.7880 ± 0.0084 | -0.0420 | no |
| Fig. 4 | Digits, classical linear SVM | 0.8800 | 0.9200 ± 0.0000 | +0.0400 | no |
