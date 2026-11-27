# E1 comparison (5 seeds, full MNIST subsets, final-round test accuracy)

Scope label: multi-seed, full data, paper rounds and batch size; **circuit, partition, learning rate and read-out are inferred, and the primary VQTN uses a hybrid eight-<Z> linear head (a documented deviation)**; not paper-accurate. 'vs paper' names a difference only when it exceeds the larger standard deviation.

| Task | Setting | Paper acc % | Hybrid VQTN acc % | Hybrid - paper | vs paper | Classical acc % | Faithful VQTN acc % | Hybrid AUC (paper) | Hybrid F1 (paper) |
|---|---|---:|---:|---:|---|---:|---:|---|---|
| 0v9 | centralized | 94.03 +- 0.06 | 94.60 +- 0.17 | +0.57 | above | 97.93 | 89.79 +- 2.13 | 0.9851 (0.9820) | 0.9460 (0.9399) |
| 0v9 | clients2 | 94.04 +- 0.03 | 94.13 +- 0.80 | +0.09 | within noise | 97.66 | 91.77 +- 1.06 | 0.9859 (0.9816) | 0.9412 (0.9400) |
| 0v9 | clients4 | 94.01 +- 0.02 | 94.51 +- 0.55 | +0.50 | within noise | 97.56 | 90.96 +- 2.64 | 0.9857 (0.9820) | 0.9451 (0.9403) |
| 0v9 | clients8 | 93.95 +- 0.09 | 94.40 +- 0.24 | +0.45 | above | 97.14 | 90.63 +- 1.18 | 0.9848 (0.9822) | 0.9439 (0.9391) |
| 0v9 | clients16 | 93.87 +- 0.14 | 94.32 +- 0.20 | +0.45 | above | 96.96 | 89.80 +- 1.98 | 0.9834 (0.9813) | 0.9432 (0.9386) |
| 2v7 | centralized | 96.61 +- 0.04 | 95.61 +- 0.30 | -1.00 | below | 97.70 | n/a | 0.9919 (0.9905) | 0.9561 (0.9664) |
| 2v7 | clients2 | 96.66 +- 0.02 | 95.37 +- 0.58 | -1.29 | below | 97.64 | n/a | 0.9918 (0.9905) | 0.9537 (0.9669) |
| 2v7 | clients4 | 96.71 +- 0.02 | 95.45 +- 0.60 | -1.26 | below | 97.06 | n/a | 0.9913 (0.9906) | 0.9545 (0.9675) |
| 2v7 | clients8 | 96.72 +- 0.07 | 95.51 +- 0.43 | -1.21 | below | 96.79 | n/a | 0.9918 (0.9906) | 0.9551 (0.9674) |
| 2v7 | clients16 | 96.56 +- 0.09 | 95.48 +- 0.25 | -1.08 | below | 96.57 | n/a | 0.9906 (0.9905) | 0.9547 (0.9659) |
| 3v6 | centralized | 98.00 +- 0.05 | 94.82 +- 1.27 | -3.18 | below | 99.24 | n/a | 0.9934 (0.9972) | 0.9481 (0.9806) |
| 3v6 | clients2 | 97.99 +- 0.08 | 94.68 +- 1.20 | -3.31 | below | 99.34 | n/a | 0.9928 (0.9972) | 0.9468 (0.9804) |
| 3v6 | clients4 | 97.97 +- 0.04 | 94.91 +- 0.70 | -3.06 | below | 99.08 | n/a | 0.9922 (0.9975) | 0.9489 (0.9803) |
| 3v6 | clients8 | 97.94 +- 0.08 | 95.02 +- 0.32 | -2.92 | below | 99.01 | n/a | 0.9912 (0.9969) | 0.9502 (0.9800) |
| 3v6 | clients16 | 97.97 +- 0.11 | 94.12 +- 0.23 | -3.85 | below | 98.72 | n/a | 0.9901 (0.9969) | 0.9409 (0.9803) |

## Federated versus non-federated (claim C1)

| Task | Clients | Hybrid fed - non-fed (points) | Verdict | Classical fed - non-fed (points) | Verdict | Paper fed - non-fed (points) |
|---|---:|---:|---|---:|---|---:|
| 0v9 | 2 | -0.47 | within noise | -0.27 | within noise | +0.01 |
| 0v9 | 4 | -0.09 | within noise | -0.37 | within noise | -0.02 |
| 0v9 | 8 | -0.20 | within noise | -0.78 | below | -0.08 |
| 0v9 | 16 | -0.28 | below | -0.97 | below | -0.16 |
| 2v7 | 2 | -0.24 | within noise | -0.06 | within noise | +0.05 |
| 2v7 | 4 | -0.17 | within noise | -0.64 | below | +0.10 |
| 2v7 | 8 | -0.10 | within noise | -0.91 | below | +0.11 |
| 2v7 | 16 | -0.14 | within noise | -1.13 | below | -0.05 |
| 3v6 | 2 | -0.13 | within noise | +0.10 | within noise | -0.01 |
| 3v6 | 4 | +0.09 | within noise | -0.16 | within noise | -0.03 |
| 3v6 | 8 | +0.20 | within noise | -0.22 | below | -0.06 |
| 3v6 | 16 | -0.70 | within noise | -0.52 | below | -0.03 |

## Rounds to reach 99 % of the final test accuracy (claim C2, communication efficiency)

| Task | Setting | Hybrid rounds to 99 % of final accuracy (mean) | Classical rounds (mean) |
|---|---|---:|---:|
| 0v9 | centralized | 2.4 | 3.2 |
| 0v9 | clients2 | 2.0 | 2.4 |
| 0v9 | clients4 | 3.0 | 2.2 |
| 0v9 | clients8 | 4.4 | 2.0 |
| 0v9 | clients16 | 6.2 | 2.6 |
| 2v7 | centralized | 2.6 | 2.4 |
| 2v7 | clients2 | 2.2 | 1.4 |
| 2v7 | clients4 | 3.8 | 1.2 |
| 2v7 | clients8 | 4.2 | 2.0 |
| 2v7 | clients16 | 5.6 | 2.0 |
| 3v6 | centralized | 4.2 | 1.2 |
| 3v6 | clients2 | 2.8 | 1.2 |
| 3v6 | clients4 | 3.8 | 2.0 |
| 3v6 | clients8 | 6.6 | 2.0 |
| 3v6 | clients16 | 8.0 | 2.8 |

## i.i.d. control

| Setting | non-i.i.d. acc % | i.i.d. acc % | i.i.d. - non-i.i.d. | Verdict |
|---|---:|---:|---:|---|
| 0v9, 2 clients | 94.13 +- 0.80 | 94.74 +- 0.39 | +0.61 | within noise |
| 0v9, 16 clients | 94.32 +- 0.20 | 94.25 +- 0.08 | -0.07 | within noise |
