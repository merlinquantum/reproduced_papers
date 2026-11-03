# E2 comparison (4-class 2, 3, 6, 7; 5 seeds; final-round test accuracy)

Scope label: multi-seed, full data; inferred circuit with a hybrid eight-<Z> head (deviation); the paper gives no non-federated reference, so the non-federated rows are ours. The paper reports no standard deviation for its single 81.80 % value, so 'vs paper' uses only our standard deviation.

| Setting | Paper acc % | Hybrid VQTN acc % | Hybrid - paper | vs paper | Classical acc % |
|---|---:|---:|---:|---|---:|
| fourclass_centralized | not reported | 76.16 +- 2.35 | n/a | n/a | 90.74 +- 0.60 |
| fourclass_clients4 | 81.80 | 73.73 +- 0.49 | -8.07 | below | 89.85 +- 1.04 |

## Per-class metrics, 4 clients (mean over 5 seeds)

| Class | Paper precision | Hybrid precision | Paper recall | Hybrid recall | Paper F1 | Hybrid F1 |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 0.7343 | 0.8298 | 0.7045 | 0.3550 | 0.7191 | 0.4948 |
| 3 | 0.8298 | 0.6585 | 0.7673 | 0.8257 | 0.7973 | 0.7319 |
| 6 | 0.8564 | 0.6659 | 0.8841 | 0.9378 | 0.8701 | 0.7787 |
| 7 | 0.8484 | 0.9026 | 0.9202 | 0.8473 | 0.8829 | 0.8740 |
