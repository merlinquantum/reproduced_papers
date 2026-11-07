# MerLin photonic adaptation versus Phase 3 (5 seeds, full data, MerLin simulation (adaptation, not the paper's model))

Final-round test accuracy %, mean +- std over seeds 1 to 5. Photonic model: superposition layer, encode even pixels, trainable mesh, encode odd pixels, trainable mesh; 8 modes, 2 photons, UNBUNCHED, click probabilities into a linear head (130 parameters).

## E1 per setting

| Task | Setting | Paper | Photonic (MerLin) | Hybrid VQTN | Photonic - hybrid | Within 2.0? | Classical MLP | Photonic rounds to 99 % |
|---|---|---:|---:|---:|---:|---|---:|---:|
| 0v9 | centralized | 94.03 | 96.38 +- 0.93 | 94.60 +- 0.17 | +1.78 | yes | 97.93 | 1.8 |
| 0v9 | clients2 | 94.04 | 96.44 +- 0.77 | 94.13 +- 0.80 | +2.31 | no | 97.66 | 3.6 |
| 0v9 | clients4 | 94.01 | 96.34 +- 0.36 | 94.51 +- 0.55 | +1.83 | yes | 97.56 | 2.8 |
| 0v9 | clients8 | 93.95 | 96.28 +- 0.52 | 94.40 +- 0.24 | +1.88 | yes | 97.14 | 5.2 |
| 0v9 | clients16 | 93.87 | 96.25 +- 0.23 | 94.32 +- 0.20 | +1.93 | yes | 96.96 | 9.4 |
| 2v7 | centralized | 96.61 | 96.03 +- 0.87 | 95.61 +- 0.30 | +0.42 | yes | 97.70 | 2.0 |
| 2v7 | clients2 | 96.66 | 96.55 +- 0.66 | 95.37 +- 0.58 | +1.18 | yes | 97.64 | 2.2 |
| 2v7 | clients4 | 96.71 | 96.39 +- 0.38 | 95.45 +- 0.60 | +0.94 | yes | 97.06 | 4.6 |
| 2v7 | clients8 | 96.72 | 96.13 +- 0.23 | 95.51 +- 0.43 | +0.61 | yes | 96.79 | 2.8 |
| 2v7 | clients16 | 96.56 | 95.73 +- 0.33 | 95.48 +- 0.25 | +0.25 | yes | 96.57 | 5.8 |
| 3v6 | centralized | 98.00 | 98.39 +- 0.38 | 94.82 +- 1.27 | +3.58 | no | 99.24 | 1.4 |
| 3v6 | clients2 | 97.99 | 98.55 +- 0.42 | 94.68 +- 1.20 | +3.86 | no | 99.34 | 2.2 |
| 3v6 | clients4 | 97.97 | 98.33 +- 0.66 | 94.91 +- 0.70 | +3.42 | no | 99.08 | 3.4 |
| 3v6 | clients8 | 97.94 | 98.34 +- 0.28 | 95.02 +- 0.32 | +3.32 | no | 99.01 | 2.4 |
| 3v6 | clients16 | 97.97 | 97.31 +- 1.13 | 94.12 +- 0.23 | +3.19 | no | 98.72 | 3.0 |

## Federated versus non-federated

| Task | Clients | Photonic fed - non-fed | Verdict | Hybrid fed - non-fed | Verdict |
|---|---:|---:|---|---:|---|
| 0v9 | 2 | +0.06 | within noise | -0.47 | within noise |
| 0v9 | 4 | -0.04 | within noise | -0.09 | within noise |
| 0v9 | 8 | -0.10 | within noise | -0.20 | within noise |
| 0v9 | 16 | -0.13 | within noise | -0.28 | below |
| 2v7 | 2 | +0.52 | within noise | -0.24 | within noise |
| 2v7 | 4 | +0.36 | within noise | -0.17 | within noise |
| 2v7 | 8 | +0.10 | within noise | -0.10 | within noise |
| 2v7 | 16 | -0.30 | within noise | -0.14 | within noise |
| 3v6 | 2 | +0.15 | within noise | -0.13 | within noise |
| 3v6 | 4 | -0.06 | within noise | +0.09 | within noise |
| 3v6 | 8 | -0.05 | within noise | +0.20 | within noise |
| 3v6 | 16 | -1.09 | within noise | -0.70 | within noise |

## Predeclared criteria

- (a) federated - non-federated within noise in at least 10 of 12 E1 comparisons: **12 of 12, MET**
- (b) within 2.0 points of the hybrid VQTN in every E1 setting: **9 of 15, NOT MET**

## Read-out sensitivity (task 0 vs 9)

| Setting | Photonic, linear head (130 params) | Photonic, fixed LexGrouping (112 params) | Faithful gate VQTN (42 params) | Paper |
|---|---:|---:|---:|---:|
| 0v9 centralized | 96.38 +- 0.93 | 94.73 +- 0.88 | 89.79 +- 2.13 | 94.03 |
| 0v9 clients2 | 96.44 +- 0.77 | 93.48 +- 1.76 | 91.77 +- 1.06 | 94.04 |
| 0v9 clients4 | 96.34 +- 0.36 | 93.68 +- 1.61 | 90.96 +- 2.64 | 94.01 |
| 0v9 clients8 | 96.28 +- 0.52 | 94.09 +- 0.82 | 90.63 +- 1.18 | 93.95 |
| 0v9 clients16 | 96.25 +- 0.23 | 94.35 +- 1.13 | 89.80 +- 1.98 | 93.87 |

## E2 (4-class)

| Setting | Paper | Photonic | Hybrid VQTN | Classical MLP |
|---|---:|---:|---:|---:|
| fourclass_centralized | not reported | 86.53 +- 1.99 | 76.16 +- 2.35 | 90.74 +- 0.60 |
| fourclass_clients4 | 81.80 | 83.81 +- 1.52 | 73.73 +- 0.49 | 89.85 +- 1.04 |
