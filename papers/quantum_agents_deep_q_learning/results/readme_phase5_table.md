| System | Role | Scope | Successful / expected runs | Params | Solved | Median episodes [95% CI] | Mean | Wall-clock, hardware |
|---|---|---|---:|---:|---:|---|---:|---|
| PQC 5 layers, tuned | paper method | paper-method-tuned | 1 / 1 | 62 | 10/10 | 214 [195, 266] | 262 | 1947 s, remote L4 |
| NN (5,5), tuned | added fair baseline | added-baseline-tuned | 1 / 1 | 67 | 10/10 | 298 [252, 1520] | 964 | 199 s, laptop CPU |
| PQC 25 layers, tuned (= paper setting) | paper method | code-faithful | 1 / 1 | 302 | 10/10 | 332 [254, 616] | 435 | 6770 s, remote L4 |
| NN (14,14), tuned | added fair baseline | added-baseline-tuned | 1 / 1 | 310 | 10/10 | 232 [214, 297] | 304 | 53 s, laptop CPU |
| PQC 5 layers, paper setting | paper method (context) | code-faithful | 1 / 1 | 62 | 10/10 | 303 [224, 337] | 303 | 2759 s, remote L4 |
| NN (5,5), untuned | baseline before tuning (context) | added-baseline | 1 / 1 | 67 | 5/10 | 4910 [622, 5000] | 3236 | 263 s, laptop CPU |
| NN (14,14), untuned | baseline before tuning (context) | added-baseline | 1 / 1 | 310 | 10/10 | 387 [240, 2164] | 1168 | 198 s, laptop CPU |
| NN (20,20), paper setting | paper baseline (context) | code-faithful | 1 / 1 | 562 | 10/10 | 250 [199, 900] | 556 | 110 s, laptop CPU |
| NN (30,30), paper setting | paper baseline (context) | code-faithful | 1 / 1 | 1142 | 10/10 | 216 [199, 272] | 234 | 29 s, laptop CPU |
| MerLin 5 layers (adaptation) | photonic adaptation | merlin-adaptation (simulated) | 1 / 1 | 82 | 10/10 | 280 [203, 400] | 304 | 6028 s, remote CPU |

| Pair | Median A | Median B | Median difference A-B [95% CI] | Mann-Whitney p | Verdict | Status |
|---|---:|---:|---|---:|---|---|
| PQC 5 layers, tuned vs NN (5,5), tuned | 214 | 298 | -84 [-1306, -16] | 0.00459 | PQC 5 layers, tuned faster | COMMENSURABLE |
| PQC 25 layers, tuned (= paper setting) vs NN (14,14), tuned | 332 | 232 | 100 [18, 392] | 0.00911 | NN (14,14), tuned faster | COMMENSURABLE |
| MerLin 5 layers (adaptation) vs NN (5,5), tuned | 280 | 298 | -18 [-1168, 95] | 0.241 | no clear separation | PARTIALLY_COMMENSURABLE |
| MerLin 5 layers (adaptation) vs PQC 5 layers, tuned | 280 | 214 | 66 [-22, 174] | 0.13 | no clear separation | PARTIALLY_COMMENSURABLE |
