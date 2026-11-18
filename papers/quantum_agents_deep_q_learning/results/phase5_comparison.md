# Phase 5: parameter-matched comparison (Cart Pole, code-faithful settings)

Episodes to solve: 0-based, unsolved censored at 5000; 10 agents (seeds 0-9) per row; 95% bootstrap intervals. All values locally executed; paper values read from figures or text where given.

| System | Role | Provenance | Scope | Successful / expected runs | Solved | Mean [95% CI] | Median [95% CI] | Best agent | Matching basis | Params | Wall-clock (s), hardware | Paper |
|---|---|---|---|---:|---:|---|---|---:|---|---:|---|---|
| PQC 5 layers, tuned | paper method | local | paper-method-tuned | 1 / 1 | 10/10 | 262 [205, 357] | 214 [195, 266] | 185 | 62 params vs NN (5,5) 67; same 6-setting grid | 62 | 1947, remote L4 | -- |
| NN (5,5), tuned | added fair baseline | local | added-baseline-tuned | 1 / 1 | 10/10 | 964 [375, 1781] | 298 [252, 1520] | 245 | 67 params vs PQC 5 layers 62; same grid | 67 | 199, laptop CPU | -- |
| PQC 25 layers, tuned (= paper setting) | paper method | local | code-faithful | 1 / 1 | 10/10 | 435 [315, 570] | 332 [254, 616] | 247 | 302 params vs NN (14,14) 310; same grid | 302 | 6770, remote L4 | ~350 (Fig. 6a); 500 (text) |
| NN (14,14), tuned | added fair baseline | local | added-baseline-tuned | 1 / 1 | 10/10 | 304 [219, 444] | 232 [214, 297] | 190 | 310 params vs PQC 25 layers 302; same grid | 310 | 53, laptop CPU | -- |
| PQC 5 layers, paper setting | paper method (context) | local | code-faithful | 1 / 1 | 10/10 | 303 [253, 363] | 303 [224, 337] | 208 | paper's Table 2/3 setting | 62 | 2759, remote L4 | ~320 (Fig. 6a) |
| NN (5,5), untuned | baseline before tuning (context) | local | added-baseline | 1 / 1 | 5/10 | 3236 [1908, 4537] | 4910 [622, 5000] | 460 | batch 16 / target 1, not tuned | 67 | 263, laptop CPU | -- |
| NN (14,14), untuned | baseline before tuning (context) | local | added-baseline | 1 / 1 | 10/10 | 1168 [447, 2103] | 387 [240, 2164] | 197 | batch 16 / target 1, not tuned | 310 | 198, laptop CPU | -- |
| NN (20,20), paper setting | paper baseline (context) | local | code-faithful | 1 / 1 | 10/10 | 556 [260, 902] | 250 [199, 900] | 197 | 562 params, not matched | 562 | 110, laptop CPU | ~430 (Fig. 6b); 250 (text) |
| NN (30,30), paper setting | paper baseline (context) | local | code-faithful | 1 / 1 | 10/10 | 234 [207, 266] | 216 [199, 272] | 191 | 1142 params, not matched | 1142 | 29, laptop CPU | ~250 (Fig. 6b) |
| MerLin 5 layers (adaptation) | photonic adaptation | local | merlin-adaptation (simulated) | 1 / 1 | 10/10 | 304 [244, 370] | 280 [203, 400] | 194 | 82 params, untuned: PARTIALLY_COMMENSURABLE | 82 | 6028, remote CPU | -- |

| Pair | Median A | Median B | Median difference A-B [95% CI] | MWU p | Verdict | Status |
|---|---:|---:|---|---:|---|---|
| PQC 5 layers, tuned vs NN (5,5), tuned | 214 | 298 | -84 [-1306, -16] | 0.00459 | PQC 5 layers, tuned faster | COMMENSURABLE |
| PQC 25 layers, tuned (= paper setting) vs NN (14,14), tuned | 332 | 232 | 100 [18, 392] | 0.00911 | NN (14,14), tuned faster | COMMENSURABLE |
| MerLin 5 layers (adaptation) vs NN (5,5), tuned | 280 | 298 | -18 [-1168, 95] | 0.241 | no clear separation | PARTIALLY_COMMENSURABLE |
| MerLin 5 layers (adaptation) vs PQC 5 layers, tuned | 280 | 214 | 66 [-22, 174] | 0.13 | no clear separation | PARTIALLY_COMMENSURABLE |
