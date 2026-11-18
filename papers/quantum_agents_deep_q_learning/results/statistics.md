# Phase 3 statistics - paper-table run set

Generated from: outdir/batches/e1_frozenlake, outdir/batches/cartpole_optimal, outdir/batches/cartpole_optimal_red_full, outdir/batches/cartpole_suboptimal, outdir/batches/e4_pqc_depth, outdir/batches/e4_nn_width

Config names are the paper-table stems; for the code-faithful set each row is the matching `_codefaithful` config.

## Per configuration (episodes-to-solve, 0-based index, unsolved censored at cap)

| Config | Label | Params | Solved | Mean (solved only) | Mean censored [95% CI] | Median censored [95% CI] | Best agent | Paper mean (source) | Paper inside CI |
|---|---|---:|---:|---:|---|---|---:|---|---|
| `e1_frozenlake_l10_original` | paper-accurate | 88 | 8/10 | 1694 | 2355 [1510, 3324] | 1847 [1115, 3574] | 912 | -- | -- |
| `e1_frozenlake_l15_original` | paper-accurate | 128 | 10/10 | 1462 | 1462 [1041, 1948] | 1336 [832, 1910] | 554 | -- | -- |
| `e1_frozenlake_l5_cap200_variant` | variant | 48 | 8/10 | 1463 | 2171 [1264, 3230] | 1626 [832, 3640] | 592 | -- | -- |
| `e1_frozenlake_l5_onlinetarget_variant` | variant | 48 | 10/10 | 1594 | 1594 [1125, 2094] | 1426 [808, 2347] | 577 | -- | -- |
| `e1_frozenlake_l5_original` | paper-accurate | 48 | 9/10 | 1997 | 2297 [1447, 3170] | 2677 [919, 3158] | 634 | -- | -- |
| `e1_frozenlake_l5_ringansatz_variant` | variant | 40 | 4/10 | 3681 | 4472 [3924, 4988] | 5000 [3717, 5000] | 2883 | -- | -- |
| `e1_frozenlake_l5_updateevery1_variant` | variant | 48 | 10/10 | 1323 | 1323 [857, 2016] | 1071 [728, 1470] | 539 | -- | -- |
| `e1_frozenlake_mlp_classical` | added-baseline | 49 | 3/10 | 1104 | 3831 [2662, 5000] | 5000 [1523, 5000] | 893 | -- | -- |
| `e2_cartpole_optimal_full_tanh_variant` | variant | 62 | 10/10 | 639 | 639 [394, 1029] | 446 [338, 701] | 247 | -- | -- |
| `e2_cartpole_optimal_full_unbounded_replay_variant` | variant | 62 | 10/10 | 303 | 303 [253, 362] | 303 [224, 337] | 208 | -- | -- |
| `e2e3_cartpole_optimal_fixed_range_180_original` | paper-accurate | 60 | 10/10 | 843 | 843 [568, 1149] | 741 [463, 1266] | 228 | -- | -- |
| `e2e3_cartpole_optimal_fixed_range_1_original` | paper-accurate | 60 | 0/10 | -- | 5000 [5000, 5000] | 5000 [5000, 5000] | None | -- | -- |
| `e2e3_cartpole_optimal_fixed_range_90_original` | paper-accurate | 60 | 10/10 | 401 | 401 [290, 540] | 334 [246, 560] | 222 | -- | -- |
| `e2e3_cartpole_optimal_full_original` | paper-accurate | 62 | 10/10 | 617 | 617 [415, 866] | 486 [341, 810] | 233 | 320 (figure) | False |
| `e2e3_cartpole_optimal_no_input_weights_original` | paper-accurate | 42 | 5/10 | 1398 | 3199 [1951, 4398] | 4134 [353, 5000] | 220 | -- | -- |
| `e2e3_cartpole_optimal_no_reuploading_original` | paper-accurate | 46 | 10/10 | 659 | 659 [439, 951] | 474 [405, 938] | 367 | -- | -- |
| `e2e3_cartpole_suboptimal_fixed_range_180_original` | paper-accurate | 60 | 9/10 | 2019 | 2317 [1656, 3077] | 2029 [1552, 2921] | 710 | -- | -- |
| `e2e3_cartpole_suboptimal_fixed_range_1_original` | paper-accurate | 60 | 0/10 | -- | 5000 [5000, 5000] | 5000 [5000, 5000] | None | -- | -- |
| `e2e3_cartpole_suboptimal_fixed_range_90_original` | paper-accurate | 60 | 8/10 | 2757 | 3206 [2398, 3978] | 3370 [2113, 4372] | 1041 | -- | -- |
| `e2e3_cartpole_suboptimal_full_original` | paper-accurate | 62 | 9/10 | 2896 | 3106 [2254, 3951] | 3156 [1719, 4439] | 1122 | -- | -- |
| `e2e3_cartpole_suboptimal_no_input_weights_original` | paper-accurate | 42 | 1/10 | 2525 | 4752 [4258, 5000] | 5000 [5000, 5000] | 2525 | -- | -- |
| `e2e3_cartpole_suboptimal_no_reuploading_original` | paper-accurate | 46 | 5/10 | 2777 | 3889 [3099, 4593] | 4366 [2850, 5000] | 1531 | -- | -- |
| `e4_cartpole_mlp_10_10_classical` | paper-accurate | 182 | 10/10 | 990 | 990 [367, 1926] | 385 [300, 1226] | 219 | 1110 (figure) | True |
| `e4_cartpole_mlp_14_14_classical` | added-baseline | 310 | 9/10 | 1021 | 1419 [686, 2367] | 808 [462, 2201] | 222 | -- | -- |
| `e4_cartpole_mlp_15_15_classical` | paper-accurate | 347 | 9/10 | 465 | 919 [375, 1874] | 528 [277, 698] | 230 | 540 (figure) | True |
| `e4_cartpole_mlp_20_20_classical` | paper-accurate | 562 | 10/10 | 1048 | 1048 [390, 1935] | 544 [255, 1934] | 190 | 430 (figure) | True |
| `e4_cartpole_mlp_24_24_classical` | paper-accurate | 770 | 9/10 | 402 | 862 [264, 1847] | 290 [212, 832] | 187 | 415 (figure) | True |
| `e4_cartpole_mlp_30_30_classical` | paper-accurate | 1142 | 10/10 | 385 | 385 [247, 593] | 266 [222, 437] | 191 | 250 (figure) | True |
| `e4_cartpole_mlp_5_5_classical` | added-baseline | 67 | 9/10 | 1498 | 1848 [1117, 2748] | 1172 [785, 2715] | 591 | -- | -- |
| `e4_cartpole_mlp_64_64_classical` | paper-accurate | 4610 | 9/10 | 592 | 1032 [229, 2144] | 230 [199, 1856] | 195 | 345 (figure) | True |
| `e4_cartpole_pqc_l10_original` | paper-accurate | 122 | 10/10 | 576 | 576 [361, 866] | 450 [319, 700] | 207 | 635 (figure) | True |
| `e4_cartpole_pqc_l15_original` | paper-accurate | 182 | 10/10 | 474 | 474 [348, 652] | 394 [304, 541] | 272 | 690 (figure) | False |
| `e4_cartpole_pqc_l20_original` | paper-accurate | 242 | 10/10 | 790 | 790 [467, 1279] | 527 [372, 834] | 330 | 770 (figure) | True |
| `e4_cartpole_pqc_l25_original` | paper-accurate | 302 | 10/10 | 428 | 428 [336, 524] | 420 [274, 563] | 237 | 350 (figure) | True |
| `e4_cartpole_pqc_l30_original` | paper-accurate | 362 | 10/10 | 692 | 692 [548, 859] | 646 [460, 858] | 370 | 940 (figure) | False |

## Pairwise comparisons

| Claim | Comparison | Median A | Median B | Median diff A-B [95% CI] | MWU p | Verdict |
|---|---|---:|---:|---|---:|---|
| C3 | optimal: full vs w/o re-uploading | 486 | 474 | 12 [-321, 371] | 0.734 | no clear separation |
| C3 | optimal: full vs w/o input weights | 486 | 4134 | -3648 [-4640, 133] | 0.0735 | no clear separation |
| C3 | sub-optimal: full vs w/o re-uploading | 3156 | 4366 | -1210 [-2902, 1177] | 0.146 | no clear separation |
| C3 | sub-optimal: full vs w/o input weights | 3156 | 5000 | -1844 [-3281, -561] | 0.00163 | separated |
| C2 | optimal: trainable w_o vs fixed [0,1] | 486 | 5000 | -4514 [-4659, -4190] | 6.39e-05 | separated |
| C2 | optimal: trainable w_o vs fixed [0,90] | 486 | 334 | 152 [-36, 497] | 0.121 | no clear separation |
| C2 | optimal: trainable w_o vs fixed [0,180] | 486 | 741 | -255 [-814, 209] | 0.345 | no clear separation |
| C2 | sub-optimal: trainable w_o vs fixed [0,1] | 3156 | 5000 | -1844 [-3281, -561] | 0.000231 | separated |
| C2 | sub-optimal: trainable w_o vs fixed [0,90] | 3156 | 3370 | -214 [-1888, 1616] | 0.85 | no clear separation |
| C2 | sub-optimal: trainable w_o vs fixed [0,180] | 3156 | 2029 | 1127 [-568, 2594] | 0.29 | no clear separation |
| C5 | best paper PQC (25 layers) vs best paper NN (20,20) | 420 | 544 | -124 [-1515, 234] | 0.623 | no clear separation |
| C5 | iso-parameter: PQC 5 layers (62) vs NN (5,5) (67) | 486 | 1172 | -686 [-2294, -180] | 0.00728 | separated |
| C5 | iso-parameter: PQC 25 layers (302) vs NN (14,14) (310) | 420 | 808 | -388 [-1808, -43] | 0.0312 | separated |
| C1 | Frozen Lake: 5 vs 15 layers (episodes) | 2677 | 1336 | 1342 [-724, 2010] | 0.307 | no clear separation |
| variant | Cart Pole: arctan (paper) vs tanh (code) encoding | 486 | 446 | 40 [-250, 400] | 0.791 | no clear separation |
| variant | Cart Pole: replay 10000 vs unbounded | 486 | 303 | 183 [24, 546] | 0.0101 | separated |
| variant | Frozen Lake: code ansatz vs Fig. 1 ring | 2677 | 5000 | -2323 [-4081, -1036] | 0.00378 | separated |

## Frozen Lake MAE (C1)

Mean MAE44 at step 10000 by depth: {'5': 0.19677954729648064, '10': 0.19201186076170745, '15': 0.18063134702105518} (paper, read from Fig. 3b: {'5': 0.19, '10': 0.17, '15': 0.155}).
Difference 5 - 15 layers: 0.0161 [95% CI 0.0056, 0.0264], MWU p = 0.0113; paper: ~0.05.

## Scans (C5)

- pqc_depth_scan: Kruskal-Wallis H = 7.68, p = 0.175; lowest mean (censored): `e4_cartpole_pqc_l25_original`.
- nn_width_scan: Kruskal-Wallis H = 6.73, p = 0.241; lowest mean (censored): `e4_cartpole_mlp_30_30_classical`.

Best individual agents: PQC `e4_cartpole_pqc_l10_original` at 207 (paper 206, 5 layers); NN `e4_cartpole_mlp_24_24_classical` at 187 (paper 186, (30,30)).
