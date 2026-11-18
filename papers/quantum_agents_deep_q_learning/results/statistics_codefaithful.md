# Phase 3 statistics - code-faithful run set

Generated from: outdir/batches/cf_frozenlake, outdir/batches/cf_cartpole_optimal, outdir/batches/cf_cartpole_suboptimal, outdir/batches/cf_e4_pqc_depth, outdir/batches/cf_e4_nn_width

Config names are the paper-table stems; for the code-faithful set each row is the matching `_codefaithful` config.

## Per configuration (episodes-to-solve, 0-based index, unsolved censored at cap)

| Config | Label | Params | Solved | Mean (solved only) | Mean censored [95% CI] | Median censored [95% CI] | Best agent | Paper mean (source) | Paper inside CI |
|---|---|---:|---:|---:|---|---|---:|---|---|
| `e1_frozenlake_l10_original` | code-faithful | 88 | 10/10 | 821 | 821 [623, 1067] | 654 [577, 1066] | 522 | -- | -- |
| `e1_frozenlake_l15_original` | code-faithful | 128 | 10/10 | 749 | 749 [638, 885] | 688 [602, 909] | 546 | -- | -- |
| `e1_frozenlake_l5_original` | code-faithful | 48 | 10/10 | 1323 | 1323 [857, 1982] | 1071 [728, 1464] | 539 | -- | -- |
| `e1_frozenlake_mlp_classical` | code-faithful | 49 | 5/10 | 1240 | 3120 [1882, 4291] | 3670 [1007, 5000] | 567 | -- | -- |
| `e2e3_cartpole_optimal_fixed_range_180_original` | code-faithful | 60 | 10/10 | 350 | 350 [279, 445] | 291 [255, 417] | 226 | -- | -- |
| `e2e3_cartpole_optimal_fixed_range_1_original` | code-faithful | 60 | 0/10 | -- | 5000 [5000, 5000] | 5000 [5000, 5000] | None | -- | -- |
| `e2e3_cartpole_optimal_fixed_range_90_original` | code-faithful | 60 | 10/10 | 279 | 279 [239, 334] | 258 [226, 304] | 210 | -- | -- |
| `e2e3_cartpole_optimal_full_original` | code-faithful | 62 | 10/10 | 303 | 303 [254, 364] | 303 [224, 337] | 208 | 320 (figure) | True |
| `e2e3_cartpole_optimal_no_input_weights_original` | code-faithful | 42 | 8/10 | 525 | 1420 [450, 2714] | 530 [296, 2798] | 239 | -- | -- |
| `e2e3_cartpole_optimal_no_reuploading_original` | code-faithful | 46 | 10/10 | 394 | 394 [308, 523] | 344 [286, 412] | 250 | -- | -- |
| `e2e3_cartpole_suboptimal_fixed_range_180_original` | code-faithful | 60 | 9/10 | 2429 | 2686 [1803, 3645] | 2352 [1438, 4410] | 878 | -- | -- |
| `e2e3_cartpole_suboptimal_fixed_range_1_original` | code-faithful | 60 | 0/10 | -- | 5000 [5000, 5000] | 5000 [5000, 5000] | None | -- | -- |
| `e2e3_cartpole_suboptimal_fixed_range_90_original` | code-faithful | 60 | 10/10 | 1649 | 1649 [1249, 2054] | 1696 [1084, 2221] | 638 | -- | -- |
| `e2e3_cartpole_suboptimal_full_original` | code-faithful | 62 | 10/10 | 1901 | 1901 [1187, 2791] | 1504 [866, 2803] | 521 | -- | -- |
| `e2e3_cartpole_suboptimal_no_input_weights_original` | code-faithful | 42 | 5/10 | 1945 | 3472 [2405, 4554] | 4539 [1474, 5000] | 1059 | -- | -- |
| `e2e3_cartpole_suboptimal_no_reuploading_original` | code-faithful | 46 | 8/10 | 2451 | 2961 [2152, 3858] | 2317 [1670, 4239] | 1429 | -- | -- |
| `e4_cartpole_mlp_10_10_classical` | code-faithful | 182 | 9/10 | 320 | 788 [276, 1747] | 341 [229, 446] | 202 | 1110 (figure) | True |
| `e4_cartpole_mlp_14_14_classical` | code-faithful | 310 | 10/10 | 1168 | 1168 [440, 2069] | 387 [240, 2164] | 197 | -- | -- |
| `e4_cartpole_mlp_15_15_classical` | code-faithful | 347 | 10/10 | 857 | 857 [278, 1771] | 306 [257, 896] | 214 | 540 (figure) | True |
| `e4_cartpole_mlp_20_20_classical` | code-faithful | 562 | 10/10 | 556 | 556 [263, 893] | 250 [199, 856] | 197 | 430 (figure) | True |
| `e4_cartpole_mlp_24_24_classical` | code-faithful | 770 | 10/10 | 248 | 248 [219, 280] | 230 [209, 287] | 193 | 415 (figure) | False |
| `e4_cartpole_mlp_30_30_classical` | code-faithful | 1142 | 10/10 | 234 | 234 [207, 267] | 216 [199, 273] | 191 | 250 (figure) | True |
| `e4_cartpole_mlp_5_5_classical` | code-faithful | 67 | 5/10 | 1473 | 3236 [1909, 4537] | 4910 [622, 5000] | 460 | -- | -- |
| `e4_cartpole_mlp_64_64_classical` | code-faithful | 4610 | 9/10 | 375 | 837 [225, 1819] | 230 [199, 797] | 197 | 345 (figure) | True |
| `e4_cartpole_pqc_l10_original` | code-faithful | 122 | 10/10 | 412 | 412 [272, 592] | 286 [248, 588] | 215 | 635 (figure) | False |
| `e4_cartpole_pqc_l15_original` | code-faithful | 182 | 10/10 | 468 | 468 [342, 604] | 428 [255, 712] | 236 | 690 (figure) | False |
| `e4_cartpole_pqc_l20_original` | code-faithful | 242 | 10/10 | 512 | 512 [373, 715] | 407 [328, 568] | 301 | 770 (figure) | False |
| `e4_cartpole_pqc_l25_original` | code-faithful | 302 | 10/10 | 435 | 435 [314, 575] | 332 [254, 616] | 247 | 350 (figure) | True |
| `e4_cartpole_pqc_l30_original` | code-faithful | 362 | 10/10 | 751 | 751 [518, 1022] | 656 [401, 1018] | 304 | 940 (figure) | True |

## Pairwise comparisons

| Claim | Comparison | Median A | Median B | Median diff A-B [95% CI] | MWU p | Verdict |
|---|---|---:|---:|---|---:|---|
| C3 | optimal: full vs w/o re-uploading | 303 | 344 | -42 [-152, 33] | 0.14 | no clear separation |
| C3 | optimal: full vs w/o input weights | 303 | 530 | -227 [-2568, 17] | 0.0411 | no clear separation |
| C3 | sub-optimal: full vs w/o re-uploading | 1504 | 2317 | -814 [-3204, 647] | 0.104 | no clear separation |
| C3 | sub-optimal: full vs w/o input weights | 1504 | 4539 | -3036 [-4047, 545] | 0.0362 | no clear separation |
| C2 | optimal: trainable w_o vs fixed [0,1] | 303 | 5000 | -4697 [-4776, -4663] | 6.39e-05 | separated |
| C2 | optimal: trainable w_o vs fixed [0,90] | 303 | 258 | 46 [-60, 102] | 0.473 | no clear separation |
| C2 | optimal: trainable w_o vs fixed [0,180] | 303 | 291 | 12 [-143, 66] | 0.427 | no clear separation |
| C2 | sub-optimal: trainable w_o vs fixed [0,1] | 1504 | 5000 | -3496 [-4134, -2197] | 6.39e-05 | separated |
| C2 | sub-optimal: trainable w_o vs fixed [0,90] | 1504 | 1696 | -192 [-1026, 1266] | 1 | no clear separation |
| C2 | sub-optimal: trainable w_o vs fixed [0,180] | 1504 | 2352 | -848 [-2906, 872] | 0.212 | no clear separation |
| C5 | best paper PQC (25 layers) vs best paper NN (20,20) | 332 | 250 | 82 [-590, 382] | 0.384 | no clear separation |
| C5 | iso-parameter: PQC 5 layers (62) vs NN (5,5) (67) | 303 | 4910 | -4606 [-4772, -319] | 0.000221 | separated |
| C5 | iso-parameter: PQC 25 layers (302) vs NN (14,14) (310) | 332 | 387 | -55 [-1895, 310] | 0.623 | no clear separation |
| C1 | Frozen Lake: 5 vs 15 layers (episodes) | 1071 | 688 | 384 [-18, 784] | 0.0539 | no clear separation |

## Frozen Lake MAE (C1)

Mean MAE44 at step 10000 by depth: {'5': 0.16941847024980455, '10': 0.1524267624787764, '15': 0.13674653148612542} (paper, read from Fig. 3b: {'5': 0.19, '10': 0.17, '15': 0.155}).
Difference 5 - 15 layers: 0.0327 [95% CI 0.0229, 0.0419], MWU p = 0.00033; paper: ~0.05.

## Scans (C5)

- pqc_depth_scan: Kruskal-Wallis H = 14.16, p = 0.0146; lowest mean (censored): `e2e3_cartpole_optimal_full_original`.
- nn_width_scan: Kruskal-Wallis H = 9.37, p = 0.0952; lowest mean (censored): `e4_cartpole_mlp_30_30_classical`.

Best individual agents: PQC `e2e3_cartpole_optimal_full_original` at 208 (paper 206, 5 layers); NN `e4_cartpole_mlp_30_30_classical` at 191 (paper 186, (30,30)).
