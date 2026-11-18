# Phase 4: MerLin adaptation vs qubit model (code-faithful settings)

Episodes to solve: 0-based index, unsolved agents censored at 5000; 10 agents (seeds 0-9);
95% bootstrap intervals; Mann-Whitney U on censored episodes. Label: merlin-adaptation
(simulated linear optics, not hardware).

| Config | MerLin solved | MerLin mean [95% CI] | Qubit solved | Qubit mean [95% CI] | MWU p | Agreement | MerLin params | Qubit params | MAE44 @10k MerLin / qubit |
|---|---:|---|---:|---|---:|---|---:|---:|---|
| `e1_frozenlake_l10_merlin` | 0/10 | 5000 [5000, 5000] | 10/10 | 821 [623, 1070] | 6.39e-05 | MerLin slower | 110 | 88 | 0.252 / 0.152 |
| `e1_frozenlake_l10_merlin_dualrail` | 0/10 | 5000 [5000, 5000] | 10/10 | 821 [622, 1069] | 6.39e-05 | MerLin slower | 154 | 88 | 0.197 / 0.152 |
| `e1_frozenlake_l10_merlin_reupload` | 10/10 | 501 [457, 547] | 10/10 | 821 [622, 1071] | 0.00171 | MerLin faster | 110 | 88 | 0.128 / 0.152 |
| `e1_frozenlake_l15_merlin` | 0/10 | 5000 [5000, 5000] | 10/10 | 749 [638, 883] | 6.39e-05 | MerLin slower | 160 | 128 | 0.254 / 0.137 |
| `e1_frozenlake_l15_merlin_reupload` | 10/10 | 545 [498, 594] | 10/10 | 749 [639, 877] | 0.00728 | MerLin faster | 160 | 128 | 0.083 / 0.137 |
| `e1_frozenlake_l5_merlin` | 0/10 | 5000 [5000, 5000] | 10/10 | 1323 [854, 1999] | 6.39e-05 | MerLin slower | 60 | 48 | 0.251 / 0.169 |
| `e1_frozenlake_l5_merlin_8modes` | 0/10 | 5000 [5000, 5000] | 10/10 | 1323 [857, 1994] | 6.39e-05 | MerLin slower | 84 | 48 | 0.372 / 0.169 |
| `e1_frozenlake_l5_merlin_dualrail` | 0/10 | 5000 [5000, 5000] | 10/10 | 1323 [859, 1998] | 6.39e-05 | MerLin slower | 84 | 48 | 0.195 / 0.169 |
| `e1_frozenlake_l5_merlin_reupload` | 7/10 | 2093 [1002, 3392] | 10/10 | 1323 [858, 1996] | 0.85 | no clear separation | 60 | 48 | 0.205 / 0.169 |
| `e2e3_cartpole_optimal_fixed_range_180_merlin` | 10/10 | 375 [294, 480] | 10/10 | 350 [277, 445] | 0.762 | no clear separation | 80 | 60 | -- |
| `e2e3_cartpole_optimal_fixed_range_1_merlin` | 0/10 | 5000 [5000, 5000] | 0/10 | 5000 [5000, 5000] | 1 | no clear separation | 80 | 60 | -- |
| `e2e3_cartpole_optimal_fixed_range_90_merlin` | 10/10 | 825 [284, 1775] | 10/10 | 279 [239, 335] | 0.406 | no clear separation | 80 | 60 | -- |
| `e2e3_cartpole_optimal_full_merlin` | 10/10 | 304 [244, 370] | 10/10 | 303 [253, 363] | 0.791 | no clear separation | 82 | 62 | -- |
| `e2e3_cartpole_optimal_no_input_weights_merlin` | 5/10 | 3122 [1908, 4298] | 8/10 | 1420 [450, 2721] | 0.0227 | MerLin slower | 62 | 42 | -- |
| `e2e3_cartpole_optimal_no_reuploading_merlin` | 3/10 | 3613 [2226, 5000] | 10/10 | 394 [308, 521] | 0.00294 | MerLin slower | 66 | 46 | -- |
| `e2e3_cartpole_suboptimal_fixed_range_180_merlin` | 7/10 | 3100 [2200, 4020] | 9/10 | 2686 [1817, 3625] | 0.495 | no clear separation | 80 | 60 | -- |
| `e2e3_cartpole_suboptimal_fixed_range_1_merlin` | 0/10 | 5000 [5000, 5000] | 0/10 | 5000 [5000, 5000] | 1 | no clear separation | 80 | 60 | -- |
| `e2e3_cartpole_suboptimal_fixed_range_90_merlin` | 5/10 | 3193 [2075, 4302] | 10/10 | 1649 [1241, 2058] | 0.159 | no clear separation | 80 | 60 | -- |
| `e2e3_cartpole_suboptimal_full_merlin` | 8/10 | 1989 [1107, 3071] | 10/10 | 1901 [1172, 2757] | 0.97 | no clear separation | 82 | 62 | -- |
| `e2e3_cartpole_suboptimal_no_input_weights_merlin` | 2/10 | 4615 [4066, 5000] | 5/10 | 3472 [2391, 4514] | 0.121 | no clear separation | 62 | 42 | -- |
| `e2e3_cartpole_suboptimal_no_reuploading_merlin` | 0/10 | 5000 [5000, 5000] | 8/10 | 2961 [2117, 3858] | 0.000751 | MerLin slower | 66 | 46 | -- |

## Preservation criteria (declared in the Phase 4 Adaptation Plan)

| Criterion | Observed | Met |
|---|---|---|
| C1 planned (basis phases once): all depths solve >= 8/10 and depth does not hurt | solved [0, 0, 0], mean episodes [5000, 5000, 5000] | no |
| C2: fixed [0,1] 0/10 (both sets); trainable w_o and fixed [0,90] >= 8/10 (best set) | [0,1] 0/10 and 0/10; w_o 10/10; [0,90] 10/10 | yes |
| C3: sub-optimal set, both ablations not faster than full (median) | median full 1314, w/o re-uploading 5000, w/o w_d 5000 | yes |
| C4: full arm, best set, >= 8/10 | 10/10 | yes |
| C1 diagnostic (basis phases re-uploaded), same criterion; not the planned model | solved [7, 10, 10], mean episodes [2093, 501, 545] | no |
