| MerLin config | Solved (MerLin / qubit) | Mean episodes MerLin [95% CI] | Mean episodes qubit [95% CI] | MWU p | Agreement | Params (MerLin / qubit) | Wall-clock s (MerLin) |
|---|---:|---|---|---:|---|---|---:|
| `e1_frozenlake_l10_merlin` | 0/10 / 10/10 | 5000 [5000, 5000] | 821 [623, 1070] | 6.4e-05 | MerLin slower | 110 / 88 | 9528 |
| `e1_frozenlake_l10_merlin_dualrail` | 0/10 / 10/10 | 5000 [5000, 5000] | 821 [622, 1069] | 6.4e-05 | MerLin slower | 154 / 88 | 1264 |
| `e1_frozenlake_l10_merlin_reupload` | 10/10 / 10/10 | 501 [457, 547] | 821 [622, 1071] | 0.0017 | MerLin faster | 110 / 88 | 484 |
| `e1_frozenlake_l15_merlin` | 0/10 / 10/10 | 5000 [5000, 5000] | 749 [638, 883] | 6.4e-05 | MerLin slower | 160 / 128 | 13163 |
| `e1_frozenlake_l15_merlin_reupload` | 10/10 / 10/10 | 545 [498, 594] | 749 [639, 877] | 0.0073 | MerLin faster | 160 / 128 | 625 |
| `e1_frozenlake_l5_merlin` | 0/10 / 10/10 | 5000 [5000, 5000] | 1323 [854, 1999] | 6.4e-05 | MerLin slower | 60 / 48 | 7639 |
| `e1_frozenlake_l5_merlin_8modes` | 0/10 / 10/10 | 5000 [5000, 5000] | 1323 [857, 1994] | 6.4e-05 | MerLin slower | 84 / 48 | 3226 |
| `e1_frozenlake_l5_merlin_dualrail` | 0/10 / 10/10 | 5000 [5000, 5000] | 1323 [859, 1998] | 6.4e-05 | MerLin slower | 84 / 48 | 803 |
| `e1_frozenlake_l5_merlin_reupload` | 7/10 / 10/10 | 2093 [1002, 3392] | 1323 [858, 1996] | 0.85 | no clear separation | 60 / 48 | 3636 |
| `e2e3_cartpole_optimal_fixed_range_180_merlin` | 10/10 / 10/10 | 375 [294, 480] | 350 [277, 445] | 0.76 | no clear separation | 80 / 60 | 10298 |
| `e2e3_cartpole_optimal_fixed_range_1_merlin` | 0/10 / 0/10 | 5000 [5000, 5000] | 5000 [5000, 5000] | 1 | no clear separation | 80 / 60 | 4219 |
| `e2e3_cartpole_optimal_fixed_range_90_merlin` | 10/10 / 10/10 | 825 [284, 1775] | 279 [239, 335] | 0.41 | no clear separation | 80 / 60 | 39479 |
| `e2e3_cartpole_optimal_full_merlin` | 10/10 / 10/10 | 304 [244, 370] | 303 [253, 363] | 0.79 | no clear separation | 82 / 62 | 6028 |
| `e2e3_cartpole_optimal_no_input_weights_merlin` | 5/10 / 8/10 | 3122 [1908, 4298] | 1420 [450, 2721] | 0.023 | MerLin slower | 62 / 42 | 44289 |
| `e2e3_cartpole_optimal_no_reuploading_merlin` | 3/10 / 10/10 | 3613 [2226, 5000] | 394 [308, 521] | 0.0029 | MerLin slower | 66 / 46 | 41502 |
| `e2e3_cartpole_suboptimal_fixed_range_180_merlin` | 7/10 / 9/10 | 3100 [2200, 4020] | 2686 [1817, 3625] | 0.49 | no clear separation | 80 / 60 | 28073 |
| `e2e3_cartpole_suboptimal_fixed_range_1_merlin` | 0/10 / 0/10 | 5000 [5000, 5000] | 5000 [5000, 5000] | 1 | no clear separation | 80 / 60 | 3296 |
| `e2e3_cartpole_suboptimal_fixed_range_90_merlin` | 5/10 / 10/10 | 3193 [2075, 4302] | 1649 [1241, 2058] | 0.16 | no clear separation | 80 / 60 | 31176 |
| `e2e3_cartpole_suboptimal_full_merlin` | 8/10 / 10/10 | 1989 [1107, 3071] | 1901 [1172, 2757] | 0.97 | no clear separation | 82 / 62 | 26146 |
| `e2e3_cartpole_suboptimal_no_input_weights_merlin` | 2/10 / 5/10 | 4615 [4066, 5000] | 3472 [2391, 4514] | 0.12 | no clear separation | 62 / 42 | 26232 |
| `e2e3_cartpole_suboptimal_no_reuploading_merlin` | 0/10 / 8/10 | 5000 [5000, 5000] | 2961 [2117, 3858] | 0.00075 | MerLin slower | 66 / 46 | 19831 |
