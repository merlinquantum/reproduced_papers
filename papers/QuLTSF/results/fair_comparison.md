# QuLTSF fair comparison (Phase 5)

Protocol: LOG.md 'Phase 5 Comparison Plan'. Rule: QuLTSF better than a system if its mean test MSE is lower by > 2x the larger seed std and DM p < 0.05 (seed-averaged per-window squared errors, lag window T); worse symmetrically; otherwise tie. Hardware: NVIDIA L4 (reproduced-papers-1), float32; MerLin exact simulation. Wall-clock is per run, with the number of runs sharing the GPU in brackets; it describes cost, not a speed claim.

## T = 96

| System | Role | Provenance | Scope | Runs (ok / expected) | Test MSE | MSE std | Test MAE | Params | Wall-clock min (parallel) | Epochs | Commensurability | QuLTSF vs system |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---|---|
| qultsf (paper Table 1) | paper method | paper-reported | single seed (2021), test drop_last | - | 0.156 | - | 0.211 | - | - | - | PARTIALLY_COMMENSURABLE (reported value) | - |
| linear (paper Table 1, quoted from Zeng et al.) | paper baseline | paper-reported | LTSF-Linear budget, single run, other pipeline | - | 0.176 | - | 0.236 | - | - | - | PARTIALLY_COMMENSURABLE (reported value) | - |
| nlinear (paper Table 1, quoted from Zeng et al.) | paper baseline | paper-reported | LTSF-Linear budget, single run, other pipeline | - | 0.182 | - | 0.232 | - | - | - | PARTIALLY_COMMENSURABLE (reported value) | - |
| dlinear (paper Table 1, quoted from Zeng et al.) | paper baseline | paper-reported | LTSF-Linear budget, single run, other pipeline | - | 0.176 | - | 0.237 | - | - | - | PARTIALLY_COMMENSURABLE (reported value) | - |
| qultsf | paper method (reproduced) | local | paper-accurate, 3 seeds | 3 / 3 | 0.1561 | 0.0001 | 0.2098 | 346,234 | 84.1 (1 run(s), 4 parallel, `sweep_table1_leftovers`); 113.2 (2 run(s), 6 parallel, `sweep_table1_r2`) | 64 | REFERENCE | - |
| linear | paper baseline, rerun | local | paper-accurate, 3 seeds | 3 / 3 | 0.1744 | 0.0005 | 0.2354 | 32,352 | 15.4 (3 run(s), 6 parallel, `sweep_table1_r2`) | 36 | COMMENSURABLE | better (margin +0.0184, p 4e-08) |
| nlinear | paper baseline, rerun | local | paper-accurate, 3 seeds | 3 / 3 | 0.1751 | 0.0002 | 0.2255 | 32,352 | 21.9 (3 run(s), 6 parallel, `sweep_table1_r2`) | 60 | COMMENSURABLE | better (margin +0.0190, p 1.2e-07) |
| dlinear | paper baseline, rerun | local | paper-accurate, 3 seeds | 3 / 3 | 0.1743 | 0.0001 | 0.2353 | 64,704 | 2.5 (1 run(s), 4 parallel, `sweep_table1_leftovers`); 8.0 (2 run(s), 6 parallel, `sweep_table1_r2`) | 35 | COMMENSURABLE | better (margin +0.0182, p 6.5e-08) |
| linear_ltsf_budget | paper baseline, rerun at LTSF-Linear budget | local | paper-accurate for Zeng et al., 3 seeds | 3 / 3 | 0.1769 | 0.0006 | 0.2384 | 32,352 | 0.5 (3 run(s), 6 parallel, `sweep_budget`) | 8 | PARTIALLY_COMMENSURABLE | better (margin +0.0208, p 1.5e-09) |
| nlinear_ltsf_budget | paper baseline, rerun at LTSF-Linear budget | local | paper-accurate for Zeng et al., 3 seeds | 3 / 3 | 0.1805 | 0.0006 | 0.2311 | 32,352 | 0.6 (3 run(s), 6 parallel, `sweep_budget`) | 10 | PARTIALLY_COMMENSURABLE | better (margin +0.0244, p 1.5e-10) |
| dlinear_ltsf_budget | paper baseline, rerun at LTSF-Linear budget | local | paper-accurate for Zeng et al., 3 seeds | 3 / 3 | 0.1751 | 0.0004 | 0.2363 | 64,704 | 0.7 (3 run(s), 6 parallel, `sweep_budget`) | 8 | PARTIALLY_COMMENSURABLE | better (margin +0.0190, p 2.2e-08) |
| persistence | indexing control (no training) | local | deterministic, 1 run | 1 / 1 | 0.2591 | - | 0.2542 | 0 | 0.1 (1 run(s), 6 parallel, `sweep_table1_r2`) | 0 | COMMENSURABLE | better (margin +0.1031, p 9.9e-28) |
| mlp | capacity-matched classical (Phase 1 plan) | local | paper-accurate settings, 3 seeds | 3 / 3 | 0.1601 | 0.0005 | 0.2148 | 346,063 | 12.6 (3 run(s), 6 parallel, `sweep_ablations`) | 56 | COMMENSURABLE | better (margin +0.0040, p 0.0024) |
| rank_bottleneck | QuLTSF skeleton without hidden layer | local | paper-accurate settings, 3 seeds | 3 / 3 | 0.1749 | 0.0002 | 0.2357 | 4,426 | 13.7 (3 run(s), 6 parallel, `sweep_ablations`) | 58 | COMMENSURABLE | better (margin +0.0188, p 1.5e-08) |
| identity_circuit | QuLTSF without circuit (C3 control) | local | paper-accurate settings, 3 seeds | 3 / 3 | 0.1549 | 0.0002 | 0.2089 | 346,144 | 16.2 (3 run(s), 6 parallel, `sweep_ablations`) | 69 | COMMENSURABLE | worse (margin -0.0011, p 1.6e-05) |
| frozen_circuit | QuLTSF with random frozen circuit (C3 control) | local | paper-accurate settings, 3 seeds | 3 / 3 | 0.1563 | 0.0002 | 0.2099 | 346,144 | 44.7 (3 run(s), 6 parallel, `sweep_ablations`) | 69 | COMMENSURABLE | tie (margin +0.0002, p 0.096) |
| identity_interferometer_merlin | MerLin adaptation, no interferometer | local | MerLin adaptation (simulation), 3 seeds | 3 / 3 | 0.1590 | 0.0002 | 0.2142 | 338,777 | 6.4 (3 run(s), 6 parallel, `sweep_merlin_control`) | 74 | COMMENSURABLE | better (margin +0.0029, p 0.004) |
| qultsf_merlin_e20 | MerLin adaptation, trained interferometer | local | MerLin adaptation (simulation), reduced scope, 1 seed | 1 / 1 | 0.1617 | - | 0.2170 | 338,959 | 127.6 (1 run(s), 2 parallel, `sweep_merlin_trained_reduced`) | 20 | PARTIALLY_COMMENSURABLE | better (margin +0.0056, p 8.1e-07) |

## T = 192

| System | Role | Provenance | Scope | Runs (ok / expected) | Test MSE | MSE std | Test MAE | Params | Wall-clock min (parallel) | Epochs | Commensurability | QuLTSF vs system |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---|---|
| qultsf (paper Table 1) | paper method | paper-reported | single seed (2021), test drop_last | - | 0.199 | - | 0.253 | - | - | - | PARTIALLY_COMMENSURABLE (reported value) | - |
| linear (paper Table 1, quoted from Zeng et al.) | paper baseline | paper-reported | LTSF-Linear budget, single run, other pipeline | - | 0.218 | - | 0.276 | - | - | - | PARTIALLY_COMMENSURABLE (reported value) | - |
| nlinear (paper Table 1, quoted from Zeng et al.) | paper baseline | paper-reported | LTSF-Linear budget, single run, other pipeline | - | 0.225 | - | 0.269 | - | - | - | PARTIALLY_COMMENSURABLE (reported value) | - |
| dlinear (paper Table 1, quoted from Zeng et al.) | paper baseline | paper-reported | LTSF-Linear budget, single run, other pipeline | - | 0.220 | - | 0.282 | - | - | - | PARTIALLY_COMMENSURABLE (reported value) | - |
| qultsf | paper method (reproduced) | local | paper-accurate, 3 seeds | 3 / 3 | 0.1987 | 0.0003 | 0.2520 | 347,290 | 84.6 (1 run(s), 4 parallel, `sweep_table1_leftovers`); 83.4 (2 run(s), 6 parallel, `sweep_table1_r2`) | 53 | REFERENCE | - |
| linear | paper baseline, rerun | local | paper-accurate, 3 seeds | 3 / 3 | 0.2160 | 0.0007 | 0.2749 | 64,704 | 13.7 (3 run(s), 6 parallel, `sweep_table1_r2`) | 35 | COMMENSURABLE | better (margin +0.0173, p 0.0002) |
| nlinear | paper baseline, rerun | local | paper-accurate, 3 seeds | 3 / 3 | 0.2185 | 0.0002 | 0.2617 | 64,704 | 22.5 (3 run(s), 6 parallel, `sweep_table1_r2`) | 63 | COMMENSURABLE | better (margin +0.0198, p 0.00016) |
| dlinear | paper baseline, rerun | local | paper-accurate, 3 seeds | 3 / 3 | 0.2166 | 0.0003 | 0.2755 | 129,408 | 2.2 (1 run(s), 4 parallel, `sweep_table1_leftovers`); 7.2 (2 run(s), 6 parallel, `sweep_table1_r2`) | 33 | COMMENSURABLE | better (margin +0.0179, p 0.00023) |
| linear_ltsf_budget | paper baseline, rerun at LTSF-Linear budget | local | paper-accurate for Zeng et al., 3 seeds | 3 / 3 | 0.2180 | 0.0013 | 0.2762 | 64,704 | 0.5 (3 run(s), 6 parallel, `sweep_budget`) | 8 | PARTIALLY_COMMENSURABLE | better (margin +0.0193, p 6.8e-05) |
| nlinear_ltsf_budget | paper baseline, rerun at LTSF-Linear budget | local | paper-accurate for Zeng et al., 3 seeds | 3 / 3 | 0.2238 | 0.0012 | 0.2670 | 64,704 | 0.6 (3 run(s), 6 parallel, `sweep_budget`) | 10 | PARTIALLY_COMMENSURABLE | better (margin +0.0251, p 5.8e-06) |
| dlinear_ltsf_budget | paper baseline, rerun at LTSF-Linear budget | local | paper-accurate for Zeng et al., 3 seeds | 3 / 3 | 0.2169 | 0.0012 | 0.2754 | 129,408 | 0.6 (3 run(s), 6 parallel, `sweep_budget`) | 7 | PARTIALLY_COMMENSURABLE | better (margin +0.0182, p 0.00015) |
| persistence | indexing control (no training) | local | deterministic, 1 run | 1 / 1 | 0.3092 | - | 0.2917 | 0 | 0.1 (1 run(s), 6 parallel, `sweep_table1_r2`) | 0 | COMMENSURABLE | better (margin +0.1105, p 7e-23) |
| identity_interferometer_merlin | MerLin adaptation, no interferometer | local | MerLin adaptation (simulation), 3 seeds | 3 / 3 | 0.2015 | 0.0003 | 0.2575 | 340,217 | 6.0 (3 run(s), 6 parallel, `sweep_merlin_control`) | 68 | COMMENSURABLE | better (margin +0.0028, p 0.00098) |

## T = 336

| System | Role | Provenance | Scope | Runs (ok / expected) | Test MSE | MSE std | Test MAE | Params | Wall-clock min (parallel) | Epochs | Commensurability | QuLTSF vs system |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---|---|
| qultsf (paper Table 1) | paper method | paper-reported | single seed (2021), test drop_last | - | 0.248 | - | 0.296 | - | - | - | PARTIALLY_COMMENSURABLE (reported value) | - |
| linear (paper Table 1, quoted from Zeng et al.) | paper baseline | paper-reported | LTSF-Linear budget, single run, other pipeline | - | 0.262 | - | 0.312 | - | - | - | PARTIALLY_COMMENSURABLE (reported value) | - |
| nlinear (paper Table 1, quoted from Zeng et al.) | paper baseline | paper-reported | LTSF-Linear budget, single run, other pipeline | - | 0.271 | - | 0.301 | - | - | - | PARTIALLY_COMMENSURABLE (reported value) | - |
| dlinear (paper Table 1, quoted from Zeng et al.) | paper baseline | paper-reported | LTSF-Linear budget, single run, other pipeline | - | 0.265 | - | 0.319 | - | - | - | PARTIALLY_COMMENSURABLE (reported value) | - |
| qultsf | paper method (reproduced) | local | paper-accurate, 3 seeds | 3 / 3 | 0.2479 | 0.0005 | 0.2959 | 348,874 | 74.6 (1 run(s), 4 parallel, `sweep_table1_leftovers`); 75.4 (2 run(s), 6 parallel, `sweep_table1_r2`) | 47 | REFERENCE | - |
| linear | paper baseline, rerun | local | paper-accurate, 3 seeds | 3 / 3 | 0.2633 | 0.0015 | 0.3151 | 113,232 | 12.6 (3 run(s), 6 parallel, `sweep_table1_r2`) | 34 | COMMENSURABLE | better (margin +0.0153, p 0.009) |
| nlinear | paper baseline, rerun | local | paper-accurate, 3 seeds | 3 / 3 | 0.2659 | 0.0003 | 0.2963 | 113,232 | 9.9 (1 run(s), 4 parallel, `sweep_table1_leftovers`); 16.0 (2 run(s), 6 parallel, `sweep_table1_r2`) | 48 | COMMENSURABLE | better (margin +0.0179, p 0.017) |
| dlinear | paper baseline, rerun | local | paper-accurate, 3 seeds | 3 / 3 | 0.2634 | 0.0020 | 0.3163 | 226,464 | 1.5 (1 run(s), 4 parallel, `sweep_table1_leftovers`); 7.1 (2 run(s), 6 parallel, `sweep_table1_r2`) | 27 | COMMENSURABLE | better (margin +0.0154, p 0.0068) |
| linear_ltsf_budget | paper baseline, rerun at LTSF-Linear budget | local | paper-accurate for Zeng et al., 3 seeds | 3 / 3 | 0.2631 | 0.0007 | 0.3132 | 113,232 | 0.4 (3 run(s), 6 parallel, `sweep_budget`) | 7 | PARTIALLY_COMMENSURABLE | better (margin +0.0151, p 0.0084) |
| nlinear_ltsf_budget | paper baseline, rerun at LTSF-Linear budget | local | paper-accurate for Zeng et al., 3 seeds | 3 / 3 | 0.2702 | 0.0003 | 0.3005 | 113,232 | 0.6 (3 run(s), 6 parallel, `sweep_budget`) | 10 | PARTIALLY_COMMENSURABLE | better (margin +0.0222, p 0.0049) |
| dlinear_ltsf_budget | paper baseline, rerun at LTSF-Linear budget | local | paper-accurate for Zeng et al., 3 seeds | 3 / 3 | 0.2624 | 0.0011 | 0.3129 | 226,464 | 0.5 (3 run(s), 6 parallel, `sweep_budget`) | 6 | PARTIALLY_COMMENSURABLE | better (margin +0.0145, p 0.011) |
| persistence | indexing control (no training) | local | deterministic, 1 run | 1 / 1 | 0.3764 | - | 0.3377 | 0 | 0.1 (1 run(s), 6 parallel, `sweep_table1_r2`) | 0 | COMMENSURABLE | better (margin +0.1285, p 2.7e-22) |
| identity_interferometer_merlin | MerLin adaptation, no interferometer | local | MerLin adaptation (simulation), 3 seeds | 3 / 3 | 0.2489 | 0.0005 | 0.2978 | 342,377 | 5.0 (3 run(s), 6 parallel, `sweep_merlin_control`) | 57 | COMMENSURABLE | tie (margin +0.0010, p 0.29) |

## T = 720

| System | Role | Provenance | Scope | Runs (ok / expected) | Test MSE | MSE std | Test MAE | Params | Wall-clock min (parallel) | Epochs | Commensurability | QuLTSF vs system |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---|---|
| qultsf (paper Table 1) | paper method | paper-reported | single seed (2021), test drop_last | - | 0.315 | - | 0.346 | - | - | - | PARTIALLY_COMMENSURABLE (reported value) | - |
| linear (paper Table 1, quoted from Zeng et al.) | paper baseline | paper-reported | LTSF-Linear budget, single run, other pipeline | - | 0.326 | - | 0.365 | - | - | - | PARTIALLY_COMMENSURABLE (reported value) | - |
| nlinear (paper Table 1, quoted from Zeng et al.) | paper baseline | paper-reported | LTSF-Linear budget, single run, other pipeline | - | 0.338 | - | 0.348 | - | - | - | PARTIALLY_COMMENSURABLE (reported value) | - |
| dlinear (paper Table 1, quoted from Zeng et al.) | paper baseline | paper-reported | LTSF-Linear budget, single run, other pipeline | - | 0.323 | - | 0.362 | - | - | - | PARTIALLY_COMMENSURABLE (reported value) | - |
| qultsf | paper method (reproduced) | local | paper-accurate, 3 seeds | 3 / 3 | 0.3163 | 0.0022 | 0.3494 | 353,098 | 79.7 (1 run(s), 4 parallel, `sweep_table1_leftovers`); 65.0 (2 run(s), 6 parallel, `sweep_table1_r2`) | 44 | REFERENCE | - |
| linear | paper baseline, rerun | local | paper-accurate, 3 seeds | 3 / 3 | 0.3255 | 0.0008 | 0.3655 | 242,640 | 11.3 (3 run(s), 6 parallel, `sweep_table1_r2`) | 31 | COMMENSURABLE | tie (margin +0.0092, p 0.13) |
| nlinear | paper baseline, rerun | local | paper-accurate, 3 seeds | 3 / 3 | 0.3346 | 0.0002 | 0.3445 | 242,640 | 5.9 (1 run(s), 4 parallel, `sweep_table1_leftovers`); 15.0 (2 run(s), 6 parallel, `sweep_table1_r2`) | 55 | COMMENSURABLE | tie (margin +0.0183, p 0.096) |
| dlinear | paper baseline, rerun | local | paper-accurate, 3 seeds | 3 / 3 | 0.3258 | 0.0004 | 0.3658 | 485,280 | 1.5 (1 run(s), 4 parallel, `sweep_table1_leftovers`); 7.2 (2 run(s), 6 parallel, `sweep_table1_r2`) | 28 | COMMENSURABLE | tie (margin +0.0095, p 0.12) |
| linear_ltsf_budget | paper baseline, rerun at LTSF-Linear budget | local | paper-accurate for Zeng et al., 3 seeds | 3 / 3 | 0.3260 | 0.0006 | 0.3646 | 242,640 | 0.5 (3 run(s), 6 parallel, `sweep_budget`) | 8 | PARTIALLY_COMMENSURABLE | tie (margin +0.0097, p 0.1) |
| nlinear_ltsf_budget | paper baseline, rerun at LTSF-Linear budget | local | paper-accurate for Zeng et al., 3 seeds | 3 / 3 | 0.3385 | 0.0003 | 0.3482 | 242,640 | 0.6 (3 run(s), 6 parallel, `sweep_budget`) | 10 | PARTIALLY_COMMENSURABLE | better (margin +0.0222, p 0.048) |
| dlinear_ltsf_budget | paper baseline, rerun at LTSF-Linear budget | local | paper-accurate for Zeng et al., 3 seeds | 3 / 3 | 0.3272 | 0.0042 | 0.3672 | 485,280 | 0.5 (3 run(s), 6 parallel, `sweep_budget`) | 7 | PARTIALLY_COMMENSURABLE | tie (margin +0.0109, p 0.084) |
| persistence | indexing control (no training) | local | deterministic, 1 run | 1 / 1 | 0.4652 | - | 0.3935 | 0 | 0.2 (1 run(s), 6 parallel, `sweep_table1_r2`) | 0 | COMMENSURABLE | better (margin +0.1489, p 2e-11) |
| mlp | capacity-matched classical (Phase 1 plan) | local | paper-accurate settings, 3 seeds | 3 / 3 | 0.3252 | 0.0033 | 0.3552 | 352,701 | 10.7 (3 run(s), 6 parallel, `sweep_ablations`) | 48 | COMMENSURABLE | better (margin +0.0089, p 0.00078) |
| rank_bottleneck | QuLTSF skeleton without hidden layer | local | paper-accurate settings, 3 seeds | 3 / 3 | 0.3255 | 0.0003 | 0.3645 | 11,290 | 12.5 (3 run(s), 6 parallel, `sweep_ablations`) | 56 | COMMENSURABLE | tie (margin +0.0092, p 0.097) |
| identity_circuit | QuLTSF without circuit (C3 control) | local | paper-accurate settings, 3 seeds | 3 / 3 | 0.3145 | 0.0013 | 0.3479 | 353,008 | 9.5 (3 run(s), 6 parallel, `sweep_ablations`) | 42 | COMMENSURABLE | tie (margin -0.0018, p 0.00039) |
| frozen_circuit | QuLTSF with random frozen circuit (C3 control) | local | paper-accurate settings, 3 seeds | 3 / 3 | 0.3164 | 0.0022 | 0.3494 | 353,008 | 27.6 (3 run(s), 6 parallel, `sweep_ablations`) | 44 | COMMENSURABLE | tie (margin +0.0001, p 0.025) |
| identity_interferometer_merlin | MerLin adaptation, no interferometer | local | MerLin adaptation (simulation), 3 seeds | 3 / 3 | 0.3203 | 0.0034 | 0.3550 | 348,137 | 4.1 (3 run(s), 6 parallel, `sweep_merlin_control`) | 44 | COMMENSURABLE | tie (margin +0.0040, p 0.00065) |
| qultsf_merlin_e20 | MerLin adaptation, trained interferometer | local | MerLin adaptation (simulation), reduced scope, 1 seed | 1 / 1 | 0.3252 | - | 0.3631 | 348,319 | 124.8 (1 run(s), 2 parallel, `sweep_merlin_trained_reduced`) | 20 | PARTIALLY_COMMENSURABLE | better (margin +0.0089, p 2.9e-05) |

## Not compared

- C4 PatchTST / MTST: INCOMPARABLE: values quoted from Zhang et al. 2024 (different pipeline, RevIN); not rerun
- Transformer baselines (FEDformer, Autoformer, Informer): INCOMPARABLE: quoted from Zeng et al. 2023, not rerun; far worse than the linear models
