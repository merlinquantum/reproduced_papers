# Sweep `S3_spsa_grid_surrogate`

- Source: `S3_spsa_grid_surrogate_20261023-161206/summary.json` (generator: `papers/photonic_QKD/utils/run_sweep.py --summarize`, 2026-10-23T16:23:14Z).
- Predeclared SPSA grid for conditioning=surrogate: a in (0.01,0.05,0.2) x c in (0.05,0.2), MNIST conv1+conv2, R=4, dim(theta)=45, S=200, EMA 0.9, seed 0, 10 epochs. Candidates ordered by increasing a (tie-break toward smaller a).
- Selection rule (predeclared): final_epoch.proxy_objective on the val (fixed first 1024 images, SPSA proxy) split, final_epoch, minimize, aggregated as single seed (seed 0), tie tolerance 0.002 → spec_order.
- Coverage: **COMPLETE SWEEP**; invalid runs: none.
- Selected: ['a0.05_c0.2']; tied within tolerance: ['a0.05_c0.2', 'a0.2_c0.05'].

| candidate | runs | status | final_epoch.proxy_objective | proxy_objective_delta_first_minus_final | final_epoch.parameter_norm | final_evaluation.val_accuracy | final_evaluation.test_accuracy |
|---|---|---|---|---|---|---|---|
| a0.01_c0.05 | 1/1 | COMPLETE | 0.1610 | 0.5749 | 15.67 | 98.48 | 98.75 |
| a0.01_c0.2 | 1/1 | COMPLETE | 0.1507 | 0.5740 | 13.07 | 98.46 | 98.61 |
| a0.05_c0.05 | 1/1 | COMPLETE | 0.1529 | 0.5648 | 16.74 | 98.88 | 98.79 |
| a0.05_c0.2 | 1/1 | COMPLETE | 0.1371 | 0.5692 | 16.94 | 98.62 | 98.69 |
| a0.2_c0.05 | 1/1 | COMPLETE | 0.1378 | 0.6091 | 21.07 | 98.72 | 98.81 |
| a0.2_c0.2 | 1/1 | COMPLETE | 0.1466 | 0.5438 | 21.07 | 98.64 | 98.66 |
