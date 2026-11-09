# Sweep `S2_spsa_grid_photonic`

- Source: `S2_spsa_grid_photonic_20261023-161206/summary.json` (generator: `papers/photonic_QKD/utils/run_sweep.py --summarize`, 2026-10-23T15:43:45Z).
- Predeclared SPSA grid for conditioning=photonic: a in (0.01,0.05,0.2) x c in (0.05,0.2), MNIST conv1+conv2, R=4, dim(theta)=45, S=200, EMA 0.9, seed 0, 10 epochs. Candidates ordered by increasing a (tie-break toward smaller a).
- Selection rule (predeclared): final_epoch.proxy_objective on the val (fixed first 1024 images, SPSA proxy) split, final_epoch, minimize, aggregated as single seed (seed 0), tie tolerance 0.002 → spec_order.
- Coverage: **COMPLETE SWEEP**; invalid runs: none.
- Selected: ['a0.01_c0.2']; tied within tolerance: ['a0.01_c0.2'].

| candidate | runs | status | final_epoch.proxy_objective | proxy_objective_delta_first_minus_final | final_epoch.parameter_norm | final_evaluation.val_accuracy | final_evaluation.test_accuracy |
|---|---|---|---|---|---|---|---|
| a0.01_c0.05 | 1/1 | COMPLETE | 0.1624 | 0.4844 | 0.31 | 98.60 | 98.80 |
| a0.01_c0.2 | 1/1 | COMPLETE | 0.1425 | 0.4643 | 1.08 | 98.76 | 98.78 |
| a0.05_c0.05 | 1/1 | COMPLETE | 0.2892 | 0.3743 | 21.07 | 97.90 | 98.37 |
| a0.05_c0.2 | 1/1 | COMPLETE | 0.1536 | 0.4258 | 10.05 | 98.66 | 98.86 |
| a0.2_c0.05 | 1/1 | COMPLETE | 0.1926 | 0.5609 | 15.75 | 98.22 | 98.44 |
| a0.2_c0.2 | 1/1 | COMPLETE | 0.2038 | 0.3700 | 21.07 | 98.18 | 98.61 |
