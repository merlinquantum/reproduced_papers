# Sweep `S1_teacher_mnist_30ep`

- Source: `S1_teacher_mnist_30ep_20261023-145803/summary.json` (generator: `papers/photonic_QKD/utils/run_sweep.py --summarize`, 2026-10-23T14:11:11Z).
- MNIST teachers (64,128,128), seeds 0-2, 30 epochs (partial RED-tier; paper 100 epochs, 5 seeds). Not a selection sweep: one candidate.
- Selection rule (predeclared): best_checkpoint.test_accuracy on the test split, best_val_epoch, maximize, aggregated as mean over seeds, tie tolerance 0.0 → spec_order.
- Coverage: **COMPLETE SWEEP**; invalid runs: none.
- Selected: ['teacher_64_128_128']; tied within tolerance: ['teacher_64_128_128'].

| candidate | runs | status | best_checkpoint.val_accuracy | best_checkpoint.val_cross_entropy | best_checkpoint.test_accuracy | final_epoch.train_accuracy | final_epoch.val_accuracy | best_checkpoint.epoch |
|---|---|---|---|---|---|---|---|---|
| teacher_64_128_128 | 3/3 | COMPLETE | 99.18 ± 0.03 | 0.0303 ± 0.0031 | 99.29 ± 0.01 | 99.61 ± 0.03 | 99.03 ± 0.09 | 26.33 ± 1.53 |
