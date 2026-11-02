# S8: causal preprocessing vs original (S1), paper hyperparameters, 80/20 split, seeds 0-4 paired

Sweep: `outdir/sweeps/s8_causal_fill_20261029-162724`. Metric: test RMSE (AQI) on the 402 observed test targets (identical dates and identical test inputs under both fills). Rule (predeclared): paired 95% t-interval, n=5; 'changed' / 'worse' if the CI excludes 0.

## (1) Per-model RMSE change: causal − original

| Model | Original mean | Causal mean | Δ mean | 95% CI | Decision |
|---|---:|---:|---:|---|---|
| LSTM(16) | 12.611 | 15.240 | +2.629 | [+1.081, +4.176] | changed |
| Q-B quantum | 15.679 | 16.302 | +0.622 | [-0.207, +1.451] | not changed |
| Q-B RBF | 16.006 | 16.415 | +0.410 | [-0.640, +1.459] | not changed |

## (2) Quantum − RBF difference: causal vs original (difference-in-differences)

- original: quantum − RBF: -0.326 [-0.591, -0.062] → excludes 0
- causal: quantum − RBF: -0.113 [-0.576, +0.349] → includes 0
- difference-in-differences (causal − original): +0.213 [-0.265, +0.691] → includes 0

## (3) Phase 5 rule under causal preprocessing: Q-B quantum − LSTM(16)

- original (S1): +3.068 [+0.757, +5.379], QK better on 0/5 → worse
- causal (S8): +1.062 [-0.972, +3.096], QK better on 1/5 → not separated
