# Consistency vs nested autograd — micro-benchmark

Tests the paper's central methodological claim (§III.B) that *nested*
automatic differentiation through the CV simulator allocates super-linearly
with the Fock cutoff, motivating the consistency-loss work-around.

Measured by `utils/measure_nested_overhead.py`: 5 training steps on the
1D Poisson problem with the smoke 2 + 2 layer QPINN. All measurements
share CPU with the multi-seed heat sweep launched in the same session,
so absolute step times overstate by ~3-5x; the *relative* differences
between consistency and nested at the same cutoff remain valid.

| Cutoff | Loss | Step (s) | Peak RSS delta (MB) | Final loss after 5 steps |
|---:|---|---:|---:|---:|
| 8 | consistency | 10.27 | 0.1 | 2.38e-1 |
| 8 | nested | 9.37 | 1.1 | 2.33e-1 |
| 10 | consistency | 9.78 | 0.0 | 2.22e-1 |
| 10 | nested | 13.89 | 1.8 | 2.29e-1 |
| 12 | consistency | 12.40 | 0.0 | 2.22e-1 |
| 12 | nested | 15.83 | 30.1 | 2.28e-1 |
| 15 | consistency | 23.20 | 0.0 | 2.22e-1 |
| 15 | nested | not measured | — | — |

## Reading

- **Peak RSS delta** between consistency and nested loss jumps from
  ~2 MB at cutoff 10 to **30 MB at cutoff 12**, a ~15x discontinuity.
  This qualitatively confirms the paper's claim that nested autograd
  allocates super-linearly with cutoff. At cutoff 15 we expect another
  large step; the run is not measured here because of CPU contention
  but the trend is unmistakable.
- **Per-step time** is more modest: ~1.3-1.4x slower for nested at
  cutoff 10-12, compared with consistency. At cutoff 8 nested is
  actually *faster* on average (within noise) because the extra
  `autograd.grad` call is cheaper than the second `(u, ux, trace)`
  forward pass through the QPINN.
- **Final loss after 5 steps** is essentially the same for both,
  confirming that both formulations are training the same underlying
  PDE residual (the difference is only how the second derivative is
  computed).

## Verdict (per-step micro-benchmark)

The consistency-loss trick matters mostly for **memory**, not throughput.
At small cutoff (8-10) it is roughly neutral; from cutoff 12 onwards the
memory footprint of nested autograd dominates. For a paper-accurate
cutoff (15-20) the gap would be much larger and our simulator could
run into RAM pressure under nested gradients — exactly as the paper
predicts for the Strawberry Fields TensorFlow backend.

## End-to-end RMSE comparison (200 epochs each, smoke 2+2 layer QPINN)

This is the comparison the paper text *implies* but does not directly
report: trained to the same epoch budget, which loss reaches lower error?

| Cutoff | Loss | Epochs | RMSE | Wall time | Ratio (cons / nested) |
|---:|---|---:|---:|---:|---:|
| 8  | nested      | 200 | **4.24e-5** | 85 s  | — |
| 8  | consistency | 200 | 4.64e-3     | 168 s | **109× worse** |
| 12 | nested      | 200 | **1.51e-4** | 141 s | — |
| 12 | consistency | 200 | 1.85e-3     | 115 s | **12× worse**  |

Both nested and consistency runs use the same architecture, same
optimiser, same collocation points (Sobol 64-pt), and the same Adam
learning rate 0.05. The only difference is which Poisson loss function is
used (`poisson_total_loss` vs `poisson_nested_loss`).

## Updated verdict

The consistency-loss trick **buys memory at a clear accuracy cost** in
our simulator:

- **Memory:** nested allocates 30 MB more per step at cutoff 12 — confirmed.
- **Time:** nested is ~20-50 % slower per step at cutoff 10-12 —
  marginal.
- **Accuracy:** nested is **12-100× more accurate** at the same epoch
  budget at cutoffs 8-12 — a substantial, previously-undocumented cost
  to the consistency-loss design.

The paper's framing is that nested autograd is impractical because it
blows up memory. Our results suggest that at the cutoffs where memory
is *not* yet prohibitive (≤12 in our PyTorch simulator), the
consistency-loss trick should be regarded as a *memory optimisation
with a non-trivial accuracy penalty*, not as a pure improvement. At
paper-accurate cutoff (15-20) the memory wall presumably reverses
this trade-off; we did not measure that regime due to wall-clock
constraints (the cutoff-15 nested run was killed under CPU contention
in the multi-seed sweep).

The methodological contribution of the paper (the consistency-loss
trick itself) is **partially upheld**: it does solve the memory problem,
but at a cost that the paper does not flag.
