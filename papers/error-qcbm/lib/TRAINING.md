# Finite-shot training

Training follows the [author implementation](https://github.com/Quandela/photonic-qcbm)
using native MerLin samples and classical SPSA. Each objective call samples under
`torch.no_grad()`, estimates probabilities, and computes a loss. There is no
backpropagation through counts or recycling. In the inspected MerLin 0.4.2 path,
autograd instead uses exact probabilities when it disables the requested shots.
See [SAMPLING.md](SAMPLING.md) for the optical and sampling pipeline.

## Probability estimates and losses

Post-selection normalizes the complete-click counts on the fixed target bins.
Recycling uses Perceval's official `photon_recycling()` function. For four input
photons it uses the four-, three- and two-click sectors; lower-click patterns
contribute to compatible completions. Perceval performs the fit, extrapolation,
clipping and normalization. The project aligns its output with the target bins.
Required sectors must be nonempty; no smoothing or replacement estimate is used.

Gaussian training uses base-two `KL(model || target)`. Zero model mass contributes
zero; positive model mass at zero target mass fails without pseudocounts. This
edge convention differs from the author's helper but does not affect the positive
Gaussian target. Financial training uses `TVD = 0.5 * sum(abs(model - target))`.
Training and reporting losses are computed from sampled probability estimates.

## SPSA and matched controls

For each update k, draw a sign vector delta with independent +1/-1 entries:

```text
L_plus  = loss from N fresh shots at theta + ck*delta
L_minus = loss from N fresh shots at theta - ck*delta
gk[j] = (L_plus - L_minus) / (2*ck*delta[j])
theta = theta - ak*gk
```

Full runs use `alpha=.602`, `gamma=.101`, `c=.1` and `A=100`:
`ak=a/(k+A)^alpha`, `ck=c/k^gamma`. The gain is calibrated once using two
initial perturbed evaluations: `a=.001*(A+1)^alpha/abs(mean(g0))`.
The denominator is `abs(mean(g0))`, not `mean(abs(g0))`. Degenerate calibration
fails. Smaller runs use A equal to 10% of their configured update count.

Initial phases follow `RandomState(seed).normal(0, 2*pi, parameter_count)`.
Conditions share initialization, perturbation seeds, sampling-seed schedules and
initial-shot budgets. They use the same gain-calibration rule, which can produce
different numerical gains. Matched seeds do not imply identical sampled outcomes.

## Reporting and shot cost

Full runs perform 1,000 updates and report a fresh sampled loss every ten updates.
An additional initial report gives 101 records at updates `0,10,...,1000`.
Plots use `Iteration Number = update/10` and omit the initial report.

An uninterrupted run uses **2,103 objective calls**: one initial report, two
calibration calls, 2,000 gradient calls and 100 subsequent reports. The author's
loop omits our extra initial report. Each call consumes N initial attempts,
including vacuum, collisions and events discarded by the estimator.

| Experiment | Shots per call | Shots per uninterrupted run |
|---|---:|---:|
| Gaussian | 200,000 | 420,600,000 |
| Financial | 60,000 | 126,180,000 |
| Belenos characteristics and controls | 50,000,000 | 105,150,000,000 |

## Resuming training

Only resumable configs support continuation; ordinary training checkpoints are
progress artifacts. Rerun a resumable config with the same `training.state_dir`
and scientific settings to continue. `training.segment_seconds` limits each
invocation. Resume restores phases, optimizer and RNG state without recalibration.
Incompatible settings or environments are rejected; use one writer per state
folder. Failed numerical runs are not automatically retried.

Interrupted work may consume shots again. Use the sampling ledger and
`shot_accounting` for physical cost; committed totals can include an inherited
retry prefix and omit replay work. Separate sampling-stream retries are not new
independent initializations and should be reported separately from originals.
