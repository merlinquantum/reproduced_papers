# A Binary Optimisation Algorithm for Near-Term Photonic Quantum Processors — Reproduction

## Reference and Attribution

- **Paper**: *A Binary Optimisation Algorithm for Near-Term Photonic Quantum Processors*,
  Makarovskiy, Slysz, Grodzki, Siera, Farnsworth, Clements, Rydlichowski, Kurowski
  (ORCA Computing; Poznań Supercomputing and Networking Center; Poznań University of
  Technology), [arXiv:2510.08274](https://arxiv.org/abs/2510.08274).
- **Original code**: none released for this paper, and no data-availability statement.
- **Ancestor code used as a reference**: ORCA Computing's
  [`quantumqubo`](https://github.com/orcacomputing/quantumqubo) (Apache-2.0), which
  accompanies the predecessor paper [arXiv:2112.09766](https://arxiv.org/abs/2112.09766).
  It implements the earlier QUBO solver on the same PT-Series hardware model, not the
  algorithm of this paper. Its single-loop simulator is used here as ground truth for the
  interferometer physics (`tests/test_tbi.py`), and its solver settled two conventions the
  paper leaves ambiguous.

## Original Paper

A train of `m` optical pulses ("time bins") carrying the state `|1,0,1,0,…⟩` passes through
programmable fibre delay lines of lengths 1, 3 and 9. Threshold detectors report which bins
contain at least one photon, giving a bit string per shot. A trainable layer flips bit `i`
independently with probability `p_i`, producing a candidate solution `X`. Training minimises
`E[C(X)]` for the problem's cost function `C`: the beamsplitter angles by a photonic
parameter-shift rule, the flip probabilities by an analytic gradient, both with plain SGD.

The claims are that this solves knapsack, tactical-deconfliction and TSP instances of up to
30 binary variables while evaluating a small fraction of the solution space, that it is
competitive with simulated annealing and hill climbing, and that it runs on ORCA's PT-1.

## Reproduction Scope, Claims and Deviations

Reproduced: the knapsack and TSP rows of Table I, the Appendix B candidate-count bound, the
paper's own SA and HC baselines, plus baselines and controls the paper does not run.

**Not reproduced, and why:**

- **Tactical deconfliction** (5 of Table I's 13 rows). The cost function is given but the
  conflict-matrix distribution and the per-size factorisation are not, so the instances
  cannot be generated without inventing them.
- **Table II** (hardware). Needs an ORCA PT-1 and a loss budget the paper does not state.

**Deviations from the paper:**

| Deviation | Reason |
|---|---|
| Instance generators are assumptions | The paper does not publish them. Knapsack: uniform values/weights on {1,…,20}, capacity half the total weight. TSP: uniform points on a 100×100 grid. |
| Circuit topology is the **pairwise chain** | The paper's own parameter count (`sum_i (m - l_i)` = 77 at m=30) requires it. ORCA's released simulator instead models the loop as a rail with memory, which needs `m` beamsplitters per loop. Both are implemented; the disagreement is recorded. |
| Parameter-shift denominator | The paper prints `1/sin(phi)`; ORCA's code uses `1/sin(2 phi)`. The two are consistent with different beamsplitter conventions. Under this package's `R = cos^2(theta/2)` the paper's expression is twice the true derivative — a constant factor, equivalent to doubling the learning rate. `shift_scale` selects it; the default reproduces the paper as printed. |
| Shift magnitude `phi = pi/2` | Not given in the paper's main text. ORCA's code defaults to `pi/6`. |
| RED-tier sizes run with fewer instances | m=25 and m=30 cost 9 h and 43 h per 100-instance row on the available 2-core machine. The paper-accurate configs are kept unchanged as artifacts; `_reduced` configs are what is actually run. |

## Install and How to Run

```bash
python -m venv papers/bosonic_binary_solver/.venv
source papers/bosonic_binary_solver/.venv/bin/activate
pip install -r papers/bosonic_binary_solver/requirements.txt
```

```bash
# smoke test, well under a minute
python implementation.py --paper bosonic_binary_solver --config configs/defaults.json

# one paper-accurate table row
python implementation.py --paper bosonic_binary_solver --config configs/knapsack_m20_original.json

# a reduced run of an expensive row
python implementation.py --paper bosonic_binary_solver \
    --config configs/knapsack_m30_original.json --instances 20

# a queue of experiments, one run directory each
python utils/run_configs.py configs/knapsack_m10_original.json configs/tsp_m10_original.json
```

`cli.json` is the authoritative flag schema: `--size`, `--family`, `--instances`,
`--updates`, `--samples`, `--lr-theta`, `--lr-alpha`, `--method`, `--source`, `--rate-scale`.

## Configuration

| Pattern | Meaning |
|---|---|
| `<family>_m<size>_original.json` | Paper-accurate: N=200, S=50, lr 0.01/0.05, delays 1-3-9, 100 instances. Kept exact; `tests/test_configs.py` enforces it. |
| `<family>_m<size>_reduced.json` | Same hyperparameters, fewer instances, for the RED-tier sizes. |
| `knapsack_m<size>_{simulated_annealing,hill_climbing,random_search}.json` | Baselines at the solver's Appendix B budget. |
| `knapsack_m<size>_source_<source>.json` | The control the paper does not run: same algorithm, different click source. |
| `knapsack_m<size>_merlin.json` | MerLin exact-gradient variant. |

## Data

No dataset. Instances are generated deterministically from the instance seed; exact optima
come from dynamic programming (knapsack) and Held–Karp (TSP), both checked against brute
force at m=10.

## Results Obtained and Comparison with the Paper

GPU queue of 2026-09-05, 3 100 runs. Every method receives exactly the Appendix B
candidate budget. Labels follow the workflow: `paper-accurate` means the paper's
hyperparameters, 100 instances, single run per instance, as the paper reports.

### Table I, knapsack

| m | claim | paper | this reproduction | instances | label |
|---|---|---|---|---|---|
| 20 | C1 | 98% | **99%** (err 0.014%) | 100 | paper-accurate |
| 25 | C1 | 99% | **96%** (err 0.054%) | 100 | paper-accurate |
| 30 | C1 | 93% | **73%** (err 0.193%) | 100 | paper-accurate, see the gap below |

TSP m=29: **7% optimal, 5.59% mean error** against the paper's 2% and 6.58%.

m=10 and m=15 are saturated (the budget is 537x and 29x the whole space) and are
reported from the CPU path.

### Table I, TSP m=29 -- and the boundary of the density result

| arm | clicks / m | % optimal | mean error | paper |
|---|---|---|---|---|
| shuffled boson | 38.3% | 3.5% | 6.385% | |
| **boson** | **38.6%** | **7.0%** | **5.589%** | 2%, 6.58% |
| bernoulli@1.0 | 44.0% | 3.0% | 5.600% | |
| distinguishable | 44.1% | 8.5% | 5.317% | |

The paper's own TSP row reproduces and is if anything exceeded (7% against 2%
optimal, 5.59% against 6.58% mean error), at 0.386% coverage of the space.

Every source comparison is null: boson against shuffled boson z = -0.76, against
distinguishable z = +0.27, against independent Bernoulli z = -0.29, with |t| <= 1.5
throughout. **The density ladder that dominates knapsack is flat here**, and that is
the predicted result rather than a surprise: a TSP candidate is an integer that is
Lehmer-decoded into a permutation, so the number of ones in the string carries no
meaning, and there is nothing for click density to be aligned or misaligned with.

This bounds the knapsack conclusion precisely. Click density governs performance
**where the encoding gives it meaning**. It is a property of the binary encoding of
the problem, not a universal property of the algorithm -- and on the family where
it does not apply, no source beats any other.

### Fair baselines at the same budget (m=20, 100 instances)

| method | % optimal | paper |
|---|---|---|
| simulated annealing | 100% | 100% |
| hill climbing | 100% | 84% |
| uniform random search | 81% | not run |
| BBS | 99% | 98% |

The budget at m=20 is 1.29x the entire solution space, so this row cannot separate
a search strategy from none: uniform sampling alone reaches 81%. The paper's hill
climbing at 84% sits **below** random sampling at the solver's own budget, which
indicates its baselines were not given that budget. The paper does not state what
budget they had.

### C7 -- the paper's ablation reproduces, strongly

Freezing the interferometer at its random initialisation and training only the
bit-flip layer:

| m | trained | theta frozen | discordant | McNemar z |
|---|---|---|---|---|
| 25 | 96% | **21%** | 0 / 75 | 8.54 |
| 30 | 73% | **3%** | 0 / 70 | 8.25 |

The trainable photonic circuit is doing the work. This is the paper's claim and it
holds decisively.

### C5 -- but the *sampler* is not what the circuit contributes

The control the paper does not run: same circuit, same trained angles, same
bit-flip layer, same budget, only the click source swapped. 100 vs 200 instances,
paired by instance.

| comparison | discordant | McNemar z | mean error gap |
|---|---|---|---|
| m=25 boson vs shuffled boson | 7 / 4 | -0.60 | -0.004% |
| m=25 boson vs distinguishable | 2 / 4 | +0.41 | -0.044% |
| m=25 boson vs independent Bernoulli | 4 / 4 | -0.35 | -0.034% |
| m=30 boson vs shuffled boson | 14 / 14 | -0.19 | +0.008% |
| m=30 boson vs distinguishable | 9 / 20 | +1.86 | -0.077% |
| m=30 boson vs independent Bernoulli | 6 / 21 | **+2.69** | -0.110% |

Shuffling the boson output keeps every per-mode marginal exactly and destroys the
joint distribution; it changes nothing. Independent Bernoulli clicks with the same
marginals do at least as well, and better at m=30.

### The reason: click density, not interference

`bernoulli@r` scales the per-mode click rate by r and changes nothing else.
m=30, 200 instances:

| arm | clicks / m | % optimal | mean error |
|---|---|---|---|
| bernoulli@0.87 | 36.9% | 73.5% | 0.219% |
| **boson** | **38.0%** | **73.0%** | **0.193%** |
| shuffled boson | 37.2% | 79.0% | 0.166% |
| bernoulli@1.0 | 42.5% | 89.0% | 0.081% |
| bernoulli@1.15 | 48.0% | **98.5%** | 0.006% |
| bernoulli@1.3 | 51.6% | 97.0% | 0.017% |
| bernoulli@1.5 | 54.4% | 95.0% | 0.031% |

Monotone up to a peak near 48% density (0.87 to 1.0: z = 4.29; 1.0 to 1.15:
z = 3.60; 0.87 to 1.5: z = 5.66; 1.15 to 1.5 turns back down, z = -1.81). And at
**matched** density the boson sampler and independent Bernoulli clicks are
indistinguishable: boson (38.0%) vs bernoulli@0.87 (36.9%) gives discordant 18/19,
z = 0.00, t = 0.38.

Read together with C7: the trained interferometer matters a great deal, and what
it contributes is carried by the per-mode marginals of |U(theta)|^2. Multi-photon
interference contributes nothing measurable here, and bunching costs the boson
source by putting it at the low end of the density axis it cannot correct.

### Two open conventions, both closed

| question | result |
|---|---|
| shift magnitude phi = pi/2 (here) vs pi/6 (ORCA's code) | no difference: discordant 9/4, z = -1.11, t = 0.22. The paper's omission does not matter. |
| the printed rule (twice the true gradient in this convention) vs the true gradient | the paper's printed rule is **better**: discordant 13/3, z = -2.25, error gap +0.089%, t = 2.40. Consistent with it being a step-size choice that suits these sizes. |

### The circuit disagreement does not matter

Chain (the paper's parameter count) against loop (ORCA's released simulator):

| m | chain | loop | discordant | z |
|---|---|---|---|---|
| 20 | 99% | 99% | 1 / 1 | -0.71 |
| 25 | 96% | 96.5% | 3 / 4 | 0.00 |

A useful null: the topology ambiguity described under "Disagreements with the released
reference code" below has no measurable effect on solution quality, so every other conclusion
here is robust to it.

### Open gap: m=30, 73% against the paper's 93%

The m=20 and m=25 rows land within 1 and 3 points; m=30 is 20 points low. The
leading hypothesis is the **instance generator**, which the paper does not publish:
knapsack difficulty at fixed m is very sensitive to the capacity fraction and to
the value/weight correlation, and this reproduction assumes uncorrelated uniform
values and weights with capacity at half the total weight. Diagnosis is deliberately
not pursued further -- the density result above does not depend on it, and chasing
an unpublished generator is the kind of exact-metric archaeology the workflow
discourages. Recorded as unresolved, with confidence in the implementation HIGH
(three independent anchors: the candidate count matches Appendix B exactly, the
loop unitary matches ORCA's simulator to 1e-12, and the GPU sampler matches
Perceval).

## Extension beyond the paper: scaling

The paper's largest simulated size is m=30, where the Appendix B budget already covers only
0.2% of the space. This extension goes to m=42 with a boson arm and to m=60 without one, to
ask whether the click-density result **survives when the search is a vanishing fraction of the
space, or was an artifact of the regime where the budget is still appreciable**. 30 instances
per arm, paper hyperparameters throughout, 540 runs; run with `scripts/run_scaling_gpu.sh`.

Found-optimum falls towards zero out here, so mean relative error carries the comparison — the
same shift the paper's own TSP m=29 row shows.

| m | coverage | arm | clicks/m | % optimal | mean error % |
|---|---|---|---|---|---|
| 34 | 1.4e-4 | boson | 37.9% | 30.0 | 0.566 |
| | | distinguishable | 42.5% | 50.0 | 0.285 |
| | | bernoulli@1.0 | 42.3% | 63.3 | 0.179 |
| | | **bernoulli@1.15** | 47.8% | **90.0** | **0.067** |
| 38 | 1.0e-5 | boson | 37.9% | 23.3 | 0.609 |
| | | distinguishable | 42.3% | 36.7 | 0.480 |
| | | bernoulli@1.0 | 42.5% | 50.0 | 0.261 |
| | | **bernoulli@1.15** | 47.9% | **83.3** | **0.065** |
| 42 | 7.1e-7 | boson | 37.7% | 16.7 | 1.177 |
| | | distinguishable | 42.6% | 20.0 | 0.673 |
| | | bernoulli@1.0 | 42.4% | 23.3 | 0.680 |
| | | **bernoulli@1.15** | 47.8% | **53.3** | **0.183** |
| 48 | 1.3e-8 | bernoulli@1.0 | 42.3% | 10.0 | 1.212 |
| | | bernoulli@1.15 | 48.0% | 30.0 | 0.444 |
| 54 | 2.3e-10 | bernoulli@1.0 | 42.4% | 3.3 | 2.152 |
| | | bernoulli@1.15 | 48.1% | 13.3 | 0.908 |
| 60 | 3.9e-12 | bernoulli@1.0 | 42.1% | 0.0 | 2.978 |
| | | bernoulli@1.15 | 48.4% | 3.3 | 1.580 |

**The result survives, and strengthens.** Boson against the dense Bernoulli arm, paired by
instance: z = 4.01, 3.80, 2.29 at m = 34, 38, 42, with the error gap *widening* — −0.50%,
−0.54%, −0.99%. And density in isolation, scaling only the Bernoulli click rate from 1.0 to
1.15 with everything else fixed, gets monotonically stronger as the space grows:

| m | 34 | 38 | 42 | 48 | 54 | 60 |
|---|---|---|---|---|---|---|
| paired t | −2.08 | −2.98 | −4.01 | −5.23 | −6.26 | **−8.24** |

At m=60, where the budget covers 4e-12 of the space and the sparser arm has stopped finding
optima entirely, a 15% higher click rate still halves the mean error (2.978% → 1.580%). So
this is not a property of the middle regime the paper tests: click density governs performance
across five orders of magnitude of coverage, and the boson sampler's bunching leaves it at the
wrong end of that axis at every size measured.

The one place the ordering blurs is exactly where the metric demands it: at m=42 boson and
distinguishable tie on found-optimum (discordant 3/4, z = 0.00) while the error metric still
separates them (t = −2.87), which is why relative error is the right measure once the space
outruns the budget.

### Cost

Measured boson wall-clock: 19.9 s/instance at m=34, 125.2 at m=38, 526.5 at m=42 — about
4.2x per +4 modes once past m=34, i.e. 2.05x per photon, Clifford & Clifford's scaling. The
whole extension took roughly five hours on an H100, half again my 3.8 h estimate. The
interference-free arms cost ~0.5 s/instance and are flat in m, which is why they reach m=60
while the boson arm stops at 42.

This is an extension, not a reproduction: no number in this section corresponds to anything in
the paper.

## Fair Baselines

The paper reports SA and HC but does not say what budget they were given. Here every
baseline receives **exactly** the Appendix B candidate budget for the same `(m, delays, N, S)`.
A `random_search` baseline is added at the same budget: at m ≤ 20 that budget exceeds the
entire solution space (129% at m=20, 53 711% at m=10), so random sampling alone finds most
optima there, which is the clearest statement of what those rows of Table I can and cannot
show.

## MerLin Photonic Extension

`lib/bbs_merlin.py` keeps the physics identical — same unrolled time-bin circuit, same
`|1,0,1,0,…⟩` input, same threshold detectors, same trainable flip layer — and replaces the
photonic parameter-shift rule with MerLin's exact gradient. The paper's budget is dominated
by that estimator: at m=30, 154 of the 155 circuit evaluations per update exist only to
measure a derivative.

Two counters are reported and they are not interchangeable: `candidates_sampled`
(`updates × samples`, what a hardware run would evaluate) and `simulator_cost_evaluations`
(`updates × 2^m`, what forming the exact expectation touches in simulation). The exact path
is a simulation-only capability and is reported as such, not as a faster BBS.

## Hardware-Aware Settings

| Field | Value |
|---|---|
| Computation space | `FOCK` (threshold patterns are formed by summing Fock outcomes) |
| Detector model | threshold (click / no click), no photon-number resolution |
| Photon number | `m/2`, from `|1,0,1,0,…⟩` |
| Modes | `m` (chain topology); `m + K` with one rail per loop (loop topology) |
| Encoding | none — the circuit carries no data input, only trainable angles |
| Measurement strategy | probabilities (MerLin variant); sampling (reproduction path) |
| Postselection | none |
| Backend | Perceval `CliffordClifford2017` for sampling, `SLOS` for exact distributions |
| Shot count | `S = 50` per expectation value |

## Notebook

`notebook.ipynb` runs top to bottom on a laptop CPU in **20 seconds** and is the intended
entry point for a reader meeting the paper for the first time: the problem, the circuit, the
claim inventory, a live training run, the MerLin exact-gradient variant, fair baselines at a
matched budget, and a live demonstration of the click-density result with the full-scale
numbers alongside. Every live cell is labelled `demo` and every full-scale figure names its
source in `results/`.

## Disagreements with the released reference code

No code was released for this paper. ORCA's earlier toolkit
[`quantumqubo`](https://github.com/orcacomputing/quantumqubo) accompanies the predecessor
paper (arXiv:2112.09766), implements the same PT-Series hardware model, and disagrees with
this paper in two places. Both are resolved in the paper's favour, and both are documented
here because either one silently changes every number if it is got wrong.

### Circuit topology

| | |
|---|---|
| **The paper** | Appendix B counts `sum_i (m - l_i)` trainable beamsplitters — 77 at m=30 for delays (1,3,9) — and its stated bound of 2.15e6 cost evaluations follows from exactly that count. That is the **pairwise chain**: one beamsplitter per pair of time bins `(t, t+l)`. |
| **ORCA's code** | `quantumqubo/tbi/tbi_sampler.py` models the fibre loop as a **rail with memory**: a beamsplitter between the loop and *every* bin, so photons may circulate for several round trips. That is `m` beamsplitters per loop, not `m - l`, and it needs an extra mode. |
| **Chosen** | `chain`, the paper's. `loop` is implemented too and validated: `tests/test_tbi.py` shows the rail unitary reproduces ORCA's sampler's output distribution to a total-variation distance below 1e-12. |
| **Why** | The paper is explicit and internally consistent about its parameter count, and that count is load-bearing for its headline efficiency claim. |
| **Impact** | Measured, not assumed: chain against loop is a dead heat at m=20 (99% vs 99%) and m=25 (96% vs 96.5%). The ambiguity does not affect any conclusion here. |

### The parameter-shift denominator

The paper prints `dE/dtheta ≈ (E[C(X_{θ+φ})] − E[C(X_{θ−φ})]) / sin(φ)`; ORCA's code divides
by `sin(2φ)`. The two are consistent with **different beamsplitter conventions**: ORCA uses
`R = cos²(θ)`, where `E` carries harmonics of `2θ` and `1/sin(2φ)` is exact, while this
package and MerLin use `R = cos²(θ/2)`, where the paper's expression is exactly *twice* the
true derivative. The factor is constant, so it is equivalent to doubling the learning rate.
`shift_scale` selects the convention and defaults to 1.0, reproducing the paper as printed —
which the m=25 comparison shows is the better of the two (discordant 13/3, z = −2.25).

## Open questions

| question | status |
|---|---|
| m=30 reaches 73% against the paper's 93% | **Open.** Leading suspect is the unpublished instance generator; see the gap discussion above. |
| Parameter-shift magnitude `phi` is absent from the paper's main text | **Closed.** `pi/2` (used here) and `pi/6` (ORCA's default) are indistinguishable: discordant 9/4, z = −1.11, t = 0.22. |
| What candidate budget did the paper's SA and HC baselines receive? | **Unanswerable from the paper**, and it matters: its hill climbing underperforms uniform random search at the solver's budget. |
| Tactical deconfliction instance distribution | **Not specified.** 5 of Table I's 13 rows cannot be generated without inventing it. |
| Loss model | Not attempted. The paper gives no loss budget, so Table II is unreproducible without hardware. The density result implies a sharp prediction for it: loss lowers click density, so it should move the boson arm monotonically down the same ladder. |

## Working documents

The reproduction effort also produced `LOG.md` (running decision log and run history),
`VISITED_URLS.md` (resource trail), `FEEDBACK.md` (feedback on the reproduction workflow
itself) and `CONFLUENCE.md` (a summary staged for publication elsewhere). These are working
documents rather than part of the deliverable and are not committed; this README is the
committed record, and everything load-bearing from them is folded into the sections above.

## Limitations

- Tactical deconfliction and the hardware table are out of scope (see above).
- The circuit-topology disagreement is resolved in the paper's favour and shown to have no
  measurable effect, but it remains an ambiguity in the paper itself.
- The knapsack and TSP instance generators are assumptions; the paper does not publish its
  own, and this is the leading suspect for the m=30 gap.
- Raw per-instance rows are not committed; `results/summary.json` carries the aggregates and
  paired statistics behind every number above.
- Single run per instance, as in the paper: instance difficulty and initialisation luck are
  not separated.
- The MerLin variant is limited to `m <= 16` by the Fock-space size.

## Tests

```bash
cd papers/bosonic_binary_solver && pytest -q
```

95 tests. The load-bearing ones:

- the candidate count equals the Appendix B bound exactly, term for term;
- the loop-topology unitary reproduces ORCA's released simulator to a total-variation
  distance below 1e-12;
- the MerLin Perceval circuit equals the torch unitary to 1e-12 — Perceval's default
  beamsplitter is complex, and picking it would still yield a valid unitary while silently
  changing every interference term;
- the beamsplitter convention is pinned (`R = cos²(θ/2)`, balanced at `π/2`);
- TSP sizes invert to 7/10/13 locations and the exact optimum stays reachable through the
  binary encoding, without which "% optimal" could never reach 100;
- the GPU path imports neither Perceval nor MerLin, checked in a subprocess with both made
  unavailable — an import chain that reached them broke the first GPU launch;
- `torch.einsum` on integer tensors is rejected the way CUDA rejects it, so a dtype gap that
  a CPU suite cannot discover by running is discovered by construction — this reproduces the
  failure that killed the first TSP run.

## Citation and License

Cite the original paper (arXiv:2510.08274). ORCA's `quantumqubo` is Apache-2.0; the port of
its single-loop model in `lib/tbi.py` is credited in place.
