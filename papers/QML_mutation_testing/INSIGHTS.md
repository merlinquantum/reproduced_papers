# INSIGHTS.md — Efficient Mutation Testing of QML Models (photonic adaptation)

Durable observations that outlive this paper. Each is backed by an artifact in
this directory; none is speculation.

## 1. A stochastic kill criterion needs a null control, and almost no mutation-testing paper has one

The paper's kill rule is "the mutant changed at least one suite prediction",
evaluated from a 1,024-shot majority vote. Sampling noise alone satisfies that
rule whenever a suite prediction is low-margin.

Measured here with 12 identity mutants per seed (`results/mutation_scores.csv`,
`results/comparison.csv`): false-kill rate **0.25 +/- 0.43** — 0.00 for the two
seeds whose model reaches 0.90 test accuracy, and **0.75** for the seed whose
model reaches 0.733. The noise floor of the metric is therefore set by the
*quality of the model under test*, not by the operators being studied, and it
can consume most of a reported mutation score.

Transferable rule: any reproduction of a shot-based, threshold-style metric
(mutation score, attack success rate, fidelity-drop detection, agreement
rates) should evaluate an identity/no-op control through the identical path
and report its rate next to the metric. It costs a few percent of runtime and
converts an assumption into a measurement.

## 2. Never mix two evaluation protocols in one verdict

The first implementation selected the test suite analytically and judged
mutants from sampled predictions. That produced an *analytic* false-kill rate
of 0.333 on identity mutants — impossible physically, and the signal that
exposed the defect. Each protocol (shot-based, analytic) must own its suite
*and* its verdict end to end; then the analytic false-kill rate is exactly zero
by construction and becomes a free correctness check on the whole harness.

Generalisation: in any reproduction where a quantity is available both exactly
and by sampling, run both paths completely and use the exact path as an
invariant test of the sampled one.

## 3. Saturating metrics hide real effects; keep a graded companion statistic

Both operator families kill 88-95% of mutants here, so the paper's binary
mutation score cannot separate them (+0.019 +/- 0.023 under the shot protocol).
The same data under a graded statistic — the *fraction* of suite samples a
mutant flips — separates them in every seed (+0.096 +/- 0.042), as does
behavioural redundancy (+0.400 +/- 0.021). The paper's qualitative claim
survives only because these companion statistics exist.

Transferable rule: when a metric is a threshold over a count, compute and
report the underlying count as well. It costs nothing at evaluation time and
it is the difference between "no effect observed" and "effect observed, metric
saturated".

## 4. Global phases make "mutate a whole layer" generate equivalent mutants in linear optics

Reading the paper's APC operator at layer granularity (its prose) rather than
gate granularity (its Table I counts), exactly 12 of 30 mutants survive in
every seed, always the same twelve: the additive perturbations (`+pi/2`,
`+pi`) applied to a whole phase column. A uniform phase across all modes is a
global phase and is unobservable in photon counting, so these are *equivalent
mutants*, not surviving ones — 40% of the operator's inventory is undetectable
by construction (`outdir/run_20260914-183447/`).

Transferable rule: when porting a gate-model perturbation scheme to photonics,
enumerate the symmetries of the readout (global phase, mode permutations,
unobservable trailing phase columns) *before* generating perturbations. The
same reasoning shows that a trailing trainable phase column after the last
mixer is invisible to photon counting, which is why the photonic ansatz here
ends on a mixing column.

## 5. Mutation-score comparisons are inventory-dependent and travel badly across modalities

The absolute values of Table I are not a property of the method; they are a
property of the gate set crossed with the circuit structure. Linear optics has
one natural one-mode parameterised element against the gate model's three
rotations, which alone changes every per-operator count. A cross-modality
reproduction of such a paper can only test the *comparative and structural*
content of the claim, and should say so before producing numbers rather than
after.

## 6. Redundancy is a better efficiency metric than mutant count

The paper argues its operators are efficient because they generate fewer
mutants (8.5x fewer). Here the reduction is only 2.3x, but the *behavioural*
redundancy measurement is much more favourable and much more stable across
seeds: 58.2% +/- 4.2 of prior-operator mutants duplicate another mutant's
behaviour on the suite, against 18.2% +/- 5.0 of directed mutants. Counting
mutants measures the generator; counting distinct behaviours measures what the
tester actually gains.
