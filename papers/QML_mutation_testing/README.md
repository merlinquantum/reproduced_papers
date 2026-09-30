# Efficient Mutation Testing of Quantum Machine Learning Models — photonic (MerLin) adaptation

> **Status: complete photonic adaptation, partial support for the paper's
> central claim.** All seven mutation operations, the prior-operator baseline,
> a null control and fair classical baselines run end to end over three seeds.
> The paper's *direction* is reproduced (directed operators beat prior
> operators in every seed on every comparative statistic); its *magnitude* is
> not (+0.055 mutation-score gap here against +0.357 in the paper), because the
> binary kill criterion saturates under this photonic mapping. See
> [Results](#results-obtained-and-comparison-with-the-paper).

## Reference and Attribution

- Paper: *Efficient Mutation Testing of Quantum Machine Learning Models*
- Authors: Emma Andrews, Prabhat Mishra (University of Florida)
- arXiv: 2605.00107v1 (30 April 2026), quant-ph
- Original repository: **none found** — the paper references no artifact,
  repository, or data-availability statement.
- This directory contains an independent re-implementation. All credit for the
  method belongs to the original authors.

## Original Paper

Mutation testing injects faults into a program to check that a test suite can
detect them. The paper carries the idea over to quantum machine learning. It
argues that the five classical quantum-circuit mutation operators (add gate,
delete gate, change gate, add measurement, delete measurement, as used by
Muskit and QMutPy) are too generic for QNNs, because a QNN has a specific
structure: a feature map that loads data into parameterized gates, and a
layered ansatz whose parameters are the learned weights.

It therefore defines **seven QML-specific mutation operations**:

| Op | Name | Target | What it does |
|---|---|---|---|
| APC | Ansatz Parameter Change | learned weights | Mutates a whole layer's rotation angles at once: zeroing, sign flip, addition of a perturbation, scaling. Perturbations are *directed* — fixed radians such as pi/2 and pi rather than Gaussian draws, because theta enters the unitary through cos(theta/2), sin(theta/2). theta = 2*pi is skipped as redundant with 0. |
| DFC | Data-sample Feature Change | input features | Mutates a *pair* of features feeding the feature map: add a fixed value, multiply by a fixed value, sign flip, one-minus. Goal: break the entanglement and symmetry of the feature map. For images: crop, rotate, flip. |
| APGC | Ansatz Parameterized Gate Change | gate type | Replaces a parameterized gate by another one *keeping the parameter value*, directed so that the replacement has a different effect (RX/RY change amplitude, RZ changes phase). |
| LS | Layer Shuffling | layer order | Swaps two ansatz layers. May produce *incompetent* mutants, which are counted separately and excluded from the mutation score. |
| ILS | In-Layer Shuffling | block order inside a layer | Swaps the blocks within one layer, e.g. entangling gates before rotations. |
| ALA | Ansatz Layer Addition | architecture | Inserts an extra layer at any position after the existing ones; new parameters are either random or copied from another layer, so that the mutant stays competent. |
| ALD | Ansatz Layer Deletion | architecture | Deletes a layer — the whole layer, the rotations only, or the entanglers only — together with its learned weights. |

The test suite is the set of the 20 evaluated test samples that the *original*
model classifies correctly; a mutant is *killed* when it changes one of those
predictions. Mutation score is killed / total.

Headline numbers for the configuration targeted here (Iris, ZFeatureMap,
RealAmplitudes): mutation score **71.59%** for the seven new operators against
**35.92%** for the prior operators, with 1,584 mutants instead of 13,448.

## Reproduction Scope (including Updates and Deviations)

### What this is

A **photonic adaptation**, not a Qiskit reproduction. The reproduction was
directed to use `merlinquantum` exclusively; Qiskit, Qiskit Machine Learning
and Qiskit-Aer (the paper's stack) are not installed and not used. The paper's
gate-model constructs are mapped onto linear optics as follows.

| Paper construct | Photonic / MerLin counterpart |
|---|---|
| n features -> n qubits | n features -> n **modes** |
| `H` column opening the ZFeatureMap | fixed 50:50 beam-splitter column |
| `P(2*x_i)` encoding gate | phase shifter of angle `encoding_scale * x_i` |
| `RY(theta_i)` column of RealAmplitudes | trainable phase-shifter column |
| CNOT entangler of RealAmplitudes | **non-trainable** beam-splitter mesh |
| computational-basis readout | photon-counting probabilities, unbunched space |
| label from bitstring counts | `LexGrouping(6, 3)` then argmax |
| 1,024 shots, most frequent result | `layer(x, shots=1024)` then argmax |

The entangler is deliberately parameter-free so that the trainable budget stays
`n_modes * n_layers`, exactly the parameter count of
`RealAmplitudes(n_qubits, reps = n_layers - 1)`. This is asserted by
`tests/test_smoke.py::test_iso_parameter_with_real_amplitudes`.

### Deviations

| ID | Deviation | Reason |
|---|---|---|
| D1 | Every trainable phase column is followed by a mixing column, whereas `RealAmplitudes` ends on rotations | A trailing phase column is invisible to photon counting: mode phases do not change photon-number statistics |
| D2 | Entanglers are fixed rather than trainable | Preserves the iso-parameter budget |
| D3 | 2 photons in 4 modes (6-outcome space) instead of 4 qubits (16-dimensional space) | The photonic policy requires at least 2 photons; the outcome space is consequently smaller and less expressive |
| D4 | No Qiskit and no QASM3 export | Directed scope |
| D5 | Python 3.12.3, not the paper's 3.13.5 | Container image |

### What this means for the paper's numbers

Mutation counts and mutation scores are a function of the available gate set
and the circuit's structure. Under a photonic mapping the mutant inventory of
each operator is necessarily different, so **Table I's absolute values are not
reproducible here by construction**. What remains testable is the structural
and comparative content: whether the seven operators have meaningful photonic
counterparts, how their mutation scores rank, and how many mutants they
generate relative to component-level baseline operators.

### Implemented so far (Iteration 1)

- Iris data pipeline with the paper's 20-sample test-suite protocol.
- Photonic ZFeatureMap + RealAmplitudes analogue as a MerLin `QuantumLayer`.
- Training, analytic and 1,024-shot evaluation, multi-seed reporting.
- Pre-declared hyperparameter grid (the paper specifies none) with full sweep
  evidence.

### Not implemented

- The mutation engine (circuit-structure representation, mutant evaluation
  harness, mutation-score computation).
- The seven mutation operations.
- The prior-operator baseline (Table II).
- Any classical baseline.

## Install and How to Run

```bash
pip install -r requirements.txt
```

Everything needed is already present in the project Docker image
(`merlinquantum` 0.4.1, `perceval-quandela` 1.2.4, torch, scikit-learn).

```bash
# From the repository root — full reproduction: train + 303 directed mutants
# + 12 control mutants + 697 prior-operator mutants, 3 seeds (~11 min)
python implementation.py --paper QML_mutation_testing --config configs/mutation_testing.json

# Original (unmutated) model only, 3 seeds (~11 s)
python implementation.py --paper QML_mutation_testing

# APC granularity ablation (paper prose reading), 3 seeds (~2 min)
python implementation.py --paper QML_mutation_testing --config configs/apc_layer_ablation.json

# Rebuild every table and figure in results/ from a run directory
python utils/make_report.py --run-dir outdir/<run>

# From inside this folder
python ../../implementation.py --config configs/defaults.json

# Quick smoke run
python ../../implementation.py --epochs 5 --shots 128

# Hyperparameter grid (36 runs, ~2 min)
python utils/select_hyperparameters.py

# Tests (TMPDIR must be container-local, see FEEDBACK.md)
TMPDIR=/home/agent/.tmp_pytest pytest tests -q
```

## Configuration

`cli.json` is the authoritative CLI schema: `--epochs`, `--lr`, `--layers`,
`--photons`, `--shots`, plus the global runtime flags (`--config`, `--outdir`,
`--seed`, `--log-level`, ...).

Key entries of `configs/defaults.json`:

| Key | Value | Note |
|---|---|---|
| `model.params.n_modes` | 4 | one mode per Iris feature |
| `model.params.n_photons` | 2 | minimum for a non-trivial photonic model |
| `model.params.n_layers` | 6 | selected on validation accuracy |
| `model.params.input_state` | `[1, 0, 1, 0]` | photons spread across the trainable modes |
| `model.params.encoding_scale` | pi | phases land in [0, pi]; avoids 2-pi aliasing |
| `training.epochs` / `lr` | 200 / 0.05 | full batch, Adam |
| `evaluation.shots` | 1024 | the paper's shot budget |
| `experiment.seeds` | `[42, 43, 44]` | model-init replicates |
| `dataset.n_suite_samples` | 20 | the paper's suite size |

## Data

Iris, loaded from `sklearn.datasets.load_iris` (bundled with scikit-learn, no
download). Stratified 60/20/20 split into 96 train / 24 validation / 30 test,
then a class-balanced 20-sample subset of the test split feeds the mutation
suite. Features are min-max scaled to [0, 1] with a scaler fitted on train only.

## Results Obtained and Comparison with the Paper

Original photonic QNN, Iris, 4 modes / 2 photons / 6 layers (24 trainable
phases), mean +/- std over model-init seeds 42/43/44 with the data split fixed.
Run: `outdir/run_20260914-182320/` (`results/source_run.txt` records which run
every table below was built from).

| Metric | Value | Per seed (42 / 43 / 44) |
|---|---|---|
| Train accuracy (analytic) | 0.809 +/- 0.060 | 0.844 / 0.740 / 0.844 |
| Validation accuracy (analytic) | 0.806 +/- 0.024 | 0.792 / 0.833 / 0.792 |
| **Test accuracy (analytic)** | **0.844 +/- 0.096** | 0.900 / 0.733 / 0.900 |
| Test accuracy (1,024 shots) | 0.811 +/- 0.069 | 0.867 / 0.733 / 0.833 |
| Mutation test-suite size (of 20) | 15.67 +/- 0.58 | 16 / 15 / 16 |
| Training wall-clock per seed | 2.7 s | |

The paper never reports the accuracy of its original models, so this table has
no counterpart in it. Fair classical baselines on the same split
(`results/accuracy.csv`):

| Model | Parameters | Test accuracy |
|---|---:|---|
| Photonic QNN (this adaptation) | 24 | 0.844 +/- 0.096 |
| Linear softmax | 15 | 0.922 |
| MLP, 3 hidden units | 27 | 0.967 |
| Majority class | 0 | 0.333 |

The photonic QNN is well above chance and well below an iso-parameter
classical classifier. This is context, not a paper claim: the paper makes no
accuracy claim. It matters here only because the *quality of the original
model sets the noise floor of every shot-based mutation score* (see the null
control below).

### Mutation testing — the paper's central claim

Three seeds, one data split, 1,024 shots, suite of 15.67 +/- 0.58 samples.
Sources: `results/mutation_scores.csv`, `results/comparison.csv`,
`results/mutation_scores.png`.

| Family | Mutants | MS (1,024 shots) | MS (analytic) | Behavioural redundancy |
|---|---:|---|---|---|
| Seven directed operators | 303 | 0.9516 +/- 0.0188 | 0.9384 +/- 0.0697 | 0.182 +/- 0.050 |
| Prior operators (ADD/DELETE/CHANGE) | 697 | 0.9330 +/- 0.0274 | 0.8833 +/- 0.1040 | 0.582 +/- 0.042 |
| **Null control (identity mutants)** | 12 | **0.2500 +/- 0.4330** | **0.0000** | — |

Paired per seed, which cancels model-quality variance:

| Quantity (directed - prior) | Value | Per seed |
|---|---|---|
| Binary MS, 1,024 shots | +0.0185 +/- 0.0227 | +0.008 / +0.003 / +0.045 |
| Binary MS, analytic | +0.0551 +/- 0.0351 | +0.028 / +0.095 / +0.042 |
| Fraction of suite samples flipped | +0.0963 +/- 0.0419 | +0.050 / +0.131 / +0.108 |
| Behavioural redundancy (prior - directed) | +0.400 +/- 0.021 | +0.379 / +0.399 / +0.422 |

Against the paper's Iris / ZFeatureMap / RealAmplitudes row:

| Quantity | Here | Paper | Agreement |
|---|---|---|---|
| Mutants, directed / prior | 303 / 697 | 1,584 / 13,448 | not comparable in absolute terms (different gate inventory) |
| Mutant reduction factor | 2.30x | 8.49x | same direction, ~3.7x smaller |
| MS, directed operators | 0.938 (analytic) | 0.716 | higher here |
| MS, prior operators | 0.883 (analytic) | 0.359 | much higher here |
| **MS separation** | **+0.055** | **+0.357** | **trend reproduced, magnitude not** |
| Generation time per mutant | 0.00057 s / 0.00045 s | 0.00025 s / 0.00032 s | same order, both far below the paper's 1 s bound |

**What this means.** The binary kill criterion saturates in this photonic
mapping: with a 15-16 sample suite and a 6-outcome space, almost any change to
a 4-mode interferometer flips at least one prediction, so both families land
near 0.9 and the metric loses its discriminating power. The paper's ranking
still holds in every seed and on every comparative statistic, and the
redundancy gap (58% of prior mutants are behavioural duplicates, against 18%
of directed ones) supports the efficiency argument more strongly than the
mutation score does.

**Null control — a caveat that applies to the paper itself.** Identity mutants
were killed at 0.25 +/- 0.43 under the paper's shot-based criterion: 0.00 for
seeds 42 and 44, **0.75 for seed 43**, the weak model (test accuracy 0.733).
When suite predictions are low-margin, 1,024-shot sampling noise alone kills
three quarters of unmutated circuits. The paper reports no such control, so an
unknown fraction of its shot-based mutation scores may be sampling noise
rather than fault detection. Under the analytic protocol the false-kill rate
is 0 by construction.

### APC granularity ablation

The paper's prose mutates "a layer at once"; its Table I counts require one
gate at a time. Both readings are implemented
(`configs/apc_layer_ablation.json`, run `outdir/run_20260914-183447/`):

| APC granularity | APC mutants | APC MS (analytic) |
|---|---:|---|
| gate (default, matches Table I counts) | 120 | 0.886 +/- 0.155 |
| layer (paper prose) | 30 | 0.589 +/- 0.019 |

Exactly 12 of the 30 layer-granularity mutants survive in every seed, and they
are always the same twelve: the additive operations (`+pi/2`, `+pi`) applied to
a whole phase column. A uniform phase on all modes is a **global phase**, which
photon counting cannot detect, so those are *equivalent mutants* rather than
surviving ones. Read at layer granularity, the paper's APC operator would
inject 40% undetectable mutants on any photonic backend, and the paper
performs no equivalent-mutant detection.

### Hardware-aware report

| Field | Value |
|---|---|
| Computation space | UNBUNCHED |
| Detector model | threshold |
| Photon number | 2 |
| Number of modes | 4 |
| Input state | `[1, 0, 1, 0]` |
| Encoding | angle encoding on modes 0-3, scale = pi rad, preceded by a fixed 50:50 beam-splitter column |
| Measurement strategy | `MeasurementStrategy.probs(computation_space=UNBUNCHED)` + `LexGrouping(6, 3)` |
| Postselection | none |
| Simulator / QPU | MerLin CPU simulator (Perceval backend), analytic for training |
| Shot count | 1,024 for suite predictions; analytic (0) during training |
| Wall-clock time | training 2.7 s per seed; full mutation run 644.6 s for 3 seeds x 1,012 mutants; APC ablation 125.6 s |
| Seeds | 42, 43, 44 (model init); 42 (data split) |

### Hyperparameter selection

The paper specifies no optimizer, learning rate, epoch count, or ansatz depth.
A pre-declared grid over `n_layers` in {2, 3, 4, 6} and `lr` in
{0.02, 0.05, 0.10}, three seeds per candidate, 36/36 runs completed
(`outdir/sweeps/hparam-8170d640/`), selected on mean **validation** accuracy:

| Candidate | Params | Val acc (mean) | Test acc (mean) |
|---|---|---|---|
| **L6_lr0.05 (selected)** | 24 | 0.806 | 0.844 |
| L6_lr0.1 (tied) | 24 | 0.806 | 0.856 |
| L6_lr0.02 | 24 | 0.792 | 0.822 |
| L4_lr0.1 | 16 | 0.764 | 0.889 |
| ... | | | see `sweep_summary.csv` |

`L6_lr0.05` and `L6_lr0.1` are tied within the declared 0.005 tolerance; the
declared tie-break (fewer layers, then smaller lr) picks `L6_lr0.05`. With only
24 validation samples, differences below roughly 0.04 are within sampling
noise: this grid does **not** establish that 6 layers is statistically superior
to 4.

## Verdict

| Claim | Metric agreement | Trend agreement | Support |
|---|---|---|---|
| C1 — seven operators definable on a layered QNN | n/a | n/a | supported |
| C2 — directed operators score much higher than prior ones | no (+0.055 vs +0.357) | yes, in every seed | **partially supported** |
| C3 — up to 9.5x fewer mutants | no (2.30x) | yes | **partially supported** |
| C4 — generation below 1 s per mutant | yes (0.57 / 0.45 ms) | yes | supported |
| C5 — suite = correctly classified subset of 20 samples | n/a | n/a | supported |

Overall: **partially supported** as a cross-modality adaptation. The paper's
qualitative position survives the move to linear optics; its quantitative
separation does not, for a measured reason (saturation of the binary kill
criterion). Two findings run against the paper's protocol rather than merely
failing to confirm it: the uncontrolled false-kill rate of its shot-based
criterion, and the equivalent mutants its APC operator produces at layer
granularity.

- Reproducibility confidence: **MEDIUM**
- Implementation confidence: **HIGH**
- Validity tier: **V3** (substitute-architecture reproduction on the paper's
  own dataset, task and protocol)
- Photonic status: **`PARTIAL_MERLIN_TRANSLATION`**

## Limitations

- **Trend agreement without metric agreement.** The paper's mutation-score
  separation is reproduced in direction but not in magnitude; the binary
  criterion saturates here, so this reproduction cannot confirm the paper's
  71.59% / 35.92% numbers, and it is not designed to (the photonic mutant
  inventory differs by construction).
- The shot-protocol results carry a measured false-kill rate of 0.25 +/- 0.43.
  Only the analytic protocol is noise-free. Single-seed shot-based mutation
  scores — the paper's setting — should not be trusted without a control.
- Three model-initialisation seeds and a single data split: enough to show
  that the binary-MS gap is inside seed variance, not enough for a formal
  significance test.
- Accuracy (0.844 +/- 0.096) is modest for Iris and the seed spread is large
  (seed 43 reaches only 0.733). Suspected causes: the small 6-outcome space
  from 2 photons in 4 modes, the rigid `LexGrouping` readout, and the
  non-trainable entangling mesh imposed by the iso-parameter constraint.
- Absolute agreement with Table I is out of reach under the photonic mandate;
  only structural and comparative agreement can be assessed.
- Table I of the paper is internally inconsistent for 4 of its 12 QNN rows
  (`sum(killed)/sum(total)` does not equal the printed mutation score). The row
  targeted here, Iris/ZFM/RA, *is* consistent (1,134/1,584 = 71.59%). Details in
  `LOG.md`.
- Single data split; only model-initialisation seeds are varied.

## Tests

18 tests. `tests/test_mutations.py` asserts the mutant-count formula of every
operator, the suppression of provably redundant LS variants, and that mutated
circuits remain executable. `tests/test_cli.py` checks the CLI schema and that the defaults describe a
valid photonic model (>= 2 photons, 20-sample suite, >= 3 seeds).
`tests/test_smoke.py` checks the iso-parameter property against
`RealAmplitudes`, the output being a normalised class distribution, rejection of
single-photon circuits, a short end-to-end training run with suite
construction, and that the runner writes its contract artifacts.

## Citation and License

Cite the original paper for the method. This reproduction inherits the
repository licence (see `/reproduced_papers/LICENSE`).
