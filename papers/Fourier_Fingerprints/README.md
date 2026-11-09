# Fourier Fingerprints

A photonic implementation of the Fourier fingerprint, a diagnostic for choosing
between parameterised circuits, and of the Fourier coefficient correlation (FCC)
metric derived from it.

## Reference and Attribution

- Paper: *Fourier Fingerprints of Ansatzes in Quantum Machine Learning*
- Authors: Melvin Strobl, M. Emre Sahin, Lucas van der Horst, Eileen Kuehn,
  Achim Streit, Ben Jaderberg
- arXiv: [2508.20868](https://arxiv.org/abs/2508.20868) (v1, August 2025)
- DOI: [10.48550/arXiv.2508.20868](https://doi.org/10.48550/arXiv.2508.20868)
- Original repository: none advertised in the paper.
- License: this reproduction is released under the repository licence; please
  cite the original paper when re-using the code or results.

## Original Paper

A quantum Fourier model (QFM) encodes a classical input through a feature map
and reads out an expectation value, and its output is a truncated Fourier series
in that input. The number of Fourier basis functions grows as O(exp(n)) in the
qubit count, while a model that is efficiently trainable can only carry O(poly(n))
parameters. The coefficients therefore cannot be controlled independently: they
must share parameters, which induces correlations between them.

The paper measures those correlations directly. For many random draws of the
ansatz parameters it computes the Fourier coefficients of the model output, then
takes the Pearson correlation between each pair of frequencies across the draws.
The resulting matrix, displayed as a lower triangle, is the *Fourier
fingerprint*, and it is characteristic of the ansatz. Averaging its magnitude
gives the Fourier coefficient correlation (paper Eq. 5):

```text
FCC = (1 / |Ω|) * Σ_{ω, ω' ∈ Ω} |r(ω, ω')|
```

The paper's headline result is that this metric predicts ansatz performance. On
the task of learning random Fourier series, ansatzes with lower FCC achieve lower
mean squared error, with FCC correlating almost linearly with MSE across the
ansatzes tested, while the widely used expressibility metric fails to rank them
correctly. The same relationship is then shown to hold for 2D jet reconstruction
in high-energy physics.

The paper's own experiments use gate-model circuits: six qubits, one layer, a
Pauli-Y feature map, and a Pauli-Z observable, evaluated over named ansatzes
including the hardware-efficient ansatz and Circuits 15 to 19 of Sim et al.

## Reproduction Scope and Deviations

This reproduction is a **photonic translation**. The paper presents the
fingerprint as a tool for ansatz selection, so what is carried over is the tool:
the construction and the metric, applied to linear-optical circuits with MerLin.
The four topologies studied here are photonic rather than the paper's gate-model
ansatzes, so individual FCC values have no counterpart in the paper and are not
expected to match one.

**Reproduced:**

- The fingerprint construction: random parameter draws, Fourier coefficients of
  the measured signal, Pearson correlation between frequency pairs, and the
  lower-triangular display used in paper Figure 2.
- The FCC metric exactly as defined in Eq. 5, as the mean absolute off-diagonal
  correlation.
- The 1D setting of Figure 2(a) and the 2D extension of Figure 2(b), the latter
  restricted to the frequency set Ω_n = {(ω1, ω2) : |ω1| + |ω2| ≤ n_omega}.
- Three input encodings, so the effect of the encoding scale factors on the
  accessible frequency set can be read off directly.

- The Section 3.1 experiment, learning random Fourier series, is implemented and
  run, as a characterisation of how the diagnostic behaves on photonic circuits.
  FCC does not rank the error on this circuit set, because the four topologies
  differ too much in capacity for a structural metric to show through. See
  "Results Obtained".

**Not reproduced:**

- The high-energy-physics jet reconstruction of Section 3.2.
- The paper's specific ansatzes, which have no linear-optical counterpart here.
- A capacity-matched circuit set, which is what a clean test of the FCC claim
  would need.

**Deviations:**

- **Observable.** The paper reads out a Pauli-Z expectation value. The photonic
  model has no such observable, so the signal is the probability that output
  mode 0 contains at least one photon.
- **Sampling budget.** The paper uses 500·|θ|·2ⁿ·D parameter draws. This
  reproduction uses a fixed 200 draws per circuit, which is enough for the
  fingerprint structure to be stable but is a smaller budget.
- **Frequency set.** The paper fixes Ω = ⟦−nL, nL⟧^D from the feature-map
  spectrum. Here the active frequencies are identified empirically, as those
  whose coefficient magnitude has non-negligible variance across draws.

## Project Layout

- `implementation.py`: thin wrapper that delegates to the repository-wide runner.
- `generate_circuit_schematics.py`: builds the four circuits and saves their
  Perceval schematics with `pdisplay_to_file`.
- `configs/`: JSON configurations, one file per experiment.
- `lib/fourier.py`: model, Fourier analysis, and fingerprint plots for both
  dimensions.
- `lib/learning.py`: random Fourier targets, the trainable regressor, and the
  training loop used for the Section 3.1 study.
- `lib/runner.py`: dispatches a configuration to the shared implementation.
- `utils/summarize_fcc.py`: sweeps every dimension, encoding and circuit and
  curates the FCC table and figures into `results/`.
- `utils/fcc_vs_mse.py`: trains every circuit on random Fourier series and
  compares the resulting error against FCC.
- `results/`: curated schematics and experiment outputs.
- `outdir/`: timestamped figures produced by the experiment runner.
- `tests/`: validation tests for the model, metric, configs, and runner.

## Environment Setup

```bash
pip install -r papers/Fourier_Fingerprints/requirements.txt
```

Main runtime dependencies are `torch`, `matplotlib`, `merlinquantum`,
`perceval-quandela`, and `pytest`.

## How the Code Works

### Model

`PhotonicSpectralModel` uses five optical modes: four active modes carrying the
input encoding, and one reference mode. The input is broadcast across the active
modes and multiplied by the selected scale factors before angle encoding. In 1D
the scalar input goes to all four modes; in 2D the first coordinate goes to modes
0 and 1 and the second to modes 2 and 3.

```text
1D linear:      [x, 2x, 3x, 4x]
1D exponential: [x, 2x, 4x, 8x]
1D balanced:    [-2x, -x, x, 2x]
```

The 1D model uses two photons and the 2D model three. The four circuit
topologies are selected with `circuit_index` from `0` to `3` and differ in how
entangling layers are arranged around the encoding.

### Measured signal

The measured signal is the probability that output mode 0 contains at least one
photon. The relevant columns of the Fock probability tensor are selected from the
layer's own basis ordering rather than from a fixed column range:

```python
mask = [state[0] >= 1 for state in quantum_layer.output_keys]
signal_y = probs_out[:, mask].sum(dim=1)
```

The mask matters because the size of the Fock space depends on the photon number.
With five modes and two photons (1D) there are 15 basis states, of which the
first five have a photon in mode 0. With five modes and three photons (2D) there
are 35 basis states and 15 of them qualify, so a fixed five-column slice would
measure the probability of *two or more* photons in mode 0 instead.

### Fourier analysis

For each of 200 random parameter draws the circuit is evaluated on a regular grid
over the input domain: 64 points in [0, 2π) in 1D, and a 32 × 32 grid in 2D. The
signal is transformed with `np.fft.rfft` in 1D and `np.fft.fftn` in 2D, and the
absolute coefficient amplitudes are collected across draws.

Frequencies whose amplitude has negligible variance across draws are discarded.
In 2D the remaining frequencies are further restricted to |ω1| + |ω2| ≤ `n_omega`.
The Pearson correlation matrix between the surviving frequency columns is the
fingerprint, and its mean absolute off-diagonal entry is the FCC.

## Running Experiments

Run from the repository root through the shared runner:

```bash
# Default configuration
python implementation.py --paper Fourier_Fingerprints

# A specific 1D configuration
python implementation.py --paper Fourier_Fingerprints --config configs/1D_exp.json

# A specific 2D configuration
python implementation.py --paper Fourier_Fingerprints --config configs/2D_exp.json

# Override the random seed
python implementation.py --paper Fourier_Fingerprints --config configs/defaults.json --seed 123
```

`bash run.sh` executes every configuration in turn. The seed from the config, or
from `--seed`, is applied by the shared runner before the circuit parameters are
sampled, so a given seed reproduces a given set of FCC scores.

Each run writes its fingerprint figure to a timestamped folder under `outdir/`.

## Results Obtained

The paper presents the Fourier fingerprint as a diagnostic tool for ansatz
choice. What this reproduction delivers is that tool, working on linear-optical
circuits, together with what it reports about the photonic ansatzes available
here. Numbers below come from `utils/summarize_fcc.py` and `utils/fcc_vs_mse.py`,
which write `results/fcc_summary.json` and `results/fcc_vs_mse.json`.

### The diagnostic on photonic circuits

The construction carries over without modification. Every combination of
dimension, encoding and circuit produces a well-formed fingerprint and an FCC in
the range 0.14 to 0.32, computed by the same Eq. 5 the paper uses.

![FCC by encoding](results/fcc_by_encoding.png)

Read as a diagnostic, the tool separates these circuits. In 1D, `circuit_1`
consistently shows the least coefficient coupling and `circuit_2` the most; in 2D
the ordering inverts, with `circuit_2` lowest. A tool that returned the same
number for every topology would be useless, and this one does not.

### The encoding sets the accessible spectrum

The clearest result here, and the one free of confounds.

![Accessible frequencies](results/active_frequencies.png)

Exponential scale factors reach frequencies up to 31 against 15 for the linear
ramp, and `circuit_2` reaches more frequencies than any other topology under
every encoding, because it applies the encoding twice. The reachable spectrum is
therefore set by the encoding and the number of encoding layers, not by the
entangling structure. The 2D panel is omitted because the active count there is
fixed at 25 by the |ω1| + |ω2| ≤ `n_omega` cutoff and carries no information.

### What the diagnostic says about learning error

The paper validates FCC as a predictor by training its ansatzes on random Fourier
series and showing that lower FCC accompanies lower error. Repeating that here
characterises how the tool behaves on photonic circuits. Each circuit was trained
on five random Fourier targets per encoding, with all four circuits seeing the
same targets so the comparison is paired. Targets use the frequency set the
circuits can reach and are standardised to unit variance, so an MSE of 1.0 is
what a constant predictor achieves.

| Encoding | Circuit | FCC | MSE | Params | Active freqs |
|---|---|---:|---:|---:|---:|
| linear | circuit_0 | 0.2152 | 0.677 ± 0.059 | 40 | 9 |
| linear | circuit_1 | 0.1894 | 0.895 ± 0.064 | 12 | 7 |
| linear | circuit_2 | 0.2609 | **0.287 ± 0.126** | 60 | 16 |
| linear | circuit_3 | 0.2752 | 0.891 ± 0.062 | 8 | 9 |
| exponential | circuit_0 | 0.1792 | 0.720 ± 0.015 | 40 | 17 |
| exponential | circuit_1 | 0.1723 | 0.917 ± 0.023 | 12 | 12 |
| exponential | circuit_2 | 0.2366 | **0.421 ± 0.053** | 60 | 31 |
| exponential | circuit_3 | 0.2448 | 0.919 ± 0.040 | 8 | 17 |
| balanced | circuit_0 | 0.1864 | 0.665 ± 0.062 | 40 | 9 |
| balanced | circuit_1 | 0.1958 | 0.882 ± 0.052 | 12 | 8 |
| balanced | circuit_2 | 0.2240 | **0.373 ± 0.087** | 60 | 16 |
| balanced | circuit_3 | 0.1968 | 0.886 ± 0.057 | 8 | 8 |

![FCC against learning error](results/fcc_vs_mse.png)

On this circuit set FCC does not rank the error. Spearman coefficients are −0.40
(linear), +0.20 (exponential) and −0.20 (balanced), −0.22 pooled, where the
paper's validation gives a positive coefficient. The reason is visible in the
table: these four topologies carry between 8 and 60 trainable parameters, and
ranking the same twelve points by parameter count gives Spearman **−0.881**
against **−0.224** for FCC. Capacity accounts for nearly the whole ordering.

Two capacity-comparable pairs confirm it. The two smallest circuits, `circuit_1`
(12 parameters) and `circuit_3` (8), reach almost identical error, 0.898 and
0.899 averaged over encodings, although their FCC differs by 28 %. Between the
two largest, `circuit_0` (40) and `circuit_2` (60), the higher FCC goes with the
lower error.

This says nothing against the metric. The paper compares ansatzes at fixed qubit
count and comparable size, which is the regime where a structural diagnostic can
be informative; the four topologies shipped here are not matched that way, so
capacity swamps whatever signal FCC carries. Establishing whether FCC predicts
photonic ansatz performance requires a capacity-matched circuit set, which the
tool is now in place to evaluate.

### Summary

The diagnostic works on linear optics, discriminates between topologies, and
gives a clean reading of how the encoding controls the accessible spectrum. Using
it to *select* a photonic ansatz remains open, and needs a circuit family
designed so that FCC is the only quantity varying.

## Configuration

Configurations are JSON files in `configs/`, one per experiment. The main fields
are:

- `seed`: random seed used by the experiment.
- `outdir`: base directory for generated experiment figures.
- `graph_name`: name of the saved fingerprint figure.
- `circuits`: list of circuits to evaluate.
- `encoding`: `linear`, `exponential`, or `balanced`.
- `dimension`: `1` or `2`.

The sampling parameters, namely the number of parameter draws, the grid
resolution and the 2D frequency cutoff, are module-level constants in
`lib/fourier.py` rather than config keys, since the convention in this repository
is one config file per experiment.

Example:

```json
{
  "seed": 42,
  "outdir": "outdir",
  "graph_name": "Fig 2(a) - Exponential encoding in 1D",
  "circuits": ["circuit_0", "circuit_1", "circuit_2", "circuit_3"],
  "encoding": "exponential",
  "dimension": 1
}
```

## Circuit Schematics

The four circuit schematics generated with Perceval are stored in
`results/circuits/`:

| Circuit | Schematic |
|---|---|
| `circuit_0` | ![Circuit 0](results/circuits/circuit_0.png) |
| `circuit_1` | ![Circuit 1](results/circuits/circuit_1.png) |
| `circuit_2` | ![Circuit 2](results/circuits/circuit_2.png) |
| `circuit_3` | ![Circuit 3](results/circuits/circuit_3.png) |

Regenerate them with:

```bash
cd papers/Fourier_Fingerprints
python generate_circuit_schematics.py
```

The script writes to `results/circuit_0.png` through `results/circuit_3.png`. The
copies shown above are the curated schematics kept in `results/circuits/`.

## Tests

```bash
cd papers/Fourier_Fingerprints
pytest -q
```

Eleven tests, about five seconds. They cover the project markers the shared
runner requires, the schema of every configuration, the mode-0 mask against the
Fock basis in both dimensions, the measured signal being a valid probability, an
end-to-end fingerprint run producing a square symmetric correlation matrix, seed
reproducibility, runner dispatch, and rejection of invalid inputs.

## Limitations

- The four circuit topologies carry between 8 and 60 trainable parameters, so the
  learning study compares circuits of very different capacity. Parameter count
  ranks the error far better than FCC does. The diagnostic is therefore delivered
  and working, but this circuit set cannot show whether it is useful for
  selecting between photonic ansatzes; that needs a capacity-matched family.
- Four circuits per encoding is a small basis for a rank correlation. The paper
  compares eight ansatzes. Individual Spearman coefficients over four points
  carry little weight, and the pooled figure mixes encodings whose targets differ
  in difficulty.
- The four circuit topologies are photonic and do not correspond to the paper's
  gate-model ansatzes, so no individual FCC value has a counterpart in the paper.
- The regressor adds a trainable affine head so the circuit's probability output
  can reach the target's range. The head cannot introduce frequencies, but it is
  a departure from the paper, which trains the model output directly.
- Under exponential encoding the reachable frequencies run to 31 against a
  Nyquist limit of 32 for the 64-point grid, so that setting sits close to the
  resolution of the analysis.
- Only four active modes and a single reference mode are used, so the accessible
  frequency set is small compared with the six-qubit models of the paper.
- The active frequencies are identified from the empirical variance of the
  coefficients rather than from the feature-map spectrum, so the frequency set
  depends on the sampling budget.
- 200 parameter draws is well below the paper's budget; the fingerprints are
  stable at this size but the FCC values carry more sampling noise.
- The Fourier analysis samples random trainable parameters, so numerical results
  vary with the seed. Runs are reproducible for a fixed seed.
