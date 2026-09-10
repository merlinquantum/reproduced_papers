# Fourier Fingerprints

Photonic reproduction of the Fourier fingerprint construction and the Fourier
coefficient correlation (FCC) metric.

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

This reproduction is a **photonic translation**. The construction is carried over
to linear optics with MerLin, and the four circuit topologies studied here are
photonic ones rather than the paper's gate-model ansatzes, so individual FCC
values are not expected to match any specific number in the paper.

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

**Not reproduced:**

- The paper's central predictive claim, that lower FCC implies lower MSE when
  learning random Fourier series (Section 3.1). Testing it requires training the
  models against random Fourier targets and correlating the resulting error with
  FCC. Nothing in this reproduction is trained; parameters are only sampled.
- The high-energy-physics jet reconstruction of Section 3.2.
- The paper's specific ansatzes, which have no linear-optical counterpart here.

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
- `lib/runner.py`: dispatches a configuration to the shared implementation.
- `utils/summarize_fcc.py`: sweeps every dimension, encoding and circuit and
  curates the FCC table and figures into `results/`.
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

Not yet recorded. `utils/summarize_fcc.py` sweeps every dimension, encoding and
circuit and writes the FCC table and figures into `results/`; the table and its
discussion belong here once that sweep and the Section 3.1 study below have been
run together.

The open question this reproduction is set up to answer is whether the paper's
relationship between FCC and model error survives the move to linear optics. The
fingerprint machinery is in place; what is missing is training the same four
circuits against random Fourier targets and checking whether the ansatz with the
lowest FCC also achieves the lowest MSE.

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

- The predictive claim that motivates the FCC metric is not tested here. Without
  it, the reproduction demonstrates that fingerprints can be computed for
  photonic circuits but not that they are useful for choosing one.
- The four circuit topologies are photonic and do not correspond to the paper's
  gate-model ansatzes, so no individual FCC value has a counterpart in the paper.
- Only four active modes and a single reference mode are used, so the accessible
  frequency set is small compared with the six-qubit models of the paper.
- The active frequencies are identified from the empirical variance of the
  coefficients rather than from the feature-map spectrum, so the frequency set
  depends on the sampling budget.
- 200 parameter draws is well below the paper's budget; the fingerprints are
  stable at this size but the FCC values carry more sampling noise.
- The Fourier analysis samples random trainable parameters, so numerical results
  vary with the seed. Runs are reproducible for a fixed seed.
