# Distributed quantum machine learning via classical communication — Reproduction

A reduced, runnable reproduction of Hwang et al.,
"Distributed quantum machine learning via classical communication"
([arXiv:2408.16327](https://arxiv.org/abs/2408.16327)).

## Reference and Attribution

- Paper: *Distributed quantum machine learning via classical communication*,
  K. Hwang, H.-T. Lim, Y.-S. Kim, D. K. Park, Y. Kim (2024).
- arXiv: <https://arxiv.org/abs/2408.16327>
- Original repository: not found (paper does not link one).
- This reproduction is independent and does not redistribute paper code or data.

## Original Paper

The paper proposes a *distributed quantum machine learning* (DQML)
scheme that wires two small quantum processor units (QPUs) together with
**classical communication** — mid-circuit measurements followed by
feedforward operations — instead of full quantum communication. They
benchmark four schemes on an 8-dimensional synthetic binary
classification task:

| Scheme | Description |
|---|---|
| non-DQML | One 4-qubit QPU; embedding repeated to cover the 8 attributes. |
| NC-DQML | Two 4-qubit QPUs, no inter-QPU operations. |
| CC-DQML | Two 4-qubit QPUs with cross-QPU **classical** feedforward gates. |
| QC-DQML | Two 4-qubit QPUs with cross-QPU **quantum** two-qubit gates. |

Each QPU runs a small QCNN: Havlicek-style ZZ embedding, L brick-wall
convolutional sub-layers, two cascaded pooling layers (4 → 2 → 1), and a
trainable *interpret function*
`f = w0 P[00] + w1 P[01] + w2 P[10] + w3 P[11]`
that turns the joint single-bit readouts into a real-valued score
trained against ±1 labels.

The main empirical claim (Table I, Fig. 4c, Fig. 4d) is that **CC-DQML
closely matches QC-DQML** at the tested depths L ≤ 9 and substantially
outperforms NC-DQML, supporting the case that classical communication is
a NISQ-friendly substitute for quantum communication on this benchmark.

## Reproduction Scope (including Updates and Deviations)

### What this reproduction covers

| Item | Status |
|---|---|
| Synthetic 8D dataset generator (Appendix B) | reproduced |
| QCNN circuit blocks (embedding, conv sub-layers, pooling) | reproduced |
| Four DQML schemes (non, NC, CC, QC) | reproduced |
| Interpret-function readout (Eq. 3) | reproduced |
| Validation-accuracy sweep over L ∈ {3, 5, 7, 9} | reproduced (3 seeds) |
| MerLin photonic classifier baseline | added (extension) |
| Iso-parameter classical MLP baseline | added (extension) |

### What is **not** reproduced (and why)

- Table II (parity-readout ablation) — out of scope; testing the
  interpret-function claim requires another full sweep.
- Fig. 3 / Fig. 5 effective-dimension and Fisher-spectrum analyses —
  rank-of-Fisher computations over 500 Haar states are expensive on this
  CPU-only host and the qualitative ordering is already evidenced by the
  accuracy sweep.
- Single-seed protocol with 10 trials — we run **3 seeds** per cell to
  fit within the container's compute budget.

### Updates and deviations (high level)

- **Simulator:** the paper uses PennyLane; we use a direct PyTorch
  state-vector simulator (`lib/simulator.py`) — autograd-compatible
  and faster on these small circuits.
- **Pooling block** reformulated via deferred measurement;
  mathematically identical output distributions.
- **Cross-QPU red block** uses a single CRX (CC) and CRX + CRZ (QC);
  matches the parameter-count ordering and rough magnitudes in
  Fig. 3c.
- **MerLin extension** is a *faithful photonic translation* of the
  paper's four schemes: each QPU becomes a photonic chip, the
  classical-feedforward red block becomes a soft-bit-weighted
  conditional phase shifter on chip 1, and quantum communication
  becomes one larger coherent chip.
- **3 seeds per cell** instead of the paper's 10 trials.

The reasoning behind these choices, the cross-QPU red-block
parameter-count derivation, the photonic recipe details (encoding
scale, photon placement, trainable bit head), and the design
narrative for the photonic side live in
[`ADDITIONAL_NOTES.md`](ADDITIONAL_NOTES.md).

## Install and How to Run

```bash
# from repo root
python -m venv .venv && source .venv/bin/activate
pip install -r papers/distributed_qml_cc/requirements.txt

# smoke test (fastest sanity run)
python implementation.py --paper distributed_qml_cc

# CC-DQML at the paper's L=9, 1000 iterations (~2 min)
python implementation.py --paper distributed_qml_cc \
    --config papers/distributed_qml_cc/configs/classification_original.json \
    --scheme cc --n-layers 9

# Full sweep over (scheme, L, seed) — produces results/sweep/sweep.json
cd papers/distributed_qml_cc
python utils/run_sweep.py --schemes non,nc,cc,qc --layers 3,5,7,9 \
    --seeds 0,1,2 --iterations 1000 --outdir results/sweep

# Figures and Table I
python utils/plot_results.py --sweep results/sweep/sweep.json

# Classical and MerLin baselines
python ../../implementation.py --config configs/classification_classical.json
python ../../implementation.py --config configs/classification_merlin.json
```

## Configuration

Configs are JSON. The schema is described in `cli.json`; key fields:

| Path | Meaning |
|---|---|
| `pipeline` | `quantum`, `classical`, or `merlin`. |
| `model.params.scheme` | `non`, `nc`, `cc`, or `qc`. |
| `model.params.n_layers` | L, number of convolutional sub-layers. |
| `training.n_iterations` | Adam optimisation steps. |
| `training.lr` | Adam learning rate (paper default 0.05). |
| `training.batch_size` | Mini-batch size (paper default 512). |
| `seeds` | List of seeds; the runner trains one model per seed. |

CLI overrides take precedence over config values; the shared runtime
also accepts `--seed`, `--dtype`, `--device`, `--outdir`, and
`--log-level`.

## Data

The dataset is **synthetic** and is regenerated on every run from
`lib.data.build_synthetic_dataset`. The recipe matches Appendix B:

1. Sample 2048 vectors uniformly from the 8D ball of radius π/4.
2. Form 32 clusters of 64 points each. Each cluster is translated by a
   distinct corner of the {±π/4}⁸ hypercube (sampled without
   replacement from the 256 corners).
3. Assign 16 clusters to label +1 and 16 to −1 uniformly at random.
4. Split 1536 train / 512 validation.

The maximum absolute Pearson correlation between any attribute and the
label is 0.18 for seed=42 (paper reports 0.239 for *their* draw).

## Results Obtained and Comparison with the Paper

### Headline table (Table I reproduction)

Validation accuracy at 1000 training iterations, mean ± std over 3
seeds. Numbers in italics are the paper's reported values (rounded
mean from Table I, first row per L).

| L | non-DQML | NC-DQML | CC-DQML | QC-DQML |
|---|---------:|--------:|--------:|--------:|
| 3 | 82.94 ± 1.13 *(70.6)* | 87.30 ± 1.39 *(84.6)* | 95.90 ± 2.22 *(90.0)* | 96.61 ± 0.64 *(89.5)* |
| 5 | 88.41 ± 0.64 *(75.5)* | 88.74 ± 0.60 *(86.3)* | 97.72 ± 1.06 *(93.1)* | 99.22 ± 0.42 *(93.2)* |
| 7 | 86.78 ± 0.18 *(75.1)* | 87.24 ± 0.88 *(86.7)* | 98.89 ± 0.56 *(96.0)* | 99.67 ± 0.09 *(95.4)* |
| 9 | 88.02 ± 0.46 *(78.1)* | 87.57 ± 0.74 *(88.1)* | 99.22 ± 0.42 *(96.8)* | 99.80 ± 0.28 *(96.0)* |

The **qualitative claims of the paper are reproduced**:

1. **CC-DQML ≈ QC-DQML at all tested L** (within 1–1.5 percentage
   points; both schemes converge to within ~1% of each other by
   L = 7).
2. **CC-DQML > NC-DQML** by a substantial margin (~9–12 percentage
   points), confirming the paper's headline message that classical
   communication unlocks most of the capacity gain of quantum
   communication.
3. **NC-DQML > non-DQML** at most L (small but consistent ~1–5%
   advantage from running two QPUs instead of one).
4. **QC-DQML converges faster than CC-DQML** during training, even
   when their final accuracies match. Visible at L=9 in
   `results/sweep/fig4c_training_curves.png`.

Our absolute numbers run a few percentage points higher than the
paper's, but the qualitative trends and the central claim (CC ≈ QC,
both ≫ NC) are unchanged. See
[`ADDITIONAL_NOTES.md §2`](ADDITIONAL_NOTES.md) for the likely causes
(embedding richness, pooling-block rotation choice, cross-QPU red-block
parameterisation).

### Figures

- `results/sweep/fig4c_training_curves.png` — reproduction of Fig. 4c
  at L=9. Mean ± std over 3 seeds.
- `results/sweep/fig4d_acc_vs_layers.png` — reproduction of Fig. 4d
  (validation accuracy at the final iteration vs. L).
- `results/sweep/table1.md` — Markdown reproduction of Table I.

### Photonic MerLin reproduction

The photonic side of the reproduction lives in two files:
`lib/merlin_model.py` (non-DQML baseline) and
`lib/merlin_distributed.py` (NC / CC / QC). All chips use angle
encoding only, `scale = 1.0`, evenly-spread input photon states, and
the FirstQuantumLayers-tutorial structure (trainable MZI mesh + angle
encoding + trainable rotations + another trainable MZI mesh). The
photonic chip → paper-scheme mapping is:

| Scheme | Photonic geometry | Inter-chip channel |
|---|---|---|
| non-DQML | 1 × `m = 8, n = 3` chip, all 8 attributes angle-encoded | n/a |
| NC-DQML | 2 × `m = 8, n = 3` chips, 4 attributes each | none |
| CC-DQML | 2 × `m = 8, n = 3` chips, 4 attributes each | classical: chip-0 soft bit weights two trainable feedforward phases angle-encoded on an extra mode of chip 1 |
| QC-DQML | 1 × `m = 16, n = 6` chip, 8 attributes on first 8 modes | implicit (one coherent chip) |

Per-chip readout uses a *trainable* `Softmax(Linear(C(m, n), 2))`
soft-bit head — the photonic analogue of the gate model's trainable
QCNN pooling tree. The two soft bits feed a learned 4-element
interpret function. See
[`ADDITIONAL_NOTES.md §4`](ADDITIONAL_NOTES.md) for the per-scheme
forward pass, the `LearnedBitHead` rationale (fixed `LexGrouping`
collapsed both NC and CC to ~52%; trainable head unlocked them), and
the photonic-baseline iteration story.

Run with (3 seeds, 800 iterations, Adam lr=0.05, batch=256):

```bash
python implementation.py --paper distributed_qml_cc --config papers/distributed_qml_cc/configs/classification_merlin.json
python implementation.py --paper distributed_qml_cc --config papers/distributed_qml_cc/configs/classification_merlin_distributed_nc.json
python implementation.py --paper distributed_qml_cc --config papers/distributed_qml_cc/configs/classification_merlin_distributed_cc.json
python implementation.py --paper distributed_qml_cc --config papers/distributed_qml_cc/configs/classification_merlin_distributed_qc.json
python utils/plot_photonic_results.py
```

| Scheme | Params | Final val acc | Best val acc |
|---|---:|---:|---:|
| non-DQML | 120 | 89.39 ± 3.29 | 90.17 ± 2.65 |
| NC-DQML | 472 | 88.22 ± 2.91 | 88.80 ± 3.15 |
| CC-DQML | 563 | **95.18 ± 2.47** | 96.29 ± 2.63 |
| QC-DQML | 16,514 | **98.50 ± 0.33** | 99.02 ± 0.16 |

The **photonic reproduction recovers the central ordering**
`NC ≲ non < CC ≪ QC`. CC pulls 7 percentage points clear of NC by
adding only ~90 trainable parameters (two feedforward phases + a soft
bit head), and QC closes the remaining 3-point gap. Figures live at
`results/photonic/fig_photonic_training_curves.png` (Fig. 4c analogue)
and `results/photonic/fig_photonic_acc_bar.png` (Fig. 4d analogue);
the headline table is `results/photonic/photonic_results_table.md`.

Note: in our photonic translation, NC sits just *below* non-DQML
(unlike the paper's gate-model where NC sits above non-DQML). This
is a consequence of our chip-size choice — see
[`ADDITIONAL_NOTES.md §5`](ADDITIONAL_NOTES.md).

### Other baselines

| Model | Params | Val acc (mean ± std, 3 seeds) | Comment |
|---|---|---|---|
| TinyMLP (hidden=8) | 137 | **98.50 ± 0.74** | Iso-parameter classical reference. The synthetic task is solved nearly perfectly by a tiny MLP. |

Saved under `results/baselines/`.

### Interpretation

The classical-MLP baseline at 98.5% means this benchmark is **not a
test of quantum advantage** in the strict sense — a small classical
network already saturates the task. Within the quantum family,
however, the relative ordering between schemes is informative and is
what the paper measures and what we reproduce. The photonic MerLin
baseline at ~60% indicates that the off-the-shelf vector-in
classifier of `MERLIN_COOKBOOK.md` pattern B is not well suited to
data clustered at hypercube corners (likely the angle-encoding /
threshold-detector subspace cannot easily separate those clusters);
this is a useful negative data point about default photonic
architectures rather than a claim against photonic computing in
general.

## Fair Baselines

### Classical MLP baseline

`lib/classical_model.py` trains a tiny 2-hidden-layer MLP (tanh
activations, hidden width 8 → 137 parameters) on the same synthetic
dataset, with the same Adam(lr=0.05) and batch size. This serves as an
iso-parameter point of reference: if the MLP already matches CC-DQML,
the quantum-classical gap is one of inductive bias, not expressivity.
Run via `configs/classification_classical.json`.

### Photonic MerLin extension

`lib/merlin_model.py` implements an 8-mode, 4-photon dual-rail photonic
classifier (cookbook pattern B). Hardware-aware fields:

| Field | Value |
|---|---|
| Computation space | UNBUNCHED |
| Detector model | threshold |
| Photon number | 4 |
| Number of modes | 8 |
| Input state | `[0,1,0,1,0,1,0,1]` |
| Encoding | angle, scale=π, one parameter per mode |
| Measurement strategy | PROBABILITIES |
| Postselection | none |
| Simulator / QPU path | MerLin CPU simulator (analytic, shots=0) |
| Seeds | 3 |

Run via `configs/classification_merlin.json`.

## Hardware-Aware Settings

The gate-model reproduction uses 4 qubits per QPU and at most two QPUs
(total ≤ 8 qubits). All simulation is analytic, single-precision,
CPU-only. No QPU access required.

## Limitations

- 3 seeds per cell instead of the paper's 10. Variance estimates are
  noisier than reported.
- The cross-QPU red block is approximated by single-rotation
  controlled gates; the paper's exact decomposition is not specified.
- The dataset generation procedure (Appendix B's "32 clusters
  distributed across an 8-dimensional space, each placed within one
  of 32 divisions randomly selected from a total of 2⁸ = 256 possible
  divisions") is mildly ambiguous; we translate each cluster by a
  distinct hypercube corner, the most natural reading.

Open exploration directions and further-work ideas are catalogued in
[`ADDITIONAL_NOTES.md §6`](ADDITIONAL_NOTES.md).

## Tests

```bash
cd papers/distributed_qml_cc
PYTHONPATH=$PWD/tests:$PYTHONPATH pytest -q
```

Tests cover dataset shape/balance, simulator gate semantics (Bell state,
RX rotation probabilities), differentiability of the DQML forward pass,
parameter-count bookkeeping, and an end-to-end smoke training run.

## Citation and License

If you build on this reproduction, please cite the original paper. This
repository's reproduction code is offered under the same license as the
rest of `reproduced_papers`.

```bibtex
@misc{hwang2024distributed,
  title  = {Distributed quantum machine learning via classical communication},
  author = {Hwang, Kiwmann and Lim, Hyang-Tag and Kim, Yong-Su and Park, Daniel K. and Kim, Yosep},
  year   = {2024},
  eprint = {2408.16327},
  archivePrefix = {arXiv}
}
```
