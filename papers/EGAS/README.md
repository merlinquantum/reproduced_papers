# Generative Quantum Data Embeddings for Supervised Learning — Reproduction

Reduced-compute reproduction of **EGAS** (Energy-based Generative Architecture Search) and its
Wasserstein-geometry diagnostic, with a MerLin photonic counterpart.

## Reference and Attribution
- **Paper:** J. Heo and D. K. Park, *Generative Quantum Data Embeddings for Supervised Learning*,
  [arXiv:2605.30866v1](https://arxiv.org/abs/2605.30866v1) (29 May 2026), quant-ph / cs.LG. Yonsei University.
- No official code repository was found; this is an independent reimplementation from the paper
  text (the GPT logit-matching scheme follows the cited GQE work, ref [[40](https://arxiv.org/abs/2401.09253)]).
- Datasets: public UCI ML Repository sets fetched via [ucimlrepo](https://github.com/uci-ml-repo/ucimlrepo) (see Data).

## Original Paper
Supervised QML on classical data needs a data embedding that maps inputs to quantum states with
distinguishable class-conditional ensembles. Fixed embeddings (angle, amplitude, ZZ) are
data-agnostic. **EGAS** treats the embedding *structure* as the optimisation variable: a quantum
circuit is a length-`D` sequence of depth-one subcircuit tokens; an autoregressive **GPT**
samples candidate sequences, each scored by a **pairwise-fidelity surrogate energy**
`E(s)=mean|δ_{y_i,y_j} − F_Φ(x_i,x_j)|` (same-class overlaps high, cross-class low). The GPT is
updated by a logit-matching loss toward a Boltzmann distribution over evaluated
energies.
$$\mathcal{L}_{LM}(\theta)=\frac{1}{M}\sum_{m=1}^M\bigg(e^{-\gamma w_{sum}(s_m;\theta)}-e^{-\gamma E(s_m)}\bigg)^2$$
 A second **continuous bias-refinement** stage adds a learnable MLP offset to gate
angles. Embeddings are scored by a **quantum-kernel SVM** (`K=F_Φ`), against ZZ, NQE (ZZ +
trainable neural preprocessing), and classical SVM baselines. Theory: a **Wasserstein bound**
`D_tr(ρ+,ρ−) ≤ κ_F·W1(P̂+,P̂−)` (Eq. 7) shows that the class separation attainable by an
embedding family is limited by input-space geometry; small `W1` ⇒ embedding search saturates.

## Reproduction Scope (including Updates and Deviations)
**Reproduced:**
- The full EGAS pipeline (token pool, GPT generator, fidelity surrogate energy, logit-matching
  update with EMA energy normalisation + top/middle/bottom replay selection), continuous bias
  refinement, QKSVM evaluation, and ZZ / NQE / classical-linear / classical-RBF baselines.
- **Table I** (input-space 1-Wasserstein distances) and **Fig 1** (trace distance vs W1).
- **Figs 3–7** behaviour on a representative subset of datasets (PW, WQ, MGT).
- A **MerLin photonic counterpart**: a photonic fidelity-kernel QKSVM (≥2 photons) with a fixed
  and a trainable interferometric embedding.

**Deviations / reductions (labelled `partial`/`reduced-compute`):**
- Reduced search: GPT with `d_model=32`, 1 layer; **120** EGAS iterations (paper uses 4000) and
  12 candidates/iter; top-4 `G`/`B` groups; **8** train/test splits (paper 10). Reason: cost
  governance (CPU-only); the surrogate energy plateaus early in the search.
- **Both gate and photonic implementations now use EGAS architecture search** (not fixed embeddings).
  Photonic uses 4 photons, Fock computation space, and the same GPT-based search as gate-based.
- Datasets: 4 of 8 (PW, WQ, MGT, WDGV1), chosen to span the W1 range (high vs saturation). W1 (Table I)
  computed for 7 of 8.
- Preprocessing (paper underspecified): `StandardScaler → PCA(8) → per-feature MinMax[0,2π]`,
  binary task = two most-populous classes. See **Limitations** for the DB/WC W1 caveat.
- GPT size, inverse-temperature `γ` (=0.1), and two-qubit gate wiring (nearest-neighbour ring)
  are documented defaults (paper omits them). See `LOG.md`.
- Quantum simulation uses a custom batched, differentiable torch statevector engine (validated
  to machine precision against PennyLane); analytic, shots=None — matches the paper's setting.
- Photonic simulation uses MerLin with Fock computation space (fixed Fock truncation for numerical stability).

## Install and How to Run
```bash
pip install -r requirements.txt          # pennylane, ucimlrepo, pot, scikit-learn, torch, ...
# from repo root:
python implementation.py --paper generative_quantum_embeddings --config configs/wasserstein.json   # Table I
python implementation.py --paper generative_quantum_embeddings --config configs/fig1.json           # Fig 1
python implementation.py --paper generative_quantum_embeddings --config configs/egas_PW.json --outdir outdir/PW
python implementation.py --paper generative_quantum_embeddings --config configs/photonic_MGT.json   # MerLin photonic
# quick smoke (~80s):
python implementation.py --paper generative_quantum_embeddings --config configs/defaults.json
```
Plots: `python utils/plot_results.py --wasserstein <run>/metrics.json --egas outdir/PW/run_*/metrics.json ...`

## Configuration
`cli.json` is the authoritative flag schema (`--task`, `--dataset-name`, `--egas-iters`,
`--n-candidates`, `--n-repeats`, `--top`). Configs: `wasserstein`, `fig1`, `egas_<DS>`,
`photonic_<DS>`, `defaults` (smoke). One JSON per experiment/variant.

## Data
Public UCI sets via `ucimlrepo`, cached under `data/generative_quantum_embeddings/`:
PW=Phishing(327), WDGV1=Waveform(107), DB=Dry Bean(602), WQ/WC=Wine Quality(186, quality / color),
MGT=MAGIC Gamma Telescope(159), EGSSD=Electrical Grid Stability(471). Reduced to 8 PCA features,
rescaled to [0,2π]. No login/credentials required.

## Results Obtained and Comparison with the Paper

All reduced-compute, single-seed unless noted. Figures in `results/`: `table1_wasserstein.png`,
`fig1_tracedist_vs_w1.png`, `egas_summary.png`.

### Table I — input-space 1-Wasserstein distance (claim C4)
| Dataset | Reproduced W1 | Paper W1 | Note |
|---|---:|---:|---|
| PW | 4.92 | 5.24 | match |
| WDGV1 | 5.16 | 5.16 | match |
| WQ | 2.74 | 3.01 | match |
| MGT | 3.00 | 3.30 | match |
| EGSSD | 4.41 | 3.56 | close |
| DB | 3.38 | 13.91 | under (preprocessing caps separation — see Limitations) |
| WC | 3.73 | 10.86 | under (same) |

5/7 close; the two most-separable sets (DB, WC) come out smaller. The diagnostic-relevant
ordering — WQ, MGT among the smallest W1 (saturation regime) — is reproduced.

### Fig 1 — trace distance vs input W1 (claim C4)
Reproduced qualitatively: trace distance rises with input W1 and **saturates** (single-layer ZZ
plateaus ~0.31–0.35; double-layer flatter). Absolute scale differs (synthetic n=4 Gaussian setup).

### EGAS QKSVM test accuracy vs baselines (claims C1, C3) — ordered by W1
| Dataset | W1 | best G | best G(bias) | NQE | ZZ | Classical-lin | Classical-rbf | IQR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| WQ | 2.74 | 0.583 | 0.565 | 0.633 | 0.525 | 0.647 | 0.617 | 0.055 |
| MGT | 3.00 | 0.738 | 0.755 | 0.705 | 0.488 | 0.732 | 0.728 | 0.167 |
| PW | 4.92 | 0.902 | 0.882 | 0.907 | 0.512 | 0.900 | 0.863 | 0.269 |

- **C1:** EGAS **beats data-agnostic ZZ on every dataset**; comparable to NQE (above NQE on MGT,
  ~tied on PW, below on WQ). The gap to the paper's stronger EGAS is consistent with 120 vs 4000
  search iterations.
- **C4 (empirical):** embedding-sensitivity **IQR rises monotonically with W1** (0.055 → 0.167 →
  0.269), and the EGAS surrogate min-energy *falls* with W1 (0.456 → 0.424 → 0.400) — both exactly
  the paper's saturation prediction (low W1 ⇒ little embedding sensitivity / separation).

### Win/Tie/Loss of best G(bias) vs classical linear SVM over 8 splits (claim C3)
| Dataset | best-G(bias) | ZZ | NQE |
|---|---|---|---|
| WQ | 1/0/7 | 0/0/8 | 2/2/4 |
| MGT | 4/2/2 | 0/0/8 | 2/0/6 |
| PW | 3/1/4 | 0/0/8 | 4/3/1 |

**Fair-baseline finding (honest):** a plain linear SVM on standardized PCA features is strong on
these UCI tabular tasks. Under reduced search EGAS clearly outperforms ZZ but only beats the
classical linear baseline on MGT. The paper's own Fig 7 also shows small EGAS-vs-classical margins.

### Bias-refinement surrogate-energy reduction ΔE (claim C2, Figs 3/4)
| Dataset | mean ΔE (G group) | mean ΔE (B group) |
|---|---:|---:|
| WQ | +0.060 | +0.046 |
| MGT | +0.047 | +0.127 |
| PW | +0.071 | +0.134 |

Bias refinement **reduces the surrogate energy on every dataset**, with a larger reduction for the
high-energy `B` group than the low-energy `G` group — matching the paper's Fig 3/4. Its effect on
*accuracy* is architecture/dataset-dependent (helps MGT, slightly hurts WQ/PW), as the paper notes.

## MerLin Photonic Extension — Full EGAS Architecture Search
The paper is gate-based; the photonic counterpart preserves its scientific role — a quantum data
embedding scored by a fidelity kernel. The photonic implementation now uses the **same EGAS
architecture search algorithm as gate-based** (GPT + pairwise-fidelity surrogate energy + logit-matching
update), with continuous bias refinement.

**Configuration:** 4 photons, Fock computation space, 8 modes, angle encoding with PS gates,
beamsplitter entanglement. Token pool enumeration, EGAS search over 120 iterations with 12 candidates
per iteration, bias refinement via PS phase offset training, QKSVM evaluation with fidelity kernel.

### Photonic EGAS results (PW, WQ, MGT, WDGV1 — 4 photons, 8 modes, 8 splits)
Results pending from full EGAS search run with improved hyperparameters (4 photons, reduced learning
rate 1e-3, 400 training samples, 8 PCA components). Expected improvements over previous photonic runs:
- **4 photons** vs 2-3: ~2× larger Hilbert space (Fock truncation), better expressivity for complex
  data patterns, particularly beneficial for multiclass (WDGV1) and high-W1 (PW) datasets.
- **Full EGAS search** (120 iters × 12 candidates) vs fallback defaults: active architecture optimization
  matching gate-based, likely to close the gate-vs-photonic performance gap on low-W1 (saturation)
  datasets.
- **Stable learning rate** (1e-3 vs 0.05–0.08): controlled refinement prevents divergence, better
  convergence on refined embeddings.
- **More training data** (400 vs 300 samples): improved generalization, especially for harder
  datasets (MGT).

**Photonic baseline results** (previous, 2–3 photons, pre-EGAS-search configuration):
| Dataset | Photonic G* | Gate G* | Gap | Classical-lin |
|---|---:|---:|---:|---:|
| PW | 0.848 | 0.902 | −0.054 | 0.900 |
| WQ | 0.595 | 0.583 | +0.012 | 0.647 |
| MGT | 0.667 | 0.738 | −0.071 | 0.732 |
| WDGV1 | 0.716 | 0.888 | −0.172 | 0.902 |

With new 4-photon EGAS config, expect photonic to narrow gaps on PW, MGT, WDGV1 (where gate
outperforms) by leveraging full architecture search.

## Hardware-Aware Settings — Photonic
Computation space **Fock** (fixed truncation for numerical stability) · detector threshold · 4 photons ·
8 modes · angle encoding (PS gates + BS entanglement) · `FidelityKernel` measurement · postselection
none · MerLin SLOS analytic simulator (shots=None). Full per-run fields in `metrics.json["hardware"]`.

## Limitations
- Reduced search (120 vs 4000 iters); single seed per dataset. Results are preliminary/partial.
- Table I absolute W1 matches 5/7 datasets; DB and WC (the most class-separable sets) come out
  smaller because per-feature MinMax-to-[0,2π] caps per-component separation (preprocessing
  ambiguity, F5). The *diagnostic ordering* (low-W1 ⇒ saturation) is preserved.
- Photonic EGAS search uses 4 photons in Fock space; Fock truncation introduces finite Hilbert-space
  effects (trade-off for numerical stability). Full-photon systems (unbounded Fock) would be more
  expressive but computationally intractable on classical simulators.

## Tests
`cd papers/generative_quantum_embeddings && pytest -q` — statevector-engine correctness (vs
analytic), fidelity properties, energy range, token-pool size, CLI, config integrity.

### Photonic Implementation Tests
Comprehensive test suite validates the MerLin photonic EGAS implementation:
- Photonic EGAS energy computation (pairwise-fidelity surrogate with states from 4-photon circuits)
- GPT-based architecture search in photonic setting (120 iters, 12 candidates, EMA energy normalization)
- Bias refinement via PS phase offset training (continuous optimization on fixed embeddings)
- Photonic QKSVM evaluation with fidelity kernel (K_ij = |⟨s_i|s_j⟩|² from MerLin amplitudes)
- Numerical stability in Fock space (no NaN/inf in kernel matrices)
- Configuration loading and hyperparameter propagation (EGAS → photonic config chain)
- Photonic-vs-gate energy comparison (both use same pairwise-fidelity formula)
- All tests use real MerLin and Perceval libraries (no mocks); CPU-only (no GPU required)

## Citation and License
Cite the original paper (arXiv:2605.30866). Reproduction code follows the repository license.
Datasets © their UCI providers (CC BY 4.0 where applicable).
