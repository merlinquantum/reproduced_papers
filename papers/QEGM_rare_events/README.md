# Quantum-Enhanced Generative Models for Rare Event Prediction — Reproduction

## Reference and Attribution

- Paper: *Quantum-Enhanced Generative Models for Rare Event Prediction* (QEGM)
- Authors: M. Z. Haider, M. U. Ghouri, T. Noreen, M. Salman
- ArXiv: [2511.02042v1](https://arxiv.org/abs/2511.02042) (Nov 2025)
- Original repository: none located as of 2026-05-27
- License notes: this reproduction code is MIT (repo default); paper text and
  figures remain © the authors.

## Original Paper

The paper introduces **QEGM**, a hybrid VAE-style framework augmented with a
Variational Quantum Circuit (VQC) for rare-event generative modeling. Two key
ingredients are claimed:

1. A *hybrid loss* (Eq. 5) that combines a reconstruction term with a
   tail-aware term focused on rare-event regions of the distribution.
2. *Quantum-randomness noise injection* (Eq. 7): the latent noise variance is
   modulated by a random scalar ``r`` produced by measuring a parameterised
   quantum circuit, with the claim that this avoids the deterministic
   correlations of PRNG-based sampling.

The paper evaluates QEGM against GAN, VAE, and diffusion baselines on a
synthetic 3-component Gaussian mixture and on real-world finance (S&P 500),
climate (temperature/precipitation), and protein (AlphaFold embeddings) data.
Reported headline numbers include a ~50% reduction in tail KL on the GMM
benchmark and rare-event recall improvements from 0.74 (Diffusion) to 0.88
(QEGM).

## Reproduction Scope (Updates and Deviations)

**Reproduced:**

* The synthetic Gaussian-mixture experiment (Section VI.C) — three components
  with means `{-3, 0, +3}` and variances `{1, 0.5, 1.5}`.
* The full QEGM architecture (encoder, decoder, hybrid loss, VQC,
  quantum-randomness noise injection per Eq. 7).
* A photonic translation of the VQC built on MerLin (Pattern A from
  `MERLIN_COOKBOOK.md`).
* A fair classical VAE baseline with the same encoder/decoder skeleton as QEGM.

**Not reproduced (blocked or out of scope):**

* The real-world experiments (Sec. VI.D — finance, climate, protein). The
  paper does not specify dataset URLs, preprocessing pipelines, encoder
  architectures, λ values, or training hyperparameters with enough precision
  to attempt these honestly. Marked as `BLOCKED` in `LOG.md`.
* The Diffusion and GAN baselines. The paper does not pin baseline
  architectures, and fitting them on top of the 3-seed × 3-variant sweep
  would exceed budget. We report the **VAE** baseline only and use a
  *matched-architecture* comparison.

**Deviations:**

* **Quantum gradients.** Paper Eq. 16 uses the parameter-shift rule. We use
  PyTorch autograd through the state-vector simulator. The forward pass is
  identical and gradients match in expectation; parameter-shift is only
  required for QPU training.
* **Latent encoding dimension.** The paper has an internal contradiction
  (`d` qubits for angle encoding in Eq. 13 vs `⌈log2 d⌉` for amplitude
  encoding in Eq. 12). We follow Eq. 13 because angle encoding is the
  formulation actually used in the noise-injection pathway.
* **Photonic counterpart.** Added on top of the gate-based variant: same
  VAE skeleton, MerLin `QuantumLayer` (Pattern A, 6 modes, 3 photons,
  threshold detection) providing the `r` values.

Reproduction tier: **V4 — synthetic-data structural reproduction**. The
synthetic experiment exercises the full architecture and the central claims
qualitatively; the real-world claims are not testable here.

## Project Layout

```
QEGM_rare_events/
├── README.md
├── LOG.md
├── INSIGHTS.md
├── FEEDBACK.md
├── CONFLUENCE.md
├── VISITED_URLS.md
├── requirements.txt
├── cli.json
├── configs/
│   ├── defaults.json
│   ├── smoke.json
│   └── synthetic_multi_seed.json
├── lib/
│   ├── data.py            # GMM dataset
│   ├── models.py          # VAE / QEGM / QEGM-MerLin
│   ├── vqc.py             # Hardware-efficient gate VQC simulator
│   ├── merlin_layer.py    # Photonic counterpart via merlinquantum
│   ├── metrics.py         # Tail KL, recall, coverage
│   ├── training.py        # Hybrid loss + training loop
│   └── runner.py          # train_and_evaluate entry point
├── utils/
│   └── plot_results.py    # Figure generation
└── tests/
    ├── common.py
    ├── test_cli.py
    └── test_smoke.py
```

## Install and How to Run

Inside the project Docker image, all dependencies are already installed.
Otherwise:

```bash
pip install -r papers/QEGM_rare_events/requirements.txt
```

Run the smoke configuration (a few minutes):

```bash
python implementation.py --paper QEGM_rare_events --config configs/smoke.json
```

Run the multi-seed reproduction (≈10–15 min on CPU):

```bash
python implementation.py --paper QEGM_rare_events --config configs/synthetic_multi_seed.json
```

Override knobs:

```bash
# Train only the photonic variant for two seeds
python implementation.py --paper QEGM_rare_events --config configs/synthetic_multi_seed.json \
    --models qegm_merlin --seeds 0,1

# Stronger tail weight
python implementation.py --paper QEGM_rare_events --config configs/synthetic_multi_seed.json \
    --lambda-tail 4.0
```

Outputs land under `papers/QEGM_rare_events/outdir/run_YYYYMMDD-HHMMSS/`:

```
run_20260527-123525/
├── config_snapshot.json
├── run.log
├── metrics.json           # per-seed metrics + summary
├── histories.json         # per-epoch training losses
├── real_samples_test.npy
├── samples_<variant>_seed<seed>.npy
├── fig_densities.png
├── fig_tail_kl.png
├── fig_rare_recall.png
└── fig_coverage.png
```

## Configuration

The full CLI is described in `cli.json`. Key flags above the shared global
ones (`--seed`, `--dtype`, `--device`, `--log-level`, `--outdir`):

- `--epochs INT`
- `--batch-size INT`
- `--lr FLOAT`
- `--n-samples INT`
- `--lambda-tail FLOAT`
- `--seeds STR` — comma-separated list, e.g. `0,1,2`
- `--models STR` — comma-separated subset of `vae,qegm,qegm_merlin`

## Data

Synthetic only. The dataset is generated on the fly in `lib/data.py` from the
means, variances, and mixture weights given in the paper. No external download.

## Results Obtained and Comparison with the Paper

Synthetic GMM (3 seeds, 60 epochs, 2048 samples; results curated in
`results/metrics*.json` and figures `results/fig_*.png`).

### Headline table — matched-architecture comparison

| Variant | Tail KL ↓ (|x|>1.5) | Rare recall | 95% coverage | Wall-clock (s) | Params |
|---|---:|---:|---:|---:|---:|
| VAE baseline (classical, matched arch.) | 0.165 ± 0.040 | 1.000 ± 0.000 | 0.946 ± 0.008 | 1.7 | 2633 |
| QEGM (gate-based VQC, paper formulation) | 0.208 ± 0.033 | 1.000 ± 0.000 | 0.915 ± 0.023 | 215.7 | 2658 |
| QEGM (MerLin photonic counterpart) | 0.149 ± 0.014 | 1.000 ± 0.000 | 0.946 ± 0.013 | 38.0 | 2724 |
| **QEGM (const r=0.5 ablation)** | **0.177 ± 0.048** | 1.000 ± 0.000 | 0.937 ± 0.018 | 3.1 | 2658 |

### Rigor ablations

#### Ablation 1 — pin `r` to a constant (`results/metrics_const_ablation.json`)

Replacing the entire trained VQC with `r = 0.5` ∈ [0, 1] produces tail KL
**0.177 ± 0.048** — *statistically indistinguishable* from the gate-based
QEGM (0.208 ± 0.033) and from the matched VAE (0.165 ± 0.040). The Eq. 7
quantum-randomness modulation is therefore **provably inert** on this
benchmark: a fixed scalar matches the trained circuit's contribution.

#### Ablation 2 — remove the tail-aware loss term, λ_tail = 0 (`results/metrics_no_tail_ablation.json`)

| Variant | Tail KL with λ_tail=2 | Tail KL with λ_tail=0 | Δ |
|---|---:|---:|---:|
| VAE | 0.165 ± 0.040 | **0.138 ± 0.035** | −16% |
| QEGM (gate) | 0.208 ± 0.033 | **0.122 ± 0.035** | −41% |
| QEGM (MerLin) | 0.149 ± 0.014 | **0.140 ± 0.018** | −6% |

Removing the tail-aware term *improves* tail KL for **every** variant.
The paper's hybrid-loss innovation (Eq. 5) is **mildly counterproductive**
on this benchmark — likely because in 1-D the tail-weighted term creates
high-variance gradients from a small number of tail samples per batch.

### Threshold and rarity-score sweeps (`results/fig_threshold_sweep.png`)

Tail KL across thresholds `|x| > {1.5, 2.0, 2.5, 3.0}` and using the
paper's rarity-score definition `s(x) = −log p(x)` with top-5 / 10 / 20%
rarest:

- At all thresholds the gate-based QEGM is **not** better than the
  matched VAE.
- The MerLin photonic counterpart is mildly better than VAE at strict
  thresholds (e.g. 0.092 ± 0.033 vs 0.160 ± 0.058 at |x|>3.0), but the
  error bars overlap — no statistically reliable advantage.

### Paper-claim audit

| Paper item | Claim tested | Paper value | Reproduced value | Verdict |
|---|---|---:|---:|---|
| Sec VI.C, Fig 5b | QEGM tail KL ≈50% lower than Diffusion | qualitative ≪ | 0.208 vs 0.165 (gate vs VAE) | **not supported** |
| Sec VI.C | Recall 0.74→0.88 (Diffusion → QEGM) | 0.88 | 1.00 (all variants, recall saturates) | metric not discriminative on this benchmark |
| Sec IV.D | Eq. 7 quantum-randomness reduces correlated noise | qualitative | const-r matches trained VQC | **mechanism provably inert** |
| Sec IV / Eq. 5 | Hybrid tail-aware loss improves tail fidelity | qualitative | λ_tail=0 *beats* λ_tail=2 on every variant | **mechanism mildly harmful** |
| Sec VI.C | QEGM "preserves fidelity across all three modes" | qualitative | All variants collapse the GMM to a single unimodal blob | **not supported** |

### Net reproduction outcome

Under a fair *matched-architecture* comparison, with two independent
controlled ablations, **neither of the paper's two headline innovations
(quantum-randomness noise injection, hybrid tail-aware loss) demonstrates
the claimed effect** on the synthetic GMM benchmark:

- The Eq. 7 modulation is *inert* — a constant `r` matches the trained
  VQC within seed variance.
- The Eq. 5 tail-aware loss is *mildly harmful* — every variant improves
  on tail KL when it is removed.

Reproduction confidence: **HIGH** on the implementation (forward pass
and gradients verified; MerLin photonic counterpart trains end-to-end;
const-r and no-tail ablations are deterministic controls). Verdict on
the paper's central claims: **unsupported** at the synthetic-benchmark
level; **unresolved** at the real-world-benchmark level because the
authors do not pin enough details to attempt those reproductions.

Caveats:

* Reproduction tier V4 (synthetic structural). External GAN/VAE/Diffusion
  baselines specified narratively in the paper cannot be reproduced
  numerically.
* All variants produce unimodal generated densities (see `fig_densities.png`);
  none recover the GMM's three-component structure. The paper claims
  QEGM "preserves fidelity across all three modes" — that claim is not
  supported under matched conditions.
* Single-domain reproduction; real-world finance/climate/protein
  experiments blocked by missing dataset URLs and hyperparameters.

## Fair Baselines

Classical fair baseline: `VAEBaseline` (in `lib/models.py`) with the same
encoder/decoder MLP, latent dimension, and hybrid loss as QEGM — only the
quantum-randomness noise modulation is removed. This isolates the contribution
of the VQC.

## MerLin Photonic Extension

`QEGMMerlin` swaps the gate-based VQC for a MerLin `QuantumLayer` (Pattern A
of `MERLIN_COOKBOOK.md`). The hardware-aware settings written to
`metrics.json["hardware_settings"]`:

| Field | Value |
|---|---|
| Computation space | `UNBUNCHED` |
| Detector model | threshold |
| Photon number | 3 |
| Number of modes | 6 |
| Input state | `[1, 0, 1, 0, 1, 0]` |
| Encoding | angle, modes 0–3, scale = π |
| Measurement strategy | `MeasurementStrategy.probs(computation_space=UNBUNCHED)` (merlin >= 0.4 API) |
| Postselection | none |
| Simulator | MerLin CPU analytic (shots=0) |

## Limitations

* Single-task (synthetic GMM); real-world tasks blocked by missing details.
* Single baseline (VAE); GAN and Diffusion baselines not reproduced.
* Reproduction tier V4.
* The paper's quantitative comparison numbers are not directly comparable.

## Tests

```bash
cd papers/QEGM_rare_events
pytest -q
```

`test_cli.py` runs a one-epoch / 64-sample / single-seed end-to-end VAE
training run to validate the shared-runtime wiring.

## Citation and License

If you cite this reproduction, please also cite the original paper:

```
@article{haider2025qegm,
  title={Quantum-Enhanced Generative Models for Rare Event Prediction},
  author={Haider, M. Z. and Ghouri, M. U. and Noreen, T. and Salman, M.},
  journal={arXiv preprint arXiv:2511.02042},
  year={2025}
}
```

Reproduction code: MIT (see repository root `LICENSE`).
