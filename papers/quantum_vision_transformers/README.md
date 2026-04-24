# Quantum Vision Transformers — Photonic Reproduction & Extensions

**Paper**: Cherrat et al., "Quantum Vision Transformers", *Quantum* 2024.
[arXiv:2209.08167](https://arxiv.org/abs/2209.08167v2)

**Framework**: [MerLin](https://merlinquantum.ai/) — native photonic linear optics.

---

## Models

### Paper reproduction (A–D)

| Model | Description | Photons | Modes | Sector |
|:---:|---|:---:|:---:|---|
| **A** | Orthogonal Patch-wise NN | 1 | d | Full 1-ph (no post-selection) |
| **B** | Quantum Orthogonal Transformer | 1 | d | Two interferometers V, W |
| **C** | Direct Quantum Attention (C2) | 1 | d | Pragmatic hybrid |
| **D** | Compound Transformer | 2 | n+d | Cross-partition post-selection |

### Paper baselines

| Model | Description | Type |
|:---:|---|---|
| **VisionTransformer** | Classical transformer baseline from Appendix A | Classical |
| **OrthoFNN** | Quantum orthogonal fully connected baseline from [5] | Quantum baseline |

### Extensions beyond the paper (D_full, E, F)

| Model | Description | Photons | Modes | Key idea |
|:---:|---|:---:|:---:|---|
| **D_full** | Full-sector Compound | 2 | n+d | Use ALL 2-ph sectors: cross (features) + pp (emergent attention) + ff (feature correlations) |
| **E** | Multi-sector Attention | 1 + 2 | n+d | Shared interferometer; 1-ph → features, 2-ph pp → attention. One set of angles serves both roles. |
| **F** | Hierarchical Compound | 3 | r+p+d | 3-photon encoding of region × patch × feature hierarchy. V⁽³⁾ jointly mixes all three levels. |

### Attention summary

| Model | Attention mechanism | How token mixing happens |
|:---:|---|---|
| **VisionTransformer** | Classical learned attention | Learned score matrix + softmax over patch tokens |
| **OrthoFNN** | None | No token-token attention; repeated tokenwise orthogonal transforms |
| **A** | None | Shared 1-photon interferometer `V` applied independently to each token |
| **B** | Quantum overlap attention | `W` produces pairwise overlap scores, `softmax` gives attention weights, `V` gives features |
| **C** | Direct quantum attention | Same quantum overlap scores as `B`, but attention is applied to inputs before the final feature transform |
| **D** | Implicit compound attention | One 2-photon `(n+d)`-mode interferometer; attention emerges from interference plus cross-partition readout |
| **D_full** | Implicit compound attention | Same as `D`, but keeps all 2-photon sectors instead of only cross-partition outputs |
| **E** | Shared-circuit multi-sector attention | One shared interferometer: 1-ph sector gives features, 2-ph patch-patch sector gives attention |
| **F** | Hierarchical implicit attention | 3-photon interference across region, patch, and feature blocks; attention is encoded in sector structure rather than an explicit matrix |

### Fock space sizes (28×28, embed_dim=16)

| Model | Modes | Photons | Basis size |
|:---:|:---:|:---:|:---:|
| A, B, C | 16 | 1 | 16 |
| D, D_full, E | 33 | 2 | 561 |
| F | 24 | 3 | 2,600 |

## Design principles

1. **Amplitude encoding** — input is a quantum state in a fixed photon-number sector.
2. **Circuit Families** — Configurable via `"circuit_family"`.
   - **generic**: Universal rectangular MZI mesh (Clements/Reck). The original repo path.
   - **butterfly**: Structured butterfly MZI circuit family. The "paper-closer" structured path, matching the layout used in the original paper. Requires power-of-two mode counts (e.g., $d=16$, or $n+d=32$ for Model D).
3. **MerLin native** — `QuantumLayer`, `StateVector.from_tensor`, `CircuitBuilder`. SLOS computes compound actions internally.
4. **Post-selection only where semantically required** — D (cross-partition), F (triple-cross).

## Project layout

```
QVT/
├── implementation.py               Compatibility shim to the shared root CLI
├── verify.py                       All-model verification checklist
├── requirements.txt
├── configs/
│   ├── model_{a,b,c}_retina.json   Paper models
│   ├── model_d_retina.json         Paper compound (cross_only)
│   ├── model_d_full_retina.json    Extension: full-sector compound
│   ├── model_e_retina.json         Extension: multi-sector attention
│   └── model_f_retina.json         Extension: hierarchical 3-photon
│   └── paper/                      Exact paper benchmark family incl. baselines
├── lib/
│   ├── photonic_primitives.py      Interferometer, readouts (Compound/Full/Triple)
│   ├── models.py                   All 6 architectures + QVTModel wrapper
│   ├── data.py                     Patch embedding (flat + hierarchical), MedMNIST
│   └── training.py                 Train loop, metrics, sector logging
└── scripts/
    ├── validation/                 Validation and smoke tests
    ├── reproduction/               Paper-facing benchmark runners
    ├── suites/                     High-level workflow wrappers
    ├── experiments/                Exploratory and subset-study runners
    ├── analysis/                   Figure generation and summaries
    ├── benchmarks/                 Profiling and backend benchmarks
    └── README.md                   Script index
```

## Quick start

```bash
cd papers/quantum_vision_transformers
pip install -r requirements.txt
python ../../implementation.py --paper quantum_vision_transformers --config configs/paper/model_a_retina.json
bash scripts/validation/validate.sh                            # check all models build and grad-flow
bash scripts/suites/run_retina_cpu_suite.sh --device cpu       # paper + butterfly CPU workflow
bash scripts/suites/run_medmnist_cpu_suite.sh --device cpu     # MedMNIST CPU workflow
python scripts/analysis/generate_figures.py outdir/            # figures (all)
python scripts/analysis/generate_figures.py outdir/ --profile lite  # lite figures only
```

`requirements.txt` now installs the vendored [third_party/merlinquantum](/C:/Users/BenjaminSTOTT/PycharmProjects/reproduced_papers/third_party/merlinquantum) checkout in editable mode, so a full repo clone is required on any machine where you want the local MerLin performance fixes.

## Shared runner usage

Run QVT through the repo-root shared CLI:

```bash
python implementation.py --paper quantum_vision_transformers --config papers/quantum_vision_transformers/configs/paper/model_a_retina.json
```

For paper-local convenience, the existing `scripts/` wrappers still work, but they now call the shared root runner internally.

## Server / GPU setup

To reproduce the current behavior on a remote server:

```bash
git clone <your repo>
cd reproduced_papers/papers/quantum_vision_transformers
pip install -r requirements.txt
python -c "import merlin; print(merlin.__file__)"
```

The last command should resolve inside `third_party/merlinquantum`, not a site-packages wheel.

## CPU vs GPU profiling

Use the benchmark script to compare the current optimized stack across devices:

```bash
python scripts/benchmarks/benchmark_device_profile.py --devices cpu cuda:0 --precision-mode gpu_friendly
```

## Retina-sized train subsets

For a data-efficiency study on MedMNIST, the repo now supports training on a
RetinaMNIST-sized subset while keeping the official validation and test splits
intact. The current runner is:

```bash
bash scripts/experiments/run_all_medmnist_butterfly_lite_retina_sized.sh --device cpu
```

This uses:
- `train_subset_size = 1080`
- stratified sampling on the training split
- full validation and test splits

The resulting runs are tagged with `data_regime="retina_sized_train"` so they
remain distinct from the standard benchmark outputs in figure generation.

For a larger but still capped training regime, use:

```bash
TRAIN_SUBSET_SIZE=5000 bash scripts/experiments/run_all_medmnist_butterfly_lite_subset.sh --device cpu
```

These runs are tagged with `data_regime="train_subset_5000"` by default, so
they stay separate from both the standard benchmark and the Retina-sized subset runs.

It runs the real 1-epoch Retina configs, writes reports to `outdir/device_profile/`, and prints timing plus memory summaries. The main outputs are:

- `outdir/device_profile/benchmark_summary.json`
- `outdir/device_profile/benchmark_summary.csv`

## Checkpoints and resume

Each run directory now stores:

- `config.json` — resolved runtime config
- `last.pt` — resumable checkpoint (model, optimizer, scheduler, RNG, history)
- `best.pt` — best validation-AUC weights
- `progress.json` — incremental per-epoch progress snapshot
- `results.json` — final completed run summary

Rerunning `implementation.py` with the same `--outdir` resumes automatically from
`last.pt` when present. Use `--resume never` to force a fresh run in an existing
directory, or `--resume must` to fail unless a checkpoint is available.

## Figures generated

| Figure | Content |
|---|---|
| `training_curves_{dataset}.pdf` | Loss, acc, val AUC per model (mean ± std across seeds) |
| `comparison_{dataset}.pdf` | Bar chart: test AUC & ACC, with paper reference lines |
| `sector_mass_{dataset}.pdf` | Per-sector probability evolution over training (D/D_full/E/F) |
| `param_comparison.pdf` | Attention params vs total, with classical ViT reference |
| `summary.csv` | All results in one flat table |

## Paper reference values (RetinaMNIST, Table 4)

| Model | AUC | ACC | Attn params / layer |
|---|:---:|:---:|:---:|
| Classical ViT | 0.736 | 55.75% | 512 (2d²) |
| A: OrthoPatchWise | 0.738 | 56.50% | 32 |
| B: OrthoTransformer | 0.749 | 56.50% | 64 |
| D: Compound | 0.729 | 56.50% | 80 |

## Citation

```bibtex
@article{cherrat2024quantum,
  title={Quantum Vision Transformers},
  author={Cherrat, El Amine and Kerenidis, Iordanis and Mathur, Natansh
          and Landman, Jonas and Strahm, Martin and Li, Yun Yvonna},
  journal={Quantum},
  year={2024}
}
```
