# Hybrid Quantum-Classical de Novo Design of MHC-Binding Peptides

## Reference and Attribution

Engdal, Funk, Bacarreza et al. (2026 preprint),
[Hybrid quantum-classical de novo design of MHC-binding peptides](https://doi.org/10.64898/2026.07.09.736951).
Original code: [orcacomputing/pep-q-gan](https://github.com/orcacomputing/pep-q-gan).
Adapted code retains [Apache-2.0](LICENSE.pep-q-gan) and [attribution notices](NOTICE.md).

## Overview

This project tests photonic latent distributions in a classical HLA-conditioned
peptide GAN. MerLin constructs a fixed interferometer; Perceval samples its output
to supply the GAN's latent vectors. Only the classical networks are trained.

Changes from the paper:

- Fixed 16-photon Fock input over 32 modes and one MerLin MZI entangling mesh,
  replacing squeezed-state GBS.
- Matched distinguishable (Dist) and indistinguishable (Indist) photons, plus a
  Gaussian reference. Sampling uses official Perceval CPU APIs.
- MHCflurry presentation screening replaces NetMHCpan evaluation due to license.

The [notebook](notebook.ipynb) visualizes the circuit and samples both priors directly, defines small conditional WGAN-GP models,
trains each for 1,000 CPU updates, and generates 64 peptide sequences.  It uses synthetic motif data by default; set
`DATA_SOURCE="author"` and `DATA_ROOT` for prepared author data. Synthetic A/B
labels are artificial motif classes, not real HLA alleles. The notebook keeps the full trainer's settings. 
The small model is a demonstration, separate from the full experiment below. 

## How to Run

### Installation

Use Python >=3.11. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv/Scripts/Activate.ps1
python -m pip install "torch>=2.6,<2.13" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r papers/pep_q_gan/requirements.txt
python -m jupyter lab papers/pep_q_gan/notebook.ipynb
```

For CUDA, install the appropriate PyTorch build using the
[official selector](https://pytorch.org/get-started/locally/), then install
`requirements-gpu.txt`. Both environments include Jupyter and MHCflurry 2.3.16;
predictor weights are downloaded separately. The notebook always runs on CPU.
For macOS, use the official platform-specific PyTorch installation instructions.

### Command line

```bash
# Inspect the fixed circuit.
python implementation.py --paper pep_q_gan

# List available options.
python implementation.py --paper pep_q_gan --help
```

From inside `papers/pep_q_gan/`, use `../../implementation.py`. In a flat checkout,
omit `papers/` from paths and use `../implementation.py` from the paper folder.
Full sampling, training and evaluation commands are in [RUNNING.md](RUNNING.md).
Full 100k-update training took roughly **five GPU(RTX 6000 Ada) hours per model** in the recorded
runs; CPU training can be very slow. 

## Configuration

[cli.json](cli.json) defines options; JSON files in `configs/` select operations.

| Configuration | Operation |
|---|---|
| `defaults.json` | Circuit inspection |
| `prior_sampling.json` / `dist_sampling.json` | Indist / Dist sampling |
| `training_full.json` | Full GAN training |
| `gaussian_baseline.json` | Gaussian baseline training |
| `evaluation.json` | MHCflurry scoring of existing generated peptides |

Runs write configuration, logs and artifacts to `outdir/run_YYYYMMDD-HHMMSS/`.
Notebook-generated outputs go to ignored `outdir/notebook_runs/`; final images are in `results/`.

## Data

### Download the two required training inputs

The full experiment uses the paper's curated [IEDB](https://www.iedb.org/)-derived
peptide data. Download these files directly from the authors:

- [HLA_pos9mers.txt](https://raw.githubusercontent.com/orcacomputing/pep-q-gan/51f7dfa162e33a3be9ee90a69dd93f92e07d7788/data/HLA_pos9mers.txt): peptide–HLA training pairs.
- [MHC_pseudo_HLA_only.dat](https://raw.githubusercontent.com/orcacomputing/pep-q-gan/51f7dfa162e33a3be9ee90a69dd93f92e07d7788/data/MHC_pseudo_HLA_only.dat): HLA conditioning pseudosequences.

Preserve filenames under external `data/pep_q_gan/`, or pass that directory with
`--dataset-root`. Full evaluation also needs the authors'
[131-HLA panel](https://raw.githubusercontent.com/orcacomputing/pep-q-gan/51f7dfa162e33a3be9ee90a69dd93f92e07d7788/data/unique_hla_allotypes.txt).
Datasets and predictor weights are not bundled. See [DATA_SOURCES.md](DATA_SOURCES.md)
for pinned sources, hashes, acquisition instructions and remaining source/terms findings.

## Results and Analysis

The full experiment used the 512-wide WGAN-GP, million-event photonic pools and
100k training updates. Each prior has 30 seeds, 131 HLA and 1,000 generation
attempts per HLA.

| Prior | Screening hits per 131,000 attempts, mean ± seed SEM |
|---|---:|
| Gaussian | 92,261 ± 537 |
| Dist | 91,606 ± 602 |
| Indist | 91,423 ± 696 |


Hits use no-flank MHCflurry presentation percentile `<0.5`. Gaussian has the highest mean in this setup. 
These results do not establish general quantum advantage. 
The changed input and evaluator prevent a direct comparison with the paper's GBS/NetMHCpan scores. 

[Figure 2](results/figure2.png) · [Figure 3](results/figure3.png) ·
[Figure 8](results/figure8.png) · [Table 1](results/table1.png).
See [figure captions](results/README.md) for details. These are archived
full-model figures, separate from the notebook's live demonstration. Only PNG/SVG
images are distributed; datasets, checkpoints and experiment-output tables remain external.

## Extensions and Next Steps

Our fixed-prior experiments found no clear benefit from Guassian and Indist over Dist.
Related work suggests two considerations for prior design:

- **Generator capacity:** [Complexity Matters](https://papers.neurips.cc/paper_files/paper/2023/hash/5e8023f07625374c6fdf3aa08bb38e0e-Abstract-Conference.html)
  (Hu et al., NeurIPS 2023) studies data-dependent latent distributions that make
  mapping to the target distribution less demanding for a limited-capacity generator.
- **Data structure:** [BourGAN](https://proceedings.neurips.cc/paper_files/paper/2018/file/8a3363abe792db2d8761d6403605aeb7-Paper.pdf)
  (Xiao et al., NeurIPS 2018) constructs a Gaussian-mixture prior reflecting the
  target data's mode structure and geometry, combined with a distance-preserving
  training objective.

These studies motivate designing priors around both the target data's structure
and the generator's capacity. Learning the prior is one possible approach:
[SLOGAN](https://openreview.net/forum?id=2-mkiUs9Jx7) (Hwang et al., ICLR 2022)
optimizes a Gaussian-mixture prior, while
[QC-AAN](https://doi.org/10.1103/PhysRevX.12.031010) (Rudolph et al., PRX 2022)
trains a quantum-circuit prior using discriminator features. Whether such
adaptation improves this peptide model remains untested. Further controls could
vary generator capacity and separate circuit seeds from GAN initialization seeds.

## Reproducibility Notes

Keep circuit/event/training seeds, package versions, thread count and pool order
fixed. Checked exact sampling replay uses a fresh backend and one thread;
cross-version or cross-device identity is not guaranteed. The notebook's synthetic
9-mer validity follows from its output alphabet and length, not biological validation.

## Testing

```bash
python -m pip install pytest
cd papers/pep_q_gan
python -m pytest -q
```

## Citation and License

Cite the original paper and preserve [LICENSE.pep-q-gan](LICENSE.pep-q-gan) and
[NOTICE.md](NOTICE.md). External data and model terms are documented separately
in [DATA_SOURCES.md](DATA_SOURCES.md).

```bibtex
@article{Engdal2026.07.09.736951,
  author = {Engdal, Emilie Sofie and Funk, Jonathan and Bacarreza, Omar and Machado, Laura and Johansen, Kristoffer Haurum and Kemming, Janine and Farnsworth, Thorin and Brasas, Valentas and Lefevre-Morand, Rodrigue Yves Louis and Slysz, Mateusz and Noerregaard, Oliver Luscombe and Sandberg, Oliver August Dall Alba and Makarovskiy, Alexander and Lodahl, Peter and Acevedo-Rocha, Carlos G. and Kurowski, Krzysztof and Hadrup, Sine Reker and Clements, William R. and Jenkins, Timothy},
  title = {Hybrid quantum-classical de novo design of MHC-binding peptides},
  elocation-id = {2026.07.09.736951},
  year = {2026},
  doi = {10.64898/2026.07.09.736951},
  URL = {https://www.biorxiv.org/content/early/2026/07/10/2026.07.09.736951},
  eprint = {https://www.biorxiv.org/content/early/2026/07/10/2026.07.09.736951.full.pdf},
  journal = {bioRxiv}
}
```
