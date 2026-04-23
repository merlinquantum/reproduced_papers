# Photonic Quantum Generative Adversarial Networks for classical data

## IMPORTANT: only in Perceval for now

## Reference and Attribution

- Paper: Photonic quantum generative adversarial networks for classical data (Optica Quantum, 2025)
- Authors: Tigran Sedrakyan, Alexia Salavrakos
- DOI/ArXiv: https://arxiv.org/abs/2405.06023
- Original repository: https://github.com/Quandela/photonic-qgan
- License and attribution notes: to complete


## Overview

"In generative learning, models are trained to produce new samples that follow the distribution of the target data. These models were historically difficult to train, until proposals such as Generative Adversarial Networks (GANs) emerged, where a generative and a discriminative model compete against each other in a minimax game. Quantum versions of the algorithm were since designed, both for the generation of classical and quantum data. While most work so far has focused on qubit-based architectures, in this article we present a quantum GAN based on linear optical circuits and Fock-space encoding, which makes it compatible with near-term photonic quantum computing. We demonstrate that the model can learn to generate images by training the model end-to-end experimentally on a single-photon quantum processor." [Original abstract of the [paper](https://opg.optica.org/opticaq/fulltext.cfm?uri=opticaq-2-6-458)]

![Photonic QGAN overview](./photonicQGAN.png)

## Project layout

```
.
├── README.md                                    # Overview, layout, and run commands
├── requirements.txt                             # Python dependencies
├── photonicQGAN.png                             # Figure for overview section
├── __init__.py                                  # Package marker
│
├── configs/
│   ├── cli.json                                 # CLI schema for runtime runner
│   ├── defaults.json                            # Default configuration for runs
│   ├── classical_comparison.json                # Classical baseline config (aligned with original notebook)
│   ├── ideal_selectable.json                    # Single selectable ideal config (one setup/input)
│   ├── Figure6.json                             # Figure 6 preset (full ideal grid, default photonic setup)
│   ├── Figure7.json                             # Figure 7 preset (tuned hp-study params, setup c/01010)
│   ├── digits_top1.json                         # Digits-mode config: top-1 hp-study candidate
│   ├── digits_top2.json                         # Digits-mode config: top-2 hp-study candidate
│   └── digits_top3.json                         # Digits-mode config: top-3 hp-study candidate
│
├── lib/
│   ├── __init__.py                              # Package marker
│   ├── runner.py                                # Runtime entrypoint and training loop
│   ├── qgan.py                                  # QGAN model wrapper
│   ├── generators.py                            # Classical and photonic patch generators
│   ├── discriminator.py                         # Discriminator network
│   ├── hp_study.py                              # Successive-halving hyperparameter search
│   └── classical_generator.dict                 # Serialized classical generator state
│
├── utils/
│   ├── __init__.py                              # Package marker
│   ├── mappings.py                              # Output mapping utilities for photonic states
│   ├── pqc.py                                   # Parametrized photonic quantum circuit helpers
│   ├── spsa.py                                  # Legacy optimizer utility (not used by current runner)
│   ├── visualize.py                             # Visualization helpers for training artifacts
│   ├── hp_study.py                              # HP study utilities (parameter analysis)
│   ├── hp_study_report.py                       # HP study reporting and ranking
│   ├── plot_config_report.py                    # Per-config training report plots
│   ├── plot_ssim_diversity.py                   # SSIM/diversity plot utilities
│   └── rank_ssim.py                             # SSIM-based ranking utilities
│
├── tests/
│   ├── common.py                                # Shared test utilities
│   ├── test_cli.py                              # CLI smoke tests
│   └── test_smoke.py                            # Basic run wiring smoke test
│
├── outdir/                                      # Generated run artifacts (logs, configs, CSVs, images)
├── results/                                     # Generated run artifacts (logs, configs, CSVs, images)
│
└── ../                                          # Parent directory
    ├── data/photonic_QGAN/
    │   └── optdigits_csv.csv                    # Dataset CSV
    └── papers/shared/photonic_QGAN/
        └── digits.py                            # Shared dataset utilities
```

## Modes

- `smoke`: quick placeholder run that writes a `done.txt` marker (no training, for test purposes).
- `digits`: trains on digit subsets from the Optdigits CSV; uses the `digits` config block
  (`arch`, `noise_dim`, `input_state`, `gen_count`, `pnr`) and the digit selection
  (`digits` list or `digit_start`/`digit_end`).
- `ideal`: sweeps a grid of generator configs for one or more target digits. The digit(s) to
  evaluate are controlled by `ideal.digits` / `--ideal-digits` (comma-separated list, e.g.
  `0,1,3`); or `ideal.digit` / `--ideal-digit` for a single digit (default `0`). One
  `ideal-{digit}/` subfolder is produced per digit, all under the same `run_<timestamp>/`
  directory. The grid source is resolved in priority order:
  1. `ideal.config_grid_path` — path to a JSON file containing a list of configs
  2. `ideal.config_grid` — inline list of configs in the config JSON
  3. `ideal.setup` + `ideal.input_state` — single selectable setup (as in `configs/ideal_selectable.json`)
  4. Default fallback — all four setups × all supported input states
  Each config is saved under `results/.../ideal-{digit}/<setup>/config_<n>_input_<state>/`.
- `hp_study`: runs `HalvingGridSearchCV` over ideal-mode training hyperparameters and ranks
  candidates using final SSIM averaged across configured setups/input-states/digits.

## Hyperparameter study results

When replacing SPSA by Adam optimizer, different hyperparameters needed to be studied to reproduce similar results as in the original paper using [Halving search](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.HalvingGridSearchCV.html#sklearn.model_selection.HalvingGridSearchCV). The study explored the following hyperparameter ranges:

| Parameter | Range | Impact |
|-----------|-------|--------|
| **Learning Rate (Generator)** | 0.001 - 0.004 | High |
| **Learning Rate (Discriminator)** | 0.0002 - 0.001 | High |
| **Adam β₁ (Momentum)** | 0.5, 0.7 | Medium |
| **Adam β₂ (RMS Decay)** | 0.99, 0.999 | High |
| **Discriminator Steps** | 1, 2 | Medium |
| **Generator Steps** | 1, 2, 3 | Medium |
| **Real Label Smoothing** | 0.9 | Low |
| **Generator Target** | 0.9 | Low |

The optimal hyperparameters identified at the final resource level (600 iterations):

```json
{
  "adam_beta1": 0.5,
  "adam_beta2": 0.99,
  "d_steps": 1,
  "g_steps": 3,
  "gen_target": 0.9,
  "lrD": 0.0002,
  "lrG": 0.004,
  "real_label": 0.9,
  "opt_iter_num": 600
}
```

With
- **Best Score (SSIM):** 0.570575
- **Rank:** 1st out of 323 evaluations
- **Resource Level:** 600 iterations
- **Training Time per Evaluation:** ~23.0 seconds

The latest study snapshot is stored under:
`papers/photonic_QGAN/results-old/run_20260213-104332/hp_study`.

Key setup used in this run:
- Setup: `setup_c`
- Input state: `01010`
- Digits: `[0, 3]`
- Successive-halving resources: `opt_iter_num` in `{200, 600}`


Importance summary (eta^2):
- `lrG`: `0.3832`
- `g_steps`: `0.3445`
- `lrD`: `0.2755`
- `d_steps`: `0.0464`
- `adam_beta1`: `0.0012`
- `adam_beta2`: `0.0003`

The image below presents the summary of the analysis of the hyperparameter study:

![Summary of HP Study](./assets/hp_study/summary_statistics.png)

Notes:
- Importance computed only on final-stage candidates is less robust here, because only 3 candidates reached `opt_iter_num=600`.
- The all-resource ranking is more informative for this specific study run.


## Running

Base command pattern:

```bash
python implementation.py --paper photonic_QGAN --config <CONFIG_JSON> --mode <MODE>
```

### Digits mode details

Purpose:
- Train one fixed generator architecture across one or several digit classes.

```bash
python implementation.py --paper photonic_QGAN --config configs/defaults.json --mode digits --digits 0,1,2 --runs 3
```

### Ideal mode details

Purpose:
- Evaluate photonic setup/input-state combinations for a single target digit dataset slice.

Typical commands:

```bash
# Full default grid, single digit
python implementation.py --paper photonic_QGAN --config configs/defaults.json --mode ideal --ideal-digit 0 --runs 5

# Multiple digits in one run (produces ideal-0/, ideal-1/, ideal-3/ under the same run directory)
python implementation.py --paper photonic_QGAN --config configs/defaults.json --mode ideal --ideal-digits 0,1,3 --runs 5

# All digits in one run
python implementation.py --paper photonic_QGAN --config configs/digits_top1.json --mode ideal --ideal-digits 0,1,2,3,4,5,6,7,8,9

# Single selectable setup/input
python implementation.py --paper photonic_QGAN --config configs/ideal_selectable.json --mode ideal --runs 5
```

### `configs/defaults.json`

Recommended baseline config.

- `--mode digits`: runs digit training using the `digits` block.
- `--mode ideal`: runs the full ideal sweep because `ideal.use_default_grid=true` (setups `a,b,c,d` with supported input states).
- `--mode hp_study`: runs successive-halving hyperparameter search.

```bash
python implementation.py --paper photonic_QGAN --config configs/defaults.json --mode digits
python implementation.py --paper photonic_QGAN --config configs/defaults.json --mode ideal
python implementation.py --paper photonic_QGAN --config configs/defaults.json --mode hp_study
```

### `configs/classical_comparison.json`

Classical baseline aligned with `photonic-qgan-main/notebooks/classical_gan.ipynb`.

- Uses `model.generator_type="classical"`.
- Uses digit `0` with `noise_dim=2`.
- Uses Adam settings from the notebook (`lr=0.0015`, betas `(0.5, 0.999)`).

```bash
python implementation.py --paper photonic_QGAN --config configs/classical_comparison.json --mode digits
```

### For Figure 6 and Figure 7 of the original paper

Figure-oriented preset based on the default photonic setup (same structure as defaults).

```bash
python implementation.py --paper photonic_QGAN --config configs/Figure6.json --mode ideal
python implementation.py --paper photonic_QGAN --config configs/Figure6.json --mode digits
```

Figure-oriented preset using the best hp-study training block (setup `c`, input `01010`, tuned Adam params).

```bash
python implementation.py --paper photonic_QGAN --config configs/Figure7.json --mode digits
python implementation.py --paper photonic_QGAN --config configs/Figure7.json --mode ideal
```

### `configs/ideal_selectable.json`

Single selectable ideal config (not the full ideal grid).

- Change `ideal.setup` (`setup_a|setup_b|setup_c|setup_d`) and `ideal.input_state`.
- Useful for controlled one-setup experiments and hp-study runs.

```bash
python implementation.py --paper photonic_QGAN --config configs/ideal_selectable.json --mode ideal
python implementation.py --paper photonic_QGAN --config configs/ideal_selectable.json --mode hp_study
```

### `configs/digits_top1.json`, `configs/digits_top2.json`, `configs/digits_top3.json`

Digits-mode configs corresponding to the top-1/top-2/top-3 hyperparameter choices from hp-study.

```bash
python implementation.py --paper photonic_QGAN --config configs/digits_top1.json --mode digits
python implementation.py --paper photonic_QGAN --config configs/digits_top2.json --mode digits
python implementation.py --paper photonic_QGAN --config configs/digits_top3.json --mode digits
```

### Run-specific archived top-3 configs

The `configs/digits_top1/2/3.json` files are derived from the hp-study run archived under
`results-old/run_20260213-104332`. They can be used directly without re-running the study:

```bash
python implementation.py --paper photonic_QGAN --config configs/digits_top1.json --mode digits
```


## Results

Here, we display the training curves and samples for digit `0` using the requested setup and input-state order.

### `results-final/all-configs-spsa`

#### setup A
<p float="left">
  <img src="./assets/results-final/all-configs-spsa/ideal-0/setup_a/config_0_input_01010/all_runs_report.png" width="45%" />
  <img src="./assets/results-final/all-configs-spsa/ideal-0/setup_a/config_0_input_1011/all_runs_report.png" width="45%" />
</p>

#### setup B
<p float="left">
  <img src="./assets/results-final/all-configs-spsa/ideal-0/setup_b/config_0_input_01010/all_runs_report.png" width="45%" />
  <img src="./assets/results-final/all-configs-spsa/ideal-0/setup_b/config_0_input_00100100/all_runs_report.png" width="45%" />
</p>

#### setup C
<p float="left">
  <img src="./assets/results-final/all-configs-spsa/ideal-0/setup_c/config_0_input_01010/all_runs_report.png" width="45%" />
  <img src="./assets/results-final/all-configs-spsa/ideal-0/setup_c/config_0_input_1011/all_runs_report.png" width="45%" />
</p>

#### setup D
<p float="left">
  <img src="./assets/results-final/all-configs-spsa/ideal-0/setup_d/config_0_input_01010/all_runs_report.png" width="45%" />
  <img src="./assets/results-final/all-configs-spsa/ideal-0/setup_d/config_0_input_00100100/all_runs_report.png" width="45%" />
</p>

### `results-final/all-configs-0`

#### setup A
<p float="left">
  <img src="./assets/results-final/all-configs-0/ideal-0/setup_a/config_9_input_01010/all_runs_report.png" width="45%" />
  <img src="./assets/results-final/all-configs-0/ideal-0/setup_a/config_6_input_1011/all_runs_report.png" width="45%" />
</p>

#### setup B
<p float="left">
  <img src="./assets/results-final/all-configs-0/ideal-0/setup_b/config_10_input_01010/all_runs_report.png" width="45%" />
  <img src="./assets/results-final/all-configs-0/ideal-0/setup_b/config_18_input_00100100/all_runs_report.png" width="45%" />
</p>

#### setup C
<p float="left">
  <img src="./assets/results-final/all-configs-0/ideal-0/setup_c/config_11_input_01010/all_runs_report.png" width="45%" />
  <img src="./assets/results-final/all-configs-0/ideal-0/setup_c/config_7_input_1011/all_runs_report.png" width="45%" />
</p>

#### setup D
<p float="left">
  <img src="./assets/results-final/all-configs-0/ideal-0/setup_d/config_12_input_01010/all_runs_report.png" width="45%" />
  <img src="./assets/results-final/all-configs-0/ideal-0/setup_d/config_20_input_00100100/all_runs_report.png" width="45%" />
</p>

### SSIM plots

<p float="left">
  <img src="./assets/results-final/all-configs-spsa/ideal-0/ssim_vs_diversity.png" width="45%" />
  <img src="./assets/results-final/all-configs-0/ideal-0/ssim_vs_diversity.png" width="45%" />
</p>
