# Repository Instructions

These instructions apply to the whole repository.

## Working Style

- Be direct and specific.
- Read the local code before changing it. Match the patterns already used by the
  target paper when those patterns do not conflict with this file.
- When a test fails, determine the root cause before changing code.
- Do not hide failures with broad default fallbacks during development. If a value,
  dependency, file, or config key is required, let the code fail clearly. Add a
  `# TODO` only when the missing behavior is intentional and should be handled
  later.
- Use clear, descriptive variable and function names. Prefer self-explanatory
  code over comments that restate each line.
- Keep changes scoped to the paper or shared module being refactored.
- Do not rewrite generated artifacts, notebooks, cached files, or unrelated
  results unless the task explicitly asks for it.

## Repository Model

This repository is a catalogue of reproduced quantum machine learning papers.
Each runnable reproduction lives under `papers/` and is executed by the
repository-level `implementation.py`.

The shared runtime discovers a paper project by these files:

```text
papers/<paper_name>/
|-- configs/defaults.json
|-- cli.json
`-- lib/runner.py
```

`lib/runner.py` must expose the project entry point:

```python
train_and_evaluate(cfg, run_dir)
```

`cfg` is the resolved configuration after defaults, config overlays, global CLI
flags, and paper-specific CLI flags have been applied. `run_dir` is the
timestamped output directory created by the shared runtime.

Some older paper folders currently keep `cli.json` under `configs/cli.json`.
Treat that as a legacy layout. When refactoring one of those papers, migrate the
CLI schema to paper-root `cli.json` so `implementation.py --list-papers` and
`implementation.py --paper <name>` can discover it.

Nested paper suites are allowed. For example:

```text
papers/fock_state_expressivity/<subproject>/
```

Each runnable subproject must still satisfy the same project markers:
`configs/defaults.json`, paper-root `cli.json`, and `lib/runner.py`.

Do not add alternative runtime entry systems such as `runtime.json` or
`runtime_entry.py`. Use the shared runtime.

## Paper Folder Structure

New and refactored papers should follow this structure:

```text
papers/<paper_name>/
|-- README.md
|-- requirements.txt
|-- notebook.ipynb
|-- cli.json
|-- configs/
|   |-- defaults.json
|   `-- <experiment>.json
|-- lib/
|   |-- __init__.py
|   |-- runner.py
|   `-- <paper modules>.py
|-- tests/
|   `-- test_<behavior>.py
|-- utils/
|   `-- <analysis or launch helpers>.py
|-- models/
|-- outdir/
`-- assets/ or images/ or figures/
```

Directory roles:

- `README.md`: paper explanation, reproduction scope, run instructions, and
  obtained results.
- `requirements.txt`: dependencies needed by this paper. Keep paper-specific
  dependencies here instead of assuming another paper's environment.
- `notebook.ipynb`: interactive exploration or reproduction notebook when useful
  for the paper.
- `configs/defaults.json`: default runnable config.
- `configs/<experiment>.json`: named experiments, ablations, or paper settings.
- `cli.json`: paper-specific CLI schema. Global options like `--config`,
  `--outdir`, `--seed`, `--dtype`, `--device`, and `--log-level` are injected by
  the shared runtime.
- `lib/`: importable implementation used by both the runner and notebooks.
- `lib/runner.py`: runtime entry point. Keep orchestration here; keep model,
  data, plotting, and training details in separate modules when they grow.
- `tests/`: smoke tests, config tests, import tests, and focused behavior tests.
- `utils/`: optional command-line helpers for plotting, aggregation, sweeps, or
  batch launching. Utilities should call into `lib/` instead of duplicating core
  logic.
- `models/`: trained model artifacts that are intentionally kept. Use `.gitkeep`
  if the directory should exist without committed models.
- `outdir/`: raw generated run directories. Treat this as disposable runtime
  output and avoid committing it unless a task explicitly curates a small file.
- `assets/`, `images/`, or `figures/`: static README images, paper diagrams, or
  curated visual material. Prefer one of these names per paper; keep references
  in README relative and valid.

Data does not belong inside each paper by default. Use the repository-level data
root:

```text
data/<paper_name>/
```

Reusable dataset or helper code belongs under:

```text
papers/shared/<topic_or_paper>/
```

Paper-local modules should import shared helpers through thin wrappers in
`papers/<paper_name>/lib/` when that keeps the paper API stable.

## Config and CLI Rules

- Configs are JSON-only.
- `configs/defaults.json` must be runnable or intentionally lightweight.
- Named configs should overlay `defaults.json` and represent clear experiments:
  dataset variants, model variants, ablations, smoke runs, or paper-accurate
  runs.
- Do not leave placeholder values such as `<<PATH>>` in runnable configs.
- Do not silently invent values when required config keys are missing. Validate
  the key and fail with a clear error.
- Keep CLI flags in `cli.json`; do not hard-code argument parsing in a paper
  runner unless there is no shared-runtime path for it.
- Keep CLI names stable when refactoring. If a flag changes, update tests and
  README commands in the same change.
- Use the top-level `dtype` convention when a paper supports precision control.
  The shared runtime normalizes dtype entries; do not duplicate dtype parsing
  unless the paper has a real model-specific need.

Standard run commands:

```bash
# From repository root
python implementation.py --paper <paper_name> --config configs/<experiment>.json

# From inside papers/<paper_name>
python ../../implementation.py --config configs/<experiment>.json

# Discover runnable projects
python implementation.py --list-papers
```

Each run should write a timestamped directory:

```text
outdir/run_YYYYMMDD-HHMMSS/
|-- config_snapshot.json
|-- run.log
`-- <paper artifacts>
```

Prefer structured artifacts for results that will be parsed later:
`metrics.json`, `summary.json`, `losses.csv`, `predictions.csv`, `.npz`, model
`state_dict` files, and figure files.

## README Requirements

Every paper README must explain the original paper, the reproduction updates,
how to run the code, and the results obtained.

Use these sections unless the paper has a stronger existing structure:

```text
# <Paper Title> - Reproduction

## Reference and Attribution
## Original Paper
## Reproduction Scope (incuding Updates and Deviations)
## Project Layout (optional as it is fixed by the template)
## Install and How to Run
## Configuration
## Data (optional if the dataset is well-known)
## Results Obtained and Comparison with the Paper
## Limitations
## Tests (optional description)
## Citation and License
```

README content requirements:

- State the paper title, authors, venue/year, DOI or arXiv link, original
  repository if any, and license or attribution notes.
- Explain the original paper in plain language: problem, model, datasets,
  metrics, and main claims.
- State exactly what this repository reproduces: datasets, models, figures,
  tables, metrics, seeds, and hardware/software assumptions.
- List updates and deviations from the paper: MerLin or photonic migration,
  optimizer changes, changed datasets, smaller smoke settings, extra baselines,
  different precision, faster configs, missing QPU access, or unfinished pieces.
- Include install commands using the paper's `requirements.txt`.
- Include run commands from the repo root and, when useful, from inside the
  paper folder.
- Show quick smoke commands separately from full paper reproduction commands.
- Document important configs and CLI flags. Point to `cli.json` as the
  authoritative schema.
- Explain data location. Use `data/<paper_name>/` or the relevant
  `papers/shared/` path.
- Explain output paths, especially `outdir/run_YYYYMMDD-HHMMSS/` for raw runs
  and `results/` for curated artifacts.
- Show obtained results with concrete metrics, tables, or figure links. Include
  enough detail to compare against the original paper.
- If results are preliminary or missing, say so directly and include the blocker
  or `# TODO` work item.
- Link README figures to committed curated files, not to disposable `outdir`
  paths.
- Update the root `README.md` paper table when a paper's result summary changes.

## Code Comments and Docstrings

Code should be self-explanatory first. Add clear comments where they explain
intent that is not obvious from names and control flow.

Good places for comments:

- Mapping code to original paper equations, algorithms, figures, or notation.
- Tensor shapes, optical modes, qubit wires, photon counts, or basis ordering
  when a wrong convention would silently corrupt results.
- Non-obvious numerical choices, optimizer behavior, seeding constraints, or
  determinism limits.
- File artifacts written by a runner when those artifacts are part of the public
  reproduction contract.
- Compatibility shims used during a refactor, with a `# TODO` and removal
  condition.

Avoid comments that restate the code:

```python
# Bad: increment epoch
epoch += 1
```

Prefer a useful intent comment:

```python
# The paper reports losses after each discriminator update, so keep generator
# and discriminator histories separate for table reproduction.
```

Public functions, classes, and modules should have Napoleon-compatible NumPy
style docstrings. Keep the section names and indentation consistent:

```python
def train_model(config, train_loader, valid_loader):
    """Train the model for one configured experiment.

    Parameters
    ----------
    config : dict
        Resolved experiment configuration.
    train_loader : torch.utils.data.DataLoader
        Training batches.
    valid_loader : torch.utils.data.DataLoader
        Validation batches.

    Returns
    -------
    dict
        Metrics collected during training.

    Raises
    ------
    ValueError
        If the configuration is missing a required key.
    """
```

For optional parameters, document the default value explicitly:

```text
layer : QuantumLayer | None
    Photonic layer used for the quantum branch. If omitted, the caller must
    provide `layer_factory`. Default value is None.
```

Do not use docstrings to excuse unclear code. Rename variables and split helper
functions first.

## Testing and Verification

- Run tests from inside the target paper when tests assume paper-local imports:

  ```bash
  cd papers/<paper_name>
  pytest -q
  ```

- Use root commands when validating shared runtime discovery or CLI wiring:

  ```bash
  python implementation.py --list-papers
  python implementation.py --paper <paper_name> --help
  python implementation.py --paper <paper_name> --config configs/<experiment>.json
  ```

- Run focused tests first, then broader tests if the change touches shared
  runtime, shared datasets, or multiple papers.
- For style checks, use the repository Ruff config:

  ```bash
  ruff format .
  ruff check .
  ```

- If a test fails after a refactor, inspect the failing assertion, config, and
  artifact path before changing implementation.
- For documentation-only changes, tests are not required unless commands,
  configs, or documented behavior changed.

## Refactoring Checklist for Existing Papers

When refactoring incoming or legacy code into this repository:

- Place the implementation under `papers/<paper_name>/`.
- Add or preserve `README.md`, `requirements.txt`, `configs/defaults.json`,
  paper-root `cli.json`, `lib/runner.py`, `tests/`, `utils/`, `models/`,
  `outdir/`, and `results/`.
- Move paper-specific runtime logic into `lib/runner.py::train_and_evaluate`.
- Move model, dataset, training, plotting, and artifact helpers into named
  modules under `lib/` or `utils/`.
- Convert ad hoc command-line parsing to `cli.json` whenever the shared runtime
  can express the options.
- Keep generated raw runs in `outdir/`; keep README-worthy figures and compact
  summaries in `results/`.
- Move reusable cross-paper code into `papers/shared/<topic_or_paper>/`.
- Keep data paths configurable through the shared data root. Do not hard-code
  absolute local paths.
- Add smoke tests for imports, config loading, and a tiny run when feasible.
- Add focused tests for any scientific or numerical behavior changed by the
  refactor.
- Update the paper README and root README summary in the same change when the
  visible reproduction behavior changes.
- Remove caches and local system files from changes: `__pycache__/`,
  `.pytest_cache/`, `.ruff_cache/`, `.DS_Store`, and large raw run directories.
