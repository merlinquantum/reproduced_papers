# Feedback on the Reproduction Workflow

## What worked well

- The `runtime_lib` / `implementation.py` runner makes the CLI and per-paper
  config story very compact. Adding a new `experiment` switch in the runner
  is a clean way to share the same code path across QPINN, classical-PINN
  and MerLin variants.
- `runtime_lib.dtypes.coerce_dtype_spec` lifting `"float64"` to
  `(label, torch.dtype)` keeps the runner ergonomic — but see "friction
  points" below.
- The MerLin Cookbook (`/home/agent/MERLIN_COOKBOOK.md`) was sufficient to
  build the linear-optics adaptation without grepping any MerLin source.
  Pattern B (vector classifier with `LexGrouping`) is close enough to what
  we needed that the only delta was wiring two linear heads instead of a
  grouping.

## Friction points and missing guidance

- The template `tests/test_cli.py` and `tests/test_smoke.py` reference the
  template folder name (`reproduction_template`) hard-coded in `from
  reproduction_template.lib import runner as tpl_runner`. The
  `PAPER_REPRODUCTION_INSTRUCTIONS.md` mentions this in §7.1.1 but it
  is easy to miss; consider making the template tests parametric on
  `PROJECT_DIR.name`.
- The shared `runtime_lib` injects a `DtypeSpec` object into the resolved
  config which is *not* JSON-serialisable. Storing `cfg` in `summary.json`
  needed a `default=str` kwarg. A small `runtime_lib.serialize_config(cfg)`
  helper would save every paper from re-discovering this.
- For paper-accurate runs of a CV-quantum simulation the budget guidance
  (`< $50 total exploratory`) is appropriate for cloud compute but does not
  map cleanly to "CPU hours on a 64 GB workstation". A row in the LOG
  template explicitly for "estimated wall-clock for a paper-accurate run"
  would be useful.

## Dataset and environment issues

- Strawberry Fields' TensorFlow backend was the only honest path to a
  literal port and the install combination was incompatible with Python
  3.12 in the container. Worth documenting in `PAPER_REPRODUCTION_INSTRUCTIONS.md`
  as a known case — and worth mentioning the Fock-truncated PyTorch
  workaround as the recommended substitute for CV reproductions.
- `scipy.stats.qmc.Sobol` warns when `n` is not a power of 2. The paper's
  Poisson run uses 258 = 2^8 + 2 (the `+2` reserves the boundary points);
  passing 258 through `qmc.Sobol(d=1).random(n=258)` works fine but is
  noisy. Could be wrapped in a `papers.shared` Sobol helper to suppress
  the warning consistently.

## Reproduction ambiguity notes

- The paper's gate-count per layer does not match Table II once you sum
  Killoran's standard ansatz (we get 14 per multi-qumode layer + 10 per
  single-qumode layer = 96 for 4 + 4 layers, paper reports 88 for 4 hidden
  "layers"). We document the discrepancy and proceed; the precise count is
  not load-bearing for the scientific claims.
- Tables I and III report `λ` weights as percentages summing to 100%. We
  read those as relative weights; whether they should be re-normalised to
  sum to 1 after multiplying by the mean square errors is left implicit.
- The "consistency loss" in Eq. 17 is written `du/dx - ux`, not
  `(du/dx - ux)²`. We assume the squared form for stability.

## MerLin integration feedback

- The decision tree "is this paper photonic-CV or photonic-linear-optics?"
  is worth surfacing in `PAPER_REPRODUCTION_INSTRUCTIONS.md` because the
  MerLin policy in §7.4 implicitly assumes linear optics. A papers/HQPINN
  cross-link would help future agents know that a CV paper can either
  (a) be skipped from MerLin, (b) get a *different* photonic adaptation,
  or (c) be classified F7 (photonic incompatibility).

## Suggestions for improving PAPER_REPRODUCTION_INSTRUCTIONS.md

- Add a "CV-photonic vs linear-optics-photonic" section under §7.4 with
  the decision tree above.
- Add an explicit note that the smoke `defaults.json` config produced by
  the template must be runnable to completion, not just to import — the
  test gate currently checks only that artefacts appear, which is too
  weak to catch off-by-one bugs in the analytic reference.
- The §7.2.1 Docker policy is clear about installing additional libraries
  inside the container; would be useful to add `scipy` to the implicit
  list of "common reproduction helpers" since it was needed for both
  Sobol sampling and the RK45 reference.

## Net assessment

The workflow scales well to a paper that is methodologically novel but
empirically small (one Poisson + one heat equation). The biggest single
friction was the absence of a TF-backed CV simulator on Python 3.12 and
the need to roll a Fock-truncated PyTorch substitute; the rest of the
reproduction slotted into the template cleanly.
