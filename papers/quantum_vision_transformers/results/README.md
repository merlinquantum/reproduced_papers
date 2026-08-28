# Results

This folder exists to match the repository contribution template (`how_to_contribute.md`).

Packaging policy — everything committed here must be:
- a key, representative figure or summary table of the reproduction (PNG/CSV only, no PDFs), and
- recomposable from raw run data: rerunning the benchmarks (see `scripts/suites/`) and then
  `python scripts/analysis/generate_figures.py outdir/ --out results/figures/`
  repopulates this folder in the same layout.

Layout: `figures/{circuit_family}/{profile}/` holds per-variant `comparison_*` accuracy figures,
`param_comparison`, `summary.csv`, and the RetinaMNIST (headline dataset) training-curve figure.
The generator emits only these per-family/profile bundles — a combined all-variants view is
redundant with them and unreadable at scale (pass `--out` for a single combined bundle if ever
needed). Sector-mass evolution figures are produced by the same script for runs that log
multi-epoch sector history.

Raw run artifacts stay untracked in `outdir/` (timestamped `run_YYYYMMDD-HHMMSS/` folders).
Extended per-dataset diagnostics and written analysis reports are not committed; they can be
regenerated from `outdir/` or found in this branch's git history.
