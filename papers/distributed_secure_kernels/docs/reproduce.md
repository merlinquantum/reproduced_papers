# Reproducing the stored results

The README gives the run commands. This note lists every config, the sweep that ran it, and the scripts that rebuild `results/` from those sweeps.

## Configs and sweeps

Each run writes `outdir/run_YYYYMMDD-HHMMSS/` with `run.log`, `run_status.json`, `config_snapshot.json` and `metrics.json`; each sweep (`utils/run_sweep.py`, 5 seeds) writes `outdir/sweep_<id>_<stamp>/` with `sweep.log`, `sweep_status.json` and `summary.{json,md}`. Run and sweep folders are not committed.

| Config (`configs/`) | Experiment | Sweep id | Feeds |
|---|---|---|---|
| `accuracy_<dataset>_original.json` | the paper's three columns, paper protocol | `table1_original` | Table 1 |
| `accuracy_<dataset>_reference.json` | exact squared-overlap and cosine kernels, majority class, paper protocol | `table1_original` | Table 1 (additional baselines) |
| `accuracy_<dataset>_corrected.json` | the paper's columns, corrected protocol | `table1_corrected` | Table 1 bis |
| `accuracy_<dataset>_classical.json` | tuned RBF / polynomial SVMs on a first γ grid, 10× off the planned one (superseded by `_fair_classical`) | `table1_corrected` | not used in the results |
| `accuracy_<dataset>_fair_classical.json` | tuned RBF / polynomial SVMs on the planned grids, corrected protocol | `fair_comparison` | Table 1 bis (RBF SVM) |
| `accuracy_{wine,parkinsons}_merlin_corrected.json` | MerLin adaptation, corrected protocol | `fair_comparison` | `results/fair_comparison.md` |
| `noise_<dataset>_original.json` | Fig. 3 | `noise_original` | Fig. 3 |
| `noise_<dataset>_decomposed_initialize.json` | diagnostic: noisy, decomposed state preparation | `noise_decomposed_initialize` | Fig. 3 (diagnostic) |
| `shots_digits_original.json` | Fig. 4 | `shots_original` | Fig. 4 |
| `accuracy_<dataset>_merlin.json`, `shots_digits_merlin.json` | MerLin adaptation, Table 1 and Fig. 4 settings | `merlin_original` | MerLin results |
| `protocol_equivalence_original.json`, `protocol_equivalence_four_qubits.json` | claim 1: per-shot Aer runs of the authors' circuits | single runs (see the README) | `results/protocol_check.png` |

## Rebuilding `results/`

Run from the repository root, passing the sweep folders produced above. `export_sweep_tables.py` must run before `paper_layout_results.py`, which reads only `results/sweep_results.json`.

| Script | Arguments | Output |
|---|---|---|
| `utils/verify_sweeps.py` | any sweep folders | recomputes every summary row from the runs' `metrics.json` (printed) |
| `utils/compare_with_paper.py` | `table1_original`, `noise_original`, `shots_original` | `results/paper_comparison.{json,md}` |
| `utils/diagnose_gaps.py` | none | `results/gap_diagnostics.json` |
| `utils/compare_merlin.py` | `table1_original`, `shots_original`, `merlin_original` | `results/merlin_comparison.{json,md}`; exits with 1 when a setting is outside 2 σ (Digits at 1,024 shots) |
| `utils/fair_comparison.py` | `--corrected`, `--fair`, `--original`, `--merlin` | `results/fair_comparison.{json,md}` |
| `utils/export_sweep_tables.py` | all seven sweeps | `results/sweep_results.{md,json}` (the JSON keeps the per-seed values the notebook reads) |
| `utils/paper_layout_results.py` | none | `results/table1.md`, `table1_bis.md`, `table1_merlin.md`, `fig3.png`, `fig4.png`, `fig4_merlin.png` |
| `utils/plot_protocol_check.py` | none (latest protocol runs) | `results/protocol_check.png` |
| `utils/pooled_protocol_check.py` | none (latest protocol runs) | pooled χ² against the swap-test law and two wrong laws (printed) |
| `utils/merlin_noise_check.py` | none | MerLin distinguishability-noise check (printed) |
