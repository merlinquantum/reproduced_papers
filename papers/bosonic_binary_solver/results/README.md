# results/

`summary.json` is the committed evidence for every BBS number in `../README.md`: one entry
per arm (instances, % optimal, mean and standard deviation of relative error, mean click
density, candidate budget, coverage, timing) and the 32 paired comparisons the discussion
relies on, each with its McNemar z and paired t.

`baselines.json` is the same for the classical baselines — simulated annealing, hill climbing
and uniform random search at the solver's Appendix B candidate budget, on the same instance
seeds — written by `utils/run_baselines.py`.

`click_density_m30.png` is the figure embedded in `../README.md`, redrawn from the two JSON
files above with `python utils/plot_density.py`.

The raw per-instance rows are **not** committed — 11 240 JSONL lines over 4 340 unique runs,
and `AGENTS.md` keeps raw run output out of the repository. They are produced by the GPU
queue:

```bash
scripts/run_gpu.sh --host USER@GPU --key ~/.ssh/KEY --clifford /path/to/gpu-clifford
scripts/run_gpu.sh --host USER@GPU --key ~/.ssh/KEY --fetch
```

and this file is regenerated from them with:

```bash
python utils/summarise.py results/gpu_results_*.jsonl --out results/summary.json
```

Reading a paired comparison: `mcnemar_z` is positive when arm **b** solves more instances,
and `mean_error_gap_percent` is negative when **b** has the lower error. The two agree
throughout; where they disagree in sign the effect is not significant either way.
