# results/

`summary.json` is the committed evidence for every number in `../README.md`: one entry per
arm (instances, % optimal, mean and standard deviation of relative error, mean click density,
candidate budget, coverage, timing) and the 17 paired comparisons the discussion relies on,
each with its McNemar z and paired t.

The raw per-instance rows are **not** committed — 6 900 JSONL lines, and `AGENTS.md` keeps
raw run output out of the repository. They are produced by the GPU queue:

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
