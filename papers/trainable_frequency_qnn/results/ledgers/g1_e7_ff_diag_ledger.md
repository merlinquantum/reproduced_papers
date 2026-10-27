### Sweep Execution Ledger — `g1_e7_ff_diag`

Sweep dir `outdir/sweeps/g1_e7_ff_diag`; status COMPLETED; 3/3 runs completed; code e7beb6a4f0 (dirty=True); started 2026-10-26T11:05:41Z, completed 2026-10-26T12:14:31Z.

| Candidate | Seed | Attempts | Status | Run log | Run status | Config snapshot | Metrics |
|---|---:|---:|---|---|---|---|---|
| ff_15k_iterations | 1 | 1 | COMPLETED | `ff_15k_iterations/seed_1/attempt_1/run_20261026-120546/run.log` | `ff_15k_iterations/seed_1/attempt_1/run_20261026-120546/run_status.json` | `ff_15k_iterations/seed_1/attempt_1/run_20261026-120546/config_snapshot.json` | `ff_15k_iterations/seed_1/attempt_1/run_20261026-120546/metrics.json` |
| ff_minmax | 1 | 1 | COMPLETED | `ff_minmax/seed_1/attempt_1/run_20261026-120546/run.log` | `ff_minmax/seed_1/attempt_1/run_20261026-120546/run_status.json` | `ff_minmax/seed_1/attempt_1/run_20261026-120546/config_snapshot.json` | `ff_minmax/seed_1/attempt_1/run_20261026-120546/metrics.json` |
| ffs | 1 | 1 | COMPLETED | `ffs/seed_1/attempt_1/run_20261026-120546/run.log` | `ffs/seed_1/attempt_1/run_20261026-120546/run_status.json` | `ffs/seed_1/attempt_1/run_20261026-120546/config_snapshot.json` | `ffs/seed_1/attempt_1/run_20261026-120546/metrics.json` |

Exact command per candidate (seed and output folder vary per run):

- `ff_15k_iterations`: `/opt/miniconda3/envs/trainable_freq/bin/python /Users/brian_quandela/Workspace/Reproduced_papers/reproduced_papers/implementation.py --paper trainable_frequency_qnn --config /Users/brian_quandela/Workspace/Reproduced_papers/reproduced_papers/papers/trainable_frequency_qnn/outdir/sweeps/g1_e7_ff_diag/candidate_configs/ff_15k_iterations__rep0.json --seed 1 --outdir /Users/brian_quandela/Workspace/Reproduced_papers/reproduced_papers/papers/trainable_frequency_qnn/outdir/sweeps/g1_e7_ff_diag/ff_15k_iterations/seed_1/attempt_1 --device cpu`
- `ff_minmax`: `/opt/miniconda3/envs/trainable_freq/bin/python /Users/brian_quandela/Workspace/Reproduced_papers/reproduced_papers/implementation.py --paper trainable_frequency_qnn --config /Users/brian_quandela/Workspace/Reproduced_papers/reproduced_papers/papers/trainable_frequency_qnn/outdir/sweeps/g1_e7_ff_diag/candidate_configs/ff_minmax__rep0.json --seed 1 --outdir /Users/brian_quandela/Workspace/Reproduced_papers/reproduced_papers/papers/trainable_frequency_qnn/outdir/sweeps/g1_e7_ff_diag/ff_minmax/seed_1/attempt_1 --device cpu`
- `ffs`: `/opt/miniconda3/envs/trainable_freq/bin/python /Users/brian_quandela/Workspace/Reproduced_papers/reproduced_papers/implementation.py --paper trainable_frequency_qnn --config /Users/brian_quandela/Workspace/Reproduced_papers/reproduced_papers/papers/trainable_frequency_qnn/outdir/sweeps/g1_e7_ff_diag/candidate_configs/ffs__rep0.json --seed 1 --outdir /Users/brian_quandela/Workspace/Reproduced_papers/reproduced_papers/papers/trainable_frequency_qnn/outdir/sweeps/g1_e7_ff_diag/ffs/seed_1/attempt_1 --device cpu`
