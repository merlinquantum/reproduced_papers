Declared rule (Phase 5 Comparison Plan, LOG.md): per model, metric = median over the 5 selection agents (seeds 100-104) of the 0-based solve episode, unsolved censored at 5000; direction: lower is better; split: online interaction on selection seeds (evaluation seeds 0-9 are disjoint); checkpoint: final (agents stop at solve); aggregation: median over agents; tie tolerance 5% of the best median, ties go to the smaller batch, then the smaller target interval.

| Model | Params | Batch | Target update | Median episodes | Solved (of 5) | Status |
|---|---:|---:|---:|---:|---:|---|
| mlp_5_5 | 67 | 16 | 1 | 1837 | 3 |  |
| mlp_5_5 | 67 | 16 | 30 | 5000 | 1 |  |
| mlp_5_5 | 67 | 32 | 1 | 2222 | 3 |  |
| mlp_5_5 | 67 | 32 | 30 | 5000 | 2 |  |
| mlp_5_5 | 67 | 64 | 1 | 2557 | 3 |  |
| mlp_5_5 | 67 | 64 | 30 | 1487 | 5 | selected |
| mlp_14_14 | 310 | 16 | 1 | 258 | 4 |  |
| mlp_14_14 | 310 | 16 | 30 | 290 | 5 |  |
| mlp_14_14 | 310 | 32 | 1 | 214 | 5 | selected |
| mlp_14_14 | 310 | 32 | 30 | 295 | 5 |  |
| mlp_14_14 | 310 | 64 | 1 | 241 | 5 |  |
| mlp_14_14 | 310 | 64 | 30 | 217 | 5 | tied within 5% |
| pqc_l5 | 62 | 16 | 1 | 327 | 5 |  |
| pqc_l5 | 62 | 16 | 30 | 227 | 5 |  |
| pqc_l5 | 62 | 32 | 1 | 206 | 5 |  |
| pqc_l5 | 62 | 32 | 30 | 255 | 5 |  |
| pqc_l5 | 62 | 64 | 1 | 221 | 5 |  |
| pqc_l5 | 62 | 64 | 30 | 194 | 5 | selected |
| pqc_l25 | 302 | 16 | 1 | 1073 | 5 |  |
| pqc_l25 | 302 | 16 | 30 | 897 | 5 |  |
| pqc_l25 | 302 | 32 | 1 | 304 | 5 |  |
| pqc_l25 | 302 | 32 | 30 | 499 | 5 |  |
| pqc_l25 | 302 | 64 | 1 | 437 | 5 |  |
| pqc_l25 | 302 | 64 | 30 | 271 | 5 | selected |

Completeness: COMPLETE SWEEP, 24 of 24 planned candidate runs observed and COMPLETED (4 models x 6 settings), validated with `validate_logging.py`; `utils/select_phase5.py` raises on any missing, duplicate or non-100-104-seed candidate. Selections and ties:

- mlp_5_5: best median 1487; selected `e5_select_mlp_5_5_b64_t30_classical`; no ties
- mlp_14_14: best median 214; selected `e5_select_mlp_14_14_b32_t1_classical`; tied: e5_select_mlp_14_14_b64_t30_classical (smaller batch wins)
- pqc_l5: best median 194; selected `e5_select_pqc_l5_b64_t30`; no ties
- pqc_l25: best median 271; selected `e5_select_pqc_l25_b64_t30`; no ties

Uncertainty: each candidate has only 5 agents, so medians carry large sampling noise, in particular for NN (5,5) (medians of 1487-5000 with 1-5 of 5 solved); the selection is a protocol step, not evidence that the chosen setting is optimal. The selected settings were evaluated once on seeds 0-9 (`results/phase5_comparison.md`).
