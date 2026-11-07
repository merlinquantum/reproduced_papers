# Curated results

Tracked record of the reproduction's key findings. Raw run artefacts
(per-fold metrics, configs, logs) live in the gitignored `outdir/`;
everything here is compact and regenerable. Full analysis: the paper
`README.md` ("Results Obtained").

## Headline findings

1. **The paper's headline MoE advantage does not reproduce.** At n=100
   CV folds, the paired mean AUCPR difference (MoE − XGBoost) is
   negative and statistically significant at the paper's headline
   thresholds for every backend tested (gate, classical ablation, both
   photonic variants), e.g. gate at γ=0.5: −0.017 [95% CI −0.028,
   −0.006], paired-t p=0.004. See `fig_mean_diff_vs_gamma.png`.
2. **The mechanism is a left-skewed minority of collapse folds.** Most
   folds slightly favor MoE (win-rates 63–70%, medians ≈ 0), but 5–11%
   of folds show catastrophic AUCPR collapse (down to −0.49), driven by
   calibration instability on tiny near-separable validation folds. See
   `fig_fold_diff_distribution.png`.
3. **The quantum block is not the differentiator**: the parameter-matched
   classical ablation behaves like the gate model, and the trainable
   photonic readout is the *worst* variant (−0.060 mean, p<0.001) —
   contrary to the hypothesis that a trainable readout would close the
   photonic gap.
4. **Table 1 latency: ranking reproduced, magnitude not.** GQC is the
   fastest and QMKL the slowest of the three, as claimed, but the
   measured speedup is 3–10×, not the paper's 542–1387×. See
   `fig_latency.png`.
5. **Router-threshold bug: found, fixed, bounded.** A `roc_curve`
   synthetic-`inf` threshold could corrupt router targets; it fired on
   ≤3 of 100 folds per config, none of them collapse folds, and
   excluding every affected fold changes no verdict (all p ≤ 0.038).
   Fixed in `lib/moe.py` with regression tests; the exclusion analysis
   is exact because the fix provably alters only the affected folds.

## File index

| File | Contents |
|---|---|
| `analysis_<config>.json` (×6) | per-threshold paired statistics + per-fold AUCPR differences for each powered (n=100) config, incl. source run id |
| `latency_benchmark.json` | measured ms/sample for QMKL / GFM / GQC (Table 1 comparison) |
| `fig_fold_diff_distribution.png` | per-fold differences at γ=0.5 across all six configs |
| `fig_mean_diff_vs_gamma.png` | mean paired difference ± 95% CI over the router-threshold sweep |
| `fig_latency.png` | measured vs paper-reported inference latency (log scale) |

## Regenerating

```bash
# from the repo root: the six powered runs (n=100 folds, ~40-90 min each)
bash papers/MoE_fraud_detection/utils/launch_powered_runs.sh
# latency benchmark
python implementation.py --paper MoE_fraud_detection --config configs/latency_benchmark.json

# from the paper directory: promote a run's analysis into results/
python utils/analyze_powered_runs.py outdir/<config>/run_<STAMP> \
    --json-out results/analysis_<config>.json
# figures (read only tracked results/ JSONs)
python utils/plot_results.py
```
