# Curated results

Tracked record of the reproduction's key findings. Raw run artefacts
(per-run configs, logs, checkpoints) live in the gitignored `outdir/`;
everything here is a compact, regenerable view of those runs. Full
analysis and tables: the paper `README.md` ("Results Obtained").

## Headline findings

Synthetic GMM, 3 seeds, 60 epochs. Both of the paper's proposed
mechanisms are **refuted** by controlled ablations:

1. **Eq. 7 quantum-randomness modulation is inert.** Pinning the VQC
   output to a constant `r = 0.5` gives tail KL 0.177 ± 0.048 —
   statistically indistinguishable from the trained gate VQC
   (0.208 ± 0.033). See `metrics_const_ablation.json` and
   `fig_tail_kl_with_const.png`.
2. **The tail-aware loss term (Eq. 5) is mildly counterproductive.**
   Setting λ_tail = 0 improves tail KL for every variant (VAE −16%,
   gate QEGM −41%, MerLin QEGM −6%). See
   `metrics_no_tail_ablation.json`.

The MerLin photonic counterpart (0.149 ± 0.014) is the best-performing
quantum-side variant and ~5.7x faster than the gate VQC.

## File index

| File | Contents |
|---|---|
| `metrics.json` | main multi-seed run: per-seed and aggregate tail KL / rare recall / coverage / wall-clock for vae, qegm, qegm_merlin |
| `metrics_const_ablation.json` | ablation 1 (constant `r = 0.5` replaces the VQC) |
| `metrics_no_tail_ablation.json` | ablation 2 (λ_tail = 0) |
| `fig_tail_kl.png`, `fig_tail_kl_with_const.png` | tail-KL comparison, without/with the const-r ablation |
| `fig_densities.png` | generated vs true densities (log scale, tail region) |
| `fig_coverage.png`, `fig_rare_recall.png` | coverage and rare-recall comparisons |
| `fig_threshold_sweep.png` | tail-KL as a function of the rarity threshold |

## Regenerating

```bash
# from the repo root — multi-seed run and both ablations (~10-15 min CPU each)
python implementation.py --paper QEGM_rare_events --config configs/synthetic_multi_seed.json
python implementation.py --paper QEGM_rare_events --config configs/ablation_const_r.json
python implementation.py --paper QEGM_rare_events --config configs/ablation_no_tail.json
# then, from the paper directory
python utils/plot_results.py outdir/run_<STAMP>       # figures
python utils/summarize_ablations.py --help            # see path flags for the three metrics files
```
