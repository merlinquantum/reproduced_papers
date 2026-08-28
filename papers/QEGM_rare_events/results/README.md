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
2. **The tail-aware loss term (Eq. 5) is counterproductive.**
   Setting λ_tail = 0 improves tail KL for every variant: mildly on the
   GMM (VAE −16%, gate QEGM −41%, MerLin QEGM −10%), strongly on real
   returns (−59% to −68%). See `metrics_no_tail_ablation.json` and
   `metrics_sp500_no_tail.json`.
3. **Both refutations transfer to real data.** On 8314 standardized
   S&P 500 daily log-returns (1990–2022), the trained gate VQC
   (tail KL 1.599 ± 0.057) is indistinguishable from the const-r
   control (1.606 ± 0.064) and the VAE (1.611 ± 0.119). See
   `metrics_sp500.json` / `fig_sp500_*.png`.

The MerLin photonic counterpart (0.155 ± 0.018 on the GMM) is the
best-performing quantum-side variant and ~6.8x faster than the gate
VQC. All curated artefacts were produced under `merlinquantum` 0.4.

## File index

| File | Contents |
|---|---|
| `metrics.json` | main multi-seed GMM run: per-seed and aggregate tail KL / rare recall / coverage / wall-clock for vae, qegm, qegm_merlin |
| `metrics_const_ablation.json` | ablation 1 (constant `r = 0.5` replaces the VQC) |
| `metrics_no_tail_ablation.json` | ablation 2 (λ_tail = 0) |
| `metrics_sp500.json` | S&P 500 extension, λ_tail=2, all four variants |
| `metrics_sp500_no_tail.json` | S&P 500 extension, λ_tail=0 (gate proxied by const-r) |
| `fig_tail_kl_with_const.png` | GMM tail-KL comparison including the const-r control |
| `fig_densities.png` | GMM generated vs real densities |
| `fig_threshold_sweep.png` | GMM tail-KL as a function of the rarity threshold |
| `fig_sp500_tail_kl.png`, `fig_sp500_densities.png` | S&P 500 tail-KL (incl. const-r) and densities |

## Regenerating

```bash
# from the repo root — synthetic multi-seed run and both ablations
python implementation.py --paper QEGM_rare_events --config configs/synthetic_multi_seed.json
python implementation.py --paper QEGM_rare_events --config configs/ablation_const_r.json
python implementation.py --paper QEGM_rare_events --config configs/ablation_no_tail.json
# S&P 500 extension (the gate VQC makes these the slow ones: ~40 min/seed)
python implementation.py --paper QEGM_rare_events --config configs/sp500_ablations.json
python implementation.py --paper QEGM_rare_events --config configs/sp500_no_tail.json
# then, from the paper directory
python utils/plot_results.py outdir/run_<STAMP>       # figures
python utils/reevaluate.py outdir/run_<STAMP> --thresholds 1.5,2.0,2.5,3.0
python utils/summarize_ablations.py --help            # see path flags for the metrics files
```
