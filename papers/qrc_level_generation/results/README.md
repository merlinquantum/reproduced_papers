# Curated results

Tracked record of the reproduction's key findings. Raw run artefacts
live in the gitignored `outdir/` (single runs) and `sweeps/` (multi-seed
scaling sweeps: generated sequences, per-run logs, aggregates);
everything here is a compact, regenerable view. Full analysis:
the paper `README.md` ("Results Obtained").

## Headline findings

1. **The paper's temperature-knob claim is confirmed on its own data.**
   Our metric implementations, applied to the authors' released Aer
   sequences, give a 2.8% broken-rate at T = 2 (paper: "below 5% up to
   T = 2") and the qualitative originality-vs-temperature trend of
   Fig. 3. See `reference_eval_metrics.json` /
   `reference_eval_originality.png`.
2. **Fresh gate-based and photonic QRCs reproduce the trend.** Our
   6-qubit gate QRC (`qrc_qubit_metrics.json`,
   `qrc_qubit_originality.png`) and the MerLin photonic analogue
   (`qrc_photonic_metrics.json`, `qrc_photonic_originality.png`) both
   show the same originality/broken-rate temperature behaviour;
   side-by-side in `originality_combined.png`.
3. **At iso output dimension (~70), the photonic UNBUNCHED reservoir
   Pareto-dominates the gate-based reservoir** on originality and
   broken-rate despite a higher teacher-forcing loss. Sweep tables in
   `sweep_tables.md`; figures `sweep_pareto.png`, `sweep_isodim_split.png`,
   `sweep_scaling_modes.png`, `sweep_scaling_photons.png`,
   `sweep_scaling_isodim.png`.

## File index

| File | Contents |
|---|---|
| `reference_eval_metrics.json` | our metrics computed on the authors' released sequences (V1) |
| `qrc_qubit_metrics.json`, `qrc_photonic_metrics.json` | trained gate / photonic QRC temperature sweeps (V3) |
| `*_originality.png`, `originality_combined.png` | originality-vs-temperature curves |
| `sweep_tables.md` | mode / photon / iso-dim scaling sweep summaries (3 seeds) |
| `sweep_*.png` | scaling-sweep figures |
| `level_*.png` | rendered level strips (original, authors' reference at T=1, our QRC at T=1 and T=30) via `utils/render_level.py` |

## Regenerating

```bash
# from the repo root
python implementation.py --paper qrc_level_generation --config configs/reference_eval.json
python implementation.py --paper qrc_level_generation --config configs/mario_qubit_paper.json
python implementation.py --paper qrc_level_generation --config configs/mario_photonic.json
# save-point check (needs the packaged Roblox beta_1 sequences)
python utils/investigate_save_point.py
# sweeps (long, ~hours; regenerates the gitignored sweeps/ from scratch)
python utils/sweep.py --sweep modes --out-root sweeps/modes
python utils/sweep.py --sweep photons --out-root sweeps/photons
python utils/sweep.py --sweep isodim --out-root sweeps/isodim
# then one-shot tables + Pareto + scaling figures into results/
python utils/finalise_sweeps.py
# level renderings (original / reference pickle / a run's generated sequences)
python utils/render_level.py --original --out results/level_original.png
python utils/render_level.py --npz outdir/<RUN>/generated_sequences.npz --temperature 1 --out results/level_qrc_T1.png
```
