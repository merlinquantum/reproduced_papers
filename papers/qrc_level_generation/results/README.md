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

## Regenerating

```bash
# from the repo root
python implementation.py --paper qrc_level_generation --config configs/reference_eval.json
python implementation.py --paper qrc_level_generation --config configs/mario_qubit_paper.json
python implementation.py --paper qrc_level_generation --config configs/mario_photonic.json
# sweeps (long): see utils/sweep.py; aggregate + tables + figures via
python utils/aggregate.py && python utils/print_sweep_table.py
python utils/plot_scaling.py && python utils/plot_pareto.py && python utils/plot_isodim_split.py
```
