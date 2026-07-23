# Reproduction artefacts

Curated copies of the most informative runs and their figures.

| File | Source run | Notes |
|---|---|---|
| `poisson_qpinn_smoke_200ep.json` | outdir/run_20260528-120233 | QPINN smoke, 2+2 layers, cutoff 8, 200 epochs |
| `poisson_qpinn_smoke_200ep_predictions.json` | same | 200-point u(x) array (predictions, reference) |
| `poisson_pinn_3000ep.json` | outdir/run_20260528-120837 | Classical PINN baseline, 90 params, 3000 ep, lr 5e-3 |
| `poisson_pinn_3000ep_predictions.json` | same | predictions |
| `poisson_merlin_600ep.json` | outdir/run_20260528-120857 | MerLin linear-optics adaptation, 600 epochs |
| `poisson_merlin_600ep_predictions.json` | same | predictions |
| `poisson_compare.png` | combined | side-by-side QPINN / PINN / MerLin prediction vs analytic |
| `heat_qpinn_smoke_250ep.json` | outdir/run_20260528-120909 | Heat QPINN smoke (60 IC + 250 full epochs, cutoff 10) |
| `heat_qpinn_smoke_250ep_predictions.json` | same | heatmap arrays |
| `heat_qpinn_grid.png` | same | prediction / reference / abs-error heatmaps |
| `heat_pinn_1000ep.json` | outdir/run_20260528-120934 | Classical PINN heat baseline (42 params target 44) |
| `heat_pinn_1000ep_predictions.json` | same | heatmap arrays |
| `heat_pinn_grid.png` | same | prediction / reference / abs-error heatmaps |

To regenerate from a fresh run:

```bash
python implementation.py --paper CV_QPINN_PDE --config configs/poisson_smoke.json
python implementation.py --paper CV_QPINN_PDE --config configs/poisson_pinn.json --epochs 3000 --lr 0.005
python implementation.py --paper CV_QPINN_PDE --config configs/poisson_merlin.json
python implementation.py --paper CV_QPINN_PDE --config configs/heat_smoke.json --epochs 250
python implementation.py --paper CV_QPINN_PDE --config configs/heat_pinn.json
python utils/plot_poisson.py outdir/run_<...>/ --out results/poisson_compare.png
python utils/plot_heat.py outdir/run_<...> --out results/heat_qpinn_grid.png
```
