| System | Layout | Role | Provenance | Scope | Runs / expected | Test acc, seed 42 (mean +- std over 5 distributions) | All training seeds | Parameters (classical) | Median training wall-clock per run (5 runs in parallel, Apple M5 CPU) |
|---|---|---|---|---|---:|---:|---:|---|---:|
| classical_central:gate | central | centralised reference (no federation) | local run | lr validation-selected, 3 seeds | 15 / 15 | 93.29 +- 0.72 | 93.58 +- 0.42 | 72100 (72100) | 1.1 s |
| classical_central:photonic_f | central | centralised reference (no federation) | local run | lr validation-selected, 3 seeds | 15 / 15 | 91.54 +- 0.50 | 91.14 +- 0.58 | 2680 (2680) | 1.6 s |
| classical_central:photonic_h | central | centralised reference (no federation) | local run | lr validation-selected, 3 seeds | 15 / 15 | 91.79 +- 0.41 | 91.70 +- 0.37 | 3214 (3214) | 1.7 s |
| classical_gate | E1 | fair classical baseline (gate-matched) | local run | lr validation-selected, 3 seeds | 15 / 15 | 93.61 +- 0.79 | 93.06 +- 0.60 | 72059 (72059) | 2.2 s |
| classical_gate | E2 | fair classical baseline (gate-matched) | local run | lr validation-selected, 3 seeds | 15 / 15 | 75.42 +- 4.05 | 75.49 +- 3.18 | 16012 (16012) | 1.0 s |
| classical_gate | E3 | fair classical baseline (gate-matched) | local run | lr validation-selected, 3 seeds | 15 / 15 | 77.74 +- 3.81 | 77.39 +- 3.30 | 72059 (72059) | 1.9 s |
| classical_gate | E4 | fair classical baseline (gate-matched) | local run | lr validation-selected, 3 seeds | 15 / 15 | 91.97 +- 0.57 | 91.89 +- 0.48 | 72059 (72059) | 1.8 s |
| classical_photonic_f | E1 | fair classical baseline (F-matched) | local run | lr validation-selected, 3 seeds | 15 / 15 | 88.53 +- 1.39 | 87.98 +- 1.44 | 2895 (2895) | 3.6 s |
| classical_photonic_f | E2 | fair classical baseline (F-matched) | local run | lr validation-selected, 3 seeds | 15 / 15 | 65.89 +- 2.14 | 67.13 +- 1.03 | 571 (571) | 1.9 s |
| classical_photonic_f | E3 | fair classical baseline (F-matched) | local run | lr validation-selected, 3 seeds | 15 / 15 | 72.29 +- 4.97 | 72.06 +- 3.31 | 2895 (2895) | 3.6 s |
| classical_photonic_f | E4 | fair classical baseline (F-matched) | local run | lr validation-selected, 3 seeds | 15 / 15 | 84.83 +- 2.01 | 86.52 +- 1.01 | 2895 (2895) | 3.8 s |
| classical_photonic_h | E1 | fair classical baseline (H-matched) | local run | lr validation-selected, 3 seeds | 15 / 15 | 89.97 +- 0.59 | 90.05 +- 1.27 | 3270 (3270) | 3.6 s |
| classical_photonic_h | E2 | fair classical baseline (H-matched) | local run | lr validation-selected, 3 seeds | 15 / 15 | 68.28 +- 2.12 | 68.23 +- 1.23 | 989 (989) | 1.7 s |
| classical_photonic_h | E3 | fair classical baseline (H-matched) | local run | lr validation-selected, 3 seeds | 15 / 15 | 73.48 +- 3.54 | 73.44 +- 2.51 | 3270 (3270) | 3.6 s |
| classical_photonic_h | E4 | fair classical baseline (H-matched) | local run | lr validation-selected, 3 seeds | 15 / 15 | 88.59 +- 1.95 | 88.63 +- 1.69 | 3270 (3270) | 3.5 s |
| gate | E1 | reproduced paper method | local run | paper-accurate, 1 seed/distribution | 5 / 5 | 92.85 +- 0.98 | 92.85 +- 0.98 | 72000 (0) | 1604.1 s |
| gate | E2 | reproduced paper method | local run | paper-accurate, 1 seed/distribution | 5 / 5 | 71.41 +- 2.48 | 71.41 +- 2.48 | 16000 (0) | 327.8 s |
| gate | E3 | reproduced paper method | local run | paper-accurate, 1 seed/distribution | 5 / 5 | 76.24 +- 3.50 | 76.24 +- 3.50 | 72000 (0) | 2112.8 s |
| gate | E4 | reproduced paper method | local run | paper-accurate, 1 seed/distribution | 5 / 5 | 90.60 +- 1.97 | 90.60 +- 1.97 | 72000 (0) | 1792.7 s |
| photonic_f | E1 | MerLin adaptation (fully photonic) | local run | adaptation, simulator, tuned | 5 / 5 | 76.87 +- 3.15 | 76.87 +- 3.15 | 2744 (0) | 968.0 s |
| photonic_f | E2 | MerLin adaptation (fully photonic) | local run | adaptation, simulator, tuned | 5 / 5 | 48.28 +- 3.36 | 48.28 +- 3.36 | 560 (0) | 192.5 s |
| photonic_f | E3 | MerLin adaptation (fully photonic) | local run | adaptation, simulator, tuned | 5 / 5 | 61.07 +- 3.74 | 61.07 +- 3.74 | 2744 (0) | 1370.3 s |
| photonic_f | E4 | MerLin adaptation (fully photonic) | local run | adaptation, simulator, tuned | 5 / 5 | 75.05 +- 4.03 | 75.05 +- 4.03 | 2744 (0) | 1329.3 s |
| photonic_h | E1 | MerLin adaptation (head) | local run | adaptation, simulator, tuned | 5 / 5 | 86.83 +- 1.77 | 86.83 +- 1.77 | 3154 (410) | 960.1 s |
| photonic_h | E2 | MerLin adaptation (head) | local run | adaptation, simulator, tuned | 5 / 5 | 58.18 +- 5.85 | 58.18 +- 5.85 | 970 (410) | 193.2 s |
| photonic_h | E3 | MerLin adaptation (head) | local run | adaptation, simulator, tuned | 5 / 5 | 64.39 +- 3.09 | 64.39 +- 3.09 | 3154 (410) | 966.8 s |
| photonic_h | E4 | MerLin adaptation (head) | local run | adaptation, simulator, tuned | 5 / 5 | 81.13 +- 4.05 | 81.13 +- 4.05 | 3154 (410) | 986.4 s |

Paper-reported (not local, not ranked): E1 over 94%, E4 about 90%.

| Pair (quantum/photonic - classical) | Layout | Mean difference (points) | std | paired t p | positive in n of 5 | vs classical 3-seed mean | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| gate - classical_gate | E1 | -0.75 | 0.72 | 0.0802 | 0 | -0.21 | no clear difference |
| gate - classical_gate | E2 | -4.01 | 2.95 | 0.0384 | 0 | -4.08 | no clear difference |
| gate - classical_gate | E3 | -1.50 | 0.81 | 0.0143 | 0 | -1.15 | no clear difference |
| gate - classical_gate | E4 | -1.38 | 2.29 | 0.2490 | 1 | -1.30 | no clear difference |
| photonic_h - classical_photonic_h | E1 | -3.13 | 2.13 | 0.0300 | 1 | -3.22 | classical higher |
| photonic_h - classical_photonic_h | E2 | -10.09 | 6.30 | 0.0231 | 0 | -10.05 | classical higher |
| photonic_h - classical_photonic_h | E3 | -9.09 | 5.14 | 0.0167 | 0 | -9.05 | classical higher |
| photonic_h - classical_photonic_h | E4 | -7.46 | 3.18 | 0.0063 | 0 | -7.50 | classical higher |
| photonic_f - classical_photonic_f | E1 | -11.66 | 4.11 | 0.0032 | 0 | -11.12 | classical higher |
| photonic_f - classical_photonic_f | E2 | -17.62 | 4.81 | 0.0012 | 0 | -18.85 | classical higher |
| photonic_f - classical_photonic_f | E3 | -11.22 | 2.28 | 0.0004 | 0 | -10.99 | classical higher |
| photonic_f - classical_photonic_f | E4 | -9.78 | 5.64 | 0.0179 | 0 | -11.47 | classical higher |

| Sweep | System | Completed / expected |
|---|---|---:|
| outdir/sweep_e1_8clients_20261016-161327_e1 | gate | 5 / 5 |
| outdir/sweep_e2_1client_20261016-164104_e2 | gate | 5 / 5 |
| outdir/sweep_e3_8x1client_20261016-164705_e3 | gate | 5 / 5 |
| outdir/sweep_e4_2x4clients_20261017-195148_e4 | gate | 5 / 5 |
| outdir/sweep_e1p_8clients_merlin_head_20261020-115926_p5 | photonic_h | 5 / 5 |
| outdir/sweep_e2p_1client_merlin_head_20261020-121619_p6 | photonic_h | 5 / 5 |
| outdir/sweep_e3p_8x1client_merlin_head_20261020-121957_p7 | photonic_h | 5 / 5 |
| outdir/sweep_e4p_2x4clients_merlin_head_20261020-123648_p8 | photonic_h | 5 / 5 |
| outdir/sweep_e1p_8clients_merlin_photonic_20261020-125406_p9 | photonic_f | 5 / 5 |
| outdir/sweep_e2p_1client_merlin_photonic_20261020-131110_p10 | photonic_f | 5 / 5 |
| outdir/sweep_e3p_8x1client_merlin_photonic_20261020-131447_p11 | photonic_f | 5 / 5 |
| outdir/sweep_e4p_2x4clients_merlin_photonic_20261020-133821_p12 | photonic_f | 5 / 5 |
| outdir/sweep_classical_20261020-144839_cg | classical_gate | 60 / 60 |
| outdir/sweep_classical_20261020-145256_ch | classical_photonic_h | 60 / 60 |
| outdir/sweep_classical_20261020-145713_cf | classical_photonic_f | 60 / 60 |
| outdir/sweep_classical_20261020-150131_cz | classical_central | 45 / 45 |
