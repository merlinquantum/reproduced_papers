# Phase 5 comparison: PQKD vs fair baselines and the MerLin adaptation (MNIST)

Protocol: LOG.md 'Phase 5 Comparison Plan — 2026-10-28T21:08Z'. Rule: final-epoch test accuracy, mean ± s.d. over seeds 0-2; differs only if |mean paired Δ| > s.d. of paired Δ; no winner from means alone.

Coverage: teacher 3/3, S4+S4b 27/27, S6+resumes 24/24 (complete). Code revisions: S4 ['c96f1fc', 'd0249ba'], S6 ['59ca70e'].

## Systems

| System | Role | Provenance | Scope | Runs | Metric | Mean | s.d. | Matching basis | Trainable params | Train h/run | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| teacher (paper) | paper reference | paper-reported | 100 epochs, 5 seeds | 5 / 5 (paper) | val acc % | 99.07 | 0.13 | parameter count inferred (paper reports C_x and ~2.25e5) | 224,394 | — | PARTIALLY_COMMENSURABLE (different budget (100 epochs, n = 5); validation metric tuned by SPSA) |
| PQKD (paper) | paper method | paper-reported | 100 epochs, 5 seeds | 5 / 5 (paper) | val acc % | 99.09 | 0.09 | parameter count inferred (paper reports C_x and ~2.25e5) | 149,439 | — | PARTIALLY_COMMENSURABLE (different budget (100 epochs, n = 5); validation metric tuned by SPSA) |
| teacher (S1) | distillation teacher, not iso-parameter | locally executed | partial RED-tier run (30 epochs, 3 seeds) | 3 / 3 | test acc % (best-val checkpoint) | 99.29 | 0.01 | 224,394 params (1.5× PQKD) | 224,394 | 1.19 | PARTIALLY_COMMENSURABLE (checkpoint rule differs (best-val vs final epoch); not a fair comparator for claim A) |
| PQKD, 8 photons | reproduced paper method | locally executed | partial RED-tier run (30 epochs, 3 seeds) | 3 / 3 | test acc % (final epoch) | 99.13 | 0.04 | reference | 149,439 | 1.45 | COMMENSURABLE |
| B1 dense KD | fair classical baseline (claim A) | locally executed | partial RED-tier run (30 epochs, 3 seeds) | 3 / 3 | test acc % (final epoch) | 99.29 | 0.08 | trainable params ±5% (−0.5%) | 148,626 | 0.94 | COMMENSURABLE |
| B2 learned mixing | upper reference (claim A) | locally executed | partial RED-tier run (30 epochs, 3 seeds) | 3 / 3 | test acc % (final epoch) | 99.26 | 0.06 | same R, +33k params | 182,418 | 1.14 | PARTIALLY_COMMENSURABLE (not iso-parameter; upper reference only) |
| B3f fixed random z | fair classical baseline (claim B) | locally executed | partial RED-tier run (30 epochs, 3 seeds) | 3 / 3 | test acc % (final epoch) | 99.18 | 0.02 | identical architecture, no θ | 149,394 | 1.13 | COMMENSURABLE |
| B3r random z per epoch | fair classical baseline (claims B, C) | locally executed | partial RED-tier run (30 epochs, 3 seeds) | 3 / 3 | test acc % (final epoch) | 98.15 | 0.35 | identical architecture, no θ | 149,394 | 1.16 | COMMENSURABLE |
| B4 photonic, θ frozen | paper control (Supp. Note 5), reproduced | locally executed | partial RED-tier run (30 epochs, 3 seeds) | 3 / 3 | test acc % (final epoch) | 99.14 | 0.09 | identical architecture, θ not trained | 149,394 | 1.13 | COMMENSURABLE |
| B5 classical generator, SPSA | fair classical baseline (claim B) | locally executed | partial RED-tier run (30 epochs, 3 seeds) | 3 / 3 | test acc % (final epoch) | 98.82 | 0.09 | dim φ = dim θ = 45, equal SPSA grid | 149,439 | 1.34 | COMMENSURABLE |
| B6 classical histograms, S = 200 | fair classical baseline (claims B, C) | locally executed | partial RED-tier run (30 epochs, 3 seeds) | 3 / 3 | test acc % (final epoch) | 98.48 | 0.09 | same Φ, S, EMA | 149,394 | 1.17 | COMMENSURABLE |
| B0 z = 0 | paper control (Fig. 4) | locally executed | partial RED-tier run (30 epochs, 3 seeds) | 3 / 3 | test acc % (final epoch) | 10.81 | 0.93 | identical architecture | 149,394 | 1.12 | INCOMPARABLE (degenerate: M = 0 zeroes every compressed kernel; metric agreement only) |
| P0 PQKD, 4 photons | MerLin adaptation reference | locally executed (MerLin adaptation) | partial RED-tier run (30 epochs, 3 seeds), 4 photons, simulated | 3 / 3 | test acc % (final epoch) | 99.19 | 0.09 | identical to PQKD, 4 photons | 149,439 | 1.54 | COMMENSURABLE (code revision differs from S4 by additive changes only) |
| F1 mixing θ, S = 200, EMA | MerLin adaptation (claim C) | locally executed (MerLin adaptation) | partial RED-tier run (30 epochs, 3 seeds), 4 photons, simulated | 3 / 3 | test acc % (final epoch) | 98.78 | 0.49 | identical architecture | 149,439 | 1.58 | COMMENSURABLE |
| F2 mixing θ, S = 200, no EMA | MerLin adaptation (claim C) | locally executed (MerLin adaptation) | partial RED-tier run (30 epochs, 3 seeds), 4 photons, simulated | 3 / 3 | test acc % (final epoch) | 97.80 | 2.19 | identical architecture | 149,439 | 1.76 | COMMENSURABLE |
| F3 mixing θ, S = 25, EMA | MerLin adaptation (claim C) | locally executed (MerLin adaptation) | partial RED-tier run (30 epochs, 3 seeds), 4 photons, simulated | 3 / 3 | test acc % (final epoch) | 98.58 | 0.63 | identical architecture | 149,439 | 1.70 | COMMENSURABLE |
| F4 mixing θ, S = 25, no EMA | MerLin adaptation (claim C) | locally executed (MerLin adaptation) | partial RED-tier run (30 epochs, 3 seeds), 4 photons, simulated | 3 / 3 | test acc % (final epoch) | 95.26 | 1.60 | identical architecture | 149,439 | 1.68 | COMMENSURABLE |
| N1 exact, autograd θ, θ₀ random | MerLin adaptation (claims A, B) | locally executed (MerLin adaptation) | partial RED-tier run (30 epochs, 3 seeds), 4 photons, simulated | 3 / 3 | test acc % (final epoch) | 99.15 | 0.08 | identical architecture, 45 trained angles | 149,439 | 1.73 | COMMENSURABLE (simulation-only gradients) |
| N2 exact, autograd θ, θ₀ = 0 | MerLin adaptation (claim B) | locally executed (MerLin adaptation) | partial RED-tier run (30 epochs, 3 seeds), 4 photons, simulated | 3 / 3 | test acc % (final epoch) | 99.15 | 0.08 | identical architecture, 45 trained angles | 149,439 | 1.66 | COMMENSURABLE (simulation-only gradients) |
| N3 classical generator, autograd | fair classical counterpart to N1/N2 (Phase 4 sweep) | locally executed | partial RED-tier run (30 epochs, 3 seeds) | 3 / 3 | test acc % (final epoch) | 99.10 | 0.02 | dim φ = 45, same optimiser | 149,439 | 1.65 | COMMENSURABLE |

Wall-clock: sum of per-epoch seconds (pauses excluded), CPU laptop, 3 parallel × 4 threads for every local sweep.

## Paired per-seed differences (final-epoch test accuracy, arm − reference)

| Claim | Arm | Reference | Δ test acc (pp) | Separated? | Signs (+/−/0) | Δ val CE |
|---|---|---|---|---|---|---|
| A | B1 (S4) | PQKD (S4) | 0.16 ± 0.05 (n=3) | yes | 3/0/0 | -0.0035 ± 0.0018 (n=3) |
| A | N1 (S6) | B1 (S4) | -0.13 ± 0.09 (n=3) | yes | 0/3/0 | 0.0039 ± 0.0042 (n=3) |
| A | P0 (S6) | B1 (S4) | -0.09 ± 0.13 (n=3) | no | 1/2/0 | 0.0028 ± 0.0019 (n=3) |
| B | B3f (S4) | PQKD (S4) | 0.05 ± 0.04 (n=3) | yes | 3/0/0 | 0.0008 ± 0.0060 (n=3) |
| B | B4 (S4) | PQKD (S4) | 0.01 ± 0.09 (n=3) | no | 1/2/0 | 0.0024 ± 0.0018 (n=3) |
| B | B5 (S4) | PQKD (S4) | -0.31 ± 0.06 (n=3) | yes | 0/3/0 | 0.0166 ± 0.0071 (n=3) |
| B | B6 (S4) | PQKD (S4) | -0.65 ± 0.08 (n=3) | yes | 0/3/0 | 0.0318 ± 0.0020 (n=3) |
| B | B3r (S4) | PQKD (S4) | -0.98 ± 0.39 (n=3) | yes | 0/3/0 | 0.0477 ± 0.0150 (n=3) |
| B | N1 (S6) | N3 (S6) | 0.05 ± 0.10 (n=3) | no | 2/1/0 | -0.0022 ± 0.0020 (n=3) |
| B | N1 (S6) | B3f (S4) | -0.02 ± 0.07 (n=3) | no | 1/2/0 | -0.0004 ± 0.0018 (n=3) |
| B | N2 (S6) | N3 (S6) | 0.05 ± 0.09 (n=3) | no | 2/1/0 | 0.0008 ± 0.0045 (n=3) |
| C | F1 (S6) | P0 (S6) | -0.41 ± 0.56 (n=3) | no | 1/2/0 | 0.0131 ± 0.0196 (n=3) |
| C | F2 (S6) | P0 (S6) | -1.40 ± 2.24 (n=3) | no | 1/2/0 | 0.0639 ± 0.0938 (n=3) |
| C | F3 (S6) | P0 (S6) | -0.61 ± 0.54 (n=3) | yes | 0/3/0 | 0.0237 ± 0.0174 (n=3) |
| C | F4 (S6) | P0 (S6) | -3.93 ± 1.53 (n=3) | yes | 0/3/0 | 0.1823 ± 0.0763 (n=3) |
| C | F3 (S6) | F4 (S6) | 3.32 ± 1.15 (n=3) | yes | 3/0/0 | -0.1586 ± 0.0708 (n=3) |
| C | F1 (S6) | F2 (S6) | 0.98 ± 2.31 (n=3) | no | 1/2/0 | -0.0507 ± 0.0990 (n=3) |
| C | B3r (S4) | B3f (S4) | -1.03 ± 0.36 (n=3) | yes | 0/3/0 | 0.0469 ± 0.0115 (n=3) |
| check | P0 (S6) | PQKD (S4) | 0.06 ± 0.08 (n=3) | no | 2/1/0 | -0.0007 ± 0.0009 (n=3) |
