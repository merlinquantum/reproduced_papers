# Scalable Quantum Reservoir Computing over Distributed Quantum Architectures — Reproduction

Reproduction of distributed quantum reservoir computing (QRC) for next-hour Greek power-load forecasting, with additional classical baselines and a photonic (MerLin) adaptation.

**Main message.**
1. **The paper's numbers reproduce:** against the paper baseline, the MAE reductions are of the same order as the paper's.
2. **The gain is not quantum:** it comes from the kernel readout; a classical kernel applied to the last 4 hours, without any reservoir, does as well as the best quantum model.
3. **The architecture holds and is easy to distribute:** it carries over to photonics with MerLin, and its branches run in parallel with the same forecast. The expected quantum advantage did not appear, but this simple architecture is the basis of our distributed proof of concept (see Next Steps).

**Where to start:** this README, then `notebook.ipynb` (the evidence for each message, about 1 min on a laptop CPU).

## Reference and Attribution

- **Paper:** Scalable Quantum Reservoir Computing over Distributed Quantum Architectures (arXiv preprint, v1, 6 May 2026).
- **Authors:** Ioannis Liliopoulos, Georgios D. Varsamis, Konstantinos Rallis, Evangelos Tsipas, Ioannis G. Karafyllidis, Georgios Ch. Sirakoulis, Panagiotis Dimitrakis.
- **arXiv:** https://arxiv.org/abs/2605.04991
- **Original repository:** none listed; no original code is reused.
- **Data:** public data from IPTO, the Independent Power Transmission Operator of Greece (ADMIE), as cited by the paper.

## Original Paper

The paper forecasts the next hourly Greek electricity load (35,064 hourly values) from the last 4 hours. A fixed reservoir turns the window into features, and only a ridge readout is trained:

```text
  4-hour window x = (x[t-3], x[t-2], x[t-1], x[t])
        |
        v
  Reservoir (fixed)  ── classical echo state network, or a network of 4-qubit quantum neurons
        |
        |  features r(t), one value per neuron
        v
  Readout (trained)  ── linear ridge, or "quantum ridge" (kernel ridge with a quantum fidelity kernel)
        |
        v
  Forecast ŷ = x̂[t+1]
```

The classical reservoir is a ReservoirPy echo state network (ESN), a recurrent network with fixed random weights. A quantum neuron is a 4-qubit circuit ([Fig. 2](results/paper_items/FIGURE_2.png)) whose output is a measured expectation value; the quantum ridge uses the kernel K(r, r') = |⟨Φ(r)|Φ(r')⟩|² of a 5- or 10-qubit circuit (Fig. 3, Eq. 6). Everything is simulated, ideally or with noise models of IBM and IonQ devices.

**The four architectures (Table I).** Every arrow carries classical data, so each branch can run on its own node or QPU:

```text
SRSR   x ──> Reservoir ──> Readout ──> ŷ

MRSR   x ─┬─> Reservoir 1 ─┐
          ├─> Reservoir 2 ─┼─> concatenate ──> Readout ──> ŷ
          └─> Reservoir k ─┘

SRMR   x ──> Reservoir ──> r(t) ─┬─> part 1 ──> Readout 1 ─┐
                                 ├─> part 2 ──> Readout 2 ─┼─> mean ──> ŷ
                                 └─> part m ──> Readout m ─┘

MRMR   x ─┬─> Reservoir 1 ──> Readout 1 ─┐
          ├─> Reservoir 2 ──> Readout 2 ─┼─> mean ──> ŷ
          └─> Reservoir k ──> Readout k ─┘
```

Each architecture has 8 sizes (10–60 neurons) × 3 reservoir types × 3 readout types; the metrics are MAE, RMSE and R² on the normalized load, one value per cell. The paper's classical reference, which we call the **paper baseline**, is the ESN with library defaults and a linear ridge.

## Reproduction Scope and Deviations

**Scope:** all four architectures, sizes and cells, on the full data, seeds 0, 1, 2, in simulation; the noise model is built from the paper's calibration medians (App. B).

**Deviations.** Where the paper is silent (data dates, ansatz angles, observable, wiring, regularization, merge rule, baseline settings, noise construction, best-cell selection), our choices and their justification are listed in [`docs/reproduction_details.md`](docs/reproduction_details.md). Two matter for the reading: the quantum reservoir keeps no memory between time steps, and we select the best cell on the evaluation split in addition to the paper's test-based rule.

**Simulation:** the paper names Qiskit and NVIDIA CUDA-Q without saying which simulations run on which. We use a small exact numpy simulation of these circuits, checked against Qiskit (why and timings: `docs/`, *Simulation engine*); the IBM and IonQ devices are noise profiles only, as in the paper; ReservoirPy 0.4.2 provides the classical reservoir.

## Project Layout

```text
papers/distributed_qrc/
├── configs/   # paper grids, MerLin, fair baselines, diagnostics/
├── docs/      # reproduction details: deviations, cell names, flags, sweep IDs, commands
├── lib/       # model, data, readouts, noise, MerLin adaptation, baselines, runner
├── utils/     # sweep coordinator, analysis, plotting and verification scripts
├── tests/     # 49 tests
└── results/   # tables and figures; paper_items/ (the paper's figures, paper vs ours); data/; checks/
```

## Install and How to Run

From the repository root:

```bash
.venv/bin/python -m pip install -r papers/distributed_qrc/requirements.txt
python implementation.py --paper distributed_qrc                                                    # smoke run, about 20 s
python implementation.py --paper distributed_qrc --config configs/e1_srsr_original.json --seed 0   # one paper grid (SRSR), 1–2 h on a CPU
```

All results come from sweeps (one config over seeds 0–2, collected in `outdir/sweep_<ID>/`), for example:

```bash
python papers/distributed_qrc/utils/run_sweep.py --sweep-id E1-R2 --config configs/e1_srsr_original.json --seeds 0 1 2
```

Sweep IDs, configs, flags and analysis commands are explained in [`docs/reproduction_details.md`](docs/reproduction_details.md). The notebook is rebuilt and executed with `utils/build_notebook.py` and `utils/run_notebook.py`.

## Configuration

Configs are JSON overlays of `configs/defaults.json`; `cli.json` lists the flags. A model is named by its cell `<reservoir type>|<readout type>`, for example `classical|linear` (the paper baseline) or `quantum_ideal|quantum_kernel_ideal` (fully quantum, ideal); the glossary is in `docs/`.

## Data

- **Source:** IPTO "System Load" daily files, from https://www.admie.gr/en/market/market-statistics/detail-data (public, no login), stored in `data/distributed_qrc/` (not committed), with a `SHA256SUMS` file to verify a copy.
- **Window:** 2020-10-22 … 2024-10-21 (35,064 values), not stated in the paper; identified from Table II and enforced by a fingerprint.
- **Splits:** chronological 24,539 / 5,258 / 5,267 (train / eval / test), 4-hour windows, min-max scale fitted on train.

## Results Obtained and Comparison with the Paper

### Message 1: the paper's numbers reproduce

The paper's headline is the **test MAE reduction** of its best quantum or hybrid cell relative to the paper baseline, `1 − MAE(best cell) / MAE(paper baseline)`, averaged over the 8 sizes of an architecture or at the best size (peak). For example, SRSR with 10 neurons in the paper: 1 − 0.0141 / 0.0458 = 69.2 %. Ours: mean ± std over 3 seeds; paper: as printed.

| Architecture | Ours, average | Paper, average | Ours, peak | Paper, peak |
|---|---|---|---|---|
| SRSR (Table IV) | 23.4 ± 1.5 % | 28.5 % | 64.9 ± 12.3 % | 69.2 % |
| MRSR (Table V) | 26.2 ± 3.6 % | 25.6 % | 57.9 ± 17.4 % | 63.3 % |
| SRMR (Table VI) | 44.1 ± 3.1 % | 54.2 % | 75.5 ± 6.5 % | 78.8 % |
| MRMR (Table VII) | 51.8 ± 5.9 % | 52.1 % | 75.4 ± 7.0 % | not given |

The reductions are of the same order as the paper's: peaks within the seed spread, averages up to 10 points lower for SRMR. Every size, with its baseline value and selected cell: `results/results_summary.md`. MAE and RMSE of every cell, paper next to ours: [Figure 4 (SRSR)](results/paper_items/FIGURE_4.png), [Figure 5 (MRSR)](results/paper_items/FIGURE_5.png), [Figure 6 (SRMR)](results/paper_items/FIGURE_6.png) and [Figure 7 (MRMR)](results/paper_items/FIGURE_7.png); the per-cell values and seed spread are in `results/data/<architecture>_vs_paper.csv`.

### Verdict per claim

Claims of the paper's introduction and conclusions (arXiv v1), plus C6, which motivates our follow-up work:

| Claim (paper) | Verdict |
|---|---|
| C1: quantum-enhanced models beat the paper baseline, average MAE reductions 25.6–54.2 % (Sec. III-B1–B4, pp. 5–8) | **Reproduced** against the paper baseline; **not confirmed** against fair classical models (Message 2) |
| C2: the quantum-kernel readout is the most consistent contributor (Sec. I, p. 2) | **Reproduced**, but a classical kernel does as well (Message 2) |
| C3: fully quantum models beat hybrids below 30 neurons and are more stable (Sec. IV, p. 9) | **Partially reproduced:** stable, but the margin over hybrids is within seed variance |
| C4: the gains persist under noise (Sec. I, p. 2) | **Partially reproduced:** the paper's noisy cells have a 19–38 % higher MAE than ours (median per architecture): its noise degrades more. Noisy models never beat fair classical ones |
| C5: the best model is a 3×20 classical reservoir + quantum kernel hybrid, MAE 0.0113 (Table VIII, p. 9) | **Not reproduced:** MAE 0.0146 ± 0.0010, tied with the best classical model (0.0137) |
| C6: the architectures suit multi-QPU deployment (qualitative; Sec. IV, p. 9) | **Not tested on hardware;** the branches run independently (Message 3) |

## Fair Baselines

### Message 2: the gain is not quantum

The paper baseline has a **linear** readout, while the best quantum models have a **kernel** readout. We added classical models with the same inputs, splits, seeds and evaluation, declared before any run: an ESN tuned on the evaluation split, an RBF kernel readout, and two controls **without reservoir**, where a kernel (the paper's, or RBF) is applied directly to the last 4 hours. The best model of each side is chosen on the evaluation split; a model is better if its mean test MAE is lower by more than 2 seed standard deviations.

| Architecture | Best quantum vs best classical | Paper's kernel, no reservoir, vs best quantum | RBF kernel, no reservoir, vs best quantum |
|---|---|---|---|
| SRSR | 0 better / 6 tie / 2 worse | 3 better / 5 tie / 0 worse | 3 / 5 / 0 |
| MRSR | 0 / 6 / 2 | 2 / 6 / 0 | 2 / 6 / 0 |
| SRMR | 0 / 5 / 3 | 4 / 4 / 0 | 4 / 4 / 0 |
| MRMR | 0 / 4 / 4 | 4 / 4 / 0 | 4 / 4 / 0 |

The best quantum model is never better than the best classical one (its MAE is 3–6 % higher on average), and a kernel on the last 4 hours, without reservoir, is never worse than it, quantum or not. What beats the paper baseline is the kernel readout, not the quantum parts. Figure and per-size values: `results/fair_comparison.png`, `results/fair_comparison_table.md`.

## Photonic Assessment and MerLin Extension

### Message 3: the architecture holds and is easy to distribute

**Status: `PARTIAL_MERLIN_TRANSLATION`** (simulation; the noise claim C4 is excluded). Each quantum neuron becomes a frozen 4-mode, 2-photon chip, and the kernel a 10-mode (or 5-mode) 2-photon chip; both are checked against Perceval to 1e-10. Test MAE change of the photonic model relative to the gate model, median over the 8 sizes:

| Architecture | Kernel readout | Linear readout |
|---|---|---|
| SRSR | +2.8 % (8/8 sizes within tolerance) | −3.1 % (8/8) |
| MRSR | +2.5 % (8/8) | −2.3 % (8/8) |
| SRMR | +3.3 % (8/8) | −1.1 % (7/8) |
| MRMR | +3.0 % (8/8) | −0.6 % (8/8) |

The photonic models behave like the gate models (tolerance declared before the runs: within 10 % or 2 seed std in at least 6/8 sizes), and like them do not beat the classical baselines. `merlin.FidelityKernel` was not used: it deviates from the exact fidelity (reported to the MerLin team).

The architecture is also easy to distribute: in MRMR, each branch (a reservoir and its own readout) needs only the input window and returns only its forecast (`lib/distributed.py`). In the notebook, an MRMR model with 5 branches of 60 quantum neurons runs in 41 s with the branches in sequence and in 15 s with one process per branch, with the same forecast.

## Hardware-Aware Settings

- **Simulation:** exact, float64, `merlinquantum` 0.4.2 and Perceval 1.3.1; no hardware, no gradients (frozen chips, closed-form readouts).
- **Neuron chip:** `UNBUNCHED`, threshold detection, 2 photons in 4 modes `[1,0,1,0]`, post-selected on 2-click events, no shots; about 18 min per run on a laptop CPU, 26 min on an L4 GPU.
- **Kernel chip:** `FOCK` amplitudes, 2 photons in 10 (or 5) modes, exact overlaps, no shots.
- **Not modelled:** loss, distinguishability, dark counts and the post-selection success rate.

## Next Steps

A proof of concept running the MRMR branches on separate nodes, then QPUs, starting with the MerLin reservoir.

## Limitations

- Simulation only; the noise model is rebuilt from calibration medians, and the paper's construction is unknown.
- Two gaps with the paper remain unexplained: its linear readouts on quantum features have a 15–33 % higher MAE than ours, and its noisy cells a 19–38 % higher one (medians per architecture).
- 3 seeds per cell; small multi-readout sizes have large seed variance.
- The classical baselines were often selected at the edge of their tuning grids, which can only under-tune them.
- MerLin adaptation with 2 photons only.

## Tests

`cd papers/distributed_qrc && pytest -q` runs 49 tests: quantum and photonic components against Qiskit, Qiskit Aer and Perceval, the data loader against Table II, the solvers, and small end-to-end runs.

## Citation and License

```bibtex
@misc{liliopoulos2026scalableqrc,
  title  = {Scalable Quantum Reservoir Computing over Distributed Quantum Architectures},
  author = {Liliopoulos, Ioannis and Varsamis, Georgios D. and Rallis, Konstantinos and Tsipas, Evangelos and Karafyllidis, Ioannis G. and Sirakoulis, Georgios Ch. and Dimitrakis, Panagiotis},
  year   = {2026},
  eprint = {2605.04991},
  archivePrefix = {arXiv}
}
```

The data is © IPTO (ADMIE), public and provided "as is". The reproduction code follows the repository's license.
