# qrc_level_generation packaged data

Subset of the Moth Quantum open-data release for *Level Generation with
Quantum Reservoir Computing* (ferreira_2025, arXiv:2505.13287), packaged
so `papers/qrc_level_generation` runs from a fresh clone.

Source: <https://github.com/moth-quantum/OpenData>
(`Level_Generation_with_Quantum_Reservoir_Computing/`). Retrieved
2026-05-27. The data remains subject to the upstream repository's terms.

Only the files the reproduction actually consumes are committed:

| Path | Used by |
|---|---|
| `mario_level_1-2.json` | all training configs (`defaults`, `mario_qubit_*`, `mario_photonic`) |
| `reference_data/SMB/6_qubits/Aer/` (11 temperatures) | `configs/reference_eval.json` — metrics on the paper-published sequences |
| `reference_data/Roblox/{4..8}_qubits/Aer/...beta_1...` | `utils/investigate_save_point.py` — save-point separation check |

The full dump additionally contains noisy-backend variants
(`FakeJames`, `FakeGarnet`, `Aer_matrixnoise`), Roblox feature encodings,
original level exports, and Roblox beta 2/3 sequences that this
reproduction does not analyse — fetch them from the upstream repository
if you extend the noise study.
