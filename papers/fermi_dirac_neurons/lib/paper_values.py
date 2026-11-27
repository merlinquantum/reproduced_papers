"""Values reported by the paper, with their provenance, for comparison tables.

Every number here was read from the PDF page screenshots or figure crops in
``data/fermi_dirac_neurons/sources/screenshots/`` (never from text extraction),
except the per-epoch curves of Figs. 7 and 8, which come from the authors'
released CSVs and are loaded by :func:`lib.data.load_reference_curve`.
"""

from __future__ import annotations

# Table II (p. 21, crop ``table02_logistic_accuracy.png``): validation
# accuracy in percent after logistic-loss training, one run per n.
TABLE_II_VALIDATION_ACCURACY_PCT = {
    "quantum_heisenberg": {2: 95.2, 3: 97.6, 4: 96.8, 5: 92.2, 6: 91.2, 7: 91.0},
    "classical_fcim": {2: 64.0, 3: 65.8, 4: 68.6, 5: 63.8, 6: 73.6, 7: 69.6},
}

# Fig. 9 (p. 23, crop ``fig09_linear_vs_tanh.png``): training squared loss at
# epoch 10 000, read by ``utils/read_fig9_axes.py`` from pixel rows on the log
# axes (about +-0.05 decade). The 2-qubit TFIM tanh curve reaches its floor
# near epoch 6500 (float64 round-off on the loss).
FIG9_FINAL_TRAIN_LOSS_READING = {
    ("tfim", 2): {
        "quantum_tfim_tanh": 8.4e-30,
        "quantum_tfim_linear": 1.3e-3,
        "classical_ising": 4.2e-2,
    },
    ("tfim", 3): {
        "quantum_tfim_tanh": 3.9e-23,
        "quantum_tfim_linear": 3.7e-3,
        "classical_ising": 1.1e-1,
    },
    ("heis", 2): {
        "quantum_heis_tanh": 2.3e-5,
        "quantum_heis_linear": 4.6e-5,
        "classical_fcim": 1.2e-1,
    },
    ("heis", 3): {
        "quantum_heis_tanh": 2.6e-5,
        "quantum_heis_linear": 1.2e-3,
        "classical_fcim": 5.5e-2,
    },
}

# Released CSV per configuration family (columns: Quantum_Loss, Classical_Loss,
# and for Fig. 8 accuracies logged every 20 epochs).
FIG7_REFERENCE_CSV = {
    ("tanh", 2): "mse_tanh_2qubit_ising.csv",
    ("tanh", 7): "mse_tanh_7qubit_ising.csv",
    ("softplus", 2): "mse_softplus_2qubit_ising.csv",
    ("softplus", 7): "mse_softplus_7qubit_ising.csv",
}
FIG8_REFERENCE_CSV = {n: f"logloss_{n}qubit_heis_fcim.csv" for n in range(2, 8)}
