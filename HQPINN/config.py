# config.py

import torch

# General hyperparameters
LR = 0.002
N_EPOCHS = 1801
PLOT_EVERY = 100

# Default dtype and device
DTYPE = torch.float64
DEVICE = torch.device("cpu")

N_SAMPLES = 200

#  Problem: 1D damped harmonic oscillator
#  ODE:     m u''(t) + μ u'(t) + k u(t) = 0,   t ∈ (0, 1]
# Physical oscillator parameters
M = 1.0
MU = 4.0
K = 400.0

# Loss weights
LAMBDA1 = 1e-1
LAMBDA2 = 1e-4

N_QUBITS = 3
N_LAYERS = 3
