# config.py

import torch

# Default dtype and device
DTYPE = torch.float64
DEVICE = torch.device("cpu")
N_LAYERS = 3
N_QUBITS = 3

EE_GAMMA = 1.4


# ==========================
#  Problem: 1D damped harmonic oscillator
#  ODE:     m u''(t) + μ u'(t) + k u(t) = 0,   t ∈ (0, 1]
# ==========================

DHO_LR = 0.002
DHO_N_EPOCHS = 1800
DHO_PLOT_EVERY = 100
DHO_N_SAMPLES = 200

M = 1.0
MU = 4.0
K = 400.0

# Loss weights
LAMBDA1 = 1e-1
LAMBDA2 = 1e-4


DHO_NUM_HIDDEN_LAYERS = 2
DHO_HIDDEN_WIDTH = 16


# ==========================
#  SEE – Smooth Euler Equation (Sec. 3.1)
#  1D Euler, solution lisse:
#  x ∈ (-1, 1), t ∈ (0, 2)
# ==========================

SEE_LR = 5e-4
SEE_N_EPOCHS = 20000
SEE_PLOT_EVERY = 1000
SEE_NX_SAMPLES = 200
SEE_NT_SAMPLES = 200

SEE_X_MIN, SEE_X_MAX = -1.0, 1.0
SEE_T_MIN, SEE_T_MAX = 0.0, 2.0

SEE_N_IC = 50
SEE_N_BC = 50
SEE_N_F = 2000


SEE_CC_NUM_HIDDEN_LAYERS = 4
SEE_CC_HIDDEN_WIDTH = 10


# ==========================
#  DEE – Discontinue Euler Equation (Sec. 3.2)
#  1D Euler, solution lisse:
#  x ∈ (0, 1), t ∈ (0, 2)
# ==========================

DEE_LR = 5e-4
DEE_N_EPOCHS = 20000
DEE_PLOT_EVERY = 1000
DEE_NX_SAMPLES = 200
DEE_NT_SAMPLES = 200

DEE_X_MIN, DEE_X_MAX = 0.0, 1.0
DEE_T_MIN, DEE_T_MAX = 0.0, 2.0

DEE_N_IC = 60
DEE_N_BC = 60
DEE_N_F = 1000

DEE_U = 0.1
DEE_P = 1.0
DEE_RHO_L = 1.4
DEE_RHO_R = 1.0
DEE_X0 = 0.5

DEE_CC_NUM_HIDDEN_LAYERS = 4
DEE_CC_HIDDEN_WIDTH = 10
