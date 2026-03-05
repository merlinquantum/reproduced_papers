# config.py

import torch

# Default dtype and device
DTYPE = torch.float64
DEVICE = torch.device("cpu")
N_LAYERS = 3
N_QUBITS = 3

GAMMA = 1.4


# ==========================
#  Problem: 1D damped harmonic oscillator
#  ODE:     m u''(t) + μ u'(t) + k u(t) = 0,   t ∈ (0, 1]
# ==========================

DHO_LR = 0.002
DHO_N_EPOCHS = 1801
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


# ==========================
#  TAF – 2D Transonic Aerofoil Flow (Sec. 3.3)
# ==========================

#           TAF_Y_MAX   →  X_top
#    -------------------------
#    |                       |
#    |        AILE           |
#    |                       |
#    -------------------------
#           TAF_Y_MIN   →  X_bot

#   TAF_X_MIN           TAF_X_MAX
TAF_R_GAS = 287.0

TAF_X_MIN = -1.0
TAF_X_MAX = 3.5
TAF_Y_MIN = -2.25
TAF_Y_MAX = 2.25

TAF_DOMAIN_SIDE = 4.5

TAF_CHORD_X0 = 0.0
TAF_CHORD_X1 = 1.0

TAF_N_BOUNDARY = 40
TAF_N_DOMAIN_TOTAL = 4000
TAF_N_DATA_INTERNAL = 400
TAF_N_WALL = 400

TAF_LR = 5e-4
TAF_ADAM_STEPS = 40000
TAF_LBFGS_STEPS = 2000
TAF_PLOT_EVERY = 500

TAF_EPSILON_LAMBDA = 0.1
TAF_P_OUT = 0.0

TAF_CC_NUM_HIDDEN_LAYERS = 4
TAF_CC_HIDDEN_WIDTH = 40

# Files produced by your generator
TAF_X_IN_FILE = "X_in.npy"
TAF_X_OUT_FILE = "X_out.npy"
TAF_X_TOP_FILE = "X_top.npy"
TAF_X_BOT_FILE = "X_bot.npy"
TAF_X_WALL_FILE = "X_wall.npy"
TAF_X_WALL_NORMALS_FILE = "X_wall_normals.npy"
TAF_X_F_FILE = "X_f.npy"
TAF_X_DATA_INT_FILE = "X_data_int.npy"
