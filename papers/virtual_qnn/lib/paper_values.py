"""Reported values of arXiv:2609.11327 used for comparison.

Table I values are transcribed from the paper. Figure values are the exact data
arrays found in the authors' reference notebooks (repository
benchizhao/Virtual-Quantum-Neural-Network @ 2bfe9d7, ``plot_figures.ipynb``,
``VQC.ipynb`` and ``VVQE_HeH+_Chebyshev.ipynb``); they are the numbers plotted
in Figs. 4-7. See LOG.md, "Reference-code findings".
"""

from __future__ import annotations

# Table I. Keys: (noise_model, level). The amplitude-damping rows appear to
# be swapped in the paper (LOG.md, E0/E1): our SDP optimum equals the "VQEM"
# row and our trained VQEM overhead equals the "minimum" row.
TABLE1_HS = {
    ("depolarizing", 0.1): 9.53e-7,
    ("depolarizing", 0.2): 1.33e-6,
    ("depolarizing", 0.3): 1.55e-6,
    ("amplitude_damping", 0.1): 2.62e-6,
    ("amplitude_damping", 0.2): 3.34e-6,
    ("amplitude_damping", 0.3): 4.53e-6,
}
TABLE1_OVERHEAD_VQEM = {
    ("depolarizing", 0.1): 1.1667,
    ("depolarizing", 0.2): 1.372,
    ("depolarizing", 0.3): 1.644,
    ("amplitude_damping", 0.1): 1.222,
    ("amplitude_damping", 0.2): 1.500,
    ("amplitude_damping", 0.3): 1.857,
}
TABLE1_OVERHEAD_MINIMUM = {
    ("depolarizing", 0.1): 1.1667,
    ("depolarizing", 0.2): 1.375,
    ("depolarizing", 0.3): 1.643,
    ("amplitude_damping", 0.1): 1.218,
    ("amplitude_damping", 0.2): 1.49467,
    ("amplitude_damping", 0.3): 1.849,
}

# Fig. 4 (amplitude damping q = 0.3), indexed by penalty alpha.
FIG4_PENALTY = [1, 3, 10, 30, 100, 300, 1000, 3000]
FIG4_OVERHEAD = [
    1,
    1,
    1.24099731,
    1.56056046,
    1.76790524,
    1.82785916,
    1.8487323,
    1.85803246,
]
FIG4_HS = [0.0582, 0.0582, 0.0240, 0.00494, 0.000448, 0.0000495, 0.00000453, 5.9605e-07]

# Fig. 5 (depolarizing p = 0.2), budgets linspace(1.0, 2.5, 16).
FIG5_BUDGET = [round(1.0 + 0.1 * i, 1) for i in range(16)]
FIG5_HS_NOISY = [
    0.12979483604431152,
    0.10425114631652832,
    0.09013450145721436,
    0.07766413688659668,
    0.05801069736480713,
    0.04298532009124756,
    0.0271700590133667,
    0.016680598258972168,
    0.011723637580871582,
    0.005405068397521973,
    0.0031843185424804688,
    0.0007063150405883789,
    0.00011456012725830078,
    0.0,
    1.1920928955078125e-07,
    0.0,
]
FIG5_HS_IDEAL = [
    0.030022025108337402,
    0.016143441200256348,
    0.0065354108810424805,
    0.001225590705871582,
    2.384185791015625e-07,
    1.1920928955078125e-07,
    2.384185791015625e-07,
    2.384185791015625e-07,
    4.76837158203125e-07,
    2.384185791015625e-07,
    2.384185791015625e-07,
    2.384185791015625e-07,
    1.1920928955078125e-07,
    0.0,
    1.1920928955078125e-07,
    1.1920928955078125e-07,
]

# Fig. 6d, budgets [1, 1.5, 2, 2.5, 3]; mean and std over 10 repetitions.
FIG6_BUDGET = [1.0, 1.5, 2.0, 2.5, 3.0]
FIG6_TRAIN_MEAN = [0.66, 0.6935, 0.719, 0.738, 0.753]
FIG6_TRAIN_STD = [0.04037, 0.04196, 0.05440, 0.04792, 0.04226]
FIG6_TEST_MEAN = [0.588, 0.642, 0.666, 0.71, 0.723]
FIG6_TEST_STD = [0.06630, 0.04976, 0.05161, 0.04796, 0.04244]

# Fig. 7 (HeH+), indexed by the reference "distance" d (half separation, bohr).
FIG7_HALF_DISTANCE = [0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.7]
FIG7_ENERGY = {
    1.0: [
        -1.618426854915561,
        -1.7634033851362156,
        -2.014347764708238,
        -2.214162753299387,
        -2.266700082969064,
        -2.3870297200240334,
        -2.4092085784986947,
        -2.43294371675648,
        -2.471704173527817,
        -2.5026326518539888,
    ],
    1.5: [
        -1.9753961010951515,
        -2.1454281498133607,
        -2.2972168028448325,
        -2.497033735214472,
        -2.601331040546258,
        -2.6245671525107563,
        -2.652771575104592,
        -2.656521847051947,
        -2.6830705165136903,
        -2.6809861430455477,
    ],
    2.0: [
        -2.2452432985147617,
        -2.4106329123267543,
        -2.4147057856135747,
        -2.541464225693688,
        -2.6836344266702703,
        -2.670437586132346,
        -2.712409600172107,
        -2.7297447167344595,
        -2.7588220825825918,
        -2.7252626318753657,
    ],
    2.5: [
        -2.43567108,
        -2.58629536,
        -2.66536354,
        -2.78589293,
        -2.810794235165451,
        -2.83268773,
        -2.8220583673085438,
        -2.822408140505216,
        -2.8054479141644793,
        -2.789995860767154,
    ],
}
