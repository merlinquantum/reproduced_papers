"""Tabulate a QNN_STOCK sweep: per-candidate seed statistics and paired comparisons.

Reads only ``sweep_status.json`` and each completed run's ``metrics.json``
(full precision). Writes ``metrics_long.csv``, ``aggregate.csv`` and
``paired.csv`` into ``--out`` (default: the sweep directory).

Usage::

    python papers/QNN_STOCK/utils/summarize_sweep.py <sweep_dir> [--out results/phase3_s1]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

# Two-sided 95 % Student-t quantiles by degrees of freedom (n_seeds - 1).
T_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
}

REPORTED_METRICS = (
    "inference/test/log_loss",
    "inference/test/kl_conditional_weighted",
    "inference/test/accuracy",
    "output/test/kl_joint",
    "output/train/kl_joint",
    "output/train/kl_conditional_sum",
    "inference/train/log_loss",
)

# (label, factor, (candidate, model), (reference candidate, reference model)).
# The difference is candidate minus reference, paired by seed and asset.
COMPARISONS = (
    (
        "CC1_vs_B2_raw",
        "context-preserving order 1 (4 params) vs logistic (4 params)",
        ("e11_qstl_cc1", "QSTL"),
        ("e3_baselines_classical", "B2_logistic"),
    ),
    (
        "CC3_vs_B0_raw",
        "context-preserving order 3 (8 params) vs frequency table (8 params)",
        ("e11_qstl_cc3", "QSTL"),
        ("e3_baselines_classical", "B0_frequency"),
    ),
    (
        "CC1_vs_B0_raw",
        "context-preserving order 1 vs frequency table",
        ("e11_qstl_cc1", "QSTL"),
        ("e3_baselines_classical", "B0_frequency"),
    ),
    (
        "CC1_vs_HEA_raw",
        "context-preserving order 1 vs paper HEA (32 params)",
        ("e11_qstl_cc1", "QSTL"),
        ("e3_qstl_original", "QSTL"),
    ),
    (
        "CC3_vs_HEA_raw",
        "context-preserving order 3 vs paper HEA (32 params)",
        ("e11_qstl_cc3", "QSTL"),
        ("e3_qstl_original", "QSTL"),
    ),
    (
        "CC1_vs_CC3_raw",
        "interaction order 1 vs 3",
        ("e11_qstl_cc1", "QSTL"),
        ("e11_qstl_cc3", "QSTL"),
    ),
    (
        "CC2_vs_CC3_raw",
        "interaction order 2 vs 3",
        ("e11_qstl_cc2", "QSTL"),
        ("e11_qstl_cc3", "QSTL"),
    ),
    (
        "CC1_vs_B2_ma5",
        "context-preserving order 1 (4 params) vs logistic (4 params)",
        ("e11_qstl_cc1_ma5", "QSTL"),
        ("e3_baselines_ma5", "B2_logistic"),
    ),
    (
        "CC3_vs_B0_ma5",
        "context-preserving order 3 (8 params) vs frequency table (8 params)",
        ("e11_qstl_cc3_ma5", "QSTL"),
        ("e3_baselines_ma5", "B0_frequency"),
    ),
    (
        "CC1_vs_B0_ma5",
        "context-preserving order 1 vs frequency table",
        ("e11_qstl_cc1_ma5", "QSTL"),
        ("e3_baselines_ma5", "B0_frequency"),
    ),
    (
        "CC1_vs_HEA_ma5",
        "context-preserving order 1 vs paper HEA (32 params)",
        ("e11_qstl_cc1_ma5", "QSTL"),
        ("e3_qstl_ma5", "QSTL"),
    ),
    (
        "CC3_vs_HEA_ma5",
        "context-preserving order 3 vs paper HEA (32 params)",
        ("e11_qstl_cc3_ma5", "QSTL"),
        ("e3_qstl_ma5", "QSTL"),
    ),
    (
        "CC1_vs_CC3_ma5",
        "interaction order 1 vs 3",
        ("e11_qstl_cc1_ma5", "QSTL"),
        ("e11_qstl_cc3_ma5", "QSTL"),
    ),
    (
        "CC2_vs_CC3_ma5",
        "interaction order 2 vs 3",
        ("e11_qstl_cc2_ma5", "QSTL"),
        ("e11_qstl_cc3_ma5", "QSTL"),
    ),
    (
        "A_raw",
        "depth 2 vs 4 layers",
        ("e3_qstl_l2", "QSTL"),
        ("e3_qstl_original", "QSTL"),
    ),
    (
        "A_ma5",
        "depth 2 vs 4 layers",
        ("e3_qstl_l2_ma5", "QSTL"),
        ("e3_qstl_ma5", "QSTL"),
    ),
    (
        "B_raw",
        "QMTL sharing vs independent 2-layer QSTL, 250 epochs per asset",
        ("e4_qmtl_original", "QMTL"),
        ("e3_qstl_l2_e250", "QSTL"),
    ),
    (
        "B_ma5",
        "QMTL sharing vs independent 2-layer QSTL, 250 epochs per asset",
        ("e4_qmtl_ma5", "QMTL"),
        ("e3_qstl_l2_e250_ma5", "QSTL"),
    ),
    (
        "paper_STL_vs_MTL_raw",
        "paper's Table 1 contrast (confounded)",
        ("e4_qmtl_original", "QMTL"),
        ("e3_qstl_original", "QSTL"),
    ),
    (
        "paper_STL_vs_MTL_ma5",
        "paper's Table 1 contrast (confounded)",
        ("e4_qmtl_ma5", "QMTL"),
        ("e3_qstl_ma5", "QSTL"),
    ),
    (
        "loader_hea_vs_exact",
        "trained HEA context loader vs exact amplitudes",
        ("e3_qstl_hea_loader", "QSTL"),
        ("e3_qstl_original", "QSTL"),
    ),
    (
        "mse_vs_fidelity",
        "MSE loss (lr 1e-5, 10k epochs) vs SWAP fidelity",
        ("e6_qstl_mse_original", "QSTL"),
        ("e3_qstl_original", "QSTL"),
    ),
    (
        "QSTL_vs_B0_raw",
        "4-layer QSTL vs frequency table",
        ("e3_qstl_original", "QSTL"),
        ("e3_baselines_classical", "B0_frequency"),
    ),
    (
        "QSTL_vs_B2_raw",
        "4-layer QSTL vs logistic",
        ("e3_qstl_original", "QSTL"),
        ("e3_baselines_classical", "B2_logistic"),
    ),
    (
        "QSTL_vs_B0_ma5",
        "4-layer QSTL vs frequency table",
        ("e3_qstl_ma5", "QSTL"),
        ("e3_baselines_ma5", "B0_frequency"),
    ),
    (
        "QSTL_vs_B2_ma5",
        "4-layer QSTL vs logistic",
        ("e3_qstl_ma5", "QSTL"),
        ("e3_baselines_ma5", "B2_logistic"),
    ),
    (
        "QMTL_vs_B4_raw",
        "QMTL vs pooled shrinkage table",
        ("e4_qmtl_original", "QMTL"),
        ("e3_baselines_classical", "B4_pooled_shrinkage"),
    ),
    (
        "QMTL_vs_B4_ma5",
        "QMTL vs pooled shrinkage table",
        ("e4_qmtl_ma5", "QMTL"),
        ("e3_baselines_ma5", "B4_pooled_shrinkage"),
    ),
    (
        "B3000_raw",
        "QMTL sharing vs independent 2-layer QSTL, 3000 epochs per asset",
        ("e4_qmtl_original__e3000", "QMTL"),
        ("e3_qstl_l2", "QSTL"),
    ),
    (
        "paper3000_raw",
        "QMTL 3000 epochs per asset vs 4-layer QSTL 3000 epochs (caption reading)",
        ("e4_qmtl_original__e3000", "QMTL"),
        ("e3_qstl_original", "QSTL"),
    ),
    (
        "QMTL3000_vs_B4_raw",
        "QMTL 3000 epochs per asset vs pooled shrinkage table",
        ("e4_qmtl_original__e3000", "QMTL"),
        ("e3_baselines_classical", "B4_pooled_shrinkage"),
    ),
    (
        "B3000_ma5",
        "QMTL sharing vs independent 2-layer QSTL, 3000 epochs per asset",
        ("e4_qmtl_ma5__e3000", "QMTL"),
        ("e3_qstl_l2_ma5", "QSTL"),
    ),
    (
        "paper3000_ma5",
        "QMTL 3000 epochs per asset vs 4-layer QSTL 3000 epochs (caption reading)",
        ("e4_qmtl_ma5__e3000", "QMTL"),
        ("e3_qstl_ma5", "QSTL"),
    ),
    (
        "QMTL3000_vs_B4_ma5",
        "QMTL 3000 epochs per asset vs pooled shrinkage table",
        ("e4_qmtl_ma5__e3000", "QMTL"),
        ("e3_baselines_ma5", "B4_pooled_shrinkage"),
    ),
    (
        "Bint250_raw",
        "interleaved QMTL (5 cycles x 50) vs independent 2-layer QSTL, 250 epochs per asset",
        ("e4_qmtl_original__c5x50", "QMTL"),
        ("e3_qstl_l2_e250", "QSTL"),
    ),
    (
        "Bint3000_raw",
        "interleaved QMTL (60 cycles x 50) vs independent 2-layer QSTL, 3000 epochs per asset",
        ("e4_qmtl_original__c60x50", "QMTL"),
        ("e3_qstl_l2", "QSTL"),
    ),
    (
        "QMTLint_c5x50_vs_B4_raw",
        "interleaved QMTL vs pooled shrinkage table",
        ("e4_qmtl_original__c5x50", "QMTL"),
        ("e3_baselines_classical", "B4_pooled_shrinkage"),
    ),
    (
        "QMTLint_c60x50_vs_B4_raw",
        "interleaved QMTL vs pooled shrinkage table",
        ("e4_qmtl_original__c60x50", "QMTL"),
        ("e3_baselines_classical", "B4_pooled_shrinkage"),
    ),
    (
        "Bint250_ma5",
        "interleaved QMTL (5 cycles x 50) vs independent 2-layer QSTL, 250 epochs per asset",
        ("e4_qmtl_ma5__c5x50", "QMTL"),
        ("e3_qstl_l2_e250_ma5", "QSTL"),
    ),
    (
        "Bint3000_ma5",
        "interleaved QMTL (60 cycles x 50) vs independent 2-layer QSTL, 3000 epochs per asset",
        ("e4_qmtl_ma5__c60x50", "QMTL"),
        ("e3_qstl_l2_ma5", "QSTL"),
    ),
    (
        "QMTLint_c5x50_vs_B4_ma5",
        "interleaved QMTL vs pooled shrinkage table",
        ("e4_qmtl_ma5__c5x50", "QMTL"),
        ("e3_baselines_ma5", "B4_pooled_shrinkage"),
    ),
    (
        "QMTLint_c60x50_vs_B4_ma5",
        "interleaved QMTL vs pooled shrinkage table",
        ("e4_qmtl_ma5__c60x50", "QMTL"),
        ("e3_baselines_ma5", "B4_pooled_shrinkage"),
    ),
    (
        "S7_PS3000_vs_QBGU_raw",
        "per-sample SGD vs QBGU, 3000 iterations x 10,000 shots",
        ("e3_qstl_original__ps_e3000", "QSTL"),
        ("e3_qstl_original", "QSTL"),
    ),
    (
        "S7_PS300k_vs_QBGU300k_raw",
        "per-sample SGD vs QBGU, 300,000 iterations x 100 shots",
        ("e3_qstl_original__ps_e300k_s100", "QSTL"),
        ("e3_qstl_original__qbgu_e300k_s100", "QSTL"),
    ),
    (
        "S7_QBGU300k_vs_QBGU_raw",
        "QBGU, 300,000 x 100 vs 3000 x 10,000 shots (equal total shots)",
        ("e3_qstl_original__qbgu_e300k_s100", "QSTL"),
        ("e3_qstl_original", "QSTL"),
    ),
    (
        "S7_PS300k_vs_PS3000_raw",
        "per-sample SGD, 300,000 x 100 vs 3000 x 10,000 shots (equal total shots)",
        ("e3_qstl_original__ps_e300k_s100", "QSTL"),
        ("e3_qstl_original__ps_e3000", "QSTL"),
    ),
    (
        "S7_ps_e300k_s100_vs_B0_raw",
        "ps_e300k_s100 vs B0_frequency",
        ("e3_qstl_original__ps_e300k_s100", "QSTL"),
        ("e3_baselines_classical", "B0_frequency"),
    ),
    (
        "S7_ps_e300k_s100_vs_B2_raw",
        "ps_e300k_s100 vs B2_logistic",
        ("e3_qstl_original__ps_e300k_s100", "QSTL"),
        ("e3_baselines_classical", "B2_logistic"),
    ),
    (
        "S7_qbgu_e300k_s100_vs_B0_raw",
        "qbgu_e300k_s100 vs B0_frequency",
        ("e3_qstl_original__qbgu_e300k_s100", "QSTL"),
        ("e3_baselines_classical", "B0_frequency"),
    ),
    (
        "S7_qbgu_e300k_s100_vs_B2_raw",
        "qbgu_e300k_s100 vs B2_logistic",
        ("e3_qstl_original__qbgu_e300k_s100", "QSTL"),
        ("e3_baselines_classical", "B2_logistic"),
    ),
    (
        "S7_PS3000_vs_QBGU_ma5",
        "per-sample SGD vs QBGU, 3000 iterations x 10,000 shots",
        ("e3_qstl_ma5__ps_e3000", "QSTL"),
        ("e3_qstl_ma5", "QSTL"),
    ),
    (
        "S7_PS300k_vs_QBGU300k_ma5",
        "per-sample SGD vs QBGU, 300,000 iterations x 100 shots",
        ("e3_qstl_ma5__ps_e300k_s100", "QSTL"),
        ("e3_qstl_ma5__qbgu_e300k_s100", "QSTL"),
    ),
    (
        "S7_QBGU300k_vs_QBGU_ma5",
        "QBGU, 300,000 x 100 vs 3000 x 10,000 shots (equal total shots)",
        ("e3_qstl_ma5__qbgu_e300k_s100", "QSTL"),
        ("e3_qstl_ma5", "QSTL"),
    ),
    (
        "S7_PS300k_vs_PS3000_ma5",
        "per-sample SGD, 300,000 x 100 vs 3000 x 10,000 shots (equal total shots)",
        ("e3_qstl_ma5__ps_e300k_s100", "QSTL"),
        ("e3_qstl_ma5__ps_e3000", "QSTL"),
    ),
    (
        "S7_ps_e300k_s100_vs_B0_ma5",
        "ps_e300k_s100 vs B0_frequency",
        ("e3_qstl_ma5__ps_e300k_s100", "QSTL"),
        ("e3_baselines_ma5", "B0_frequency"),
    ),
    (
        "S7_ps_e300k_s100_vs_B2_ma5",
        "ps_e300k_s100 vs B2_logistic",
        ("e3_qstl_ma5__ps_e300k_s100", "QSTL"),
        ("e3_baselines_ma5", "B2_logistic"),
    ),
    (
        "S7_qbgu_e300k_s100_vs_B0_ma5",
        "qbgu_e300k_s100 vs B0_frequency",
        ("e3_qstl_ma5__qbgu_e300k_s100", "QSTL"),
        ("e3_baselines_ma5", "B0_frequency"),
    ),
    (
        "S7_qbgu_e300k_s100_vs_B2_ma5",
        "qbgu_e300k_s100 vs B2_logistic",
        ("e3_qstl_ma5__qbgu_e300k_s100", "QSTL"),
        ("e3_baselines_ma5", "B2_logistic"),
    ),
    (
        "S8_ps_b1_vs_QBGU_raw",
        "context-preserving order 1: per-sample SGD (ps_b1) vs QBGU, 3000 x 10,000 shots",
        ("e11_qstl_cc1__ps_b1", "QSTL"),
        ("e11_qstl_cc1", "QSTL"),
    ),
    (
        "S8_ps_b100_vs_QBGU_raw",
        "context-preserving order 1: per-sample SGD (ps_b100) vs QBGU, 3000 x 10,000 shots",
        ("e11_qstl_cc1__ps_b100", "QSTL"),
        ("e11_qstl_cc1", "QSTL"),
    ),
    (
        "S8_ps_b10000_vs_QBGU_raw",
        "context-preserving order 1: per-sample SGD (ps_b10000) vs QBGU, 3000 x 10,000 shots",
        ("e11_qstl_cc1__ps_b10000", "QSTL"),
        ("e11_qstl_cc1", "QSTL"),
    ),
    (
        "S8_QBGU_vs_B0_raw",
        "context-preserving order 1 (QBGU) vs B0_frequency",
        ("e11_qstl_cc1", "QSTL"),
        ("e3_baselines_classical", "B0_frequency"),
    ),
    (
        "S8_QBGU_vs_B2_raw",
        "context-preserving order 1 (QBGU) vs B2_logistic",
        ("e11_qstl_cc1", "QSTL"),
        ("e3_baselines_classical", "B2_logistic"),
    ),
    (
        "S8_ps_b10000_vs_B0_raw",
        "context-preserving order 1 (ps_b10000) vs B0_frequency",
        ("e11_qstl_cc1__ps_b10000", "QSTL"),
        ("e3_baselines_classical", "B0_frequency"),
    ),
    (
        "S8_ps_b10000_vs_B2_raw",
        "context-preserving order 1 (ps_b10000) vs B2_logistic",
        ("e11_qstl_cc1__ps_b10000", "QSTL"),
        ("e3_baselines_classical", "B2_logistic"),
    ),
    (
        "S8_ps_b1_vs_QBGU_ma5",
        "context-preserving order 1: per-sample SGD (ps_b1) vs QBGU, 3000 x 10,000 shots",
        ("e11_qstl_cc1_ma5__ps_b1", "QSTL"),
        ("e11_qstl_cc1_ma5", "QSTL"),
    ),
    (
        "S8_ps_b100_vs_QBGU_ma5",
        "context-preserving order 1: per-sample SGD (ps_b100) vs QBGU, 3000 x 10,000 shots",
        ("e11_qstl_cc1_ma5__ps_b100", "QSTL"),
        ("e11_qstl_cc1_ma5", "QSTL"),
    ),
    (
        "S8_ps_b10000_vs_QBGU_ma5",
        "context-preserving order 1: per-sample SGD (ps_b10000) vs QBGU, 3000 x 10,000 shots",
        ("e11_qstl_cc1_ma5__ps_b10000", "QSTL"),
        ("e11_qstl_cc1_ma5", "QSTL"),
    ),
    (
        "S8_QBGU_vs_B0_ma5",
        "context-preserving order 1 (QBGU) vs B0_frequency",
        ("e11_qstl_cc1_ma5", "QSTL"),
        ("e3_baselines_ma5", "B0_frequency"),
    ),
    (
        "S8_QBGU_vs_B2_ma5",
        "context-preserving order 1 (QBGU) vs B2_logistic",
        ("e11_qstl_cc1_ma5", "QSTL"),
        ("e3_baselines_ma5", "B2_logistic"),
    ),
    (
        "S8_ps_b10000_vs_B0_ma5",
        "context-preserving order 1 (ps_b10000) vs B0_frequency",
        ("e11_qstl_cc1_ma5__ps_b10000", "QSTL"),
        ("e3_baselines_ma5", "B0_frequency"),
    ),
    (
        "S8_ps_b10000_vs_B2_ma5",
        "context-preserving order 1 (ps_b10000) vs B2_logistic",
        ("e11_qstl_cc1_ma5__ps_b10000", "QSTL"),
        ("e3_baselines_ma5", "B2_logistic"),
    ),
)


# S9 (E13): trained context loaders against exact loading, per circuit and data.
_S9_BASES = {
    ("paper HEA", "raw"): "e3_qstl_original",
    ("paper HEA", "ma5"): "e3_qstl_ma5",
    ("context-preserving order 1", "raw"): "e11_qstl_cc1",
    ("context-preserving order 1", "ma5"): "e11_qstl_cc1_ma5",
}
COMPARISONS += tuple(
    (
        f"S9_{'HEA' if circuit == 'paper HEA' else 'CC1'}_{loader}_vs_exact_{data}",
        f"{circuit}, {loader} vs exact loading",
        (f"{base}__{loader}", "QSTL"),
        (base, "QSTL"),
    )
    for (circuit, data), base in _S9_BASES.items()
    for loader in ("hea_loader", "hea_loader_fig4")
) + tuple(
    (
        f"S9_CC1_hea_loader_fig4_vs_B2_{data}",
        "context-preserving order 1 with the Fig. 4 loader vs B2_logistic",
        (f"{base}__hea_loader_fig4", "QSTL"),
        (
            "e3_baselines_ma5" if data == "ma5" else "e3_baselines_classical",
            "B2_logistic",
        ),
    )
    for (circuit, data), base in _S9_BASES.items()
    if circuit != "paper HEA"
)


# S10 (E14): multi-step prediction, per data and horizon.
_S10_ARMS = {
    "A": "e14_qstl_seq",
    "B_l2": "e14_qstl_ent",
    "B_l3": "e14_qstl_ent_l3",
}


def _s10_comparisons() -> tuple:
    comparisons = []
    for data, suffix, hea, baselines in (
        ("raw", "", "e3_qstl_original", "e3_baselines_classical"),
        ("ma5", "_ma5", "e3_qstl_ma5", "e3_baselines_ma5"),
    ):
        for horizon in ("h2", "h3"):
            tag = f"{data}_{horizon}"
            circuit_a = (f"e14_qstl_seq{suffix}__{horizon}", "QSTL")
            for arm, config in _S10_ARMS.items():
                candidate = (f"{config}{suffix}__{horizon}", "QSTL")
                for baseline in ("B0_frequency", "B2_logistic"):
                    comparisons.append(
                        (
                            f"S10_{arm}_vs_{baseline[:2]}_{tag}",
                            f"circuit {arm} vs {baseline}",
                            candidate,
                            (f"{baselines}__{horizon}", baseline),
                        )
                    )
                if arm != "A":
                    comparisons.append(
                        (
                            f"S10_{arm}_vs_A_{tag}",
                            f"circuit {arm} vs circuit A",
                            candidate,
                            circuit_a,
                        )
                    )
            comparisons.append(
                (
                    f"S10_HEA_vs_B0_{tag}",
                    "paper HEA vs B0_frequency",
                    (f"{hea}__{horizon}", "QSTL"),
                    (f"{baselines}__{horizon}", "B0_frequency"),
                )
            )
    return tuple(comparisons)


COMPARISONS += _s10_comparisons()


# S11 (E14b): context and horizon grown together, four training variants.
_S11_LABELS = tuple(
    f"{size}{variant}"
    for size in ("t5h2", "t5h3", "t6h4")
    for variant in ("", "_exact", "_safe", "_exact_safe")
)
# S12 (E14c) reuses the S11 comparisons with L-BFGS training; its names keep
# the "S11_" prefix and end in "_lbfgs".
_S11_LABELS += tuple(f"{size}_lbfgs" for size in ("t5h2", "t5h3", "t6h4"))


def _s11_comparisons() -> tuple:
    comparisons = []
    for data, suffix, baselines in (
        ("raw", "", "e3_baselines_classical"),
        ("ma5", "_ma5", "e3_baselines_ma5"),
    ):
        for label in _S11_LABELS:
            circuit_a = (f"e14_qstl_seq{suffix}__{label}", "QSTL")
            for arm, config in _S10_ARMS.items():
                candidate = (f"{config}{suffix}__{label}", "QSTL")
                for baseline in ("B0_frequency", "B2_logistic"):
                    comparisons.append(
                        (
                            f"S11_{arm}_vs_{baseline[:2]}_{data}_{label}",
                            f"circuit {arm} vs {baseline}",
                            candidate,
                            (f"{baselines}__{label}", baseline),
                        )
                    )
                if arm != "A":
                    comparisons.append(
                        (
                            f"S11_{arm}_vs_A_{data}_{label}",
                            f"circuit {arm} vs circuit A",
                            candidate,
                            circuit_a,
                        )
                    )
    return tuple(comparisons)


COMPARISONS += _s11_comparisons()


def _leaves(value, prefix=""):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _leaves(child, f"{prefix}/{key}" if prefix else key)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield prefix, float(value)


def load_rows(sweep_dir: Path) -> list[dict]:
    """Return one row per (run, asset, model or loader, metric path)."""

    status = json.loads((sweep_dir / "sweep_status.json").read_text(encoding="utf-8"))
    rows = []
    for run in status["runs"]:
        if run["status"] != "COMPLETED":
            raise ValueError(
                f"run {run['run_id']} is {run['status']}; refusing to tabulate"
            )
        metrics = json.loads(
            (sweep_dir / run["metrics_path"]).read_text(encoding="utf-8")
        )
        for asset, block in metrics["assets"].items():
            entries = dict(block.get("models", {}))
            if "loader" in block:
                entries["context_loader"] = block["loader"]
            for model, model_metrics in entries.items():
                for path, value in _leaves(model_metrics):
                    if not math.isfinite(value):
                        raise ValueError(f"non-finite {path} in {run['run_id']}")
                    rows.append(
                        {
                            "candidate": run["candidate"]["config"],
                            "seed": run["seed"],
                            "asset": asset,
                            "model": model,
                            "metric": path,
                            "value": value,
                        }
                    )
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        key = (row["candidate"], row["asset"], row["model"], row["metric"])
        groups[key].append(row["value"])
    out = []
    for (candidate, asset, model, metric), values in sorted(groups.items()):
        out.append(
            {
                "candidate": candidate,
                "asset": asset,
                "model": model,
                "metric": metric,
                "n": len(values),
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
            }
        )
    return out


def paired(rows: list[dict]) -> list[dict]:
    index = {
        (r["candidate"], r["model"], r["asset"], r["metric"], r["seed"]): r["value"]
        for r in rows
    }
    out = []
    for label, factor, (cand, model), (ref, ref_model) in COMPARISONS:
        assets = sorted(
            {r["asset"] for r in rows if r["candidate"] == cand and r["model"] == model}
        )
        for asset in assets:
            for metric in REPORTED_METRICS:
                seeds = sorted(
                    s
                    for (c, m, a, me, s) in index
                    if (c, m, a, me) == (cand, model, asset, metric)
                    and (ref, ref_model, asset, metric, s) in index
                )
                if not seeds:
                    continue
                diffs = [
                    index[(cand, model, asset, metric, s)]
                    - index[(ref, ref_model, asset, metric, s)]
                    for s in seeds
                ]
                mean = statistics.fmean(diffs)
                std = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
                half = (
                    T_975[len(diffs) - 1] * std / math.sqrt(len(diffs))
                    if len(diffs) > 1
                    else math.nan
                )
                low, high = mean - half, mean + half
                verdict = (
                    "not distinguishable"
                    if low <= 0.0 <= high
                    else ("lower" if high < 0 else "higher")
                )
                out.append(
                    {
                        "comparison": label,
                        "factor": factor,
                        "asset": asset,
                        "metric": metric,
                        "candidate": f"{cand}:{model}",
                        "reference": f"{ref}:{ref_model}",
                        "n_pairs": len(diffs),
                        "mean_diff": mean,
                        "std_diff": std,
                        "ci95_low": low,
                        "ci95_high": high,
                        "candidate_vs_reference": verdict,
                    }
                )
    return out


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    out = args.out or args.sweep_dir
    out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.sweep_dir)
    _write(out / "metrics_long.csv", rows)
    _write(out / "aggregate.csv", aggregate(rows))
    _write(out / "paired.csv", paired(rows))
    print(f"rows={len(rows)} written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
