"""Phase 5 comparison of QNN_STOCK (LOG.md, "Phase 5 Comparison Plan").

Reads each sweep's ``sweep_status.json``, every run's ``run_status.json`` and
``metrics.json`` (full precision, through ``summarize_sweep.load_rows``),
checks coverage against the expected run counts of the plan, and writes into
``--out``:

- ``coverage.csv``: expected, observed and completed runs per sweep;
- ``systems.csv``: seed statistics, parameters and wall-clock per system;
- ``pairs.csv``: paired-by-seed differences with a 95 % t interval;
- ``claims.csv``: claim outcomes from the plan's decision rule;
- ``summary.md``: the same tables in Markdown.

Usage::

    python papers/QNN_STOCK/utils/compare_phase5.py --sweeps-root papers/QNN_STOCK/outdir/sweeps --out papers/QNN_STOCK/results/phase5
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER_DIR))

from utils.summarize_sweep import T_975, load_rows  # noqa: E402

# Sweep label -> (directory prefix, expected runs, machine, concurrent workers).
SWEEPS = {
    "S1": ("phase3_s1_core_", 80, "remote Linux machine", 16),
    "S2": ("phase3_s2_context_preserving_", 50, "remote Linux machine", 16),
    "S3": ("phase3_s3_train_size_", 560, "remote Linux machine", 32),
    "S4": ("phase3_s4_qmtl_fig7_", 50, "remote Linux machine", 16),
    "S5": ("phase3_s5_qmtl_3000_", 45, "remote Linux machine", 16),
    "S6": ("phase3_s6_qmtl_interleaved_", 60, "remote Linux machine", 16),
    "S7": ("phase3_s7_qbgu_vs_sgd_", 50, "remote Linux machine", 16),
    "S8": ("phase3_s8_cc_minibatch_", 50, "remote Linux machine", 16),
    "S9": ("phase3_s9_loader_", 70, "remote Linux machine", 16),
    "S10": ("phase3_s10_multi_step_", 120, "remote Linux machine", 16),
    "S11": ("phase3_s11_scale_up_", 480, "remote Linux machine", 16),
    "S12": ("phase3_s12_lbfgs_", 120, "remote Linux machine", 16),
    "P1": ("phase4_p1_photonic_", 150, "local laptop (8 threads)", 6),
    "P5": ("phase5_c7_c8_baselines_", 15, "remote Linux machine", 5),
    "C5": ("phase5_c5_lr_grid_", 50, "local laptop (8 threads)", 6),
}

# Learning-rate label that the C5 selection rule (lowest seed-mean train KL per
# loss and asset) picks for both losses and assets; checked in main().
C5_SELECTED_LR = "lr1e-1"
C5_LOSSES = ("e6_qstl_mse_original", "e3_qstl_original")

LOG_LOSS = "inference/test/log_loss"
FIDELITY = "final_fidelity"
TRAIN_KL = "output/train/kl_joint"


def _pair(
    claim,
    role,
    label,
    sweep,
    candidate,
    reference,
    metric=LOG_LOSS,
    reference_sweep=None,
    status="COMMENSURABLE",
    note="",
):
    """One planned comparison: ``candidate`` minus ``reference``, paired by seed.

    ``candidate`` and ``reference`` are ``"<config>:<model>"``.
    """

    return {
        "claim": claim,
        "role": role,
        "label": label,
        "sweep": sweep,
        "reference_sweep": reference_sweep or sweep,
        "candidate": candidate,
        "reference": reference,
        "metric": metric,
        "status": status,
        "note": note,
    }


# Claim -> (kind, direction the paper asserts for "candidate minus reference").
# "advantage": the claim needs the candidate to be better.
# "parity": the claim says the candidate learns or matches the reference.
CLAIMS = {
    "C1": ("parity", "lower"),
    "C2": ("advantage", "lower"),
    "C3": ("advantage", "higher"),
    "C4": ("parity", None),
    "C5": ("advantage", "higher"),
    "C6": ("parity", None),
    "C7": ("parity", "lower"),
    "C8": ("parity", "lower"),
    "P4": ("advantage", "lower"),
}

# Claims whose primary pair changes more than one factor: no outcome beyond UNRESOLVED.
CONFOUNDED_CLAIMS = {
    "P4": (
        "primary pair confounded both ways: the photonic model has 9-18x the parameters, "
        "the baselines are validation-tuned and the photonic model is not"
    ),
}

CROSS_SWEEP_NOTE = (
    "paired across sweeps on the same machine: quantum runs in S1/S4, baselines in P5"
)

_QSTL = "e3_qstl_original:QSTL"
_QSTL_MA5 = "e3_qstl_ma5:QSTL"
_BASE = "e3_baselines_classical"
_BASE_MA5 = "e3_baselines_ma5"

PAIRS = [
    # C1: the paper QSTL against the target estimator and the capacity match.
    _pair("C1", "primary", "QSTL vs B0, raw", "S4", _QSTL, f"{_BASE}:B0_frequency"),
    _pair(
        "C1", "primary", "QSTL vs B0, MA5", "S4", _QSTL_MA5, f"{_BASE_MA5}:B0_frequency"
    ),
    _pair(
        "C1", "primary", "QSTL vs B3 31 params, raw", "S4", _QSTL, f"{_BASE}:B3_mlp_h6"
    ),
    _pair(
        "C1",
        "primary",
        "QSTL vs B3 31 params, MA5",
        "S4",
        _QSTL_MA5,
        f"{_BASE_MA5}:B3_mlp_h6",
    ),
    _pair("C1", "secondary", "QSTL vs B2, raw", "S4", _QSTL, f"{_BASE}:B2_logistic"),
    _pair(
        "C1",
        "secondary",
        "QSTL vs B2, MA5",
        "S4",
        _QSTL_MA5,
        f"{_BASE_MA5}:B2_logistic",
    ),
    # C2: sharing at equal epochs and parameters per asset (factor B).
    _pair(
        "C2",
        "primary",
        "QMTL vs QSTL 2-layer, 250/asset, raw",
        "S4",
        "e4_qmtl_original:QMTL",
        "e3_qstl_l2_e250:QSTL",
    ),
    _pair(
        "C2",
        "primary",
        "QMTL vs QSTL 2-layer, 250/asset, MA5",
        "S4",
        "e4_qmtl_ma5:QMTL",
        "e3_qstl_l2_e250_ma5:QSTL",
    ),
    _pair(
        "C2",
        "secondary",
        "QMTL vs QSTL 2-layer, 3000/asset, raw",
        "S5",
        "e4_qmtl_original__e3000:QMTL",
        "e3_qstl_l2:QSTL",
    ),
    _pair(
        "C2",
        "secondary",
        "QMTL vs QSTL 2-layer, 3000/asset, MA5",
        "S5",
        "e4_qmtl_ma5__e3000:QMTL",
        "e3_qstl_l2_ma5:QSTL",
    ),
    _pair(
        "C2",
        "secondary",
        "QMTL vs B4 pooled, raw",
        "S4",
        "e4_qmtl_original:QMTL",
        f"{_BASE}:B4_pooled_shrinkage",
    ),
    _pair(
        "C2",
        "secondary",
        "QMTL vs B4 pooled, MA5",
        "S4",
        "e4_qmtl_ma5:QMTL",
        f"{_BASE_MA5}:B4_pooled_shrinkage",
    ),
    _pair(
        "C2",
        "secondary",
        "QMTL vs B3 16 params, raw",
        "S4",
        "e4_qmtl_original:QMTL",
        f"{_BASE}:B3_mlp_h3",
    ),
    _pair(
        "C2",
        "secondary",
        "QMTL vs B3 16 params, MA5",
        "S4",
        "e4_qmtl_ma5:QMTL",
        f"{_BASE_MA5}:B3_mlp_h3",
    ),
    _pair(
        "C2",
        "secondary",
        "paper contrast QMTL 250/asset vs QSTL 3000, train KL, raw",
        "S4",
        "e4_qmtl_original:QMTL",
        _QSTL,
        metric=TRAIN_KL,
        status="PARTIALLY_COMMENSURABLE",
        note="Table 1 contrast: depth, parameters, epochs and order differ at once",
    ),
    _pair(
        "C2",
        "secondary",
        "paper contrast QMTL 250/asset vs QSTL 3000, train KL, MA5",
        "S4",
        "e4_qmtl_ma5:QMTL",
        _QSTL_MA5,
        metric=TRAIN_KL,
        status="PARTIALLY_COMMENSURABLE",
        note="Table 1 contrast: depth, parameters, epochs and order differ at once",
    ),
    # C3: fit reached at equal epochs per asset.
    _pair(
        "C3",
        "primary",
        "QMTL vs QSTL 2-layer fidelity, 250/asset, raw",
        "S4",
        "e4_qmtl_original:QMTL",
        "e3_qstl_l2_e250:QSTL",
        metric=FIDELITY,
    ),
    _pair(
        "C3",
        "primary",
        "QMTL vs QSTL 2-layer fidelity, 250/asset, MA5",
        "S4",
        "e4_qmtl_ma5:QMTL",
        "e3_qstl_l2_e250_ma5:QSTL",
        metric=FIDELITY,
    ),
    _pair(
        "C3",
        "secondary",
        "QMTL vs QSTL 2-layer fidelity, 3000/asset, raw",
        "S5",
        "e4_qmtl_original__e3000:QMTL",
        "e3_qstl_l2:QSTL",
        metric=FIDELITY,
    ),
    _pair(
        "C3",
        "secondary",
        "QMTL vs QSTL 2-layer fidelity, 3000/asset, MA5",
        "S5",
        "e4_qmtl_ma5__e3000:QMTL",
        "e3_qstl_l2_ma5:QSTL",
        metric=FIDELITY,
    ),
    # C4: QBGU against per-sample SGD on the same circuit, equal shots and iterations.
    _pair(
        "C4",
        "primary",
        "per-sample vs QBGU, paper HEA, raw",
        "S7",
        "e3_qstl_original__ps_e3000:QSTL",
        _QSTL,
    ),
    _pair(
        "C4",
        "primary",
        "per-sample vs QBGU, paper HEA, MA5",
        "S7",
        "e3_qstl_ma5__ps_e3000:QSTL",
        _QSTL_MA5,
    ),
    _pair(
        "C4",
        "secondary",
        "per-sample batch 10,000 vs QBGU, context-preserving, raw",
        "S8",
        "e11_qstl_cc1__ps_b10000:QSTL",
        "e11_qstl_cc1:QSTL",
    ),
    _pair(
        "C4",
        "secondary",
        "per-sample batch 10,000 vs QBGU, context-preserving, MA5",
        "S8",
        "e11_qstl_cc1_ma5__ps_b10000:QSTL",
        "e11_qstl_cc1_ma5:QSTL",
    ),
    # C5: MSE against fidelity on the shared learning-rate grid (sweep C5,
    # LOG.md amendment 2026-11-26T04:20:00Z), each loss at the lr selected on
    # the training metric; main() checks that the rule selects C5_SELECTED_LR.
    _pair(
        "C5",
        "primary",
        "MSE vs fidelity loss, shared lr grid, selected lr 0.1, train KL, raw",
        "C5",
        f"e6_qstl_mse_original__{C5_SELECTED_LR}:QSTL",
        f"e3_qstl_original__{C5_SELECTED_LR}:QSTL",
        metric=TRAIN_KL,
    ),
    _pair(
        "C5",
        "secondary",
        "MSE vs fidelity loss, paper lrs (1e-5 vs 0.1), train KL, raw",
        "C5",
        "e6_qstl_mse_original__lr1e-5:QSTL",
        "e3_qstl_original__lr1e-1:QSTL",
        metric=TRAIN_KL,
        status="PARTIALLY_COMMENSURABLE",
        note="paper learning rates differ by 1e4",
    ),
    _pair(
        "C5",
        "secondary",
        "MSE vs fidelity loss, paper settings (S1), train KL, raw",
        "S1",
        "e6_qstl_mse_original:QSTL",
        _QSTL,
        metric=TRAIN_KL,
        status="PARTIALLY_COMMENSURABLE",
        note="paper learning rates differ by 1e4; MSE 10,000 vs fidelity 3000 epochs",
    ),
    # C6: the paper's trained loader (Fig. 4) against exact amplitudes.
    _pair(
        "C6",
        "primary",
        "Fig. 4 loader vs exact, paper HEA, raw",
        "S9",
        "e3_qstl_original__hea_loader_fig4:QSTL",
        _QSTL,
    ),
    _pair(
        "C6",
        "primary",
        "Fig. 4 loader vs exact, paper HEA, MA5",
        "S9",
        "e3_qstl_ma5__hea_loader_fig4:QSTL",
        _QSTL_MA5,
    ),
    _pair(
        "C6",
        "secondary",
        "Fig. 4 loader vs exact, context-preserving, raw",
        "S9",
        "e11_qstl_cc1__hea_loader_fig4:QSTL",
        "e11_qstl_cc1:QSTL",
    ),
    _pair(
        "C6",
        "secondary",
        "Fig. 4 loader vs exact, context-preserving, MA5",
        "S9",
        "e11_qstl_cc1_ma5__hea_loader_fig4:QSTL",
        "e11_qstl_cc1_ma5:QSTL",
    ),
    # C7: QMTL at K = 4 and 8 against baselines on the same assets.
    _pair(
        "C7",
        "primary",
        "QMTL K=4 vs B0",
        "S4",
        "e7_qmtl_k4_original:QMTL",
        "p5_baselines_k4:B0_frequency",
        reference_sweep="P5",
        note=CROSS_SWEEP_NOTE,
    ),
    _pair(
        "C7",
        "primary",
        "QMTL K=4 vs B4 pooled",
        "S4",
        "e7_qmtl_k4_original:QMTL",
        "p5_baselines_k4:B4_pooled_shrinkage",
        reference_sweep="P5",
        note=CROSS_SWEEP_NOTE,
    ),
    _pair(
        "C7",
        "primary",
        "QMTL K=8 vs B0",
        "S4",
        "e7_qmtl_k8_original:QMTL",
        "p5_baselines_k8:B0_frequency",
        reference_sweep="P5",
        note=CROSS_SWEEP_NOTE,
    ),
    _pair(
        "C7",
        "primary",
        "QMTL K=8 vs B4 pooled",
        "S4",
        "e7_qmtl_k8_original:QMTL",
        "p5_baselines_k8:B4_pooled_shrinkage",
        reference_sweep="P5",
        note=CROSS_SWEEP_NOTE,
    ),
    # C8: d = 4, T = 2 against baselines on the same quantization.
    _pair(
        "C8",
        "primary",
        "QSTL d=4 vs B0",
        "S1",
        "e8_qstl_d4:QSTL",
        "p5_baselines_d4:B0_frequency",
        reference_sweep="P5",
        note=CROSS_SWEEP_NOTE,
    ),
    _pair(
        "C8",
        "primary",
        "QSTL d=4 vs B2",
        "S1",
        "e8_qstl_d4:QSTL",
        "p5_baselines_d4:B2_logistic",
        reference_sweep="P5",
        note=CROSS_SWEEP_NOTE,
    ),
]

# P4: MerLin adaptation (2 indistinguishable photons, PNR) against B2 and B0.
for _size in ("t3", "t5", "t6"):
    for _data, _suffix, _base in (("raw", "", _BASE), ("MA5", "_ma5", _BASE_MA5)):
        PAIRS.append(
            _pair(
                "P4",
                "primary",
                f"photonic 2 PNR vs B2, {_size.upper()} {_data}",
                "P1",
                f"p4_photonic_n2_fock{_suffix}_merlin__{_size}:PHOTONIC",
                f"{_base}__{_size}:B2_logistic",
                status="PARTIALLY_COMMENSURABLE",
                note="parameters not matched (photonic 9-18x more); baselines tuned on validation, photonic not",
            )
        )
        PAIRS.append(
            _pair(
                "P4",
                "secondary",
                f"photonic 2 PNR vs B0, {_size.upper()} {_data}",
                "P1",
                f"p4_photonic_n2_fock{_suffix}_merlin__{_size}:PHOTONIC",
                f"{_base}__{_size}:B0_frequency",
                status="PARTIALLY_COMMENSURABLE",
                note="parameters not matched; baselines tuned on validation, photonic not",
            )
        )

# Extensions run with the baselines in the same sweep (advantage evidence).
EXTENSION_PAIRS = [
    _pair(
        "E11",
        "extension",
        "context-preserving order 1 vs B2, raw",
        "S2",
        "e11_qstl_cc1:QSTL",
        f"{_BASE}:B2_logistic",
    ),
    _pair(
        "E11",
        "extension",
        "context-preserving order 1 vs B2, MA5",
        "S2",
        "e11_qstl_cc1_ma5:QSTL",
        f"{_BASE_MA5}:B2_logistic",
    ),
    _pair(
        "E11",
        "extension",
        "context-preserving order 3 vs B0, raw",
        "S2",
        "e11_qstl_cc3:QSTL",
        f"{_BASE}:B0_frequency",
    ),
    _pair(
        "E11",
        "extension",
        "context-preserving order 3 vs B0, MA5",
        "S2",
        "e11_qstl_cc3_ma5:QSTL",
        f"{_BASE_MA5}:B0_frequency",
    ),
]
for _n in ("n25", "n50", "n100", "n250", "n500", "n2000", "nfull"):
    for _data, _suffix, _base in (("raw", "", _BASE), ("MA5", "_ma5", _BASE_MA5)):
        EXTENSION_PAIRS.append(
            _pair(
                "F",
                "extension",
                f"order 1 vs B2, {_n} windows, {_data}",
                "S3",
                f"e11_qstl_cc1{_suffix}__{_n}:QSTL",
                f"{_base}__{_n}:B2_logistic",
            )
        )
for _size in ("t5h2", "t5h3", "t6h4"):
    for _data, _suffix, _base in (("raw", "", _BASE), ("MA5", "_ma5", _BASE_MA5)):
        EXTENSION_PAIRS.append(
            _pair(
                "E14",
                "extension",
                f"circuit A vs B2, {_size}, {_data}, L-BFGS",
                "S12",
                f"e14_qstl_seq{_suffix}__{_size}_lbfgs:QSTL",
                f"{_base}__{_size}_lbfgs:B2_logistic",
            )
        )
        EXTENSION_PAIRS.append(
            _pair(
                "E14",
                "extension",
                f"circuit B 3 layers vs A, {_size}, {_data}, L-BFGS",
                "S12",
                f"e14_qstl_ent_l3{_suffix}__{_size}_lbfgs:QSTL",
                f"e14_qstl_seq{_suffix}__{_size}_lbfgs:QSTL",
            )
        )


def resolve_sweeps(root: Path) -> dict[str, Path]:
    """Map each sweep label to its single directory under ``root``."""

    found = {}
    for label, (prefix, *_rest) in SWEEPS.items():
        matches = sorted(
            p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)
        )
        if len(matches) != 1:
            raise FileNotFoundError(
                f"sweep {label}: expected one directory {prefix}*, found {[m.name for m in matches]}"
            )
        found[label] = matches[0]
    return found


def coverage(label: str, sweep_dir: Path) -> dict:
    status = json.loads((sweep_dir / "sweep_status.json").read_text(encoding="utf-8"))
    runs = status["runs"]
    ids = [run["run_id"] for run in runs]
    expected = SWEEPS[label][1]
    completed = sum(run["status"] == "COMPLETED" for run in runs)
    return {
        "sweep": label,
        "directory": sweep_dir.name,
        "machine": SWEEPS[label][2],
        "workers": SWEEPS[label][3],
        "expected_runs": expected,
        "observed_runs": len(runs),
        "completed_runs": completed,
        "duplicate_run_ids": len(ids) - len(set(ids)),
        "complete": completed == expected == len(runs) and len(ids) == len(set(ids)),
    }


def wall_clock(sweep_dir: Path) -> dict[str, list[float]]:
    """Elapsed seconds per candidate config, from each run's ``run_status.json``."""

    status = json.loads((sweep_dir / "sweep_status.json").read_text(encoding="utf-8"))
    seconds = defaultdict(list)
    for run in status["runs"]:
        record = json.loads((sweep_dir / run["run_status"]).read_text(encoding="utf-8"))
        start = datetime.fromisoformat(record["started_at"].replace("Z", "+00:00"))
        end = datetime.fromisoformat(record["completed_at"].replace("Z", "+00:00"))
        seconds[run["candidate"]["config"]].append((end - start).total_seconds())
    return seconds


def index_rows(rows: list[dict]) -> dict:
    return {
        (r["candidate"], r["model"], r["asset"], r["metric"], r["seed"]): r["value"]
        for r in rows
    }


def paired_cells(pair: dict, indexes: dict) -> list[dict]:
    """Paired-by-seed difference for every asset the two systems share."""

    cand_cfg, cand_model = pair["candidate"].split(":")
    ref_cfg, ref_model = pair["reference"].split(":")
    cand_index = indexes[pair["sweep"]]
    ref_index = indexes[pair["reference_sweep"]]
    assets = sorted(
        {
            k[2]
            for k in cand_index
            if k[0] == cand_cfg and k[1] == cand_model and k[3] == pair["metric"]
        }
        & {
            k[2]
            for k in ref_index
            if k[0] == ref_cfg and k[1] == ref_model and k[3] == pair["metric"]
        }
    )
    if not assets:
        raise KeyError(
            f"no shared asset for {pair['label']} ({pair['candidate']} vs {pair['reference']}, {pair['metric']})"
        )
    out = []
    for asset in assets:
        seeds = sorted(
            k[4]
            for k in cand_index
            if k[:4] == (cand_cfg, cand_model, asset, pair["metric"])
        )
        differences = []
        for seed in seeds:
            ref_key = (ref_cfg, ref_model, asset, pair["metric"], seed)
            if ref_key not in ref_index:
                raise KeyError(
                    f"{pair['label']}: reference missing seed {seed} for {asset}"
                )
            differences.append(
                cand_index[(cand_cfg, cand_model, asset, pair["metric"], seed)]
                - ref_index[ref_key]
            )
        n = len(differences)
        mean = statistics.fmean(differences)
        half = T_975[n - 1] * statistics.stdev(differences) / math.sqrt(n)
        low, high = mean - half, mean + half
        verdict = (
            "not distinguishable"
            if low <= 0.0 <= high
            else ("lower" if high < 0 else "higher")
        )
        out.append(
            {
                **{
                    k: pair[k]
                    for k in (
                        "claim",
                        "role",
                        "label",
                        "sweep",
                        "reference_sweep",
                        "candidate",
                        "reference",
                        "metric",
                        "status",
                        "note",
                    )
                },
                "asset": asset,
                "n_pairs": n,
                "mean_difference": mean,
                "ci95_low": low,
                "ci95_high": high,
                "candidate_vs_reference": verdict,
            }
        )
    return out


def check_c5_selection(index: dict) -> None:
    """Fail unless the C5 rule selects ``C5_SELECTED_LR`` for every loss and asset."""

    for loss in C5_LOSSES:
        for asset in sorted({k[2] for k in index if k[0].startswith(loss)}):
            means = {}
            for (config, _model, key_asset, metric, _seed), value in index.items():
                if (
                    key_asset == asset
                    and metric == TRAIN_KL
                    and config.startswith(f"{loss}__")
                ):
                    means.setdefault(config.split("__")[1], []).append(value)
            best = min(means, key=lambda lr: statistics.fmean(means[lr]))
            if best != C5_SELECTED_LR:
                raise RuntimeError(
                    f"C5 rule selects {best} for {loss} on {asset}, not {C5_SELECTED_LR}"
                )


def claim_outcome(claim: str, cells: list[dict]) -> tuple[str, str]:
    """Apply the plan's decision rule to the primary cells of one claim."""

    kind, direction = CLAIMS[claim]
    primary = [c for c in cells if c["claim"] == claim and c["role"] == "primary"]
    if claim in CONFOUNDED_CLAIMS:
        return "UNRESOLVED", CONFOUNDED_CLAIMS[claim]
    verdicts = [c["candidate_vs_reference"] for c in primary]
    n = len(verdicts)
    if kind == "parity":
        # Learning or matching the reference: worse with an interval excluding 0 refutes it.
        worse = {"lower": "higher", "higher": "lower", None: None}[direction]
        if direction is None:
            against = sum(v != "not distinguishable" for v in verdicts)
        else:
            against = sum(v == worse for v in verdicts)
        if against == 0:
            return "SUPPORTED", f"no primary cell against the claim ({n} cells)"
        return "UNSUPPORTED", f"{against} of {n} primary cells against the claim"
    favourable = sum(v == direction for v in verdicts)
    opposite = sum(v not in (direction, "not distinguishable") for v in verdicts)
    if favourable == n:
        return "SUPPORTED", f"advantage in all {n} primary cells"
    if opposite > 0:
        return (
            "UNSUPPORTED",
            f"opposite direction in {opposite} of {n} primary cells, advantage in {favourable}",
        )
    if favourable == 0:
        return (
            "UNSUPPORTED",
            f"no advantage in any of {n} primary cells (all not distinguishable)",
        )
    return (
        "PARTIALLY_SUPPORTED",
        f"advantage in {favourable} of {n} primary cells, none opposite",
    )


def system_rows(rows_by_sweep: dict, clocks: dict, pairs: list[dict]) -> list[dict]:
    systems = {}
    for pair in pairs:
        for sweep, system in (
            (pair["sweep"], pair["candidate"]),
            (pair["reference_sweep"], pair["reference"]),
        ):
            systems.setdefault((sweep, system, pair["metric"]), pair["claim"])
    out = []
    for (sweep, system, metric), claim in sorted(systems.items()):
        cfg, model = system.split(":")
        values = defaultdict(list)
        params = {}
        for r in rows_by_sweep[sweep]:
            if r["candidate"] == cfg and r["model"] == model:
                if r["metric"] == metric:
                    values[r["asset"]].append(r["value"])
                elif r["metric"] == "n_params":
                    params[r["asset"]] = r["value"]
        seconds = clocks[sweep][cfg]
        for asset, vals in sorted(values.items()):
            out.append(
                {
                    "claim": claim,
                    "sweep": sweep,
                    "system": system,
                    "asset": asset,
                    "metric": metric,
                    "n_seeds": len(vals),
                    "mean": statistics.fmean(vals),
                    "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                    "n_params": params.get(asset),
                    "wall_clock_s_median": statistics.median(seconds),
                    "wall_clock_s_max": max(seconds),
                    "machine": SWEEPS[sweep][2],
                    "concurrent_workers": SWEEPS[sweep][3],
                    "provenance": "locally executed",
                }
            )
    return out


def consistency_check(indexes: dict) -> list[dict]:
    """B0 and B2 on AAPL / GOOGL raw must not depend on the extra K = 4 assets."""

    out = []
    for model in ("B0_frequency", "B2_logistic"):
        for asset in ("AAPL", "GOOGL"):
            for seed in range(42, 47):
                old = indexes["S4"][(_BASE, model, asset, LOG_LOSS, seed)]
                new = indexes["P5"][("p5_baselines_k4", model, asset, LOG_LOSS, seed)]
                out.append(
                    {
                        "model": model,
                        "asset": asset,
                        "seed": seed,
                        "S4": old,
                        "P5": new,
                        "abs_difference": abs(new - old),
                    }
                )
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _md_table(
    rows: list[dict], columns: list[str], formats: dict | None = None
) -> list[str]:
    formats = formats or {}
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        cells = []
        for col in columns:
            value = row.get(col)
            cells.append(
                formats[col].format(value)
                if col in formats and value is not None
                else str(value)
            )
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweeps-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    dirs = resolve_sweeps(args.sweeps_root)
    cover = [coverage(label, path) for label, path in dirs.items()]
    incomplete = [c["sweep"] for c in cover if not c["complete"]]
    if incomplete:
        raise RuntimeError(f"incomplete sweeps: {incomplete}")

    rows_by_sweep = {label: load_rows(path) for label, path in dirs.items()}
    indexes = {label: index_rows(rows) for label, rows in rows_by_sweep.items()}
    clocks = {label: wall_clock(path) for label, path in dirs.items()}

    check_c5_selection(indexes["C5"])
    cells = [cell for pair in PAIRS for cell in paired_cells(pair, indexes)]
    extension_cells = [
        cell for pair in EXTENSION_PAIRS for cell in paired_cells(pair, indexes)
    ]
    claims = []
    for claim in CLAIMS:
        outcome, reason = claim_outcome(claim, cells)
        claims.append(
            {
                "claim": claim,
                "kind": CLAIMS[claim][0],
                "outcome": outcome,
                "reason": reason,
            }
        )
    claims.append(
        {
            "claim": "C9",
            "kind": "robustness",
            "outcome": "UNRESOLVED",
            "reason": "EVIDENCE_MISSING: E10 noise study not implemented in Phase 3",
        }
    )
    claims.append(
        {
            "claim": "C10",
            "kind": "resources",
            "outcome": "OUT OF SCOPE",
            "reason": "theoretical, no experiment (Phase 1)",
        }
    )
    systems = system_rows(rows_by_sweep, clocks, PAIRS + EXTENSION_PAIRS)
    check = consistency_check(indexes)

    args.out.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out / "coverage.csv", cover)
    _write_csv(args.out / "pairs.csv", cells + extension_cells)
    _write_csv(args.out / "claims.csv", claims)
    _write_csv(args.out / "systems.csv", systems)
    _write_csv(args.out / "consistency_k4_vs_s4.csv", check)

    fmt = {
        "mean_difference": "{:+.4f}",
        "ci95_low": "{:+.4f}",
        "ci95_high": "{:+.4f}",
        "mean": "{:.4f}",
        "std": "{:.4f}",
        "wall_clock_s_median": "{:.0f}",
    }
    lines = ["# Phase 5 comparison (generated by utils/compare_phase5.py)", ""]
    lines.append(
        f"Runs: {sum(c['completed_runs'] for c in cover)} completed of {sum(c['expected_runs'] for c in cover)} expected in {len(cover)} sweeps."
    )
    lines.append(
        f"Consistency: B0/B2 on AAPL, GOOGL in P5 (K = 4 run) vs S4, max |difference| = {max(c['abs_difference'] for c in check):.2e}."
    )
    lines += ["", "## Coverage", ""]
    lines += _md_table(
        cover,
        [
            "sweep",
            "directory",
            "machine",
            "workers",
            "expected_runs",
            "completed_runs",
            "complete",
        ],
    )
    lines += ["", "## Claim outcomes (decision rule of the Phase 5 plan)", ""]
    lines += _md_table(claims, ["claim", "kind", "outcome", "reason"])
    lines += ["", "## Paired comparisons (candidate minus reference)", ""]
    lines += _md_table(
        cells,
        [
            "claim",
            "role",
            "label",
            "asset",
            "metric",
            "n_pairs",
            "mean_difference",
            "ci95_low",
            "ci95_high",
            "candidate_vs_reference",
            "status",
        ],
        fmt,
    )
    lines += ["", "## Extensions (advantage evidence: baselines in the same sweep)", ""]
    lines += _md_table(
        extension_cells,
        [
            "claim",
            "label",
            "asset",
            "n_pairs",
            "mean_difference",
            "ci95_low",
            "ci95_high",
            "candidate_vs_reference",
        ],
        fmt,
    )
    lines += ["", "## Systems", ""]
    lines += _md_table(
        systems,
        [
            "claim",
            "sweep",
            "system",
            "asset",
            "metric",
            "n_seeds",
            "mean",
            "std",
            "n_params",
            "wall_clock_s_median",
            "machine",
            "concurrent_workers",
        ],
        fmt,
    )
    (args.out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for claim in claims:
        print(f"{claim['claim']}: {claim['outcome']} ({claim['reason']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
