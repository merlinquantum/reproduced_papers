"""Phase 4 analysis: photonic learning (P4-L), MerLin trainability (E13) and the Kerr ablation (E13-K).

Usage (from the repository root)::

    python3 papers/fermi_dirac_neurons/utils/analyze_p4.py learning \
        papers/fermi_dirac_neurons/outdir/sweeps/p4_learning_v1
    python3 papers/fermi_dirac_neurons/utils/analyze_p4.py trainability \
        papers/fermi_dirac_neurons/outdir/sweeps/p4_trainability_v1
    python3 papers/fermi_dirac_neurons/utils/analyze_p4.py kerr \
        papers/fermi_dirac_neurons/outdir/sweeps/p4_kerr_v1

Rules declared in LOG.md ("Phase 4 Adaptation Plan — 2026-11-10T12:30:08Z")
before any Phase 4 run:

* P4-L: paired advantage of ``photonic_tanh`` over each baseline with
  ``utils/sweep_statistics.paired_comparison`` (decades, two-sided Wilcoxon,
  0.1-decade tolerance). Preservation of the Phase 3 behaviour:
  (i) C4, ``photonic_tanh`` vs ``photon_count_tanh`` "supported" on both
  metrics in the haar arm at every size; (ii) C8, fock-arm validation-MSE
  advantage over ``photon_count_tanh`` is "tie" or "not_supported" at every
  size; (iii) C6, ``photonic_tanh`` vs ``photonic_linear`` "supported" on
  training loss in the haar arm at every size. Detector and shot ablations:
  median over seeds of log10(ablated / PNR validation MSE), descriptive.
* E13 / E13-K: slope b of log10 Var vs log10 d by weighted least squares over
  the family's sizes with m >= 4 (point SE from the 95 % bootstrap CI);
  "consistent with 1/d" if |b + 1| <= 2 SE, "slower" if b > -1 + 2 SE,
  "faster" if b < -1 - 2 SE. X2(a): fock_first and haar_interferometer cells
  of photonic_tanh (d f) "slower" in the N = m/2 family. X2(b): haar_full
  cells "consistent" or "faster". X2(c): b(kappa) - b(0) beyond 2 combined SE.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR.parent))
sys.path.insert(0, str(PROJECT_DIR.parents[1]))

from fermi_dirac_neurons.utils.sweep_statistics import (  # noqa: E402
    load_values,
    paired_comparison,
)

LEARN_PATTERN = re.compile(r"p4_learn_m(\d+)_n(\d+)_(haar|fock)")
SIZE_PATTERN = re.compile(r"p4_(?:grad|kerr)_m(\d+)_n(\d+)")
BASELINES = ("photon_count_tanh", "photonic_linear", "correlation_tanh")
METRICS = ("train_loss", "validation_mse")
FAMILIES = {
    "N=2": lambda m, n: n == 2,
    "N=3": lambda m, n: n == 3,
    "N=m/2": lambda m, n: 2 * n == m,
}
MIN_MODES_IN_FIT = 4
# Variances below this are round-off of an identically zero gradient.
ZERO_VARIANCE = 1e-20


def _completed_runs(sweep_dir: Path):
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    if status["status"] != "COMPLETED":
        raise ValueError(f"{sweep_dir} is {status['status']}, not COMPLETED")
    for run in status["runs"]:
        yield run, json.loads((sweep_dir / run["metrics_path"]).read_text())


def learning(sweep_dir: Path) -> dict:
    """P4-L paired comparisons, preservation verdicts and readout ablations."""
    values, seeds = load_values(sweep_dir)
    configs = sorted({config for config, _, _ in values})
    comparisons = []
    for config in configs:
        m_text, n_text, arm = LEARN_PATTERN.fullmatch(config).groups()
        for baseline in BASELINES:
            for metric in METRICS:
                comparisons.append(
                    {
                        "n_modes": int(m_text),
                        "n_photons": int(n_text),
                        "arm": arm,
                        **paired_comparison(
                            values,
                            seeds,
                            (config, "photonic_tanh"),
                            (config, baseline),
                            metric,
                        ),
                    }
                )

    def verdict(arm, baseline, metric, size):
        (row,) = [
            row
            for row in comparisons
            if row["arm"] == arm
            and row["other_model"] == baseline
            and row["metric"] == metric
            and (row["n_modes"], row["n_photons"]) == size
        ]
        return row["verdict"]

    sizes = sorted({(row["n_modes"], row["n_photons"]) for row in comparisons})
    preservation = {
        "C4_noncommuting_beats_commuting_on_quantum_data": {
            f"m{m}_n{n}": all(
                verdict("haar", "photon_count_tanh", metric, (m, n)) == "supported"
                for metric in METRICS
            )
            for m, n in sizes
        },
        "C8_no_advantage_with_classical_data": {
            f"m{m}_n{n}": verdict("fock", "photon_count_tanh", "validation_mse", (m, n))
            in ("tie", "not_supported")
            for m, n in sizes
        },
        "C6_tanh_beats_linear": {
            f"m{m}_n{n}": verdict("haar", "photonic_linear", "train_loss", (m, n))
            == "supported"
            for m, n in sizes
        },
    }
    ratios = defaultdict(list)
    medians = defaultdict(list)
    for run, metrics in _completed_runs(sweep_dir):
        record = metrics["readout_ablations"]["photonic_tanh"]
        for key, value in record.items():
            if key != "mse_pnr":
                ratios[(run["config"], key)].append(
                    math.log10(value / record["mse_pnr"])
                )
        for name, model in metrics["models"].items():
            medians[(run["config"], name, "validation_mse")].append(
                model["final"]["validation"]["mse"]
            )
            medians[(run["config"], name, "train_loss")].append(
                model["last_logged_train_loss"]
            )
    ablations = [
        {
            "config": config,
            "readout": key.removeprefix("mse_"),
            "median_log10_ratio_to_pnr": float(np.median(values_)),
            "seeds": len(values_),
        }
        for (config, key), values_ in sorted(ratios.items())
    ]
    loss_table = [
        {
            "config": config,
            "model": name,
            "metric": metric,
            "median": float(np.median(values_)),
            "min": float(np.min(values_)),
            "max": float(np.max(values_)),
        }
        for (config, name, metric), values_ in sorted(medians.items())
    ]
    return {
        "sweep_dir": str(sweep_dir),
        "seeds": seeds,
        "comparisons": comparisons,
        "preservation": preservation,
        "readout_ablations": ablations,
        "losses": loss_table,
    }


def _weighted_slope(points: list[tuple[int, dict]]) -> tuple[float, float]:
    log_d = np.array([math.log10(d) for d, _ in points])
    values = np.array([math.log10(record["variance"]) for _, record in points])
    errors = np.array(
        [
            (
                math.log10(record["variance_ci95"][1])
                - math.log10(record["variance_ci95"][0])
            )
            / 3.92
            for _, record in points
        ]
    )
    weights = 1.0 / np.maximum(errors, 1e-6) ** 2
    design = np.vstack([np.ones_like(log_d), log_d]).T
    covariance = np.linalg.inv(design.T @ (weights[:, None] * design))
    coefficients = covariance @ design.T @ (weights * values)
    return float(coefficients[1]), float(math.sqrt(covariance[1, 1]))


def _classify(slope: float, error: float) -> str:
    if abs(slope + 1.0) <= 2 * error:
        return "consistent with 1/d"
    return "slower than 1/d" if slope > -1.0 + 2 * error else "faster than 1/d"


def _scaling(sweep_dir: Path, cell_keys: tuple[str, ...]) -> list[dict]:
    cells = defaultdict(list)
    for run, metrics in _completed_runs(sweep_dir):
        m_text, n_text = SIZE_PATTERN.fullmatch(run["config"]).groups()
        m, n = int(m_text), int(n_text)
        for record in metrics["records"]:
            key = tuple(record[name] for name in cell_keys)
            cells[key].append((m, n, metrics["fock_dimension"], record))
    rows = []
    for key, entries in sorted(cells.items(), key=lambda item: str(item[0])):
        for family, member in FAMILIES.items():
            points = sorted(
                (d, record)
                for m, n, d, record in entries
                if member(m, n) and m >= MIN_MODES_IN_FIT
            )
            if len(points) < 3:
                continue
            if max(record["variance"] for _, record in points) < ZERO_VARIANCE:
                # A mesh phase that only sets an output phase leaves photon
                # counts unchanged: its gradient is zero up to round-off.
                rows.append(
                    {
                        **dict(zip(cell_keys, key)),
                        "family": family,
                        "dimensions": [d for d, _ in points],
                        "variances": [record["variance"] for _, record in points],
                        "slope_log_var_per_log_d": None,
                        "slope_se": None,
                        "verdict": "zero gradient (inert parameter)",
                    }
                )
                continue
            if any(record["variance"] <= 0 for _, record in points):
                raise ValueError(f"Non-positive variance in cell {key} {family}")
            slope, error = _weighted_slope(points)
            rows.append(
                {
                    **dict(zip(cell_keys, key)),
                    "family": family,
                    "dimensions": [d for d, _ in points],
                    "variances": [record["variance"] for _, record in points],
                    "slope_log_var_per_log_d": slope,
                    "slope_se": error,
                    "verdict": _classify(slope, error),
                }
            )
    return rows


def trainability(sweep_dir: Path) -> dict:
    """E13 slopes and the X2(a), X2(b) verdicts."""
    keys = ("model", "input", "temperature", "energy_range", "parameter", "quantity")
    rows = _scaling(sweep_dir, keys)
    half = [
        row
        for row in rows
        if row["family"] == "N=m/2"
        and row["model"] == "photonic_tanh"
        and row["quantity"] == "output"
    ]
    half = [row for row in half if row["slope_log_var_per_log_d"] is not None]
    linear_optics = [
        row for row in half if row["input"] in ("fock_first", "haar_interferometer")
    ]
    haar_full = [row for row in half if row["input"] == "haar_full"]
    return {
        "sweep_dir": str(sweep_dir),
        "slopes": rows,
        "zero_gradient_cells": sum(
            row["slope_log_var_per_log_d"] is None for row in rows
        ),
        "X2a": {
            "cells": len(linear_optics),
            "slower_than_1_over_d": sum(
                row["verdict"] == "slower than 1/d" for row in linear_optics
            ),
            "supported": all(
                row["verdict"] == "slower than 1/d" for row in linear_optics
            ),
        },
        "X2b": {
            "cells": len(haar_full),
            "consistent_or_faster": sum(
                row["verdict"] != "slower than 1/d" for row in haar_full
            ),
            "supported": all(row["verdict"] != "slower than 1/d" for row in haar_full),
        },
    }


def kerr(sweep_dir: Path) -> dict:
    """E13-K slopes per kappa and the X2(c) differences against kappa = 0."""
    rows = _scaling(sweep_dir, ("kerr", "input", "parameter", "quantity"))
    reference = {
        (row["input"], row["parameter"], row["family"]): row
        for row in rows
        if row["kerr"] == 0.0
    }
    differences = []
    for row in rows:
        if row["kerr"] == 0.0:
            continue
        base = reference[(row["input"], row["parameter"], row["family"])]
        difference = row["slope_log_var_per_log_d"] - base["slope_log_var_per_log_d"]
        combined = math.hypot(row["slope_se"], base["slope_se"])
        differences.append(
            {
                "kerr": row["kerr"],
                "input": row["input"],
                "parameter": row["parameter"],
                "family": row["family"],
                "slope_difference": difference,
                "combined_se": combined,
                "verdict": "changes the decay"
                if abs(difference) > 2 * combined
                else "not distinguishable",
            }
        )
    return {
        "sweep_dir": str(sweep_dir),
        "simulator": "dense numpy Fock-space diagonalization (not MerLin)",
        "slopes": rows,
        "X2c": differences,
    }


def main() -> None:
    """Dispatch to the requested part and write its summary."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("part", choices=["learning", "trainability", "kerr"])
    parser.add_argument("sweep_dir", type=Path)
    args = parser.parse_args()
    summary = {"learning": learning, "trainability": trainability, "kerr": kerr}[
        args.part
    ](args.sweep_dir)
    output = PROJECT_DIR / "results" / f"p4_{args.part}_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
