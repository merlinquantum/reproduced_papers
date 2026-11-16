"""Pooled chi-square of the noiseless protocol checks (App. A) over several runs.

For every random state pair, each protocol run stores the ideal swap-test
probability ``p0 = 1/2 + 1/2 |<a|b>|^2`` and the fraction of ancilla zeros
measured on Aer for the distributed and the centralised circuit. The counts are
compared with ``Binomial(shots, p)`` for three laws: the swap-test law, and two
wrong laws the test must reject (ancilla at random, ``p = 1/2``; overlap not
squared, ``p = 1/2 + 1/2 |<a|b>|``). The chi-square sums one degree of freedom
per pair over all runs. Without arguments, the latest completed run of each
protocol config in ``outdir/`` is used.

Usage::

    .venv/bin/python papers/distributed_secure_kernels/utils/pooled_protocol_check.py [<run dir> ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_DIR = Path(__file__).resolve().parents[1]
PROTOCOL_EXPERIMENTS = (
    "protocol_equivalence_original",
    "protocol_equivalence_four_qubits",
)
CIRCUITS = ("distributed", "centralised")
LAWS = {
    "swap-test law": lambda p0: p0,
    "wrong law, ancilla at random": lambda p0: np.full_like(p0, 0.5),
    "wrong law, overlap not squared": lambda p0: 0.5 + 0.5 * np.sqrt(2 * p0 - 1),
}


def latest_runs(outdir: Path) -> list[Path]:
    """Latest completed run directory of each protocol experiment."""

    latest: dict[str, Path] = {}
    for run_dir in sorted(outdir.glob("run_*")):
        status_path, metrics_path = (
            run_dir / "run_status.json",
            run_dir / "metrics.json",
        )
        if not (status_path.exists() and metrics_path.exists()):
            continue
        if json.loads(status_path.read_text()).get("status") != "COMPLETED":
            continue
        experiment = json.loads(metrics_path.read_text()).get("experiment", {})
        if (
            isinstance(experiment, dict)
            and experiment.get("id") in PROTOCOL_EXPERIMENTS
        ):
            latest[experiment["id"]] = run_dir
    missing = [name for name in PROTOCOL_EXPERIMENTS if name not in latest]
    if missing:
        sys.exit(f"No completed run in {outdir} for: {', '.join(missing)}")
    return [latest[name] for name in PROTOCOL_EXPERIMENTS]


def pooled(run_dirs: list[Path]) -> dict[str, dict[str, dict[str, float]]]:
    """Pooled chi-square, degrees of freedom and p-value per law and circuit."""

    data = [json.loads((run_dir / "metrics.json").read_text()) for run_dir in run_dirs]
    results: dict[str, dict[str, dict[str, float]]] = {}
    for law_name, law in LAWS.items():
        results[law_name] = {}
        for circuit in CIRCUITS:
            statistic, dof = 0.0, 0
            for metrics in data:
                shots = int(metrics["experiment"]["shots"])
                p0 = np.asarray(metrics["noiseless"]["p0_ideal"])
                zeros = shots * np.asarray(
                    metrics["noiseless"][f"{circuit}_zero_fraction"]
                )
                probability = law(p0)
                statistic += float(
                    np.sum(
                        (zeros - shots * probability) ** 2
                        / (shots * probability * (1 - probability))
                    )
                )
                dof += len(p0)
            results[law_name][circuit] = {
                "chi_square": statistic,
                "dof": dof,
                "p_value": float(stats.chi2.sf(statistic, dof)),
            }
    return results


def main() -> None:
    run_dirs = [Path(arg) for arg in sys.argv[1:]] or latest_runs(
        PROJECT_DIR / "outdir"
    )
    print("Runs: " + ", ".join(run_dir.name for run_dir in run_dirs))
    for law_name, per_circuit in pooled(run_dirs).items():
        print(law_name)
        for circuit, total in per_circuit.items():
            print(
                f"  {circuit}: chi2 = {total['chi_square']:.1f} / {total['dof']}, "
                f"p = {total['p_value']:.2g}"
            )


if __name__ == "__main__":
    main()
