"""Runtime entrypoints for the Quantum random kitchen sinks reproduction."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from utils.visualization import build_decision_payload

from lib.training import run_rks_experiments

from .data import load_moons

DEFAULT_GAMMAS = list(range(1, 11))
LOGGER = logging.getLogger(__name__)


def summarize_results(
    results: list[dict[str, Any]],
    r_values: Iterable[int],
    gamma_values: Iterable[int],
) -> str:
    lines = ["Quantum vs Classical Random Kitchen Sinks", ""]
    for method in ["quantum", "classical"]:
        lines.append(method.capitalize())
        for r in r_values:
            for gamma in gamma_values:
                subset = [
                    entry
                    for entry in results
                    if entry["method"] == method
                    and entry["r"] == r
                    and entry["gamma"] == gamma
                ]
                if subset:
                    avg = sum(entry["accuracy"] for entry in subset) / len(subset)
                    lines.append(f"  R={r:>3}, gamma={gamma:>2}: {avg:.4f}")
        lines.append("")
    return "\n".join(lines)


def train_and_evaluate(cfg: dict[str, Any], run_dir: Path) -> None:
    dataset_cfg = cfg.get("data", {})
    dataset = load_moons(dataset_cfg)
    visualization_dir = run_dir / "visualization_data"
    visualization_dir.mkdir(parents=True, exist_ok=True)

    results = run_rks_experiments(
        dataset,
        cfg.get("sweep", {}),
        cfg.get("model", {}),
        cfg.get("training", {}),
        cfg.get("classifier", {}),
        int(cfg.get("seed", 1337)),
    )

    metrics = [
        {
            "method": entry["method"],
            "r": entry["r"],
            "gamma": entry["gamma"],
            "repeat": entry["repeat"],
            "accuracy": entry["accuracy"],
        }
        for entry in results
    ]
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    x_train, x_test, y_train, y_test = dataset
    dataset_payload = {
        "x_train": x_train.tolist(),
        "x_test": x_test.tolist(),
        "y_train": y_train.tolist(),
        "y_test": y_test.tolist(),
    }
    (visualization_dir / "dataset.json").write_text(
        json.dumps(dataset_payload, indent=2)
    )

    r_values = cfg.get("sweep", {}).get("r_values", [1])
    gamma_values = cfg.get("sweep", {}).get("gamma_values", DEFAULT_GAMMAS)

    summary = summarize_results(results, r_values, gamma_values)
    (run_dir / "summary.txt").write_text(summary)

    decision_quantum = build_decision_payload(
        results, dataset, r_values, gamma_values, "quantum"
    )
    decision_classical = build_decision_payload(
        results, dataset, r_values, gamma_values, "classical"
    )
    (visualization_dir / "combined_decisions_quantum.json").write_text(
        json.dumps(decision_quantum, indent=2)
    )
    (visualization_dir / "combined_decisions_classical.json").write_text(
        json.dumps(decision_classical, indent=2)
    )

    best_quantum = max(
        (entry for entry in results if entry["method"] == "quantum"),
        key=lambda e: e["accuracy"],
    )
    best_classical = max(
        (entry for entry in results if entry["method"] == "classical"),
        key=lambda e: e["accuracy"],
    )

    LOGGER.info(
        "Best quantum accuracy %.4f (R=%s, gamma=%s)",
        best_quantum["accuracy"],
        best_quantum["r"],
        best_quantum["gamma"],
    )
    LOGGER.info(
        "Best classical accuracy %.4f (R=%s, gamma=%s)",
        best_classical["accuracy"],
        best_classical["r"],
        best_classical["gamma"],
    )

    kernel_payload_quantum = {
        "kernel": best_quantum["kernel_train"].tolist(),
        "y_train": y_train.tolist(),
        "title": f"Quantum kernel R={best_quantum['r']}, γ={best_quantum['gamma']}",
    }
    kernel_payload_classical = {
        "kernel": best_classical["kernel_train"].tolist(),
        "y_train": y_train.tolist(),
        "title": f"Classical kernel R={best_classical['r']}, γ={best_classical['gamma']}",
    }
    if len(r_values) == 1 and len(gamma_values) == 1:
        (visualization_dir / "kernel_quantum.json").write_text(
            json.dumps(kernel_payload_quantum, indent=2)
        )
        (visualization_dir / "kernel_classical.json").write_text(
            json.dumps(kernel_payload_classical, indent=2)
        )
    else:
        (visualization_dir / "kernel_quantum_best.json").write_text(
            json.dumps(kernel_payload_quantum, indent=2)
        )
        (visualization_dir / "kernel_classical_best.json").write_text(
            json.dumps(kernel_payload_classical, indent=2)
        )

    LOGGER.info("Artifacts saved to %s", run_dir.resolve())


__all__ = ["train_and_evaluate", "summarize_results"]
