"""Aggregate the four-seed paper-scale synthetic experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

_ENCODING_NAMES = {
    "quantum": "quantum",
    "rrwp": "rrwp",
    "laplacian": "laplacian",
    "none": "gcn",
}

_EXPECTED_DATASET_SETTINGS = {
    "per_class": 400,
    "length_range": [100, 400],
    "seed": 314159,
    "pe_dim": 20,
    "crossing_range": [2, 2],
}


def aggregate_runs(outdir: Path) -> dict:
    """Collect one complete run for every method and seed.

    Parameters
    ----------
    outdir : pathlib.Path
        QPE_GNN runtime output directory.

    Returns
    -------
    dict
        Per-method scores with population mean and standard deviation.

    Raises
    ------
    ValueError
        If a method is missing any required seed.
    """
    collected: dict[str, dict[int, tuple[str, float]]] = {
        method: {} for method in _ENCODING_NAMES.values()
    }
    for metrics_path in sorted(outdir.glob("run_*/metrics.json")):
        config_path = metrics_path.with_name("config_snapshot.json")
        if not config_path.is_file():
            continue
        config = json.loads(config_path.read_text())
        if config.get("dataset") != "ladder_concat" or config.get("epochs") != 200:
            continue
        dataset_kwargs = config.get("dataset_kwargs", {})
        if config.get("split_seed") != 1729 or any(
            dataset_kwargs.get(key) != expected_value
            for key, expected_value in _EXPECTED_DATASET_SETTINGS.items()
        ):
            continue
        node_encoding = dataset_kwargs.get("node_encoding")
        method = _ENCODING_NAMES.get(node_encoding)
        if method is None:
            continue
        metrics = json.loads(metrics_path.read_text())
        seed = int(metrics["seed"])
        collected[method][seed] = (
            str(metrics_path.parent),
            float(metrics["test_metric"]["value"]),
        )

    summary = {"required_seeds": [0, 1, 2, 3], "methods": {}}
    for method, seed_results in collected.items():
        missing_seeds = set(summary["required_seeds"]) - set(seed_results)
        if missing_seeds:
            raise ValueError(
                f"{method} is missing seeds: {', '.join(map(str, sorted(missing_seeds)))}"
            )
        scores = np.asarray(
            [seed_results[seed][1] for seed in summary["required_seeds"]]
        )
        summary["methods"][method] = {
            "scores": scores.tolist(),
            "mean": float(scores.mean()),
            "standard_deviation": float(scores.std(ddof=0)),
            "run_directories": [
                seed_results[seed][0] for seed in summary["required_seeds"]
            ],
        }
    return summary


def main() -> None:
    """Parse paths, aggregate runs, and write the JSON report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("outdir"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/synthetic_original_summary.json"),
    )
    arguments = parser.parse_args()
    summary = aggregate_runs(arguments.outdir)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
