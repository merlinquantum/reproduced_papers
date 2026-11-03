"""Summarise fitting runs (E1, E2) into result tables, read only from metrics.json.

Usage, from the paper directory::

    python utils/summarize_fits.py \
        --run "E2 quantum s=1, 200 steps=outdir/run_A" \
        --run "E2 B2 s=2, 200 steps=outdir/run_B" \
        --output results/e2_fit_summary

Writes ``<output>.json`` (full precision) and ``<output>.md`` (tables). For
every case the tables give mean +- sample std, min and max of the final
full-grid MSE over seeds, the B1 floor and the number of seeds within 1e-4 of
B1. For cases that differ only in ``repetitions`` (same layout and run), it
also gives paired per-seed comparisons between consecutive ``r``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _load(run_dir: Path) -> dict:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    if status["status"] != "COMPLETED":
        raise ValueError(f"{run_dir} is not COMPLETED: {status['status']}")
    if metrics["experiment"] != "fit":
        raise ValueError(f"{run_dir} is not a fit run")
    return metrics


def _check_values(runs: list[dict], run_dir: Path) -> None:
    seen = set()
    for run in runs:
        key = (run["case"], run["model_kind"], run["seed"])
        if key in seen:
            raise ValueError(f"Duplicate result {key} in {run_dir}")
        seen.add(key)
        if not math.isfinite(run["final_mse"]):
            raise ValueError(f"Non-finite MSE {key} in {run_dir}")


def _family(model: dict) -> str:
    """Return the repetition family: the qubit layout, or the photonic route."""
    if model["layout"] == "photonic":
        return f"photonic/{model['route']}"
    return model["layout"]


def _degree(case: dict) -> int:
    """Return a case's predicted degree: the largest frequency of its model's spectrum (recorded by the run)."""
    return max(case["b1_frequencies"])


def _repetition_pairs(metrics: dict) -> list[dict]:
    """Paired per-seed comparisons between consecutive r for each layout."""
    cases = {case["name"]: case for case in metrics["cases"]}
    # Only cases with the same layout, target and encoding scaling form an r sequence.
    groups: dict[tuple, list[dict]] = {}
    for case in metrics["cases"]:
        key = (
            _family(case["model"]),
            case["model"]["encoding_scaling"],
            json.dumps(case["target"], sort_keys=True),
        )
        groups.setdefault(key, []).append(case)
    pairs = []
    for (layout, _, _), group_cases in groups.items():
        ordered = sorted(group_cases, key=_degree)
        for smaller, larger in zip(ordered, ordered[1:]):
            if _degree(smaller) == _degree(larger):
                continue
            for kind in metrics["model_kinds"]:
                small = {
                    r["seed"]: r["final_mse"]
                    for r in metrics["runs"]
                    if r["case"] == smaller["name"] and r["model_kind"] == kind
                }
                large = {
                    r["seed"]: r["final_mse"]
                    for r in metrics["runs"]
                    if r["case"] == larger["name"] and r["model_kind"] == kind
                }
                seeds = sorted(set(small) & set(large))
                pairs.append(
                    {
                        "layout": layout,
                        "model_kind": kind,
                        "from_case": smaller["name"],
                        "to_case": larger["name"],
                        "from_r": _degree(cases[smaller["name"]]),
                        "to_r": _degree(cases[larger["name"]]),
                        "n_seeds": len(seeds),
                        "seeds_where_larger_r_is_lower": sum(
                            large[s] < small[s] for s in seeds
                        ),
                    }
                )
    return pairs


def summarise(labelled_runs: list[tuple[str, Path]]) -> dict:
    """Build the summary for ``(label, run_dir)`` pairs."""
    report = {"runs": []}
    for label, run_dir in labelled_runs:
        metrics = _load(run_dir)
        _check_values(metrics["runs"], run_dir)
        report["runs"].append(
            {
                "label": label,
                "run_dir": str(run_dir),
                "steps": metrics["training"]["steps"],
                "seeds": metrics["seeds"],
                "success_tolerance": metrics["success_tolerance"],
                "summary": metrics["summary"],
                "repetition_pairs": _repetition_pairs(metrics),
            }
        )
    return report


def to_markdown(report: dict) -> str:
    """Render the summary as Markdown tables."""
    lines = []
    for entry in report["runs"]:
        lines.append(f"### {entry['label']}")
        lines.append("")
        lines.append(
            f"Run `{entry['run_dir']}`, {entry['steps']} steps, seeds {entry['seeds']}; "
            f"success = final MSE within {entry['success_tolerance']:g} of B1."
        )
        lines.append("")
        lines.append(
            "| Case | Model | Params | Final MSE mean ± std | Min | Max | B1 MSE | Success |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for row in entry["summary"]:
            successes = round(row["success_fraction"] * row["n_seeds"])
            lines.append(
                f"| {row['case']} | {row['model_kind']} | {row['n_parameters']} | "
                f"{row['final_mse_mean']:.3e} ± {row['final_mse_std']:.1e} | "
                f"{row['final_mse_min']:.2e} | {row['final_mse_max']:.2e} | "
                f"{row['b1_mse']:.3e} | {successes}/{row['n_seeds']} |"
            )
        if entry["repetition_pairs"]:
            lines.append("")
            lines.append(
                "| Family | Model | degree r → r' | Seeds with lower MSE at r' |"
            )
            lines.append("|---|---|---|---:|")
            for pair in entry["repetition_pairs"]:
                lines.append(
                    f"| {pair['layout']} | {pair['model_kind']} | {pair['from_r']} → {pair['to_r']} | "
                    f"{pair['seeds_where_larger_r_is_lower']}/{pair['n_seeds']} |"
                )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="'<label>=<run directory>'; repeat for several runs",
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Output path without suffix"
    )
    args = parser.parse_args()
    labelled_runs = []
    for item in args.run:
        label, separator, path = item.rpartition("=")
        if not separator:
            raise ValueError(f"--run needs '<label>=<run directory>', got {item!r}")
        labelled_runs.append((label, Path(path)))
    report = summarise(labelled_runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    args.output.with_suffix(".md").write_text(to_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
