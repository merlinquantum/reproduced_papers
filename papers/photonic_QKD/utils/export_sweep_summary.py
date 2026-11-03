"""Export a sweep coordinator summary into a curated ``results/`` table.

Reads only ``spec.json`` and ``summary.json`` written by ``utils/run_sweep.py``
and renders every candidate (not only the selected one) with the predeclared
selection rule, run counts, coverage and ties, so README tables can link a
committed artifact instead of the disposable ``outdir``.

Usage (from the repository root)::

    python papers/photonic_QKD/utils/export_sweep_summary.py <sweep dir> --name s2_spsa_grid_photonic

Writes ``results/<name>.json`` (a copy of the summary plus the spec's
selection block) and ``results/<name>.md``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PAPER_DIR = Path(__file__).resolve().parents[1]


def _cell(entry: dict[str, Any] | None, digits: int) -> str:
    if entry is None:
        return "—"
    if entry["std"] is None:
        return f"{entry['mean']:.{digits}f}"
    return f"{entry['mean']:.{digits}f} ± {entry['std']:.{digits}f}"


def render(sweep_dir: Path, spec: dict[str, Any], summary: dict[str, Any]) -> str:
    """Return a Markdown table of every candidate in the sweep summary."""
    selection = spec["selection"]
    metrics = list(spec["report_metrics"])
    if selection["metric"] not in metrics:
        metrics.insert(0, selection["metric"])
    lines = [
        f"# Sweep `{summary['sweep_id']}`",
        "",
        f"- Source: `{sweep_dir.name}/summary.json` (generator: `{summary['generator']}`, {summary['generated_at']}).",
        f"- {spec.get('description', '').strip()}",
        f"- Selection rule (predeclared): {selection['metric']} on the {selection['split']} split, "
        f"{selection['checkpoint']}, {selection['direction']}, aggregated as {selection['aggregation']}, "
        f"tie tolerance {selection['tie_tolerance']} → {selection['tie_break']}.",
        f"- Coverage: **{summary['coverage']}**; invalid runs: {summary['invalid_runs'] or 'none'}.",
        f"- Selected: {summary['selected_candidates'] or 'none'}; tied within tolerance: {summary['tied_within_tolerance'] or 'none'}.",
        "",
    ]
    header = ["candidate", "runs", "status", *metrics]
    lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in summary["rows"]:
        cells = [
            row["candidate"],
            f"{row['successful_runs']}/{row['expected_runs']}",
            row["status"],
        ]
        cells += [
            _cell(
                row.get(metric), 4 if "objective" in metric or "cross" in metric else 2
            )
            for metric in metrics
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("sweep_dir", type=Path)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    sweep_dir = args.sweep_dir.resolve()
    spec = json.loads((sweep_dir / "spec.json").read_text())
    summary = json.loads((sweep_dir / "summary.json").read_text())
    results_dir = PAPER_DIR / "results"
    (results_dir / f"{args.name}.json").write_text(
        json.dumps(
            {"sweep_dir": sweep_dir.name, "selection": spec["selection"], **summary},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    text = render(sweep_dir, spec, summary)
    (results_dir / f"{args.name}.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
