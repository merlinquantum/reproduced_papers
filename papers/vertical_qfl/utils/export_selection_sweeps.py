"""Export the complete candidate tables of every selection or diagnostic sweep.

Reads each sweep's ``sweep_status.json`` and ``summary.json`` (structured artifacts
only) and writes ``results/selection_sweeps.md`` plus a JSON copy of each summary under
``results/sweeps/``, so the full tables survive outside the disposable ``outdir/``.

Example
-------
    python utils/export_selection_sweeps.py --sweep T1=outdir/sweep_..._t1 --sweep D1=...
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sweep", action="append", required=True, metavar="LABEL=DIR")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results")
    parser.add_argument(
        "--grouped-selection",
        action="append",
        default=[],
        metavar="LABEL=FILE",
        help="Selection JSON from select_hyperparameters.py --group-by, for sweeps whose "
        "rule was applied per group after the sweep",
    )
    args = parser.parse_args()
    grouped = dict(item.split("=", 1) for item in args.grouped_selection)
    (args.out_dir / "sweeps").mkdir(parents=True, exist_ok=True)
    lines = [
        "# Selection and diagnostic sweeps (complete candidate tables)",
        "",
        "Metric: final-epoch validation accuracy (20% of the training split), maximised; "
        "aggregation: mean (and sample std) over the listed permutation seeds; checkpoint: last epoch; "
        "the test split was never used for selection. Generated from each sweep's summary.json.",
    ]
    for item in args.sweep:
        label, directory = item.split("=", 1)
        path = Path(directory)
        status = json.loads((path / "sweep_status.json").read_text())
        summary = json.loads((path / "summary.json").read_text())
        shutil.copy(
            path / "summary.json", args.out_dir / "sweeps" / f"{label}_summary.json"
        )
        selection = summary["selection"]
        completed = sum(r["status"] == "COMPLETED" for r in status["runs"])
        complete = (
            "COMPLETE SWEEP"
            if completed == status["expected_runs"]
            else "PARTIAL SWEEP"
        )
        if label in grouped:
            chosen = json.loads(Path(grouped[label]).read_text())
            group_keys = chosen["rule"]["group_by"]
            lines += [
                "",
                f"## {label}: `{status['sweep_id']}`",
                "",
                f"Status {status['status']} ({complete}), {completed}/{status['expected_runs']} runs. "
                f"The rule was applied separately per group ({', '.join(group_keys)}) by "
                f"`utils/select_hyperparameters.py --group-by {','.join(group_keys)}` "
                f"(tie tolerance {chosen['rule']['tie_tolerance']}, ties to the smallest "
                f"{', '.join(chosen['rule']['tie_break_ascending'])}); source `{grouped[label]}`.",
            ]
            for group, entry in chosen["selections_by_group"].items():
                key = json.loads(group)
                members = [
                    c
                    for c in selection["candidates"]
                    if all(c["overrides"][k] == v for k, v in key.items())
                ]
                lines += [
                    "",
                    f"### {label} group {json.dumps(key, sort_keys=True)}",
                    "",
                    f"Selected: {entry['selected']}; tied within tolerance: {entry['tied_within_tolerance']}.",
                    "",
                    "| Candidate | Successful / expected | Mean | Std |",
                    "|---|---:|---:|---:|",
                ]
                for c in sorted(
                    members, key=lambda c: -c["final_validation_accuracy"]["mean"]
                ):
                    v = c["final_validation_accuracy"]
                    std = "n/a" if v["std_ddof1"] is None else f"{v['std_ddof1']:.4f}"
                    lines.append(
                        f"| {json.dumps(c['overrides'], sort_keys=True)} | {c['successful_runs']} / {c['expected_runs']} | {v['mean']:.4f} | {std} |"
                    )
            shutil.copy(
                path / "summary.json", args.out_dir / "sweeps" / f"{label}_summary.json"
            )
            continue
        lines += [
            "",
            f"## {label}: `{status['sweep_id']}`",
            "",
            f"Status {status['status']} ({complete}), "
            f"{completed}/{status['expected_runs']} runs. Selected: {status['selected_candidates'] or 'none'}; "
            f"tied within tolerance: {selection.get('tied', [])}.",
            "",
            "| Candidate | Successful / expected | Mean | Std |",
            "|---|---:|---:|---:|",
        ]
        ranked = sorted(
            selection["candidates"],
            key=lambda c: -c["final_validation_accuracy"]["mean"],
        )
        for c in ranked:
            v = c["final_validation_accuracy"]
            std = "n/a" if v["std_ddof1"] is None else f"{v['std_ddof1']:.4f}"
            lines.append(
                f"| {json.dumps(c['overrides'], sort_keys=True)} | {c['successful_runs']} / {c['expected_runs']} | {v['mean']:.4f} | {std} |"
            )
    (args.out_dir / "selection_sweeps.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.out_dir / 'selection_sweeps.md'}")


if __name__ == "__main__":
    main()
