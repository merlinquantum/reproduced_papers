"""Apply the predeclared selection rule to the union of several selection sweeps.

Reads each sweep's ``summary.json`` (never logs), merges candidates (the first
occurrence of a repeated candidate wins; a repeat with a different mean is reported),
and writes ``results/photonic_selection.json``.

Example
-------
    python utils/select_hyperparameters.py \
        --sweep T1=outdir/sweep_..._t1 --sweep T2=outdir/sweep_..._t2 \
        --metric final_validation_accuracy --tie-tolerance 0.005 \
        --tie-break lr,client_encoding_scale
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sweep", action="append", required=True, metavar="LABEL=DIR")
    parser.add_argument("--metric", required=True)
    parser.add_argument("--tie-tolerance", type=float, required=True)
    parser.add_argument(
        "--tie-break", required=True, help="comma-separated override keys, ascending"
    )
    parser.add_argument(
        "--out", type=Path, default=PROJECT_DIR / "results/photonic_selection.json"
    )
    parser.add_argument(
        "--group-by",
        default="",
        help="Comma-separated override keys; the rule is applied separately per group",
    )
    args = parser.parse_args()

    merged: dict[str, dict] = {}
    repeats = []
    for item in args.sweep:
        label, directory = item.split("=", 1)
        status = json.loads((Path(directory) / "sweep_status.json").read_text())
        if status["status"] != "COMPLETED":
            raise ValueError(
                f"{label} is {status['status']}: no selection from a partial sweep"
            )
        summary = json.loads((Path(directory) / "summary.json").read_text())
        for candidate in summary["selection"]["candidates"]:
            if candidate["successful_runs"] != candidate["expected_runs"]:
                raise ValueError(
                    f"{label}: incomplete candidate {candidate['overrides']}"
                )
            key = json.dumps(candidate["overrides"], sort_keys=True)
            mean = candidate[args.metric]["mean"]
            if not math.isfinite(mean):
                raise ValueError(f"Non-finite {args.metric} for {key}")
            if key in merged:
                repeats.append(
                    {
                        "overrides": candidate["overrides"],
                        "kept_from": merged[key]["sweep"],
                        "repeat_from": label,
                        "kept_mean": merged[key]["mean"],
                        "repeat_mean": mean,
                        "identical": merged[key]["mean"] == mean,
                    }
                )
                continue
            merged[key] = {
                "overrides": candidate["overrides"],
                "sweep": label,
                "mean": mean,
                "std_ddof1": candidate[args.metric]["std_ddof1"],
            }
    keys = [k for k in args.tie_break.split(",") if k]
    group_keys = [k for k in args.group_by.split(",") if k]

    def select(members: list[dict]) -> tuple[dict, list[dict]]:
        best = max(c["mean"] for c in members)
        tied = [c for c in members if c["mean"] >= best - args.tie_tolerance]
        tied.sort(key=lambda c: tuple(c["overrides"][k] for k in keys))
        return tied[0]["overrides"], [c["overrides"] for c in tied]

    candidates = sorted(merged.values(), key=lambda c: -c["mean"])
    groups: dict[str, list[dict]] = {}
    for c in candidates:
        label = json.dumps({k: c["overrides"][k] for k in group_keys}, sort_keys=True)
        groups.setdefault(label, []).append(c)
    selections = {}
    for label, members in groups.items():
        chosen, tied = select(members)
        selections[label] = {"selected": chosen, "tied_within_tolerance": tied}
    result = {
        "rule": {
            "metric": args.metric,
            "direction": "maximize",
            "aggregation": "mean over permutation seeds",
            "tie_tolerance": args.tie_tolerance,
            "tie_break_ascending": keys,
            "group_by": group_keys,
        },
        "candidates_ranked_by_mean": candidates,
        "selections_by_group": selections,
        "repeated_candidates": repeats,
    }
    if not group_keys:
        only = next(iter(selections.values()))
        result["selected"] = only["selected"]
        result["tied_within_tolerance"] = only["tied_within_tolerance"]
    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {"selections_by_group": selections, "repeated_candidates": repeats},
            indent=2,
        )
    )
    for c in candidates:
        print(f"{c['sweep']}  {c['overrides']}  mean={c['mean']:.4f}")


if __name__ == "__main__":
    main()
