#!/usr/bin/env python3
"""Apply the predeclared S3 selection rule (LOG.md, Sweep Records, S3).

Per model family, select the candidate with the lowest mean validation RMSE
(observed targets, full precision) over its seeds. Candidates within 1 % of the
family best are tied; ties are broken by fewer trainable parameters, then
smaller learning rate (closest to the paper's 1e-3); for ridge, larger alpha.
Diagnostic candidates (``D2_*``, ``ANG_*``) and persistence are reported but not
selected. Requires complete coverage of every family member.

Writes ``selection.json`` / ``selection.md`` to the sweep directory and to
``papers/QK_LSTM/results/s3_val_tuning/``.

Usage::

    .venv/bin/python papers/QK_LSTM/utils/select_s3.py <S3_SWEEP_DIR>
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
TIE_TOLERANCE = 0.01
FAMILIES = {
    "QB_quantum": r"^QB_quantum_lr(?P<lr>[\d.]+)$",
    "B3B_rbf": r"^B3B_rbf_lr(?P<lr>[\d.]+)$",
    "QA_quantum": r"^QA_quantum_N(?P<n>\d+)_lr(?P<lr>[\d.]+)$",
    "B3A_rbf": r"^B3A_rbf_N(?P<n>\d+)_lr(?P<lr>[\d.]+)$",
    "B1_lstm16": r"^B1_lstm16_lr(?P<lr>[\d.]+)$",
    "B2_lstm3": r"^B2_lstm3_lr(?P<lr>[\d.]+)$",
    "B0b_ridge": r"^B0b_ridge_alpha(?P<alpha>[\d.]+)$",
}


def main(argv: list[str]) -> int:
    sweep_dir = Path(argv[0]).resolve()
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    plan = json.loads((sweep_dir / "plan.json").read_text())
    expected_seeds = set(plan["seeds"])

    per_candidate: dict[str, dict] = {}
    for run in status["runs"]:
        entry = per_candidate.setdefault(
            run["candidate"], {"val": {}, "test": {}, "params": set()}
        )
        if run["status"] != "COMPLETED":
            continue
        metrics = json.loads((sweep_dir / run["metrics_path"]).read_text())
        entry["val"][run["seed"]] = metrics["splits"]["validation"]["observed_targets"][
            "rmse"
        ]
        entry["test"][run["seed"]] = metrics["splits"]["test"]["observed_targets"][
            "rmse"
        ]
        entry["params"].add(metrics["trainable_parameters"])

    def summary(label: str) -> dict:
        entry = per_candidate[label]
        values = [entry["val"][s] for s in sorted(entry["val"])]
        return {
            "candidate": label,
            "complete": set(entry["val"]) == expected_seeds,
            "n_seeds": len(values),
            "validation_rmse_mean": statistics.fmean(values) if values else None,
            "validation_rmse_std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "test_rmse_mean_not_used_for_selection": statistics.fmean(
                entry["test"].values()
            )
            if entry["test"]
            else None,
            "trainable_parameters": sorted(entry["params"]),
        }

    selection = {
        "sweep_id": status["sweep_id"],
        "rule": __doc__.split("\n\n")[1].replace("\n", " "),
        "families": {},
        "diagnostics": [],
    }
    for family, pattern in FAMILIES.items():
        members = []
        for label in per_candidate:
            match = re.match(pattern, label)
            if match:
                row = summary(label)
                row.update({k: float(v) for k, v in match.groupdict().items()})
                members.append(row)
        if not members or not all(m["complete"] for m in members):
            selection["families"][family] = {
                "status": "PARTIAL",
                "members": members,
                "selected": None,
            }
            continue
        best = min(m["validation_rmse_mean"] for m in members)
        tied = [
            m
            for m in members
            if m["validation_rmse_mean"] - best <= TIE_TOLERANCE * best
        ]
        if family == "B0b_ridge":
            tied.sort(key=lambda m: -m["alpha"])
        else:
            tied.sort(key=lambda m: (max(m["trainable_parameters"]), m["lr"]))
        selection["families"][family] = {
            "status": "COMPLETE",
            "best_validation_rmse_mean": best,
            "tied_within_1pct": [m["candidate"] for m in tied],
            "selected": tied[0]["candidate"],
            "members": sorted(members, key=lambda m: m["validation_rmse_mean"]),
        }
    for label in per_candidate:
        if label.startswith(("D2_", "ANG_")) or label == "B0_persistence_val":
            selection["diagnostics"].append(summary(label))

    lines = [
        f"# {status['sweep_id']}: validation-only selection (predeclared rule)",
        "",
    ]
    for family, result in selection["families"].items():
        lines.append(
            f"## {family}: selected `{result['selected']}` ({result['status']})"
        )
        lines.append("")
        lines.append("| Candidate | Params | Seeds | Val RMSE mean ± std | Tied |")
        lines.append("|---|---:|---:|---:|---|")
        for member in result["members"]:
            tied = (
                "yes"
                if member["candidate"] in result.get("tied_within_1pct", [])
                else ""
            )
            lines.append(
                f"| {member['candidate']} | {'/'.join(map(str, member['trainable_parameters']))} | {member['n_seeds']} | {member['validation_rmse_mean']:.3f} ± {member['validation_rmse_std']:.3f} | {tied} |"
            )
        lines.append("")
    lines.append("## Diagnostics (not selected; validation split, seeds 0-2)")
    lines.append("")
    lines.append("| Candidate | Params | Val RMSE mean ± std |")
    lines.append("|---|---:|---:|")
    for row in sorted(
        selection["diagnostics"], key=lambda r: r["validation_rmse_mean"]
    ):
        lines.append(
            f"| {row['candidate']} | {'/'.join(map(str, row['trainable_parameters']))} | {row['validation_rmse_mean']:.3f} ± {row['validation_rmse_std']:.3f} |"
        )
    markdown = "\n".join(lines) + "\n"

    results_dir = PROJECT_DIR / "results" / plan["sweep_name"]
    results_dir.mkdir(parents=True, exist_ok=True)
    for directory in (sweep_dir, results_dir):
        (directory / "selection.json").write_text(
            json.dumps(selection, indent=2) + "\n", encoding="utf-8"
        )
        (directory / "selection.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
