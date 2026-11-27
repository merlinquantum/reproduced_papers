"""Phase 5 comparison table and statistics from structured run artifacts.

Usage (from the repository root)::

    python3 papers/contextuality_inductive_bias/utils/fair_comparison_table.py \\
        --final-dir papers/contextuality_inductive_bias/outdir/sweeps/fair_final \\
        --robustness-dir papers/contextuality_inductive_bias/outdir/sweeps/fair_robustness \\
        --selection papers/contextuality_inductive_bias/results/fair_tuning_selection.json \\
        --out-dir papers/contextuality_inductive_bias/results \\
        --untuned-reference papers/contextuality_inductive_bias/outdir/sweeps/fig8_surrogates \\
        --untuned-grid '{"--lr": 0.0001, "--protocol": "paper", "--spectrum": 2}'

For every model of the final sweep it enumerates the planned runs (from
``sweep_status.json``), keeps COMPLETED runs with finite metrics, and reports:
- successful / expected runs;
- mean and sample std of the final-step test KL (primary);
- the test KL at the step of minimum validation NLL (validation early
  stopping, secondary);
- the final validation and train NLL, the parameter count, and the mean
  ``train_seconds`` (wall-clock of the training loop).

Pairwise comparisons use Welch two-sided t-tests (95% CI on the difference)
and Mann-Whitney U. A difference counts as resolved only if both p < 0.05.
Data robustness is paired over data seeds (sign count, paired t).

``--untuned-reference SWEEP_DIR --untuned-grid JSON`` adds one Phase 3 system
at its paper settings (the surrogate L = 2, which the plan keeps untuned). It is
read from that sweep's ``summary.json``, labelled PARTIALLY_COMMENSURABLE and
kept out of the statistics. Every system also gets a commensurability status.
Writes ``fair_comparison.{csv,md,json}``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep_io import describe  # noqa: E402

ROLES = {
    "biased_quantum": "paper method (fair-tuned)",
    "generic_quantum": "paper baseline (fair-tuned)",
    "surrogate": "paper baseline B0 (fair-tuned)",
    "bias_fourier": "additional fair baseline B2",
    "mlp_biased": "additional fair baseline B3",
    "mlp_unbiased": "additional fair baseline B4",
    "quadratic": "ceiling B5 (class contains the truth)",
    "photonic_merlin": "MerLin ADAPTATION (Phase 4)",
}
PAIRS = [
    ("quantum (biased) L=2", None),
    ("quantum (biased) L=1", None),
    ("photonic MerLin (group readout, PNR) N=2", None),
    ("B2 bias-encoded Fourier L=1", "surrogate L=1"),
    ("B3 bias-encoded MLP h=7", "B4 unbiased MLP h=9"),
    ("quantum (biased) L=2", "quantum (no bias, Fig. 10 CNOTs) L=2"),
    ("quantum (biased) L=1", "quantum (no bias, Fig. 10 CNOTs) L=1"),
]


def _runs(sweep_dir: Path) -> dict[str, dict]:
    status = json.loads((sweep_dir / "sweep_status.json").read_text(encoding="utf-8"))
    groups: dict[str, dict] = defaultdict(
        lambda: {"expected": 0, "records": [], "excluded": []}
    )
    for run in status["runs"]:
        if run["status"] != "COMPLETED":
            # Group failed runs by their planned grid point; resolved to a name below.
            groups[json.dumps(run["grid_point"], sort_keys=True)]["excluded"].append(
                (run["run_id"], run["status"])
            )
            groups[json.dumps(run["grid_point"], sort_keys=True)]["expected"] += 1
            continue
        run_dir = sweep_dir / run["run_dir"]
        run_status = json.loads(
            (run_dir / "run_status.json").read_text(encoding="utf-8")
        )
        metrics = json.loads(
            (run_dir / run_status["metrics_path"]).read_text(encoding="utf-8")
        )
        key = json.dumps(run["grid_point"], sort_keys=True)
        groups[key]["expected"] += 1
        groups[key]["candidate"] = run_status["candidate"]
        values = {
            "final_test_kl": metrics["final"]["test_kl"],
            "early_stop_test_kl": metrics["min_validation_nll"]["test_kl"],
            "final_validation_nll": metrics["final"]["validation_nll"],
            "final_train_nll": metrics["final"]["train_nll"],
            "train_seconds": metrics["train_seconds"],
            "n_parameters": metrics["n_parameters"],
            "data_seed": run_status["candidate"].get("train_seed"),
        }
        if not all(
            math.isfinite(v) for k, v in values.items() if k not in ("data_seed",)
        ):
            groups[key]["excluded"].append(
                (run["run_id"], "INVALID (non-finite metric)")
            )
            continue
        groups[key]["records"].append({"seed": run["seed"], **values})
    named = {}
    for key, group in groups.items():
        if "candidate" not in group:
            raise ValueError(f"No completed run for {key}; cannot name the system")
        name = describe(group["candidate"])
        if (
            group["candidate"]["model"]["kind"] == "photonic_merlin"
            and "computation_space" not in group["candidate"]["model"]
        ):
            raise ValueError(f"{key}: photonic candidate without computation_space")
        named.setdefault(
            name,
            {
                "expected": 0,
                "records": [],
                "excluded": [],
                "candidate": group["candidate"],
                "grid_point": key,
            },
        )
        named[name]["expected"] += group["expected"]
        named[name]["records"] += group["records"]
        named[name]["excluded"] += group["excluded"]
    return named


def _compare(a: np.ndarray, b: np.ndarray) -> dict:
    va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
    dof = (va + vb) ** 2 / (va**2 / (a.size - 1) + vb**2 / (b.size - 1))
    half = stats.t.ppf(0.975, dof) * math.sqrt(va + vb)
    diff = a.mean() - b.mean()
    welch = stats.ttest_ind(a, b, equal_var=False).pvalue
    mw = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
    return {
        "diff": diff,
        "ci_low": diff - half,
        "ci_high": diff + half,
        "welch_p": welch,
        "mw_p": mw,
        "resolved": bool(welch < 0.05 and mw < 0.05),
    }


COMMENSURABILITY = {
    "photonic_merlin": (
        "COMMENSURABLE",
        "same data, protocol, test set, steps and tuning budget; a simulated MerLin ADAPTATION (exact probabilities, PNR detection), not the reproduction",
    ),
    "surrogate": (
        "COMMENSURABLE",
        "PARTIAL coverage (9/10): one seed failed on both attempts (HardTanh saturation under Adam)",
    ),
}
DEFAULT_COMMENSURABILITY = (
    "COMMENSURABLE",
    "same data, protocol, test set, steps, seeds and tuning budget",
)


def _untuned_reference(sweep_dir: Path, grid_point: dict) -> dict:
    summary = json.loads((sweep_dir / "summary.json").read_text(encoding="utf-8"))
    matches = [c for c in summary["candidates"] if c["grid_point"] == grid_point]
    if len(matches) != 1:
        raise ValueError(
            f"{sweep_dir}: expected exactly one candidate {grid_point}, found {len(matches)}"
        )
    candidate = matches[0]
    kl = candidate["aggregates"]["final.test_kl"]
    seconds = candidate["aggregates"]["train_seconds"]
    return {
        "system": f"surrogate L={grid_point['--spectrum']} (not tuned, paper App. F settings)",
        "provenance": f"locally executed (Phase 3, {summary['sweep_id']})",
        "grid_point": grid_point,
        "successful": candidate["successful_runs"],
        "expected": candidate["expected_runs"],
        "seeds": sorted(r["seed"] for r in candidate["runs"]),
        "final_test_kl_mean": kl["mean"],
        "final_test_kl_std": kl["std"],
        "train_seconds_mean": seconds["mean"],
        "status": "PARTIALLY_COMMENSURABLE",
        "reason": "same data, protocol paper, test set and 2000 steps, but plain SGD at the fixed paper lr, no validation tuning, Phase 3 seeds; excluded from the statistics",
    }


def build(
    final_dir: Path,
    robustness_dir: Path,
    selection_path: Path,
    out_dir: Path,
    untuned_reference: tuple[Path, dict] | None = None,
) -> dict:
    final = _runs(final_dir)
    robustness = _runs(robustness_dir)
    selection = {
        json.dumps(
            dict(
                item["model"],
                **{"--lr": item["selected"]["lr"], "--l2": item["selected"]["l2"]},
            ),
            sort_keys=True,
        ): item["selected"]
        for item in json.loads(selection_path.read_text(encoding="utf-8"))
    }
    rows = []
    for name, group in sorted(
        final.items(),
        key=lambda kv: np.mean([r["final_test_kl"] for r in kv[1]["records"]]),
    ):
        values = {
            k: np.array([r[k] for r in group["records"]], dtype=float)
            for k in (
                "final_test_kl",
                "early_stop_test_kl",
                "final_validation_nll",
                "final_train_nll",
                "train_seconds",
            )
        }
        selected = selection[group["grid_point"]]
        rows.append(
            {
                "system": name,
                "role": ROLES[group["candidate"]["model"]["kind"]],
                "provenance": "locally executed (fair_final)",
                "scope": "fair-tuned, paper-accurate data"
                + ("; PARTIAL" if len(group["records"]) < group["expected"] else ""),
                "successful": len(group["records"]),
                "expected": group["expected"],
                "excluded": group["excluded"],
                "lr": selected["lr"],
                "l2": selected["l2"],
                "n_parameters": int(group["records"][0]["n_parameters"]),
                **{f"{k}_mean": float(v.mean()) for k, v in values.items()},
                **{f"{k}_std": float(v.std(ddof=1)) for k, v in values.items()},
                "matching_basis": "same data/protocol/steps/tuning budget; capacity per Comparison Plan",
                "commensurability": COMMENSURABILITY.get(
                    group["candidate"]["model"]["kind"], DEFAULT_COMMENSURABILITY
                )[0],
                "commensurability_reason": COMMENSURABILITY.get(
                    group["candidate"]["model"]["kind"], DEFAULT_COMMENSURABILITY
                )[1],
            }
        )
    by_name = {
        name: np.array([r["final_test_kl"] for r in g["records"]])
        for name, g in final.items()
    }
    early = {
        name: np.array([r["early_stop_test_kl"] for r in g["records"]])
        for name, g in final.items()
    }
    comparisons = []
    for reference, other in PAIRS:
        others = [other] if other else [n for n in by_name if n != reference]
        for name in others:
            for metric, table in (
                ("final_test_kl", by_name),
                ("early_stop_test_kl", early),
            ):
                comparisons.append(
                    {
                        "metric": metric,
                        "reference": reference,
                        "other": name,
                        "n": f"{len(table[reference])} vs {len(table[name])}",
                        **_compare(table[reference], table[name]),
                    }
                )
    paired = []
    seeds_by_name = {
        name: {r["data_seed"]: r["final_test_kl"] for r in g["records"]}
        for name, g in robustness.items()
    }
    for reference in (
        "quantum (biased) L=2",
        "quantum (biased) L=1",
        "photonic MerLin (group readout, PNR) N=2",
        "B2 bias-encoded Fourier L=1",
    ):
        for name in seeds_by_name:
            if name == reference:
                continue
            common = sorted(set(seeds_by_name[reference]) & set(seeds_by_name[name]))
            a = np.array([seeds_by_name[reference][s] for s in common])
            b = np.array([seeds_by_name[name][s] for s in common])
            diff = a - b
            paired.append(
                {
                    "reference": reference,
                    "other": name,
                    "data_seeds": common,
                    "mean_diff": float(diff.mean()),
                    "reference_lower": int((diff < 0).sum()),
                    "n": len(common),
                    "paired_t_p": float(stats.ttest_rel(a, b).pvalue)
                    if len(common) > 1
                    else math.nan,
                }
            )
    robustness_means = {
        name: {"mean": float(np.mean(list(v.values()))), "per_seed": v}
        for name, v in seeds_by_name.items()
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "fair_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=[k for k in rows[0] if k != "excluded"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({k: v for k, v in row.items() if k != "excluded"})
    lines = [
        "### Phase 5 fair comparison (protocol paper, author training data, seeds 10-19, 2000 Adam steps)",
        "",
        "| System | Role | Provenance | Scope | Successful / expected | Final test KL (mean +- std) | Early-stopped test KL (mean +- std) | Selected lr, L2 | Parameters | Wall-clock per run (s) | Excluded |",
        "|---|---|---|---|---:|---:|---:|---|---:|---:|---|",
    ]
    for row in rows:
        excluded = "; ".join(f"{rid} {st}" for rid, st in row["excluded"]) or "none"
        lines.append(
            f"| {row['system']} | {row['role']} | {row['provenance']} | {row['scope']} | {row['successful']}/{row['expected']} | "
            f"{row['final_test_kl_mean']:.3e} +- {row['final_test_kl_std']:.1e} | {row['early_stop_test_kl_mean']:.3e} +- {row['early_stop_test_kl_std']:.1e} | "
            f"{row['lr']:g}, {row['l2']:g} | {row['n_parameters']} | {row['train_seconds_mean']:.0f} | {excluded} |"
        )
    reference_row = (
        _untuned_reference(*untuned_reference) if untuned_reference else None
    )
    if reference_row:
        lines += [
            "",
            "### Untuned reference (not in the statistics)",
            "",
            "| System | Provenance | Successful / expected | Seeds | Final test KL (mean +- std) | Wall-clock per run (s) | Status |",
            "|---|---|---:|---|---:|---:|---|",
            f"| {reference_row['system']} | {reference_row['provenance']} | {reference_row['successful']}/{reference_row['expected']} | "
            f"{', '.join(map(str, reference_row['seeds']))} | {reference_row['final_test_kl_mean']:.3e} +- {reference_row['final_test_kl_std']:.1e} | "
            f"{reference_row['train_seconds_mean']:.0f} | {reference_row['status']}: {reference_row['reason']} |",
        ]
    lines += [
        "",
        "### Commensurability",
        "",
        "| System | Status | Reason |",
        "|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['system']} | {row['commensurability']} | {row['commensurability_reason']} |"
        )
    if reference_row:
        lines.append(
            f"| {reference_row['system']} | {reference_row['status']} | {reference_row['reason']} |"
        )
    lines.append(
        "| Paper-reported Fig. 8 values (read off) | INCOMPARABLE | author-code evaluation (unscaled test inputs, D12), untuned optimisers, test-selected curves; compared separately in `fig8_vs_paper.md` |"
    )
    lines += [
        "",
        "### Pairwise comparisons (Welch 95% CI on reference - other; resolved = Welch and Mann-Whitney p < 0.05)",
        "",
        "| Metric | Reference | Other | n | Diff | 95% CI | Welch p | MW p | Resolved |",
        "|---|---|---|---|---:|---|---:|---:|---|",
    ]
    for c in comparisons:
        lines.append(
            f"| {c['metric']} | {c['reference']} | {c['other']} | {c['n']} | {c['diff']:+.2e} | [{c['ci_low']:+.2e}, {c['ci_high']:+.2e}] | {c['welch_p']:.1e} | {c['mw_p']:.1e} | {'yes' if c['resolved'] else 'no'} |"
        )
    lines += [
        "",
        "### Data robustness (fair_robustness: init seed 10, data seeds 101-103, final test KL)",
        "",
        "| System | data seed 101 | 102 | 103 | mean |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, info in sorted(robustness_means.items(), key=lambda kv: kv[1]["mean"]):
        per = info["per_seed"]
        lines.append(
            f"| {name} | "
            + " | ".join(f"{per[s]:.3e}" for s in sorted(per))
            + f" | {info['mean']:.3e} |"
        )
    lines += [
        "",
        "| Reference | Other | Mean paired diff | Reference lower | Paired t p |",
        "|---|---|---:|---:|---:|",
    ]
    for p in paired:
        lines.append(
            f"| {p['reference']} | {p['other']} | {p['mean_diff']:+.2e} | {p['reference_lower']}/{p['n']} | {p['paired_t_p']:.2g} |"
        )
    (out_dir / "fair_comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    result = {
        "rows": rows,
        "untuned_reference": reference_row,
        "comparisons": comparisons,
        "robustness": robustness_means,
        "paired": paired,
    }
    (out_dir / "fair_comparison.json").write_text(
        json.dumps(result, indent=2, default=float) + "\n", encoding="utf-8"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--final-dir", type=Path, required=True)
    parser.add_argument("--robustness-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--untuned-reference",
        type=Path,
        help="Phase 3 sweep directory holding an untuned reference system",
    )
    parser.add_argument(
        "--untuned-grid", type=json.loads, help="Grid point of that system, as JSON"
    )
    args = parser.parse_args(argv)
    if (args.untuned_reference is None) != (args.untuned_grid is None):
        parser.error("--untuned-reference and --untuned-grid go together")
    untuned = (
        (args.untuned_reference, args.untuned_grid) if args.untuned_reference else None
    )
    result = build(
        args.final_dir, args.robustness_dir, args.selection, args.out_dir, untuned
    )
    print(
        f"{len(result['rows'])} systems, {len(result['comparisons'])} comparisons, {len(result['paired'])} paired -> {args.out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
