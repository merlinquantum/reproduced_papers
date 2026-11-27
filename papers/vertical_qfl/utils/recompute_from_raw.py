"""Independent, read-only recomputation of every reported number from raw run metrics.

Reads only ``sweep_status.json`` and each run's ``metrics.json`` / ``config_snapshot.json``
under ``outdir/`` (never summaries or logs), recomputes coverage, aggregates, paired
differences and selections with the predeclared rules, and compares them with the
generated result files and with the numbers quoted in README.md, CONFLUENCE.md, LOG.md
and notebook.ipynb. Writes ``results/recomputation_check.md``; exits non-zero on any
mismatch.

Example
-------
    python utils/recompute_from_raw.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTDIR = PROJECT_DIR / "outdir"
RESULTS = PROJECT_DIR / "results"
PERMUTATIONS = [42, 43, 44, 45, 46]
TIE = 0.005
LAYOUTS = {
    "E1": ([0, 1, 2, 3, 4, 5, 6, 7], "server_wide"),
    "E2": ([0], "server_narrow"),
    "E3": ([0] * 8, "server_wide"),
    "E4": ([0, 1, 2, 3, 0, 1, 2, 3], "server_wide"),
}
EXPERIMENT_SWEEPS = {
    "E1": "sweep_e1_8clients_2026*_e1",
    "E2": "sweep_e2_1client_2026*_e2",
    "E3": "sweep_e3_8x1client_2026*_e3",
    "E4": "sweep_e4_2x4clients_2026*_e4",
    "S5": "sweep_e1_8clients_logits_control_*",
    "E1p": "sweep_e1p_8clients_merlin_2026*_p1",
    "E2p": "sweep_e2p_1client_merlin_2026*_p2",
    "E3p": "sweep_e3p_8x1client_merlin_2026*_p3",
    "E4p": "sweep_e4p_2x4clients_merlin_2026*_p4",
    "E1h": "sweep_e1p_8clients_merlin_head_*",
    "E2h": "sweep_e2p_1client_merlin_head_*",
    "E3h": "sweep_e3p_8x1client_merlin_head_*",
    "E4h": "sweep_e4p_2x4clients_merlin_head_*",
    "E1f": "sweep_e1p_8clients_merlin_photonic_*",
    "E2f": "sweep_e2p_1client_merlin_photonic_*",
    "E3f": "sweep_e3p_8x1client_merlin_photonic_*",
    "E4f": "sweep_e4p_2x4clients_merlin_photonic_*",
}
SELECTION_SWEEPS = {
    "T1": "sweep_e1_8clients_merlin_tuning_*_t1",
    "T2": "sweep_e1_8clients_merlin_tuning_*_t2",
    "D1": "sweep_e1_8clients_merlin_diagnostic_2026*_d1",
    "D2": "sweep_e1_8clients_merlin_diagnostic_d2_*",
    "D3": "sweep_e1_8clients_merlin_diagnostic_d3_*",
    "C-T": "sweep_classical_tuning_*_ct",
}
CLASSICAL_SWEEPS = {
    "classical_gate": "_cg",
    "classical_photonic_h": "_ch",
    "classical_photonic_f": "_cf",
}

problems: list[str] = []
report: list[str] = []


def one(pattern: str) -> Path:
    matches = sorted(OUTDIR.glob(pattern))
    if len(matches) != 1:
        raise SystemExit(f"expected one sweep for {pattern}, found {matches}")
    return matches[0]


def raw_runs(sweep: Path) -> list[dict]:
    """Every run of a sweep, read from its own metrics and config files."""
    status = json.loads((sweep / "sweep_status.json").read_text())
    runs = []
    for run in status["runs"]:
        if run["status"] != "COMPLETED":
            problems.append(f"{sweep.name}: run {run['run_id']} is {run['status']}")
            continue
        metrics = json.loads((sweep / run["metrics_path"]).read_text())
        config = json.loads((sweep / run["config_path"]).read_text())
        runs.append(
            {
                "metrics": metrics,
                "config": config,
                "seed": run["seed"],
                "permutation": config["dataset"]["permutation_seed"],
            }
        )
    if len(runs) != status["expected_runs"]:
        problems.append(
            f"{sweep.name}: {len(runs)} completed of {status['expected_runs']} expected"
        )
    return runs


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), statistics.stdev(values)


def check(
    label: str, recomputed: float, stored: float, tolerance: float = 1e-9
) -> None:
    if abs(recomputed - stored) > tolerance:
        problems.append(f"{label}: recomputed {recomputed!r} != stored {stored!r}")


def final_by_permutation(runs: list[dict], seed: int | None = 42) -> dict[int, float]:
    out: dict[int, list[float]] = {}
    for run in runs:
        if seed is None or run["seed"] == seed:
            out.setdefault(run["permutation"], []).append(
                run["metrics"]["final_test_accuracy"]
            )
    return {p: statistics.fmean(v) for p, v in out.items()}


def main() -> None:
    total_runs = 0
    # 1. Experiment sweeps (gate model, loss control, photonic v1 and v2).
    experiments = {}
    for label, pattern in EXPERIMENT_SWEEPS.items():
        runs = raw_runs(one(pattern))
        total_runs += len(runs)
        values = final_by_permutation(runs)
        if sorted(values) != PERMUTATIONS:
            problems.append(f"{label}: permutation seeds {sorted(values)}")
        experiments[label] = values
    statistics_file = json.loads((RESULTS / "statistics.json").read_text())
    for label in ["E1", "E2", "E3", "E4"]:
        stored = statistics_file["experiments"][label]["final_epoch_test_accuracy"]
        mean, std = mean_std([experiments[label][p] for p in PERMUTATIONS])
        check(f"statistics.json {label} mean", mean, stored["mean"])
        check(f"statistics.json {label} std", std, stored["std_ddof1"])
    photonic_file = json.loads((RESULTS / "photonic_v2_statistics.json").read_text())
    for label, stored in photonic_file["experiments"].items():
        mean, std = mean_std([experiments[label][p] for p in PERMUTATIONS])
        check(
            f"photonic_v2_statistics.json {label} mean",
            mean,
            stored["final_epoch_test_accuracy"]["mean"],
        )
        check(
            f"photonic_v2_statistics.json {label} std",
            std,
            stored["final_epoch_test_accuracy"]["std_ddof1"],
        )
    for name, stored in photonic_file["paired_differences"].items():
        a, b = name.split("-")
        diffs = [experiments[a][p] - experiments[b][p] for p in PERMUTATIONS]
        check(
            f"photonic_v2_statistics.json pair {name}",
            statistics.fmean(diffs),
            stored["mean"],
        )
        if sum(d > 0 for d in diffs) != stored["n_positive"]:
            problems.append(f"pair {name}: positive count differs")
    s5 = [experiments["S5"][p] - experiments["E1"][p] for p in PERMUTATIONS]
    report.append(
        f"- Loss control S5 - E1: {100 * statistics.fmean(s5):+.2f} points, positive in {sum(d > 0 for d in s5)}/5"
    )

    # 2. Classical baselines (all training seeds) and the Phase 5 comparison file.
    comparison = json.loads((RESULTS / "phase5_comparison.json").read_text())
    rows = {(r["system"], r["layout"]): r for r in comparison["systems"]}
    classical = {}
    for system, tag in CLASSICAL_SWEEPS.items():
        runs = raw_runs(one(f"sweep_classical_*{tag}"))
        total_runs += len(runs)
        for layout, (slices, server) in LAYOUTS.items():
            subset = [
                r
                for r in runs
                if r["config"]["model"]["client_slices"] == slices
                and r["config"]["model"]["server_encoding"] == server
            ]
            if len(subset) != 15:
                problems.append(f"{system} {layout}: {len(subset)} runs, expected 15")
            classical[(system, layout)] = subset
            for seed, key in ((42, "seed42"), (None, "all_seeds")):
                values = final_by_permutation(subset, seed)
                mean, std = mean_std([values[p] for p in PERMUTATIONS])
                check(
                    f"phase5 {system} {layout} mean_{key}",
                    mean,
                    rows[(system, layout)][f"mean_{key}"],
                )
                check(
                    f"phase5 {system} {layout} std_{key}",
                    std,
                    rows[(system, layout)][f"std_{key}"],
                )
    central_runs = raw_runs(one("sweep_classical_*_cz"))
    total_runs += len(central_runs)
    for size in ["gate", "photonic_h", "photonic_f"]:
        subset = [r for r in central_runs if r["config"]["model"]["size"] == size]
        mean, _ = mean_std([final_by_permutation(subset)[p] for p in PERMUTATIONS])
        check(
            f"phase5 central {size}",
            mean,
            rows[(f"classical_central:{size}", "central")]["mean_seed42"],
        )
    quantum = {"gate": "", "photonic_h": "h", "photonic_f": "f"}
    for pair in comparison["pairs"]:
        q_system, c_system = [x.strip() for x in pair["pair"].split(" - ")]
        layout = pair["layout"]
        q = experiments[layout + quantum[q_system]]
        c = final_by_permutation(classical[(c_system, layout)])
        diffs = [q[p] - c[p] for p in PERMUTATIONS]
        check(
            f"phase5 pair {pair['pair']} {layout}",
            statistics.fmean(diffs),
            pair["mean_difference"],
        )
        mean_q, std_q = mean_std([q[p] for p in PERMUTATIONS])
        _, std_c = mean_std([c[p] for p in PERMUTATIONS])
        no_clear = abs(100 * statistics.fmean(diffs)) < 1.5 or abs(
            statistics.fmean(diffs)
        ) <= max(std_q, std_c)
        verdict = (
            "no clear difference"
            if no_clear
            else (
                "quantum/photonic higher"
                if statistics.fmean(diffs) > 0
                else "classical higher"
            )
        )
        if verdict != pair["verdict"]:
            problems.append(
                f"pair {pair['pair']} {layout}: verdict {verdict} != stored {pair['verdict']}"
            )

    # 3. Selection and diagnostic sweeps: per-candidate means and the declared rules.
    selections = {}
    for label, pattern in SELECTION_SWEEPS.items():
        runs = raw_runs(one(pattern))
        total_runs += len(runs)
        groups: dict[str, list[float]] = {}
        for run in runs:
            key = json.dumps(
                run["config"].get("candidate_overrides", {}), sort_keys=True
            )
            groups.setdefault(key, []).append(
                run["metrics"]["final_validation_accuracy"]
            )
        selections[label] = {k: statistics.fmean(v) for k, v in groups.items()}
    union = {**selections["T2"], **selections["T1"]}
    best = max(union.values())
    tied = sorted(
        (json.loads(k) for k, v in union.items() if v >= best - TIE),
        key=lambda o: (o["lr"], o["client_encoding_scale"]),
    )
    stored = json.loads((RESULTS / "photonic_selection.json").read_text())["selected"]
    if tied[0] != stored:
        problems.append(f"T1+T2 selection {tied[0]} != stored {stored}")
    report.append(
        f"- T1+T2 union selection: {tied[0]} (validation mean {100 * best:.2f}%)"
    )
    for label, keys in (
        ("D1", ("server_head", "server_modes", "epochs")),
        ("D2", ("server_head", "n_photons", "reuploads")),
        ("D3", ("server_head", "n_photons")),
    ):
        best = max(selections[label].values())
        tied = sorted(
            (json.loads(k) for k, v in selections[label].items() if v >= best - TIE),
            key=lambda o: tuple(o[k] for k in keys),
        )
        status = json.loads(
            (one(SELECTION_SWEEPS[label]) / "sweep_status.json").read_text()
        )
        if tied[0] != status["selected_candidates"][0]:
            problems.append(
                f"{label} selection {tied[0]} != stored {status['selected_candidates'][0]}"
            )
        report.append(
            f"- {label} selection: {tied[0]} (validation mean {100 * best:.2f}%)"
        )
    classical_selection = json.loads(
        (RESULTS / "classical_selection.json").read_text()
    )["selections_by_group"]
    by_group: dict[str, dict] = {}
    for key, value in selections["C-T"].items():
        overrides = json.loads(key)
        group = json.dumps(
            {"architecture": overrides["architecture"], "size": overrides["size"]},
            sort_keys=True,
        )
        by_group.setdefault(group, {})[overrides["lr"]] = value
    for group, lrs in by_group.items():
        best = max(lrs.values())
        chosen = min(lr for lr, v in lrs.items() if v >= best - TIE)
        if chosen != classical_selection[group]["selected"]["lr"]:
            problems.append(
                f"C-T {group}: lr {chosen} != stored {classical_selection[group]['selected']['lr']}"
            )
        report.append(f"- C-T {group}: lr {chosen}")

    # 4. Headline numbers quoted in the documents.
    headline = {
        "E1 gate": 100 * statistics.fmean(experiments["E1"].values()),
        "E2 gate": 100 * statistics.fmean(experiments["E2"].values()),
        "E3 gate": 100 * statistics.fmean(experiments["E3"].values()),
        "E4 gate": 100 * statistics.fmean(experiments["E4"].values()),
        "E1 photonic H": 100 * statistics.fmean(experiments["E1h"].values()),
        "E1 photonic F": 100 * statistics.fmean(experiments["E1f"].values()),
        "E1 classical gate-matched": 100
        * rows[("classical_gate", "E1")]["mean_seed42"],
    }
    documents = {
        name: (PROJECT_DIR / name).read_text()
        for name in ("README.md", "CONFLUENCE.md", "LOG.md")
    }
    notebook = json.loads((PROJECT_DIR / "notebook.ipynb").read_text())
    documents["notebook.ipynb"] = json.dumps(notebook)
    for name, value in headline.items():
        text = f"{value:.2f}"
        missing = [doc for doc, content in documents.items() if text not in content]
        if missing:
            problems.append(f"{name} {text} not found in {missing}")
        report.append(
            f"- {name}: {text}% (found in {', '.join(d for d in documents if d not in missing)})"
        )

    lines = [
        "# Independent recomputation from raw run metrics",
        "",
        "Generated by `python utils/recompute_from_raw.py`, which reads only `sweep_status.json`, `metrics.json` and `config_snapshot.json` of each run.",
        "",
        f"Runs read: {total_runs}. Mismatches: {len(problems)}.",
        "",
    ] + report
    if problems:
        lines += ["", "## Mismatches"] + [f"- {p}" for p in problems]
    (RESULTS / "recomputation_check.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
