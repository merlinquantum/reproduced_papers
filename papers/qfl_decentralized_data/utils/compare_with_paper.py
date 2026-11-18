"""Compare the finished sweeps with the paper's Tables II, III and IV.

``python utils/compare_with_paper.py <sweeps_dir> <results_dir>`` reads the structured
``metrics.json`` of every run of the reporting sweeps and writes
``comparison_e1.md``, ``comparison_e2.md`` and ``comparison.json`` into ``results_dir``.
A difference is called meaningful only if it exceeds the larger standard deviation of
the two quantities compared (the paper reports a standard deviation over 5 runs too).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

# Paper Table II: mean +- std accuracy (%), per task and setting ("N/A" = non-federated).
PAPER_ACCURACY = {
    "0v9": {"N/A": (94.03, 0.06), "2": (94.04, 0.03), "4": (94.01, 0.02), "8": (93.95, 0.09), "16": (93.87, 0.14)},
    "2v7": {"N/A": (96.61, 0.04), "2": (96.66, 0.02), "4": (96.71, 0.02), "8": (96.72, 0.07), "16": (96.56, 0.09)},
    "3v6": {"N/A": (98.00, 0.05), "2": (97.99, 0.08), "4": (97.97, 0.04), "8": (97.94, 0.08), "16": (97.97, 0.11)},
}  # fmt: skip
# Paper Table III (no standard deviations reported).
PAPER_AUC = {
    "0v9": {"N/A": 0.9820, "2": 0.9816, "4": 0.9820, "8": 0.9822, "16": 0.9813},
    "2v7": {"N/A": 0.9905, "2": 0.9905, "4": 0.9906, "8": 0.9906, "16": 0.9905},
    "3v6": {"N/A": 0.9972, "2": 0.9972, "4": 0.9975, "8": 0.9969, "16": 0.9969},
}  # fmt: skip
PAPER_F1 = {
    "0v9": {"N/A": 0.9399, "2": 0.9400, "4": 0.9403, "8": 0.9391, "16": 0.9386},
    "2v7": {"N/A": 0.9664, "2": 0.9669, "4": 0.9675, "8": 0.9674, "16": 0.9659},
    "3v6": {"N/A": 0.9806, "2": 0.9804, "4": 0.9803, "8": 0.9800, "16": 0.9803},
}  # fmt: skip
# Paper Table IV (4-class 2, 3, 6, 7; 4 clients).
PAPER_E2 = {
    "precision": [0.7343, 0.8298, 0.8564, 0.8484],
    "recall": [0.7045, 0.7673, 0.8841, 0.9202],
    "f1": [0.7191, 0.7973, 0.8701, 0.8829],
    "accuracy": 0.8180,
}
SETTINGS = ("centralized", "clients2", "clients4", "clients8", "clients16")
PAPER_KEY = {
    "centralized": "N/A",
    "clients2": "2",
    "clients4": "4",
    "clients8": "8",
    "clients16": "16",
}
CONVERGENCE_FRACTION = 0.99


def load_runs(sweep_dir: Path) -> dict[str, list[dict]]:
    """Group the completed runs' metrics by candidate name."""
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    grouped: dict[str, list[dict]] = {}
    for run in status["runs"]:
        if run["status"] == "COMPLETED":
            metrics = json.loads((sweep_dir / run["metrics_path"]).read_text())
            grouped.setdefault(run["candidate_name"], []).append(metrics)
    return grouped


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), (
        statistics.stdev(values) if len(values) > 1 else float("nan")
    )


def final(runs: list[dict], metric: str) -> list[float]:
    return [run["history"][-1][metric] for run in runs]


def rounds_to_fraction(run: dict) -> int:
    """First round whose test accuracy reaches 99 % of the final-round accuracy."""
    target = CONVERGENCE_FRACTION * run["history"][-1]["test_accuracy"]
    return next(
        entry["round"] for entry in run["history"] if entry["test_accuracy"] >= target
    )


def verdict(difference: float, spread_a: float, spread_b: float) -> str:
    """Name a difference only when it exceeds the larger standard deviation."""
    threshold = max(spread_a, spread_b)
    if abs(difference) <= threshold:
        return "within noise"
    return "above" if difference > 0 else "below"


def build_e1(sweeps: Path) -> tuple[str, dict]:
    hybrid = load_runs(sweeps / "e1_vqtn_hybrid")
    classical = load_runs(sweeps / "control_classical_e1")
    faithful = load_runs(sweeps / "sensitivity_faithful_0v9")
    iid = load_runs(sweeps / "control_iid_vqtn_0v9")
    rows = [
        "| Task | Setting | Paper acc % | Hybrid VQTN acc % | Hybrid - paper | vs paper | Classical acc % | Faithful VQTN acc % | Hybrid AUC (paper) | Hybrid F1 (paper) |",
        "|---|---|---:|---:|---:|---|---:|---:|---|---|",
    ]
    federated_vs_central = [
        "| Task | Clients | Hybrid fed - non-fed (points) | Verdict | Classical fed - non-fed (points) | Verdict | Paper fed - non-fed (points) |",
        "|---|---:|---:|---|---:|---|---:|",
    ]
    convergence = [
        "| Task | Setting | Hybrid rounds to 99 % of final accuracy (mean) | Classical rounds (mean) |",
        "|---|---|---:|---:|",
    ]
    data: dict = {}
    for task in PAPER_ACCURACY:
        centre_h = mean_std(
            [100 * v for v in final(hybrid[f"task{task}_centralized"], "test_accuracy")]
        )
        centre_c = mean_std(
            [
                100 * v
                for v in final(classical[f"task{task}_centralized"], "test_accuracy")
            ]
        )
        for setting in SETTINGS:
            name = f"task{task}_{setting}"
            h = [100 * v for v in final(hybrid[name], "test_accuracy")]
            c = [100 * v for v in final(classical[name], "test_accuracy")]
            hm, hs = mean_std(h)
            cm, _ = mean_std(c)
            pm, ps = PAPER_ACCURACY[task][PAPER_KEY[setting]]
            faithful_cell = "n/a"
            if name in faithful:
                fm, fs = mean_std(
                    [100 * v for v in final(faithful[name], "test_accuracy")]
                )
                faithful_cell = f"{fm:.2f} +- {fs:.2f}"
            auc_h = statistics.fmean(final(hybrid[name], "test_auc"))
            f1_h = statistics.fmean(final(hybrid[name], "test_f1"))
            paper_auc, paper_f1 = (
                PAPER_AUC[task][PAPER_KEY[setting]],
                PAPER_F1[task][PAPER_KEY[setting]],
            )
            rows.append(
                f"| {task} | {setting} | {pm:.2f} +- {ps:.2f} | {hm:.2f} +- {hs:.2f} | {hm - pm:+.2f} | "
                f"{verdict(hm - pm, hs, ps)} | {cm:.2f} | {faithful_cell} | {auc_h:.4f} ({paper_auc:.4f}) | {f1_h:.4f} ({paper_f1:.4f}) |"
            )
            data[name] = {
                "paper": [pm, ps],
                "hybrid": [hm, hs],
                "classical": [cm, mean_std(c)[1]],
            }
            hr = statistics.fmean(rounds_to_fraction(run) for run in hybrid[name])
            cr = statistics.fmean(rounds_to_fraction(run) for run in classical[name])
            convergence.append(f"| {task} | {setting} | {hr:.1f} | {cr:.1f} |")
            if setting != "centralized":
                diff_h = hm - centre_h[0]
                diff_c = cm - centre_c[0]
                paper_diff = pm - PAPER_ACCURACY[task]["N/A"][0]
                federated_vs_central.append(
                    f"| {task} | {setting[7:]} | {diff_h:+.2f} | {verdict(diff_h, hs, centre_h[1])} | "
                    f"{diff_c:+.2f} | {verdict(diff_c, mean_std(c)[1], centre_c[1])} | {paper_diff:+.2f} |"
                )
    iid_rows = [
        "| Setting | non-i.i.d. acc % | i.i.d. acc % | i.i.d. - non-i.i.d. | Verdict |",
        "|---|---:|---:|---:|---|",
    ]
    for k in (2, 16):
        non = mean_std(
            [100 * v for v in final(hybrid[f"task0v9_clients{k}"], "test_accuracy")]
        )
        eq = mean_std(
            [100 * v for v in final(iid[f"task0v9_clients{k}_iid"], "test_accuracy")]
        )
        iid_rows.append(
            f"| 0v9, {k} clients | {non[0]:.2f} +- {non[1]:.2f} | {eq[0]:.2f} +- {eq[1]:.2f} | {eq[0] - non[0]:+.2f} | {verdict(eq[0] - non[0], eq[1], non[1])} |"
        )
    text = (
        "# E1 comparison (5 seeds, full MNIST subsets, final-round test accuracy)\n\n"
        "Scope label: multi-seed, full data, paper rounds and batch size; **circuit, partition, learning rate and read-out are inferred, and the primary VQTN uses a hybrid eight-<Z> linear head (a documented deviation)**; not paper-accurate. 'vs paper' names a difference only when it exceeds the larger standard deviation.\n\n"
        + "\n".join(rows)
        + "\n\n## Federated versus non-federated (claim C1)\n\n"
        + "\n".join(federated_vs_central)
        + "\n\n## Rounds to reach 99 % of the final test accuracy (claim C2, communication efficiency)\n\n"
        + "\n".join(convergence)
        + "\n\n## i.i.d. control\n\n"
        + "\n".join(iid_rows)
        + "\n"
    )
    return text, data


def build_e2(sweeps: Path) -> tuple[str, dict]:
    hybrid = load_runs(sweeps / "e2_vqtn_hybrid")
    classical = load_runs(sweeps / "control_classical_e2")
    lines = [
        "| Setting | Paper acc % | Hybrid VQTN acc % | Hybrid - paper | vs paper | Classical acc % |",
        "|---|---:|---:|---:|---|---:|",
    ]
    data = {}
    for setting in ("fourclass_centralized", "fourclass_clients4"):
        h = [100 * v for v in final(hybrid[setting], "test_accuracy")]
        c = [100 * v for v in final(classical[setting], "test_accuracy")]
        hm, hs = mean_std(h)
        cm, cs = mean_std(c)
        paper = 100 * PAPER_E2["accuracy"]
        paper_cell = f"{paper:.2f}" if setting.endswith("clients4") else "not reported"
        diff = f"{hm - paper:+.2f}" if setting.endswith("clients4") else "n/a"
        judged = verdict(hm - paper, hs, 0.0) if setting.endswith("clients4") else "n/a"
        lines.append(
            f"| {setting} | {paper_cell} | {hm:.2f} +- {hs:.2f} | {diff} | {judged} | {cm:.2f} +- {cs:.2f} |"
        )
        data[setting] = {"hybrid": [hm, hs], "classical": [cm, cs]}
    per_class = [
        "| Class | Paper precision | Hybrid precision | Paper recall | Hybrid recall | Paper F1 | Hybrid F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    reports = [run["final_report"] for run in hybrid["fourclass_clients4"]]
    for index, label in enumerate(reports[0]["classes"]):
        cells = []
        for key in ("precision", "recall", "f1"):
            mean = statistics.fmean(
                report[f"per_class_{key}"][index] for report in reports
            )
            cells += [f"{PAPER_E2[key][index]:.4f}", f"{mean:.4f}"]
        per_class.append(f"| {label} | " + " | ".join(cells) + " |")
    text = (
        "# E2 comparison (4-class 2, 3, 6, 7; 5 seeds; final-round test accuracy)\n\n"
        "Scope label: multi-seed, full data; inferred circuit with a hybrid eight-<Z> head (deviation); the paper gives no non-federated reference, so the non-federated rows are ours. The paper reports no standard deviation for its single 81.80 % value, so 'vs paper' uses only our standard deviation.\n\n"
        + "\n".join(lines)
        + "\n\n## Per-class metrics, 4 clients (mean over 5 seeds)\n\n"
        + "\n".join(per_class)
        + "\n"
    )
    return text, data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweeps_dir", type=Path)
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args(argv)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    e1_text, e1_data = build_e1(args.sweeps_dir)
    e2_text, e2_data = build_e2(args.sweeps_dir)
    (args.results_dir / "comparison_e1.md").write_text(e1_text, encoding="utf-8")
    (args.results_dir / "comparison_e2.md").write_text(e2_text, encoding="utf-8")
    (args.results_dir / "comparison.json").write_text(
        json.dumps({"e1": e1_data, "e2": e2_data}, indent=2), encoding="utf-8"
    )
    print(e1_text)
    print(e2_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
