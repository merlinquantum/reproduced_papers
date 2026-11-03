"""Cross-experiment statistics from validated sweep summaries.

Reads each sweep's ``summary.json`` (never logs), pairs runs by permutation seed
(the same random feature distribution), and writes ``results/statistics.json``
and ``results/statistics.md``. Optionally adds the authors' released final-epoch
accuracies as a reference column.

Example
-------
    python utils/aggregate_results.py \
        --sweep E1=outdir/sweep_e1_... --sweep E2=outdir/sweep_e2_... \
        --authors-dir ../../data/vertical_qfl/vqfl_authors_assets/VQFL/exps
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from scipy import stats

PROJECT_DIR = Path(__file__).resolve().parents[1]
TEST_SET_SIZE = 319
AUTHORS_FILE = {
    "E1": "exp.json",
    "E2": "exp1.json",
    "E3": "exp8x1.json",
    "E4": "exp2x4.json",
}
FIRST_PERMUTATION_SEED = 42  # authors' Setting_1 uses permutation seed 42
COMPARISONS = [("E1", "E2"), ("E1", "E3"), ("E1", "E4"), ("E3", "E2"), ("E4", "E3")]


def load_sweep(path: Path) -> dict[int, dict]:
    status = json.loads((path / "sweep_status.json").read_text())
    if status["status"] != "COMPLETED":
        raise ValueError(f"{path} is {status['status']}, not a complete sweep")
    summary = json.loads((path / "summary.json").read_text())
    return {
        row["permutation_seed"]: row
        for row in summary["runs"]
        if row["status"] == "COMPLETED"
    }


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), statistics.stdev(values)


def authors_final_accuracy(authors_dir: Path, label: str) -> dict[int, float]:
    out = {}
    for index in range(5):
        data = json.loads(
            (authors_dir / f"Setting_{index + 1}" / AUTHORS_FILE[label]).read_text()
        )
        out[FIRST_PERMUTATION_SEED + index] = data["all_test_acc"][-1]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sweep", action="append", required=True, metavar="LABEL=DIR")
    parser.add_argument("--authors-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "results")
    parser.add_argument("--name", default="statistics", help="Output file stem")
    parser.add_argument(
        "--compare",
        action="append",
        metavar="A-B",
        help="Paired comparison of two labels (repeatable); default: the Phase 3 set",
    )
    args = parser.parse_args()

    sweeps = {}
    for item in args.sweep:
        label, directory = item.split("=", 1)
        sweeps[label] = load_sweep(Path(directory))
    seeds = sorted(set.intersection(*(set(rows) for rows in sweeps.values())))
    if len(seeds) != 5:
        raise ValueError(f"Expected 5 shared permutation seeds, found {seeds}")

    result: dict = {
        "permutation_seeds": seeds,
        "n_test_images": TEST_SET_SIZE,
        "experiments": {},
        "paired_differences": {},
    }
    for label, rows in sweeps.items():
        final = [rows[s]["final_test_accuracy"] for s in seeds]
        best = [rows[s]["best_epoch_test_accuracy_selected_on_test"] for s in seeds]
        mean, std = mean_std(final)
        entry = {
            "final_epoch_test_accuracy": {
                "mean": mean,
                "std_ddof1": std,
                "values": final,
            },
            "best_epoch_test_accuracy_selected_on_test": dict(
                zip(("mean", "std_ddof1"), mean_std(best))
            ),
            # Binomial standard error of one accuracy measured on a 319-image test set.
            "binomial_se_single_run": math.sqrt(mean * (1 - mean) / TEST_SET_SIZE),
        }
        if args.authors_dir and label in AUTHORS_FILE:
            theirs = authors_final_accuracy(args.authors_dir, label)
            their_values = [theirs[s] for s in seeds]
            their_mean, their_std = mean_std(their_values)
            entry["authors_final_epoch"] = {
                "mean": their_mean,
                "std_ddof1": their_std,
                "values": their_values,
            }
            entry["reproduced_minus_authors_mean"] = mean - their_mean
        result["experiments"][label] = entry

    comparisons = (
        [tuple(item.split("-", 1)) for item in args.compare]
        if args.compare
        else COMPARISONS
    )
    for a, b in comparisons:
        if a in sweeps and b in sweeps:
            diffs = [
                sweeps[a][s]["final_test_accuracy"]
                - sweeps[b][s]["final_test_accuracy"]
                for s in seeds
            ]
            mean, std = mean_std(diffs)
            t_test = stats.ttest_1samp(diffs, 0.0)
            result["paired_differences"][f"{a}-{b}"] = {
                "mean": mean,
                "std_ddof1": std,
                "paired_t_pvalue": float(t_test.pvalue),
                "n_positive": sum(d > 0 for d in diffs),
                "all_five_positive": all(d > 0 for d in diffs),
                "values": diffs,
            }

    args.out_dir.mkdir(exist_ok=True)
    (args.out_dir / f"{args.name}.json").write_text(json.dumps(result, indent=2) + "\n")

    lines = [
        "| Experiment | Final-epoch test acc (mean +- std, n=5) | Best-epoch (on test) | Authors' files | Reproduced - authors |",
        "|---|---|---|---|---|",
    ]
    for label, e in result["experiments"].items():
        f, b = (
            e["final_epoch_test_accuracy"],
            e["best_epoch_test_accuracy_selected_on_test"],
        )
        a = e.get("authors_final_epoch")
        lines.append(
            f"| {label} | {100 * f['mean']:.2f} +- {100 * f['std_ddof1']:.2f} | {100 * b['mean']:.2f} +- {100 * b['std_ddof1']:.2f} | "
            + (f"{100 * a['mean']:.2f} +- {100 * a['std_ddof1']:.2f}" if a else "n/a")
            + " | "
            + (f"{100 * e['reproduced_minus_authors_mean']:+.2f}" if a else "n/a")
            + " |"
        )
    lines += [
        "",
        "| Paired difference (same feature distribution) | mean (points) | std | paired t p-value | positive in n of 5 |",
        "|---|---|---|---|---|",
    ]
    for name, d in result["paired_differences"].items():
        lines.append(
            f"| {name} | {100 * d['mean']:+.2f} | {100 * d['std_ddof1']:.2f} | {d['paired_t_pvalue']:.4f} | {d['n_positive']} |"
        )
    (args.out_dir / f"{args.name}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
