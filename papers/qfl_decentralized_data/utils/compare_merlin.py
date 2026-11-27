"""Compare the MerLin photonic adaptation with the Phase 3 results (Phase 4).

``python utils/compare_merlin.py <sweeps_dir> <results_dir>`` reads the runs'
``metrics.json`` files and applies the two criteria predeclared in LOG.md ("Phase 4
Adaptation Plan"): (a) federated minus non-federated within noise in at least 10 of the
12 E1 comparisons; (b) mean test accuracy within 2.0 points of the hybrid VQTN in every
E1 setting. It writes ``comparison_merlin.md``, ``comparison_merlin.json`` and the figure
``merlin_accuracy_vs_clients.png`` into ``results_dir``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_with_paper import (  # noqa: E402
    PAPER_ACCURACY,
    PAPER_E2,
    PAPER_KEY,
    SETTINGS,
    final,
    load_runs,
    mean_std,
    rounds_to_fraction,
    verdict,
)

PRESERVE_TOLERANCE_POINTS = 2.0
MIN_WITHIN_NOISE = 10
SCOPE_LABEL = (
    "5 seeds, full data, MerLin simulation (adaptation, not the paper's model)"
)


def accuracy(runs: list[dict]) -> tuple[float, float]:
    return mean_std([100 * value for value in final(runs, "test_accuracy")])


def build(sweeps: Path) -> tuple[str, dict]:
    photonic = load_runs(sweeps / "e1_merlin")
    hybrid = load_runs(sweeps / "e1_vqtn_hybrid")
    classical = load_runs(sweeps / "control_classical_e1")
    lex = load_runs(sweeps / "sensitivity_merlin_lexgroup_0v9")
    faithful = load_runs(sweeps / "sensitivity_faithful_0v9")
    table = [
        "| Task | Setting | Paper | Photonic (MerLin) | Hybrid VQTN | Photonic - hybrid | Within 2.0? | Classical MLP | Photonic rounds to 99 % |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    fed = [
        "| Task | Clients | Photonic fed - non-fed | Verdict | Hybrid fed - non-fed | Verdict |",
        "|---|---:|---:|---|---:|---|",
    ]
    within_noise, within_tolerance, cells = 0, 0, 0
    data: dict = {}
    for task in PAPER_ACCURACY:
        centre_p = accuracy(photonic[f"task{task}_centralized"])
        centre_h = accuracy(hybrid[f"task{task}_centralized"])
        for setting in SETTINGS:
            name = f"task{task}_{setting}"
            pm, ps = accuracy(photonic[name])
            hm, hs = accuracy(hybrid[name])
            cm, _ = accuracy(classical[name])
            paper = PAPER_ACCURACY[task][PAPER_KEY[setting]][0]
            ok = abs(pm - hm) <= PRESERVE_TOLERANCE_POINTS
            within_tolerance += ok
            cells += 1
            rounds = statistics.fmean(rounds_to_fraction(run) for run in photonic[name])
            table.append(
                f"| {task} | {setting} | {paper:.2f} | {pm:.2f} +- {ps:.2f} | {hm:.2f} +- {hs:.2f} | "
                f"{pm - hm:+.2f} | {'yes' if ok else 'no'} | {cm:.2f} | {rounds:.1f} |"
            )
            data[name] = {"photonic": [pm, ps], "hybrid": [hm, hs], "classical": cm}
            if setting != "centralized":
                diff_p, diff_h = pm - centre_p[0], hm - centre_h[0]
                verdict_p = verdict(diff_p, ps, centre_p[1])
                within_noise += verdict_p == "within noise"
                fed.append(
                    f"| {task} | {setting[7:]} | {diff_p:+.2f} | {verdict_p} | {diff_h:+.2f} | "
                    f"{verdict(diff_h, hs, centre_h[1])} |"
                )
    criterion_a = within_noise >= MIN_WITHIN_NOISE
    criterion_b = within_tolerance == cells
    sensitivity = [
        "| Setting | Photonic, linear head (130 params) | Photonic, fixed LexGrouping (112 params) | Faithful gate VQTN (42 params) | Paper |",
        "|---|---:|---:|---:|---:|",
    ]
    for setting in SETTINGS:
        name = f"task0v9_{setting}"
        a, b, c = (
            accuracy(photonic[name]),
            accuracy(lex[name]),
            accuracy(faithful[name]),
        )
        sensitivity.append(
            f"| 0v9 {setting} | {a[0]:.2f} +- {a[1]:.2f} | {b[0]:.2f} +- {b[1]:.2f} | "
            f"{c[0]:.2f} +- {c[1]:.2f} | {PAPER_ACCURACY['0v9'][PAPER_KEY[setting]][0]:.2f} |"
        )
    e2_photonic = load_runs(sweeps / "e2_merlin")
    e2_hybrid = load_runs(sweeps / "e2_vqtn_hybrid")
    e2_classical = load_runs(sweeps / "control_classical_e2")
    e2 = [
        "| Setting | Paper | Photonic | Hybrid VQTN | Classical MLP |",
        "|---|---:|---:|---:|---:|",
    ]
    for setting in ("fourclass_centralized", "fourclass_clients4"):
        paper = (
            f"{100 * PAPER_E2['accuracy']:.2f}"
            if setting.endswith("4")
            else "not reported"
        )
        a, b, c = (
            accuracy(e2_photonic[setting]),
            accuracy(e2_hybrid[setting]),
            accuracy(e2_classical[setting]),
        )
        e2.append(
            f"| {setting} | {paper} | {a[0]:.2f} +- {a[1]:.2f} | {b[0]:.2f} +- {b[1]:.2f} | {c[0]:.2f} +- {c[1]:.2f} |"
        )
    text = (
        f"# MerLin photonic adaptation versus Phase 3 ({SCOPE_LABEL})\n\n"
        "Final-round test accuracy %, mean +- std over seeds 1 to 5. Photonic model: superposition layer, encode even pixels, trainable mesh, encode odd pixels, trainable mesh; 8 modes, 2 photons, UNBUNCHED, click probabilities into a linear head (130 parameters).\n\n"
        "## E1 per setting\n\n"
        + "\n".join(table)
        + "\n\n## Federated versus non-federated\n\n"
        + "\n".join(fed)
        + "\n\n## Predeclared criteria\n\n"
        + f"- (a) federated - non-federated within noise in at least {MIN_WITHIN_NOISE} of 12 E1 comparisons: **{within_noise} of 12, {'MET' if criterion_a else 'NOT MET'}**\n"
        + f"- (b) within {PRESERVE_TOLERANCE_POINTS} points of the hybrid VQTN in every E1 setting: **{within_tolerance} of {cells}, {'MET' if criterion_b else 'NOT MET'}**\n"
        + "\n## Read-out sensitivity (task 0 vs 9)\n\n"
        + "\n".join(sensitivity)
        + "\n\n## E2 (4-class)\n\n"
        + "\n".join(e2)
        + "\n"
    )
    summary = {"criterion_a_within_noise": within_noise, "criterion_a_met": criterion_a,
               "criterion_b_within_tolerance": within_tolerance, "criterion_b_cells": cells,
               "criterion_b_met": criterion_b, "e1": data}  # fmt: skip
    return text, summary


def plot(sweeps: Path, outputs: list[Path]) -> None:
    series = {
        "photonic MerLin (ours)": (load_runs(sweeps / "e1_merlin"), "tab:purple", "D"),
        "hybrid VQTN (ours)": (load_runs(sweeps / "e1_vqtn_hybrid"), "tab:blue", "o"),
        "classical MLP (ours)": (
            load_runs(sweeps / "control_classical_e1"),
            "tab:green",
            "s",
        ),
    }
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    positions = range(len(SETTINGS))
    for axis, task in zip(axes, PAPER_ACCURACY):
        for label, (runs, colour, marker) in series.items():
            stats = [accuracy(runs[f"task{task}_{setting}"]) for setting in SETTINGS]
            axis.errorbar(
                positions,
                [m for m, _ in stats],
                [s for _, s in stats],
                label=label,
                color=colour,
                marker=marker,
                capsize=3,
            )
        paper = PAPER_ACCURACY[task]
        axis.errorbar(
            positions,
            [paper[PAPER_KEY[s]][0] for s in SETTINGS],
            [paper[PAPER_KEY[s]][1] for s in SETTINGS],
            label="paper (Table II)",
            color="black",
            linestyle="--",
            marker="x",
            capsize=3,
        )
        axis.set_xticks(list(positions), ["non-fed", "2", "4", "8", "16"])
        axis.set_xlabel("clients")
        axis.set_title(f"MNIST {task[0]} vs {task[2]}")
        axis.grid(alpha=0.3)
    axes[0].set_ylabel("test accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
    figure.suptitle(f"E1 with the MerLin adaptation ({SCOPE_LABEL})", fontsize=9)
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    for path in outputs:
        path.mkdir(parents=True, exist_ok=True)
        figure.savefig(path / "merlin_accuracy_vs_clients.png", dpi=150)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweeps_dir", type=Path)
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args(argv)
    text, summary = build(args.sweeps_dir)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "comparison_merlin.md").write_text(text, encoding="utf-8")
    (args.results_dir / "comparison_merlin.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    plot(args.sweeps_dir, [args.results_dir, args.sweeps_dir / "e1_merlin"])
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
