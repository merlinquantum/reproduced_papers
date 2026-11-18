"""Phase 4: compare the MerLin photonic classifier adaptation with the gate-model VQC.

Usage::

    python3 papers/virtual_qnn/utils/compare_merlin.py \
        --sweeps-root papers/virtual_qnn/outdir/sweeps --out papers/virtual_qnn/results

Reads the sweep ``summary.json`` files and per-run ``metrics.json``; writes
``merlin_comparison.json``/``.md`` and the figure ``fig6d_merlin_adaptation.png``
(also saved into the photonic sweep directories). Applies the comparison rule
predeclared in LOG.md ("Phase 4 Adaptation Plan").
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR.parents[1]))
sys.path.insert(0, str(PROJECT_DIR.parent))

from virtual_qnn.utils import plot_style as style  # noqa: E402

BUDGETS = [1.0, 1.5, 2.0, 2.5, 3.0]
SWEEPS = {
    "photonic_noisy": ("e4_vqc_merlin_budget_enc01", "--vqc-merlin-budget"),
    "photonic_noise_free": ("e4_vqc_merlin_noiseless_enc01", "--vqc-merlin-budget"),
    "gate_model": ("e4_vqc_budget", "--vqc-budget"),
}


def _candidates(root: Path, sweep: str, axis: str) -> dict[float, dict]:
    summary = style.load_summary(root / sweep)
    return {c["grid_point"][axis]: c for c in summary["candidates"]}


def _by_repetition(candidate: dict, metric: str) -> dict[int, float]:
    return {run["repetition"]: run[metric] for run in candidate["runs"]}


def _paired(a: dict[int, float], b: dict[int, float]) -> dict:
    keys = sorted(set(a) & set(b))
    diff = np.array([a[k] - b[k] for k in keys])
    test = stats.ttest_rel([a[k] for k in keys], [b[k] for k in keys])
    return {
        "n_pairs": len(keys),
        "mean_difference": float(diff.mean()),
        "std_difference": float(diff.std(ddof=1)),
        "t_statistic": float(test.statistic),
        "p_value_two_sided": float(test.pvalue),
        "pairs_improved": int((diff > 0).sum()),
        "pairs_tied": int((diff == 0).sum()),
    }


def _hardware(root: Path, sweep: str) -> dict:
    status = json.loads((root / sweep / "sweep_status.json").read_text())
    run = status["runs"][0]
    hardware = json.loads((root / sweep / run["metrics_path"]).read_text())["hardware"]
    elapsed = [
        a["elapsed_s"]
        for r in status["runs"]
        for a in r["attempts"]
        if a["outcome"] == "COMPLETED"
    ]
    return {
        **hardware,
        "seeds": "seed = seed_data = r, r = 1..10",
        "wall_clock_per_run_s": {
            "median": float(np.median(elapsed)),
            "max": float(np.max(elapsed)),
        },
        "shot_count": "n/a (exact probabilities; virtual estimator needs ~γ² more shots than γ=1)",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sweeps-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.sweeps_root
    data = {name: _candidates(root, *spec) for name, spec in SWEEPS.items()}
    table = []
    for name, cands in data.items():
        for budget in BUDGETS:
            if budget not in cands:
                continue
            agg = cands[budget]["aggregates"]
            table.append(
                {
                    "model": name,
                    "budget": budget,
                    "n_parameters": agg["n_parameters"]["mean"],
                    "train_mean": agg["train_accuracy"]["mean"],
                    "train_std": agg["train_accuracy"]["std"],
                    "test_mean": agg["test_accuracy"]["mean"],
                    "test_std": agg["test_accuracy"]["std"],
                    "n": cands[budget]["successful_runs"],
                }
            )
    noisy = data["photonic_noisy"]
    test_means = [noisy[b]["aggregates"]["test_accuracy"]["mean"] for b in BUDGETS]
    pooled = float(
        np.sqrt(
            (
                noisy[1.0]["aggregates"]["test_accuracy"]["std"] ** 2
                + noisy[3.0]["aggregates"]["test_accuracy"]["std"] ** 2
            )
            / 2
        )
    )
    t = lambda name, b: _by_repetition(data[name][b], "test_accuracy")  # noqa: E731
    paired = {
        "photonic_noisy_g3_minus_g1": _paired(
            t("photonic_noisy", 3.0), t("photonic_noisy", 1.0)
        ),
        "photonic_noise_free_g3_minus_g1": _paired(
            t("photonic_noise_free", 3.0), t("photonic_noise_free", 1.0)
        ),
        "photonic_noise_free_g1_minus_noisy_g1": _paired(
            t("photonic_noise_free", 1.0), t("photonic_noisy", 1.0)
        ),
        "photonic_noisy_g1_minus_gate_g1": _paired(
            t("photonic_noisy", 1.0), t("gate_model", 1.0)
        ),
        "photonic_noisy_g3_minus_gate_g3": _paired(
            t("photonic_noisy", 3.0), t("gate_model", 3.0)
        ),
    }
    g31 = paired["photonic_noisy_g3_minus_g1"]
    preserved = (
        all(b >= a - 0.01 for a, b in zip(test_means, test_means[1:]))
        and (test_means[-1] - test_means[0] > pooled)
        and g31["p_value_two_sided"] < 0.05
    )
    report = {
        "scope": "MerLin photonic ADAPTATION of E4 (not the original method); exact simulation; 10 paired repetitions",
        "table": table,
        "paired_tests": paired,
        "behaviour_preserved": preserved,
        "rule": "test accuracy non-decreasing in γ (within 0.01) AND test(γ=3) − test(γ=1) > pooled std AND paired p < 0.05",
        "trend": {
            "test_means": test_means,
            "gain_g3_minus_g1": test_means[-1] - test_means[0],
            "pooled_std": pooled,
        },
        "hardware": {
            "photonic_noisy": _hardware(root, SWEEPS["photonic_noisy"][0]),
            "photonic_noise_free": _hardware(root, SWEEPS["photonic_noise_free"][0]),
        },
        "superseded": "e4_vqc_merlin_budget (encoding restricted outputs to x1−x2; LOG.md Amendment 2) is excluded",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "merlin_comparison.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Phase 4 — MerLin photonic adaptation of the virtual classifier (generated by utils/compare_merlin.py)",
        "",
        "Scope: **adaptation**, not the original method. Exact simulation, 10 paired repetitions (same data sets as the gate-model E4).",
        "",
        "| Model | γ | Params | Train | Test |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in table:
        lines.append(
            f"| {row['model']} | {row['budget']} | {row['n_parameters']:.0f} | {row['train_mean']:.3f} ± {row['train_std']:.3f} | {row['test_mean']:.3f} ± {row['test_std']:.3f} |"
        )
    lines += [
        "",
        "| Paired comparison (test accuracy) | mean diff | t | p | improved / tied |",
        "|---|---:|---:|---:|---|",
    ]
    for name, q in paired.items():
        lines.append(
            f"| {name} | {q['mean_difference']:+.3f} | {q['t_statistic']:.2f} | {q['p_value_two_sided']:.2g} | {q['pairs_improved']} / {q['pairs_tied']} of {q['n_pairs']} |"
        )
    lines += [
        "",
        f"Behaviour preserved under the predeclared rule: **{preserved}** (gain {test_means[-1] - test_means[0]:+.3f} vs pooled std {pooled:.3f}).",
        "",
        "Hardware-aware fields:",
        "",
    ]
    for label, hw in report["hardware"].items():
        lines.append(
            f"- **{label}**: " + "; ".join(f"{k} = {v}" for k, v in hw.items())
        )
    (args.out / "merlin_comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    style.apply_style()
    fig, ax = style.plt.subplots(figsize=(6.4, 4.0))
    series = [
        ("photonic_noisy", "Photonic VQNN, η = 0.9 (MerLin)", "o", "-"),
        ("photonic_noise_free", "Photonic VQNN, noise-free (MerLin)", "s", "--"),
        ("gate_model", "Gate-model VQNN, depolarizing noise (Phase 3)", "^", "-"),
    ]
    for index, (name, label, marker, linestyle) in enumerate(series):
        budgets = [b for b in BUDGETS if b in data[name]]
        mean = np.array(
            [data[name][b]["aggregates"]["test_accuracy"]["mean"] for b in budgets]
        )
        sd = np.array(
            [data[name][b]["aggregates"]["test_accuracy"]["std"] for b in budgets]
        )
        color = style.SERIES[index]
        ax.fill_between(
            budgets, mean - sd, mean + sd, color=color, alpha=0.15, linewidth=0
        )
        ax.plot(
            budgets, mean, color=color, marker=marker, linestyle=linestyle, label=label
        )
    ax.set_xticks(BUDGETS)
    ax.set_ylim(0.5, 1.0)
    ax.set_xlabel("Sampling-overhead budget γ (γ = 1: conventional classifier)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Virtual classifier: photonic adaptation vs gate model", loc="left")
    ax.legend(fontsize=8, loc="lower right")
    style.scope_note(
        ax,
        "Scope: MerLin adaptation (not the original), 2 photons / 4 modes, PNR detection, exact simulation; 10 paired reps; band = ±1 std.",
    )
    paths = style.save(
        fig,
        "fig6d_merlin_adaptation",
        [
            args.out,
            root / SWEEPS["photonic_noisy"][0],
            root / SWEEPS["photonic_noise_free"][0],
        ],
    )
    print(
        "\n".join(
            str(p)
            for p in [
                args.out / "merlin_comparison.json",
                args.out / "merlin_comparison.md",
                *paths,
            ]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
