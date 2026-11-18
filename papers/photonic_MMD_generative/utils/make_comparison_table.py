"""Phase 5: fair-comparison table for C4 (Fig. 3a) and C8 part 2 (Fig. 3e).

Usage (from the paper folder)::

    python utils/make_comparison_table.py \
        --fig3a outdir/sweep_fig3a_bs_<stamp> \
        --b6 outdir/sweep_fig3a_bs_rbm_tuned_<stamp> \
        --fig3a-merlin outdir/sweep_fig3a_bs_merlin_<stamp> \
        --fig3e outdir/sweep_fig3e_movies_<stamp>

Reads every run's ``run_status.json`` and ``metrics.json`` (full precision,
never logs or summaries), rejects missing, duplicate, non-COMPLETED or
non-finite entries, checks commensurability (dataset id, test split hash,
sample count), and applies the rule frozen in LOG.md "Phase 5 Comparison
Plan": mean and std (ddof=1) over seeds, SE = sqrt(s1²/n1 + s2²/n2), "not
separated" when |Δ| <= 2 SE. Writes ``results/phase5_comparison.json`` and
``results/phase5_comparison.md``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

PAPER_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PAPER_DIR / "results"
SEEDS = [0, 1, 2, 3, 4]
SIGMAS = ["2.0", "3.0", "4.5"]
METRICS = ("gaussian_mmd_binarized", "walsh_mmd", "gaussian_mmd_raw")
BINARY_MODELS = {"paper RBM", "B6 tuned RBM", "B4 marginal", "uniform random"}
M = 256
# Real trainable parameters (LOG.md "Phase 5 Comparison Plan", resource matching).
PARAMETERS = {
    "QCBM (Phase 3, JAX + Perceval)": 2 * M * M,
    "QCBM (Phase 4 MerLin adaptation)": 2 * M * M,
    "B5 distinguishable twin": 2 * M * M,
    "B5 twin (MerLin backend)": 2 * M * M,
    "paper RBM": M * 64 + M + 64,
    "B4 marginal": M,
    "uniform random": 0,
    "test-to-test (noise floor)": 0,
}


def load_runs(sweep_dir: Path) -> list[dict]:
    """Completed runs of a sweep with their status and metrics; validates coverage."""
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    runs, seen = [], set()
    for entry in status["runs"]:
        if entry["status"] != "COMPLETED":
            raise ValueError(
                f"{sweep_dir.name}: run {entry['run_id']} is {entry['status']}"
            )
        run_status = json.loads((sweep_dir / entry["run_status"]).read_text())
        if run_status["status"] != "COMPLETED":
            raise ValueError(f"{entry['run_id']}: run_status {run_status['status']}")
        key = (entry["candidate"]["label"], entry["seed"])
        if key in seen:
            raise ValueError(f"{sweep_dir.name}: duplicate run {key}")
        seen.add(key)
        runs.append(
            {
                "label": entry["candidate"]["label"],
                "seed": entry["seed"],
                "run_id": entry["run_id"],
                "run_dir": sweep_dir / entry["run_dir"],
                "status": run_status,
                "metrics": json.loads((sweep_dir / entry["metrics_path"]).read_text()),
            }
        )
    if len(runs) != status["expected_runs"]:
        raise ValueError(
            f"{sweep_dir.name}: {len(runs)}/{status['expected_runs']} runs"
        )
    return runs


def stats(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"non-finite values {values}")
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)),
        "n": int(arr.size),
        "values": [float(v) for v in arr],
    }


def compare(method: dict, baseline: dict) -> dict:
    """Frozen decision rule: Δ = method − baseline, separated when |Δ| > 2 SE."""
    delta = method["mean"] - baseline["mean"]
    se = math.sqrt(
        method["std"] ** 2 / method["n"] + baseline["std"] ** 2 / baseline["n"]
    )
    if abs(delta) <= 2 * se:
        verdict = "not separated"
    else:
        verdict = "method better" if delta < 0 else "baseline better"
    return {
        "delta": delta,
        "two_se": 2 * se,
        "ratio_baseline_over_method": baseline["mean"] / method["mean"],
        "verdict": verdict,
    }


def identity(run: dict) -> tuple:
    subset = run["status"]["dataset"]["subset"]
    return (run["metrics"]["dataset"]["id"], subset["test_sha256"])


def collect_fig3a(fig3a, b6, merlin) -> tuple[dict, dict]:
    """System -> sigma -> metric -> per-seed values, and data identities seen."""
    table = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    identities = defaultdict(set)
    inside = {
        "rbm": "paper RBM",
        "marginal": "B4 marginal",
        "random": "uniform random",
        "test_to_test": "test-to-test (noise floor)",
    }
    for run in fig3a:
        model, sigma = run["label"].split("_sigma")
        system = (
            "QCBM (Phase 3, JAX + Perceval)"
            if model == "qcbm"
            else "B5 distinguishable twin"
        )
        identities[system].add(identity(run))
        sample = run["metrics"]["sample_metrics"]
        for name, entry in sample.items():
            target = system if name == "model" else inside[name]
            identities[target].add(identity(run))
            for metric in METRICS:
                if metric in entry:
                    table[target][sigma][metric][run["seed"]] = entry[metric]
            if "num_samples" in entry and entry["num_samples"] != 2000:
                raise ValueError(
                    f"{run['run_id']} {name}: {entry['num_samples']} samples"
                )
        table[system][sigma]["train_seconds"][run["seed"]] = run["metrics"]["training"][
            "seconds"
        ]
        if "rbm" in sample:
            # The paper RBM's fit time is only in run.log (timing, not a ranked metric).
            match = re.search(
                r"BASELINE_TRAINED \| model=rbm \| seconds=([0-9.]+)",
                (run["run_dir"] / "run.log").read_text(),
            )
            table["paper RBM"][sigma]["train_seconds"][run["seed"]] = float(
                match.group(1)
            )
    for run in b6:
        identities["B6 tuned RBM"].add(identity(run))
        tuning = run["metrics"]["tuning"]
        for sigma, entries in run["metrics"]["sample_metrics_by_sigma"].items():
            entry = entries["rbm_tuned"]
            if entry["num_samples"] != 2000:
                raise ValueError(f"{run['run_id']}: {entry['num_samples']} samples")
            for metric in METRICS:
                table["B6 tuned RBM"][sigma][metric][run["seed"]] = entry[metric]
            table["B6 tuned RBM"][sigma]["train_seconds"][run["seed"]] = tuning[
                "total_train_seconds"
            ]
            table["B6 tuned RBM"][sigma]["selected_parameters"][run["seed"]] = tuning[
                "selected"
            ][sigma]["num_parameters"]
            # Validation score of the selected candidate (before refit), to show selection optimism.
            table["B6 tuned RBM"][sigma]["selected_validation_mmd"][run["seed"]] = (
                tuning["selected"][sigma]["validation_mmd"]
            )
    for run in merlin:
        system = (
            "QCBM (Phase 4 MerLin adaptation)"
            if run["label"].startswith("qcbm")
            else "B5 twin (MerLin backend)"
        )
        identities[system].add(identity(run))
        entry = run["metrics"]["sample_metrics"]["model"]
        for metric in METRICS:
            table[system]["3.0"][metric][run["seed"]] = entry[metric]
        table[system]["3.0"]["train_seconds"][run["seed"]] = run["metrics"]["training"][
            "seconds"
        ]
    return table, identities


def to_stats(per_seed: dict) -> dict:
    if sorted(per_seed) != SEEDS:
        raise ValueError(f"seeds {sorted(per_seed)} != {SEEDS}")
    return stats([per_seed[s] for s in SEEDS])


def main() -> None:
    """Build, check and write the Phase 5 comparison."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("fig3a", "b6", "fig3a-merlin", "fig3e"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    fig3a, b6 = load_runs(args.fig3a), load_runs(args.b6)
    merlin, fig3e = load_runs(args.fig3a_merlin), load_runs(args.fig3e)

    table, identities = collect_fig3a(fig3a, b6, merlin)
    reference_identity = identities["QCBM (Phase 3, JAX + Perceval)"]
    if len(reference_identity) != 1:
        raise ValueError(
            f"Fig. 3a QCBM runs use several datasets: {reference_identity}"
        )
    commensurability = {}
    for system, ids in identities.items():
        same = ids == reference_identity
        if not same:
            raise ValueError(f"{system} evaluated on different data: {ids}")
        commensurability[system] = (
            "PARTIALLY_COMMENSURABLE on gaussian_mmd_raw (binary model cannot emit "
            "counts >= 2; 60% of test samples have one); COMMENSURABLE on "
            "gaussian_mmd_binarized and walsh_mmd"
            if system in BINARY_MODELS
            else "COMMENSURABLE (same dataset id, test hash, 2000 samples, K per seed)"
        )

    c4_systems = {}
    for system, by_sigma in table.items():
        c4_systems[system] = {}
        for sigma, metrics in by_sigma.items():
            entry = {}
            for metric, per_seed in metrics.items():
                entry[metric] = to_stats(per_seed)
            c4_systems[system][sigma] = entry

    method_names = (
        "QCBM (Phase 3, JAX + Perceval)",
        "QCBM (Phase 4 MerLin adaptation)",
    )
    baselines_c4 = (
        "paper RBM",
        "B6 tuned RBM",
        "B5 distinguishable twin",
        "B4 marginal",
        "uniform random",
    )
    c4_comparisons = []
    for method in method_names:
        for sigma in SIGMAS:
            if sigma not in c4_systems[method]:
                continue
            for baseline in baselines_c4:
                if sigma not in c4_systems[baseline]:
                    continue
                for metric in METRICS:
                    c4_comparisons.append(
                        {
                            "method": method,
                            "baseline": baseline,
                            "sigma": sigma,
                            "metric": metric,
                            "primary": metric == "gaussian_mmd_binarized",
                            "status": "PARTIALLY_COMMENSURABLE"
                            if metric == "gaussian_mmd_raw"
                            and baseline in BINARY_MODELS
                            else "COMMENSURABLE",
                            **compare(
                                c4_systems[method][sigma][metric],
                                c4_systems[baseline][sigma][metric],
                            ),
                        }
                    )

    # Frozen C4 decision rule on the primary metric, Phase 3 QCBM vs every classical baseline.
    primary = [
        c for c in c4_comparisons if c["primary"] and c["method"] == method_names[0]
    ]
    wins_by_sigma = {
        sigma: all(
            c["verdict"] == "method better" for c in primary if c["sigma"] == sigma
        )
        for sigma in SIGMAS
    }
    any_win = {
        sigma: any(
            c["verdict"] == "method better" for c in primary if c["sigma"] == sigma
        )
        for sigma in SIGMAS
    }
    if all(wins_by_sigma.values()):
        c4_outcome = "SUPPORTED"
    elif not any(any_win.values()):
        c4_outcome = "UNSUPPORTED"
    else:
        c4_outcome = "PARTIALLY_SUPPORTED"

    # C8 part 2: Fig. 3e, QCBM vs paper RBM, same K per seed inside each run.
    c8 = {}
    for label in ("polynomial2", "gaussian2.5", "gaussian3.5"):
        runs = [r for r in fig3e if r["label"] == label]
        qcbm = to_stats(
            {
                r["seed"]: r["metrics"]["sample_metrics"]["model"]["walsh_mmd"]
                for r in runs
            }
        )
        rbm = to_stats(
            {
                r["seed"]: r["metrics"]["sample_metrics"]["rbm"]["walsh_mmd"]
                for r in runs
            }
        )
        marginal = to_stats(
            {
                r["seed"]: r["metrics"]["sample_metrics"]["marginal"]["walsh_mmd"]
                for r in runs
            }
        )
        c8[label] = {
            "primary": label == "polynomial2",
            "qcbm": qcbm,
            "paper_rbm": rbm,
            "b4_marginal": marginal,
            "qcbm_vs_rbm": compare(qcbm, rbm),
            "qcbm_vs_b4": compare(qcbm, marginal),
            "status": "COMMENSURABLE (same run, same K, same test set, 2000 samples)",
        }
    c8_verdict = c8["polynomial2"]["qcbm_vs_rbm"]["verdict"]
    c8_outcome = {
        "method better": "SUPPORTED",
        "baseline better": "UNSUPPORTED",
        "not separated": "UNRESOLVED",
    }[c8_verdict]

    result = {
        "protocol": "LOG.md 'Phase 5 Comparison Plan — 2026-10-15T18:10:33Z'",
        "sources": {
            "fig3a": str(args.fig3a),
            "b6": str(args.b6),
            "fig3a_merlin": str(args.fig3a_merlin),
            "fig3e": str(args.fig3e),
        },
        "run_counts": {
            "fig3a": [len(fig3a), 30],
            "b6": [len(b6), 5],
            "fig3a_merlin": [len(merlin), 10],
            "fig3e": [len(fig3e), 15],
        },
        "paper_reported": "Fig. 3a / 3e plots only, not digitized: EVIDENCE_MISSING for numeric comparison; "
        "qualitative statements only (QCBM < RBM, random on BS; QCBM slightly < RBM on movies, polynomial).",
        "parameters": {
            **PARAMETERS,
            "B6 tuned RBM": "selected per seed and sigma, see c4.systems",
        },
        "commensurability": commensurability,
        "c4": {
            "systems": c4_systems,
            "comparisons": c4_comparisons,
            "wins_all_baselines_by_sigma": wins_by_sigma,
            "outcome_by_frozen_rule": c4_outcome,
        },
        "c8_part2": {"kernels": c8, "outcome_by_frozen_rule": c8_outcome},
        "merlin_c8": "N/A: no MerLin Fig. 3e run (Phase 4 scope: Fig. 3a sigma 3, Fig. 3b)",
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "phase5_comparison.json").write_text(json.dumps(result, indent=2))
    (RESULTS_DIR / "phase5_comparison.md").write_text(render_markdown(result))
    print(
        json.dumps(
            {
                "c4": c4_outcome,
                "c8_part2": c8_outcome,
                "run_counts": result["run_counts"],
            }
        )
    )


def fmt(s: dict) -> str:
    return f"{s['mean']:.3e} ± {s['std']:.1e}"


def render_markdown(result: dict) -> str:
    systems = result["c4"]["systems"]
    lines = [
        "# Phase 5 — Fair comparison (generated by `utils/make_comparison_table.py`)",
        "",
        f"Protocol: {result['protocol']}. Mean ± std (ddof=1) over seeds 0–4, test split, "
        "2000 samples per model, unbiased estimators. Lower is better. Δ separated when |Δ| > 2 SE.",
        "",
        "Paper-reported values: " + result["paper_reported"],
        "",
        "## C4 — boson-sampling data (Fig. 3a), m=256, n=16",
        "",
        "Primary metric: Gaussian MMD² on binarized samples. Locally executed on NVIDIA L4 machine "
        "`gpu-reproduced-papers-1` (QCBM: GPU/JAX; RBMs: CPU/sklearn).",
        "",
        "| System | Role | Provenance | Scope | Runs | sigma | Binarized (primary) | Walsh | Raw counts | Parameters | Train time / seed (s) |",
        "|---|---|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    roles = {
        "QCBM (Phase 3, JAX + Perceval)": (
            "method",
            "Phase 3 reproduction",
            "paper-accurate settings, new data instance",
        ),
        "QCBM (Phase 4 MerLin adaptation)": (
            "method (adaptation)",
            "Phase 4 MerLin",
            "adaptation, simulated, sigma 3 only",
        ),
        "paper RBM": ("paper baseline", "Phase 3", "paper settings"),
        "B6 tuned RBM": ("added fair baseline", "Phase 5", "tuned on validation"),
        "B5 distinguishable twin": ("added baseline", "Phase 3", "iso-architecture"),
        "B5 twin (MerLin backend)": (
            "added baseline",
            "Phase 4",
            "iso-architecture, sigma 3",
        ),
        "B4 marginal": ("added baseline", "Phase 3", "first-order statistics"),
        "uniform random": ("paper baseline", "Phase 3", "floor"),
        "test-to-test (noise floor)": ("reference", "Phase 3", "metric noise"),
    }
    for system, (role, provenance, scope) in roles.items():
        for sigma in SIGMAS:
            entry = systems.get(system, {}).get(sigma)
            if entry is None:
                continue
            cells = [fmt(entry[m]) if m in entry else "—" for m in METRICS]
            if "selected_parameters" in entry:
                params = "/".join(
                    str(int(v)) for v in entry["selected_parameters"]["values"]
                )
            else:
                params = str(result["parameters"][system])
            seconds = (
                f"{entry['train_seconds']['mean']:.0f}"
                if "train_seconds" in entry
                else "—"
            )
            n = entry[METRICS[1]]["n"]
            lines.append(
                f"| {system} | {role} | {provenance} | {scope} | {n}/5 | {sigma} | "
                + " | ".join(cells)
                + f" | {params} | {seconds} |"
            )
    lines += [
        "",
        "Train time: QCBM and B5 = optimizer seconds on the L4 GPU (JAX / torch); paper RBM = fit "
        "seconds on the same machine's CPU, read from `run.log` (`BASELINE_TRAINED`); B6 = all 6 grid fits "
        "+ refits per seed (the whole tuning protocol) on the same CPU. B4 and random need no training.",
        "",
        "Raw-count values of binary models (paper RBM, B6, B4, random) are PARTIALLY_COMMENSURABLE: "
        "they cannot emit the photon counts ≥ 2 present in 60% of test samples.",
        "",
        "### Pairwise decisions (QCBM − baseline)",
        "",
        "| Method | Baseline | sigma | Metric | Δ | 2 SE | Baseline / method | Decision | Status |",
        "|---|---|---|---|---:|---:|---:|---|---|",
    ]
    for c in result["c4"]["comparisons"]:
        lines.append(
            f"| {c['method']} | {c['baseline']} | {c['sigma']} | {c['metric']}{' (primary)' if c['primary'] else ''} | "
            f"{c['delta']:.2e} | {c['two_se']:.2e} | {c['ratio_baseline_over_method']:.1f} | {c['verdict']} | {c['status']} |"
        )
    lines += [
        "",
        f"C4 outcome by the frozen rule (Phase 3 QCBM vs every classical baseline, primary metric, all sigma): "
        f"**{result['c4']['outcome_by_frozen_rule']}** (wins by sigma: {result['c4']['wins_all_baselines_by_sigma']}).",
        "",
        "## C8 part 2 — movies (Fig. 3e), m=100, n=10; paper RBM baseline reused",
        "",
        "Primary: polynomial degree-2 kernel, `walsh_mmd` (same K operators for every model in a run).",
        "",
        "| Kernel | QCBM | paper RBM | B4 marginal | QCBM − RBM: Δ / 2 SE / decision | QCBM − B4: decision |",
        "|---|---:|---:|---:|---|---|",
    ]
    for label, k in result["c8_part2"]["kernels"].items():
        r = k["qcbm_vs_rbm"]
        lines.append(
            f"| {label}{' (primary)' if k['primary'] else ''} | {fmt(k['qcbm'])} | {fmt(k['paper_rbm'])} | {fmt(k['b4_marginal'])} | "
            f"{r['delta']:.2e} / {r['two_se']:.2e} / {r['verdict']} | {k['qcbm_vs_b4']['verdict']} |"
        )
    lines += [
        "",
        f"C8 part 2 outcome by the frozen rule: **{result['c8_part2']['outcome_by_frozen_rule']}**. "
        f"MerLin: {result['merlin_c8']}.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
