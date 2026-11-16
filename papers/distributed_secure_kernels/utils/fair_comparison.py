"""Fair comparison, generated from the sweeps' structured artifacts.

Reads ``summary.json`` and every run's ``metrics.json`` of:

- ``table1_corrected`` (corrected protocol: quantum, exact, linear, majority),
- ``fair_comparison`` (corrected protocol: tuned RBF / polynomial with the grids
  fixed before any run; MerLin adaptation on Wine and Parkinson's),
- ``table1_original`` and ``merlin_original`` (paper protocol, for the
  paper-baseline comparison).

Decision rule (frozen before the comparison runs): two systems are equivalent
when |delta| <= 2 sigma_pooled (sigma_pooled = sqrt((s1^2 + s2^2) / 2), s the std
over seeds of the 5-fold mean); otherwise different. Writes
``results/fair_comparison.{json,md}``.

Usage::

    .venv/bin/python papers/distributed_secure_kernels/utils/fair_comparison.py \
        --corrected <table1_corrected> --fair <fair_comparison> \
        --original <table1_original> --merlin <merlin_original>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASETS = ("wine", "parkinsons", "framingham")
# Subset sizes and features (paper subset; corrected protocol, no oversampling).
SUBSET = {"wine": (142, 13), "parkinsons": (156, 22), "framingham": (2999, 14)}
REGISTER_MODES = {"wine": 6, "parkinsons": 8, "framingham": 6}

# (system key, label, role, provenance, sweep key, candidate pattern, method)
CORRECTED_SYSTEMS = [
    (
        "quantum_distributed",
        "Distributed quantum kernel (swap test, 1,024 shots)",
        "paper method",
        "local, gate model (exact simulation)",
        "corrected",
        "accuracy_{d}_corrected",
        "quantum_distributed",
    ),
    (
        "quantum_centralised",
        "Centralised quantum kernel (swap test, 1,024 shots)",
        "paper method",
        "local, gate model (exact simulation)",
        "corrected",
        "accuracy_{d}_corrected",
        "quantum_centralised",
    ),
    (
        "merlin",
        "MerLin photonic swap test (1,024 shots)",
        "adaptation",
        "local, MerLin simulation",
        "fair",
        "accuracy_{d}_merlin_corrected",
        "merlin_photonic_swap_test",
    ),
    (
        "exact_squared_overlap",
        "Exact squared-overlap kernel (infinite shots)",
        "reference",
        "local, closed form",
        "corrected",
        "accuracy_{d}_corrected",
        "exact_squared_overlap",
    ),
    (
        "exact_cosine",
        "Exact cosine kernel",
        "reference",
        "local, closed form",
        "corrected",
        "accuracy_{d}_corrected",
        "exact_cosine",
    ),
    (
        "classical_linear",
        "Linear SVM (paper's classical column), C tuned",
        "paper baseline",
        "local",
        "corrected",
        "accuracy_{d}_corrected",
        "classical_linear",
    ),
    (
        "tuned_rbf",
        "Tuned RBF SVM (fair baseline)",
        "fair baseline",
        "local",
        "fair",
        "accuracy_{d}_fair_classical",
        "tuned_rbf",
    ),
    (
        "tuned_poly",
        "Tuned polynomial SVM (fair baseline)",
        "fair baseline",
        "local",
        "fair",
        "accuracy_{d}_fair_classical",
        "tuned_poly",
    ),
    (
        "majority_class",
        "Majority class",
        "floor",
        "local",
        "corrected",
        "accuracy_{d}_corrected",
        "majority_class",
    ),
]
PAIRWISE = [
    ("quantum_distributed", "quantum_centralised"),
    ("quantum_distributed", "tuned_rbf"),
    ("quantum_distributed", "tuned_poly"),
    ("quantum_distributed", "classical_linear"),
    ("quantum_distributed", "exact_cosine"),
    ("quantum_distributed", "majority_class"),
    ("merlin", "tuned_rbf"),
    ("merlin", "quantum_distributed"),
]


def _summary(sweep: Path) -> dict:
    data = json.loads((sweep / "summary.json").read_text())
    if data["coverage"] != "COMPLETE SWEEP":
        raise ValueError(f"{sweep}: {data['coverage']}")
    return {(r["candidate"], r["method"], r["model"]): r for r in data["rows"]}


def _seconds(sweep: Path, candidate: str, method: str) -> float:
    status = json.loads((sweep / "sweep_status.json").read_text())
    values = [
        json.loads((sweep / run["metrics_path"]).read_text())["methods"][method][
            "seconds"
        ]
        for run in status["runs"]
        if Path(run["candidate"]["config"]).stem == candidate
    ]
    return float(np.median(values))


def _resources(system: str, dataset: str) -> str:
    n, d = SUBSET[dataset]
    train = round(0.8 * n)
    entries = train * (train + 1) // 2 + (n - train) * train
    qubits = 4 * math.ceil(math.log2(d)) + 1
    if system in ("quantum_distributed", "quantum_centralised"):
        circuit = (
            f"{qubits} qubits"
            if system == "quantum_distributed"
            else f"{2 * math.ceil(math.log2(d)) + 1} qubits"
        )
        return f"{entries:,} kernel entries × 1,024 shots per outer fold; {circuit}"
    if system == "merlin":
        return f"{entries:,} kernel entries × 1,024 shots per outer fold; 4 photons, {2 * REGISTER_MODES[dataset]} modes"
    if system == "majority_class":
        return "none"
    return f"{entries:,} closed-form kernel entries per outer fold"


def build(paths: dict[str, Path]) -> dict:
    summaries = {key: _summary(path) for key, path in paths.items()}
    systems, pairs = [], []
    for dataset in DATASETS:
        models = (
            ["kernel_svm", "kernel_pca_nested_cv"]
            if dataset != "wine"
            else ["kernel_svm"]
        )
        for model in models:
            by_key = {}
            for (
                key,
                label,
                role,
                provenance,
                sweep,
                pattern,
                method,
            ) in CORRECTED_SYSTEMS:
                candidate = pattern.format(d=dataset)
                use_model = "kernel_svm" if key == "majority_class" else model
                row = summaries[sweep].get((candidate, method, use_model))
                if row is None:
                    status = (
                        "N/A — not run: MerLin corrected protocol on Framingham omitted (plan); kernel equals the gate-model kernel to 2.6e-8"
                        if key == "merlin"
                        else "N/A"
                    )
                    systems.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "system": key,
                            "label": label,
                            "role": role,
                            "status": status,
                        }
                    )
                    continue
                entry = {
                    "dataset": dataset,
                    "model": model,
                    "system": key,
                    "label": label,
                    "role": role,
                    "provenance": provenance,
                    "scope": "corrected protocol, 5 seeds"
                    + (", MerLin adaptation (simulation)" if key == "merlin" else ""),
                    "runs": f"{row['successful_runs']}/{row['expected_runs']}",
                    "accuracy": row["accuracy"]["seed_mean"],
                    "accuracy_seed_std": row["accuracy"]["seed_std"],
                    "balanced_accuracy": row["balanced_accuracy"]["seed_mean"],
                    "balanced_accuracy_seed_std": row["balanced_accuracy"]["seed_std"],
                    "matching": "same samples, folds, preprocessing, C grid, inner CV",
                    "resources": _resources(key, dataset),
                    "median_seconds_per_seed": _seconds(
                        paths[sweep], candidate, method
                    ),
                    "status": "COMMENSURABLE",
                }
                systems.append(entry)
                by_key[key] = entry
            for first, second in PAIRWISE:
                if first not in by_key or second not in by_key:
                    continue
                for metric in ("accuracy", "balanced_accuracy"):
                    a, b = by_key[first], by_key[second]
                    pooled = math.sqrt(
                        (a[f"{metric}_seed_std"] ** 2 + b[f"{metric}_seed_std"] ** 2)
                        / 2
                    )
                    delta = a[metric] - b[metric]
                    if abs(delta) <= 2 * pooled or (pooled == 0 and abs(delta) < 1e-12):
                        verdict = "equivalent"
                    else:
                        verdict = f"{'first' if delta > 0 else 'second'} higher"
                    pairs.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "first": first,
                            "second": second,
                            "metric": metric,
                            "delta": delta,
                            "two_sigma_pooled": 2 * pooled,
                            "verdict": verdict,
                        }
                    )
    # Paper protocol: the paper's own comparison (quantum vs linear SVM), for reference.
    original = summaries["original"]
    merlin = summaries["merlin"]
    paper_protocol = []
    for dataset in DATASETS:
        for model in (
            ["kernel_svm", "kernel_pca_best_on_test_folds"]
            if dataset != "wine"
            else ["kernel_svm"]
        ):
            linear = original[
                (f"accuracy_{dataset}_original", "classical_linear", model)
            ]["accuracy"]
            distributed = original[
                (f"accuracy_{dataset}_original", "quantum_distributed", model)
            ]["accuracy"]
            photonic = merlin[
                (f"accuracy_{dataset}_merlin", "merlin_photonic_swap_test", model)
            ]["accuracy"]
            pooled = math.sqrt(
                (linear["seed_std"] ** 2 + distributed["seed_std"] ** 2) / 2
            )
            delta = distributed["seed_mean"] - linear["seed_mean"]
            paper_protocol.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "linear_svm": linear["seed_mean"],
                    "linear_svm_seed_std": linear["seed_std"],
                    "distributed_quantum": distributed["seed_mean"],
                    "distributed_quantum_seed_std": distributed["seed_std"],
                    "merlin": photonic["seed_mean"],
                    "merlin_seed_std": photonic["seed_std"],
                    "delta_quantum_minus_linear": delta,
                    "two_sigma_pooled": 2 * pooled,
                    "verdict": "equivalent"
                    if abs(delta) <= 2 * pooled
                    else ("quantum higher" if delta > 0 else "linear SVM higher"),
                }
            )
    return {
        "rule": "equivalent when |delta| <= 2 sigma_pooled (seed std, ddof = 1)",
        "systems": systems,
        "pairwise": pairs,
        "paper_protocol": paper_protocol,
        "sources": {k: str(v.relative_to(PROJECT_DIR)) for k, v in paths.items()},
    }


# Readable model names for the markdown tables (the JSON keeps the codes).
MODEL_LABELS = {
    "kernel_svm": "kernel SVM",
    "kernel_pca_nested_cv": "kernel PCA",
    "kernel_pca_best_on_test_folds": "kernel PCA",
}


def _markdown(report: dict) -> str:
    lines = [
        "Corrected protocol (oversampling and selection inside training folds), stratified 5-fold CV, mean ± std over seeds 0-4. "
        "Accuracy is the primary metric; balanced accuracy is needed for Framingham (majority rate 0.847). "
        "Quantum values come from an exact simulation, MerLin values from MerLin simulation; nothing ran on hardware. "
        "All rows use the same samples, folds, preprocessing, C grid and inner cross-validation.",
        "",
        "| Dataset | Model | System | Role | Runs | Accuracy | Balanced accuracy | Resources per outer fold | Median s per seed |",
        "|---|---|---|---|---|---:|---:|---|---:|",
    ]
    for s in report["systems"]:
        if "accuracy" not in s:
            lines.append(
                f"| {s['dataset']} | {MODEL_LABELS[s['model']]} | {s['label']} | {s['role']} "
                f"| {s['status'].removeprefix('N/A — ')} | — | — | — | — |"
            )
            continue
        lines.append(
            f"| {s['dataset']} | {MODEL_LABELS[s['model']]} | {s['label']} | {s['role']} | {s['runs']} "
            f"| {s['accuracy']:.4f} ± {s['accuracy_seed_std']:.4f} | {s['balanced_accuracy']:.4f} ± {s['balanced_accuracy_seed_std']:.4f} "
            f"| {s['resources']} | {s['median_seconds_per_seed']:.1f} |"
        )
    lines += [
        "",
        "Pairwise decisions (rule: " + report["rule"] + "):",
        "",
        "| Dataset | Model | First | Second | Metric | Difference | 2 σ pooled | Verdict |",
        "|---|---|---|---|---|---:|---:|---|",
    ]
    for p in report["pairwise"]:
        lines.append(
            f"| {p['dataset']} | {MODEL_LABELS[p['model']]} | {p['first']} | {p['second']} | {p['metric']} | {p['delta']:+.4f} | {p['two_sigma_pooled']:.4f} | {p['verdict']} |"
        )
    lines += [
        "",
        "Paper protocol (the paper's own comparison, whose protocol leaks; see docs/protocols.md):",
        "",
        "| Dataset | Model | Linear SVM | Distributed quantum | MerLin | Quantum − linear | 2 σ pooled | Verdict |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for p in report["paper_protocol"]:
        lines.append(
            f"| {p['dataset']} | {MODEL_LABELS[p['model']]} | {p['linear_svm']:.4f} ± {p['linear_svm_seed_std']:.4f} | {p['distributed_quantum']:.4f} ± {p['distributed_quantum_seed_std']:.4f} "
            f"| {p['merlin']:.4f} ± {p['merlin_seed_std']:.4f} | {p['delta_quantum_minus_linear']:+.4f} | {p['two_sigma_pooled']:.4f} | {p['verdict']} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ("corrected", "fair", "original", "merlin"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = {
        k: getattr(args, k).resolve()
        for k in ("corrected", "fair", "original", "merlin")
    }
    report = build(paths)
    results = PROJECT_DIR / "results"
    (results / "fair_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    text = _markdown(report)
    (results / "fair_comparison.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
