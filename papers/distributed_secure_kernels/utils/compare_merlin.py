"""Compare the MerLin adaptation with the gate-model distributed kernel (noiseless).

Rule (frozen before the MerLin runs): the adaptation preserves the
original behaviour when, for every setting, |delta| <= 2 sigma_pooled, with
sigma_pooled = sqrt((s_gate^2 + s_merlin^2) / 2) and s the std over seeds of the
5-fold mean accuracy. Writes ``results/merlin_comparison.{json,md}``.

Usage::

    .venv/bin/python papers/distributed_secure_kernels/utils/compare_merlin.py \
        <table1_original sweep dir> <shots_original sweep dir> <merlin_original sweep dir>
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

# (label, gate candidate, gate method, merlin candidate, merlin method, model)
PAIRS = [
    (
        "Wine, kernel SVM",
        "accuracy_wine_original",
        "quantum_distributed",
        "accuracy_wine_merlin",
        "merlin_photonic_swap_test",
        "kernel_svm",
    ),
    (
        "Parkinson's, kernel SVM",
        "accuracy_parkinsons_original",
        "quantum_distributed",
        "accuracy_parkinsons_merlin",
        "merlin_photonic_swap_test",
        "kernel_svm",
    ),
    (
        "Parkinson's, kernel PCA",
        "accuracy_parkinsons_original",
        "quantum_distributed",
        "accuracy_parkinsons_merlin",
        "merlin_photonic_swap_test",
        "kernel_pca_best_on_test_folds",
    ),
    (
        "Framingham, kernel SVM",
        "accuracy_framingham_original",
        "quantum_distributed",
        "accuracy_framingham_merlin",
        "merlin_photonic_swap_test",
        "kernel_svm",
    ),
    (
        "Framingham, kernel PCA",
        "accuracy_framingham_original",
        "quantum_distributed",
        "accuracy_framingham_merlin",
        "merlin_photonic_swap_test",
        "kernel_pca_best_on_test_folds",
    ),
] + [
    (
        f"Digits, {s} shots",
        "shots_digits_original",
        f"quantum_distributed_{s}_shots",
        "shots_digits_merlin",
        f"merlin_photonic_swap_test_{s}_shots",
        "kernel_svm",
    )
    for s in (128, 256, 512, 1024)
]


def _rows(sweep_dirs: list[Path]) -> dict:
    rows = {}
    for sweep_dir in sweep_dirs:
        summary = json.loads((sweep_dir / "summary.json").read_text())
        if summary["coverage"] != "COMPLETE SWEEP":
            raise ValueError(f"{sweep_dir} is a {summary['coverage']}")
        for row in summary["rows"]:
            rows[(row["candidate"], row["method"], row["model"])] = row["accuracy"]
    return rows


def compare(sweep_dirs: list[Path]) -> list[dict]:
    rows = _rows(sweep_dirs)
    table = []
    for label, gate_c, gate_m, merlin_c, merlin_m, model in PAIRS:
        gate, merlin = rows[(gate_c, gate_m, model)], rows[(merlin_c, merlin_m, model)]
        pooled = math.sqrt((gate["seed_std"] ** 2 + merlin["seed_std"] ** 2) / 2)
        delta = merlin["seed_mean"] - gate["seed_mean"]
        table.append(
            {
                "setting": label,
                "gate_seed_mean": gate["seed_mean"],
                "gate_seed_std": gate["seed_std"],
                "merlin_seed_mean": merlin["seed_mean"],
                "merlin_seed_std": merlin["seed_std"],
                "delta": delta,
                "two_sigma_pooled": 2 * pooled,
                # A zero pooled std (both collapse to the majority class) is equivalent only if identical.
                "preserved": abs(delta) <= 2 * pooled
                if pooled > 0
                else abs(delta) < 1e-12,
            }
        )
    return table


def main(argv: list[str]) -> int:
    table = compare([Path(arg) for arg in argv])
    results = PROJECT_DIR / "results"
    (results / "merlin_comparison.json").write_text(json.dumps(table, indent=2) + "\n")
    lines = [
        "Metric: stratified 5-fold CV accuracy, mean ± std over seeds 0-4, paper protocol, 1,024 shots unless stated. "
        "Gate model: our distributed quantum column (noiseless). MerLin: photonic swap test (adaptation). "
        "Preserved: |difference| ≤ 2 σ_pooled.",
        "",
        "| Setting | Gate model | MerLin adaptation | Difference | 2 σ pooled | Preserved |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in table:
        lines.append(
            f"| {row['setting']} | {row['gate_seed_mean']:.4f} ± {row['gate_seed_std']:.4f} "
            f"| {row['merlin_seed_mean']:.4f} ± {row['merlin_seed_std']:.4f} | {row['delta']:+.4f} "
            f"| {row['two_sigma_pooled']:.4f} | {'yes' if row['preserved'] else 'no'} |"
        )
    text = "\n".join(lines) + "\n"
    (results / "merlin_comparison.md").write_text(text)
    print(text)
    return 0 if all(row["preserved"] for row in table) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
