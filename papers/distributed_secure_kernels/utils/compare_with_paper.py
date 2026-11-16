"""Compare sweep summaries with the paper's reported values (Table 1, Fig. 3, Fig. 4).

Reads the ``summary.json`` of the paper-protocol sweeps (``table1_original``,
``noise_original``, ``shots_original``) and writes
``results/paper_comparison.{json,md}``.

Matching rule (fixed before the sweeps): a value is *matched* when our
mean over seeds lies within the paper's fold std, or within +-0.02 when the
paper gives no std. Paper values come from Table 1 and Fig. 4 of the paper; the
noise values (Fig. 3 shows bars only) come from the authors' released logs.

Usage::

    .venv/bin/python papers/distributed_secure_kernels/utils/compare_with_paper.py \
        <table1_original sweep dir> <noise_original sweep dir> <shots_original sweep dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
NO_STD_TOLERANCE = 0.02

# (experiment, candidate, method, model, label, paper value, paper std, source)
PAPER_VALUES = [
    (
        "Table 1",
        "accuracy_wine_original",
        "classical_linear",
        "kernel_svm",
        "Wine, kernel SVM, centralised classical",
        0.9860,
        0.0172,
        "Table 1",
    ),
    (
        "Table 1",
        "accuracy_wine_original",
        "quantum_centralised",
        "kernel_svm",
        "Wine, kernel SVM, centralised quantum",
        0.8805,
        None,
        "Table 1",
    ),
    (
        "Table 1",
        "accuracy_wine_original",
        "quantum_distributed",
        "kernel_svm",
        "Wine, kernel SVM, distributed quantum",
        0.8874,
        0.0259,
        "Table 1",
    ),
    (
        "Table 1",
        "accuracy_parkinsons_original",
        "classical_linear",
        "kernel_svm",
        "Parkinson's, kernel SVM, centralised classical",
        0.8196,
        0.0644,
        "Table 1",
    ),
    (
        "Table 1",
        "accuracy_parkinsons_original",
        "quantum_centralised",
        "kernel_svm",
        "Parkinson's, kernel SVM, centralised quantum",
        0.7875,
        None,
        "Table 1",
    ),
    (
        "Table 1",
        "accuracy_parkinsons_original",
        "quantum_distributed",
        "kernel_svm",
        "Parkinson's, kernel SVM, distributed quantum",
        0.7983,
        0.0798,
        "Table 1",
    ),
    (
        "Table 1",
        "accuracy_parkinsons_original",
        "classical_linear",
        "kernel_pca_best_on_test_folds",
        "Parkinson's, kernel PCA, centralised classical",
        0.7872,
        0.0716,
        "Table 1",
    ),
    (
        "Table 1",
        "accuracy_parkinsons_original",
        "quantum_centralised",
        "kernel_pca_best_on_test_folds",
        "Parkinson's, kernel PCA, centralised quantum",
        0.7451,
        None,
        "Table 1",
    ),
    (
        "Table 1",
        "accuracy_parkinsons_original",
        "quantum_distributed",
        "kernel_pca_best_on_test_folds",
        "Parkinson's, kernel PCA, distributed quantum",
        0.7660,
        0.0744,
        "Table 1",
    ),
    (
        "Table 1",
        "accuracy_framingham_original",
        "classical_linear",
        "kernel_svm",
        "Framingham, kernel SVM, centralised classical",
        0.6788,
        0.0108,
        "Table 1",
    ),
    (
        "Table 1",
        "accuracy_framingham_original",
        "quantum_centralised",
        "kernel_svm",
        "Framingham, kernel SVM, centralised quantum",
        0.6308,
        None,
        "Table 1",
    ),
    (
        "Table 1",
        "accuracy_framingham_original",
        "quantum_distributed",
        "kernel_svm",
        "Framingham, kernel SVM, distributed quantum",
        0.6340,
        0.0143,
        "Table 1",
    ),
    (
        "Table 1",
        "accuracy_framingham_original",
        "classical_linear",
        "kernel_pca_best_on_test_folds",
        "Framingham, kernel PCA, centralised classical",
        0.6788,
        0.0095,
        "Table 1",
    ),
    (
        "Table 1",
        "accuracy_framingham_original",
        "quantum_centralised",
        "kernel_pca_best_on_test_folds",
        "Framingham, kernel PCA, centralised quantum",
        0.6249,
        None,
        "Table 1",
    ),
    (
        "Table 1",
        "accuracy_framingham_original",
        "quantum_distributed",
        "kernel_pca_best_on_test_folds",
        "Framingham, kernel PCA, distributed quantum",
        0.6422,
        0.0092,
        "Table 1",
    ),
    (
        "Fig. 3",
        "noise_framingham_original",
        "quantum_distributed_noise_level_1",
        "kernel_svm",
        "Framingham, kernel SVM, 0.1 % noise",
        0.5807,
        0.0098,
        "authors' log heart_dataset_kernel_1060336.txt",
    ),
    (
        "Fig. 3",
        "noise_framingham_original",
        "quantum_distributed_noise_level_2",
        "kernel_svm",
        "Framingham, kernel SVM, 1 % noise",
        0.5814,
        0.0072,
        "authors' log heart_dataset_kernel_1060336.txt",
    ),
    (
        "Fig. 3",
        "noise_parkinsons_original",
        "quantum_distributed_noise_level_1",
        "kernel_svm",
        "Parkinson's, kernel SVM, 0.1 % noise",
        0.7873,
        0.0655,
        "authors' log park_dataset_kernel_1055216.txt",
    ),
    (
        "Fig. 3",
        "noise_parkinsons_original",
        "quantum_distributed_noise_level_2",
        "kernel_svm",
        "Parkinson's, kernel SVM, 1 % noise",
        0.6330,
        0.0087,
        "authors' log park_dataset_kernel_1055216.txt",
    ),
    (
        "Fig. 3",
        "noise_parkinsons_original",
        "quantum_distributed_noise_level_1",
        "kernel_pca_6_components",
        "Parkinson's, 6-component kernel PCA, 0.1 % noise",
        0.7445,
        0.0410,
        "authors' log park_dataset_kernel_1055216.txt",
    ),
    (
        "Fig. 3",
        "noise_parkinsons_original",
        "quantum_distributed_noise_level_2",
        "kernel_pca_6_components",
        "Parkinson's, 6-component kernel PCA, 1 % noise",
        0.6276,
        0.0049,
        "authors' log park_dataset_kernel_1055216.txt",
    ),
    (
        "Fig. 4",
        "shots_digits_original",
        "quantum_distributed_128_shots",
        "kernel_svm",
        "Digits, 128 shots",
        0.71,
        None,
        "Fig. 4 labels / authors' logs",
    ),
    (
        "Fig. 4",
        "shots_digits_original",
        "quantum_distributed_256_shots",
        "kernel_svm",
        "Digits, 256 shots",
        0.71,
        None,
        "Fig. 4 labels / authors' logs",
    ),
    (
        "Fig. 4",
        "shots_digits_original",
        "quantum_distributed_512_shots",
        "kernel_svm",
        "Digits, 512 shots",
        0.74,
        None,
        "Fig. 4 labels / authors' logs",
    ),
    (
        "Fig. 4",
        "shots_digits_original",
        "quantum_distributed_1024_shots",
        "kernel_svm",
        "Digits, 1,024 shots",
        0.83,
        None,
        "Fig. 4 labels / authors' logs",
    ),
    (
        "Fig. 4",
        "shots_digits_original",
        "classical_linear",
        "kernel_svm",
        "Digits, classical linear SVM",
        0.88,
        None,
        "authors' logs (Fig. 4 line)",
    ),
]


def compare(sweep_dirs: list[Path]) -> list[dict]:
    """One row per paper value, with our seed mean, seed std and the match verdict."""

    rows = {}
    for sweep_dir in sweep_dirs:
        summary = json.loads((sweep_dir / "summary.json").read_text())
        if summary["coverage"] != "COMPLETE SWEEP":
            raise ValueError(f"{sweep_dir} is a {summary['coverage']}")
        for row in summary["rows"]:
            rows[(row["candidate"], row["method"], row["model"])] = row
    table = []
    for (
        experiment,
        candidate,
        method,
        model,
        label,
        paper,
        paper_std,
        source,
    ) in PAPER_VALUES:
        ours = rows[(candidate, method, model)]["accuracy"]
        tolerance = paper_std if paper_std is not None else NO_STD_TOLERANCE
        delta = ours["seed_mean"] - paper
        table.append(
            {
                "experiment": experiment,
                "label": label,
                "paper": paper,
                "paper_std": paper_std,
                "source": source,
                "ours_seed_mean": ours["seed_mean"],
                "ours_seed_std": ours["seed_std"],
                "ours_mean_fold_std": ours["mean_fold_std"],
                "delta": delta,
                "tolerance": tolerance,
                "matched": abs(delta) <= tolerance,
            }
        )
    return table


def main(argv: list[str]) -> int:
    table = compare([Path(arg) for arg in argv])
    results = PROJECT_DIR / "results"
    results.mkdir(exist_ok=True)
    (results / "paper_comparison.json").write_text(json.dumps(table, indent=2) + "\n")
    lines = [
        "Metric: stratified 5-fold CV accuracy. Ours: mean ± std over seeds 0-4. "
        "Matched: |difference| ≤ the paper's fold std, or ≤ 0.02 when the paper gives none.",
        "",
        "| Experiment | Setting | Paper | Ours (5 seeds) | Difference | Matched |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in table:
        paper = f"{row['paper']:.4f}" + (
            f" ± {row['paper_std']:.4f}" if row["paper_std"] is not None else ""
        )
        ours = f"{row['ours_seed_mean']:.4f} ± {row['ours_seed_std']:.4f}"
        lines.append(
            f"| {row['experiment']} | {row['label']} | {paper} | {ours} | {row['delta']:+.4f} | {'yes' if row['matched'] else 'no'} |"
        )
    text = "\n".join(lines) + "\n"
    (results / "paper_comparison.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
