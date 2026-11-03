"""Write the S4 spec (PQKD vs baselines B0-B6) from the S2/S3 SPSA-grid selections.

The arms, seeds, metrics and fixed configuration are predeclared in LOG.md
("Sweep Plan"); only the SPSA step sizes come from the grids. PQKD and every
non-learnable source take the S2 (photonic) selection; B5 takes the S3
(surrogate) selection, so both learnable generators get equally tuned SPSA.

Usage (from the repository root)::

    python papers/photonic_QKD/utils/make_s4_spec.py \
        --teacher-sweep <S1 dir> --photonic-grid <S2 dir> --surrogate-grid <S3 dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PAPER_DIR.parents[1]

# Seed-major order with the three Fock-space arms (pqkd, B4, B6) spaced three
# apart, so at most two ~2.2 GB photonic runs overlap on a 7.5 GB machine.
ARMS = [
    ("pqkd", ["--conditioning", "photonic"], "photonic"),
    ("B1_kd_dense", ["--student-variant", "dense"], "photonic"),
    ("B0_zero", ["--conditioning", "zero"], "photonic"),
    ("B4_fixed_photonic", ["--conditioning", "photonic_fixed"], "photonic"),
    (
        "B2_dictionary_trainable",
        ["--student-variant", "dictionary_trainable"],
        "photonic",
    ),
    ("B3f_random_fixed", ["--conditioning", "random_fixed"], "photonic"),
    ("B6_classical_histogram", ["--conditioning", "classical_histogram"], "photonic"),
    ("B3r_random_resampled", ["--conditioning", "random_resampled"], "photonic"),
    ("B5_surrogate", ["--conditioning", "surrogate"], "surrogate"),
]


def selected_spsa(grid_dir: Path) -> tuple[str, str]:
    """Return the ``(a, c)`` arguments of the grid's selected candidate."""
    status = json.loads((grid_dir / "sweep_status.json").read_text())
    if status["status"] != "COMPLETED":
        raise RuntimeError(
            f"{grid_dir} is {status['status']}; S4 needs a complete grid"
        )
    selected = status["selected_candidates"]
    if len(selected) != 1:
        raise RuntimeError(f"{grid_dir} selected {selected}; expected exactly one")
    spec = json.loads((grid_dir / "spec.json").read_text())
    args = next(c["args"] for c in spec["candidates"] if c["name"] == selected[0])
    return args[args.index("--spsa-a") + 1], args[args.index("--spsa-c") + 1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--teacher-sweep", type=Path, required=True)
    parser.add_argument("--photonic-grid", type=Path, required=True)
    parser.add_argument("--surrogate-grid", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=PAPER_DIR / "sweeps" / "S4_mnist_pqkd_baselines.json",
    )
    args = parser.parse_args()
    spsa = {
        "photonic": selected_spsa(args.photonic_grid.resolve()),
        "surrogate": selected_spsa(args.surrogate_grid.resolve()),
    }
    candidates = [
        {
            "name": name,
            "args": [*arm_args, "--spsa-a", spsa[key][0], "--spsa-c", spsa[key][1]],
        }
        for name, arm_args, key in ARMS
    ]
    spec = {
        "sweep_id": "S4_mnist_pqkd_baselines",
        "description": (
            "PQKD vs fair baselines B0-B6 on MNIST (conv1+conv2, R=4, dim(theta)=45, S=200, EMA 0.9), "
            f"30 epochs, seeds 0-2 (partial RED-tier). SPSA (a, c): photonic {spsa['photonic']} from "
            f"{args.photonic_grid.name}, surrogate {spsa['surrogate']} from {args.surrogate_grid.name}. "
            "Comparison sweep: no winner is declared; differences are read as paired per-seed deltas vs pqkd."
        ),
        "fixed_args": ["--config", "configs/pqkd_merlin.json"],
        "candidates": candidates,
        "seeds": [0, 1, 2],
        "run_order": "seed_major",
        "max_parallel": 3,
        "threads_per_run": 4,
        "teacher_sweep": str(
            args.teacher_sweep.resolve().relative_to(REPO_ROOT.resolve())
        ),
        "selection": {
            "metric": "final_evaluation.test_accuracy",
            "direction": "maximize",
            "split": "test",
            "checkpoint": "final_epoch",
            "aggregation": "mean over seeds",
            "tie_tolerance": 0.0,
            "tie_break": "report_all",
        },
        "report_metrics": [
            "final_evaluation.test_accuracy",
            "final_evaluation.val_accuracy",
            "final_evaluation.val_cross_entropy",
            "final_evaluation.test_accuracy_drop_vs_teacher",
            "final_evaluation.delta_test_accuracy_vs_zero_feature",
            "final_evaluation.test_accuracy_fresh_feature",
            "proxy_objective_delta_first_minus_final",
            "parameters.student_total_trainable",
            "parameters.compression_overall",
            "final_epoch.train_accuracy",
        ],
    }
    args.out.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
