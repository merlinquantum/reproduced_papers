"""Export the compact, curated results the notebook loads (results/notebook_data.json).

Usage (from papers/power_of_QNN):
    python utils/export_notebook_data.py --output results/notebook_data.json

Raw run directories (outdir/) are not committed, so the notebook reads its stored evidence
from this file. Every block records the run directory it came from. Per-seed values are
kept for the Phase 05 comparisons so the notebook can show individual seeds and recompute
every decision.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from figure_io import PAPER_N, UPSTREAM, load_metrics

RESULTS = Path(__file__).resolve().parents[1] / "results"


def _effdim_block(run_dir: str, names: list[str] | None = None) -> dict:
    metrics = load_metrics(Path(run_dir))
    models = {}
    for name, entry in metrics["models"].items():
        if names is not None and name not in names:
            continue
        models[name] = {
            "d": entry["d"],
            "draws": [draw["draw_seed"] for draw in entry["draws"]],
            "ed_mean": entry["normalized_effective_dimension_mean"],
            "ed_std": entry["normalized_effective_dimension_std"],
            "mean_trace": entry["mean_trace_mean"],
            "mean_rank": entry["mean_rank_mean"],
            "hardware": entry.get("hardware"),
        }
    return {
        "run": run_dir,
        "n_values": metrics["n_values"],
        "theta": [metrics["fisher"]["theta_low"], metrics["fisher"]["theta_high"]],
        "models": models,
    }


def _training_block(run_dir: str) -> dict:
    metrics = load_metrics(Path(run_dir))
    return {
        "run": run_dir,
        "models": {
            name: {
                "d": entry["d"],
                "final_loss_mean": entry["final_loss_mean"],
                "final_loss_std": entry["final_loss_std"],
                "fisher_rao_norm_mean": entry["fisher_rao_norm_mean"],
                "init": entry["effective_training"]["init"],
                "optimizer": entry["effective_training"]["optimizer"],
                "lr": entry["effective_training"]["lr"],
            }
            for name, entry in metrics["models"].items()
        },
    }


def _comparison_block(run_dirs: list[str], unify_b2: bool) -> dict:
    block = {"runs": run_dirs, "models": {}}
    for run_dir in run_dirs:
        metrics = load_metrics(Path(run_dir))
        block.setdefault("comparison", metrics["comparison"])
        for name, entry in metrics["models"].items():
            arrays = np.load(Path(run_dir) / entry["arrays"])
            # The photonic 6-mode design was registered as B2_6m_cluster (design search 4) and
            # B2_d40 (breast cancer, MNIST): one
            # design, so the Phase 05 blocks use one name. Screens keep their own names.
            key = "B2_d40" if unify_b2 and name.startswith("B2") else name
            block["models"][key] = {
                "run": run_dir,
                "d": entry["d"],
                "hardware": entry.get("hardware"),
                "grid_mean_loss": entry["grid_mean_loss"],
                "selected_lr": entry["selected_lr"],
                "tied_lrs": entry["tied_lrs"],
                "wall_clock_s": sum(entry["grid_seconds"].values())
                + entry["final_seconds"],
                "loss_curve_mean": entry["loss_curve_mean"],
                "per_seed": {
                    f"{split}_{kind}": arrays[f"{split}_{kind}"].tolist()
                    for split in metrics["comparison"]["splits"]
                    for kind in ("loss", "accuracy")
                },
            }
    return block


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    phase3 = json.loads((RESULTS / "phase03_runs.json").read_text())
    phase4 = json.loads((RESULTS / "phase04_runs.json").read_text())
    phase5 = json.loads((RESULTS / "phase05_runs.json").read_text())
    amendment5 = json.loads((RESULTS / "phase04_amendment5_runs.json").read_text())
    design = json.loads((RESULTS / "phase04_design_runs.json").read_text())

    published = np.loadtxt(
        UPSTREAM
        / "Effective_dimension_plots/plot_fig/source_data/effective_dimension.txt"
    )
    generalisation = load_metrics(Path(phase3["generalisation_original"]))
    data = {
        "provenance": "utils/export_notebook_data.py from the run directories listed in each block",
        "paper": {
            "n_values": PAPER_N,
            "fig3a": {
                "source": "authors' source data, tag v1.0.0 (Effective_dimension_plots/plot_fig/source_data/effective_dimension.txt)",
                "qnn": published[0].tolist(),
                "classical_d28_array": published[1].tolist(),
                "easy_qnn": published[2].tolist(),
            },
            "table1_loss": {
                "qnn": 0.2314,
                "easy_qnn": 0.4305,
                "classical_41112": 0.3790,
            },
            "table1_fisher_rao": {
                "qnn": 117.84,
                "easy_qnn": 104.89,
                "classical_41112": 46.45,
            },
        },
        "phase3": {
            "effdim_original": _effdim_block(phase3["effdim_d40_original"]),
            "effdim_paper_domain": _effdim_block(phase3["effdim_d40_paper_domain"]),
            "spectrum_scaling": _effdim_block(phase3["spectrum_scaling_original"]),
            "training_original": _training_block(phase3["training_iris_original"]),
            "training_paper_domain": _training_block(
                phase3["training_iris_paper_domain"]
            ),
            "generalisation": {
                "run": phase3["generalisation_original"],
                "fractions": generalisation["fractions"],
            },
        },
        "phase4": {
            "planned_design": _effdim_block(phase4["effdim_d40_merlin"]),
            "a4_confirm_u01": _effdim_block(
                design["Amendment 4 confirmation, U[0,1] (A7, A8, Q2, Q3)"]["run"]
            ),
            "a5_capacity_screen": _effdim_block(
                design["Amendment 5 capacity screen (B1-B6, Q3)"]["run"]
            ),
            "a5_capacity_confirm": _effdim_block(
                design["Amendment 5 capacity confirmation (B2, Q3)"]["run"]
            ),
            "a5_loss_screen": _comparison_block(
                [design["Amendment 5 loss screen (B1-B6)"]["run"]], unify_b2=False
            ),
        },
        "phase5": {
            name: _comparison_block(
                amendment5.get(name, [run] if isinstance(run, str) else run),
                unify_b2=True,
            )
            for name, run in phase5.items()
        },
    }
    args.output.write_text(json.dumps(data, default=float), encoding="utf-8")
    print(f"wrote {args.output} ({args.output.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
