"""Promote a raw outdir run into the curated, tracked results/ directory.

Policy: ``outdir/`` (gitignored) keeps the complete raw artefacts of every
run — config snapshot, loss history, prediction arrays, model weights,
logs. ``results/`` (tracked) keeps only what a reader needs: one compact
metrics JSON per headline run, figures, and findings write-ups. This
script is the single supported path from raw run to curated artefact so
the layout stays consistent across runs.

Usage (from the paper directory or the repo root):

    python utils/curate_results.py <run_dir> --label poisson_merlin_600ep [--plot]

``<run_dir>`` may be the timestamped run directory itself or a parent
that contains exactly one ``run_*/summary.json`` (the per-seed launcher
layout). Writes ``results/<label>.json`` — metrics, provenance, and key
hyper-parameters only; no prediction arrays, no full config dump — and,
with ``--plot``, ``results/<label>.png`` via the plot utility matching
the experiment (plot_heat for heat_*, plot_poisson otherwise).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]

_TRAINING_KEYS = (
    "pretrain_epochs",
    "epochs",
    "lr",
    "collocation_points",
    "nx",
    "nt",
    "n_ic",
    "n_bc",
    "lambdas",
    "use_nested_loss",
    "lr_schedule",
)


def resolve_run_dir(path: Path) -> Path:
    """Return the directory that actually holds summary.json."""
    if (path / "summary.json").exists():
        return path
    nested = sorted(path.glob("run_*/summary.json"))
    if len(nested) == 1:
        return nested[0].parent
    if not nested:
        raise FileNotFoundError(f"No summary.json under {path}")
    raise ValueError(
        f"{path} contains {len(nested)} runs; pass the run_* directory directly"
    )


def compact_summary(summary: dict, label: str, source_run: str) -> dict:
    """Reduce a raw summary.json to the curated schema."""
    cfg = summary.get("cfg", {})
    training = cfg.get("training", {})
    out = {
        "label": label,
        "experiment": summary.get("experiment"),
        "source_run": source_run,
        "seed": cfg.get("seed"),
        "n_params": summary.get("n_params"),
        "metrics": summary.get("metrics"),
        "best": summary.get("best"),
        "wall_time_sec": summary.get("wall_time_sec"),
        "model": cfg.get("model"),
        "training": {k: training[k] for k in _TRAINING_KEYS if k in training},
    }
    for key in ("hidden_layers", "qlayer_output_size", "merlin_hardware"):
        if key in summary:
            out[key] = summary[key]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument(
        "--label",
        required=True,
        help="basename for results/<label>.json (and .png with --plot)",
    )
    ap.add_argument(
        "--plot",
        action="store_true",
        help="also render results/<label>.png from the run predictions",
    )
    ap.add_argument("--results-dir", type=Path, default=PROJECT / "results")
    args = ap.parse_args()

    run_dir = resolve_run_dir(args.run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    compact = compact_summary(summary, args.label, run_dir.name)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.results_dir / f"{args.label}.json"
    out_json.write_text(json.dumps(compact, indent=2) + "\n")
    print(f"wrote {out_json}")

    if args.plot:
        experiment = str(compact.get("experiment", ""))
        script = "plot_heat.py" if experiment.startswith("heat") else "plot_poisson.py"
        out_png = args.results_dir / f"{args.label}.png"
        subprocess.run(
            [
                sys.executable,
                str(PROJECT / "utils" / script),
                str(run_dir),
                "--out",
                str(out_png),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
