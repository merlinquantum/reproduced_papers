"""Cross-product sweep runner for the QRC level-generation reproduction.

For each (config_point, seed) pair, launches the shared runtime as a
subprocess (so logging, snapshots, and metric files behave identically to a
manual run), then aggregates every ``metrics.json`` into a single
``aggregated.json`` keyed by ``config_point``.

Three pre-defined sweeps are bundled:

- ``modes``: photonic backend, fix ``n_photons=3``, sweep ``n_modes``.
- ``photons``: photonic backend, fix ``n_modes=6``, sweep ``n_photons``.
- ``isodim``: photonic vs gate-based reservoirs near matched output dim.

Run via the project's repo-root:

    python papers/qrc_level_generation/utils/sweep.py --sweep modes \
        --out-root papers/qrc_level_generation/sweeps/modes --seeds 3
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from math import comb
from pathlib import Path

_PAPER_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PAPER_DIR.parents[1]


@dataclass
class Point:
    """One configuration in a sweep."""

    name: str
    base_config: str  # path relative to paper dir, e.g. configs/mario_photonic.json
    extra_args: list[str] = field(default_factory=list)
    label: str = ""

    def display(self) -> str:
        return self.label or self.name


def _modes_sweep() -> list[Point]:
    pts: list[Point] = []
    for n_modes in [4, 6, 8]:
        pts.append(
            Point(
                name=f"phot_m{n_modes}_p3",
                base_config="configs/mario_photonic.json",
                extra_args=["--n-modes", str(n_modes), "--n-photons", "3"],
                label=f"photonic m={n_modes}, p=3 (dim={comb(n_modes, 3)})",
            )
        )
    return pts


def _photons_sweep() -> list[Point]:
    pts: list[Point] = []
    for n_photons in [1, 2, 3]:
        pts.append(
            Point(
                name=f"phot_m6_p{n_photons}",
                base_config="configs/mario_photonic.json",
                extra_args=["--n-modes", "6", "--n-photons", str(n_photons)],
                label=f"photonic m=6, p={n_photons} (dim={comb(6, n_photons)})",
            )
        )
    return pts


def _isodim_sweep() -> list[Point]:
    pts: list[Point] = []
    # Gate-based reservoirs at q=4,5,6 (output dims 16, 32, 64).
    for q in [4, 5, 6]:
        pts.append(
            Point(
                name=f"qubit_q{q}",
                base_config="configs/mario_qubit_paper.json",
                extra_args=[
                    "--n-qubits",
                    str(q),
                    "--evaluate-reference-sequences",
                    "false",
                ],
                label=f"qubit q={q} (dim={2**q})",
            )
        )
    # Photonic configurations near those dimensions.
    photonic_points = [
        ("phot_m6_p2", 6, 2),  # dim 15 (≈ q=4)
        ("phot_m6_p3", 6, 3),  # dim 20
        ("phot_m8_p3", 8, 3),  # dim 56 (≈ q=5)
        ("phot_m8_p4", 8, 4),  # dim 70
    ]
    for name, n_modes, n_photons in photonic_points:
        pts.append(
            Point(
                name=name,
                base_config="configs/mario_photonic.json",
                extra_args=["--n-modes", str(n_modes), "--n-photons", str(n_photons)],
                label=f"photonic m={n_modes}, p={n_photons} (dim={comb(n_modes, n_photons)})",
            )
        )
    return pts


_SWEEPS: dict[str, callable] = {
    "modes": _modes_sweep,
    "photons": _photons_sweep,
    "isodim": _isodim_sweep,
}


def _run_one(
    point: Point, seed: int, out_root: Path, extra_overrides: list[str]
) -> Path:
    """Launch ``implementation.py`` for a single (point, seed) pair."""
    point_dir = out_root / point.name / f"seed_{seed}"
    point_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "implementation.py",
        "--paper",
        "qrc_level_generation",
        "--config",
        point.base_config,
        "--seed",
        str(seed),
        "--outdir",
        str(point_dir.resolve()),
        *point.extra_args,
        *extra_overrides,
    ]
    env = os.environ.copy()
    # Reduce verbosity per sub-run (the outer sweep prints structured progress).
    env.setdefault("PYTHONUNBUFFERED", "1")
    subprocess.run(cmd, cwd=_REPO_ROOT, env=env, check=True)

    # Find the timestamped run directory produced by the shared runtime.
    candidates = sorted(point_dir.glob("run_*"))
    if not candidates:
        raise RuntimeError(f"No run_* directory under {point_dir}")
    return candidates[-1]


def _load_metrics(run_dir: Path) -> dict:
    metrics_path = run_dir / "metrics.json"
    with metrics_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _aggregate(out_root: Path, sweep_points: list[Point], seeds: list[int]) -> dict:
    """Collect every run's metrics into a single summary dict."""
    summary: dict = {"points": {}, "seeds": list(seeds)}
    for point in sweep_points:
        per_seed: list[dict] = []
        for seed in seeds:
            point_dir = out_root / point.name / f"seed_{seed}"
            # Prefer the most recent run that actually completed (has metrics.json).
            completed = [
                run
                for run in sorted(point_dir.glob("run_*"))
                if (run / "metrics.json").exists()
            ]
            if not completed:
                continue
            metrics = _load_metrics(completed[-1])
            per_seed.append(metrics)
        summary["points"][point.name] = {
            "label": point.display(),
            "base_config": point.base_config,
            "extra_args": point.extra_args,
            "per_seed_metrics": per_seed,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep",
        required=True,
        choices=sorted(_SWEEPS.keys()),
        help="Which preset sweep to run.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Where to write per-(point, seed) runs and the aggregated summary.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=3,
        help="Number of seeds (0, 1, ..., seeds-1).",
    )
    parser.add_argument(
        "--temperatures",
        type=str,
        default="0.5,1,2,5",
        help="Comma-separated temperatures forwarded to every sub-run.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=20,
        help="Generated sequences per temperature per seed.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="FNN epochs forwarded to every sub-run.",
    )
    parser.add_argument(
        "--gen-length",
        type=int,
        default=157,
        help="Length of each generated sequence.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned commands without executing them.",
    )
    args = parser.parse_args()

    sweep_points = _SWEEPS[args.sweep]()
    seeds = list(range(args.seeds))
    args.out_root.mkdir(parents=True, exist_ok=True)

    overrides = [
        "--temperatures",
        args.temperatures,
        "--n-samples",
        str(args.n_samples),
        "--epochs",
        str(args.epochs),
        "--gen-length",
        str(args.gen_length),
    ]

    for point in sweep_points:
        for seed in seeds:
            print(f"[sweep:{args.sweep}] {point.name} seed={seed}", flush=True)
            if args.dry_run:
                continue
            _run_one(point, seed, args.out_root, overrides)

    if args.dry_run:
        print("[sweep] dry run complete; skipping aggregation.")
        return

    summary = _aggregate(args.out_root, sweep_points, seeds)
    summary_path = args.out_root / "aggregated.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[sweep] wrote {summary_path}")


if __name__ == "__main__":
    main()
