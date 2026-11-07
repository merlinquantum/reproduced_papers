"""One-stop post-processing: tables + Pareto + scaling figures from the three sweeps.

Run after all three sweep.py invocations have completed and produced
``sweeps/<name>/aggregated.json``.

Writes:
    results/sweep_pareto.png
    results/sweep_scaling_modes.png
    results/sweep_scaling_photons.png
    results/sweep_scaling_isodim.png
    results/sweep_tables.md
"""

from __future__ import annotations

import argparse
import io
import subprocess
import sys
from pathlib import Path

_PAPER_DIR = Path(__file__).resolve().parents[1]
_UTILS_DIR = _PAPER_DIR / "utils"
_RESULTS_DIR = _PAPER_DIR / "results"
_SWEEPS_DIR = _PAPER_DIR / "sweeps"


def _capture(cmd: list[str]) -> str:
    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return res.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=_RESULTS_DIR / "reference_eval_metrics.json",
        help="Metrics file with the published-Aer reference curve to overlay on Pareto.",
    )
    args = parser.parse_args()

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    sweeps = {
        "modes": _SWEEPS_DIR / "modes" / "aggregated.json",
        "photons": _SWEEPS_DIR / "photons" / "aggregated.json",
        "isodim": _SWEEPS_DIR / "isodim" / "aggregated.json",
    }

    for name, path in sweeps.items():
        if not path.exists():
            print(f"[finalise] missing {path}; skipping {name}")

    # 1) Combined Pareto figure.
    available = [str(p) for p in sweeps.values() if p.exists()]
    if available:
        cmd = [
            sys.executable,
            str(_UTILS_DIR / "plot_pareto.py"),
            "--aggregated",
            *available,
            "--out",
            str(_RESULTS_DIR / "sweep_pareto.png"),
            "--title",
            "QRC Pareto front: originality(L=2) vs broken-rate (n_seeds=3)",
        ]
        if args.reference.exists():
            cmd.extend(["--reference", str(args.reference)])
        subprocess.run(cmd, check=True)

    # 2) Scaling figures (one per sweep, plus combined).
    if sweeps["modes"].exists():
        subprocess.run(
            [
                sys.executable,
                str(_UTILS_DIR / "plot_scaling.py"),
                "--aggregated",
                str(sweeps["modes"]),
                "--labels",
                "photonic, p=3, modes ∈ {4,6,8}",
                "--temperature",
                "1.0",
                "--out",
                str(_RESULTS_DIR / "sweep_scaling_modes.png"),
                "--title",
                "Sweep A: vary modes at fixed photons (T=1)",
            ],
            check=True,
        )

    if sweeps["photons"].exists():
        subprocess.run(
            [
                sys.executable,
                str(_UTILS_DIR / "plot_scaling.py"),
                "--aggregated",
                str(sweeps["photons"]),
                "--labels",
                "photonic, modes=6, p ∈ {1,2,3}",
                "--temperature",
                "1.0",
                "--out",
                str(_RESULTS_DIR / "sweep_scaling_photons.png"),
                "--title",
                "Sweep B: vary photons at fixed modes (T=1)",
            ],
            check=True,
        )

    if sweeps["isodim"].exists():
        # Split isodim into qubit and photonic series for clearer comparison.
        # For now, plot all together with a single label.
        subprocess.run(
            [
                sys.executable,
                str(_UTILS_DIR / "plot_scaling.py"),
                "--aggregated",
                str(sweeps["isodim"]),
                "--labels",
                "isodim (qubit + photonic)",
                "--temperature",
                "1.0",
                "--out",
                str(_RESULTS_DIR / "sweep_scaling_isodim.png"),
                "--title",
                "Sweep C: iso-output-dim, qubit vs photonic (T=1)",
            ],
            check=True,
        )

    # 3) Markdown tables.
    table_md = io.StringIO()
    table_md.write("# QRC scaling sweep tables\n\n")
    for name, path in sweeps.items():
        if not path.exists():
            continue
        table_md.write(f"\n## Sweep: {name}\n")
        text = _capture(
            [
                sys.executable,
                str(_UTILS_DIR / "print_sweep_table.py"),
                "--aggregated",
                str(path),
                "--temperatures",
                "0.5",
                "1.0",
                "2.0",
                "5.0",
            ]
        )
        table_md.write(text)
        table_md.write("\n")
    (_RESULTS_DIR / "sweep_tables.md").write_text(table_md.getvalue(), encoding="utf-8")

    print(f"\n[finalise] artefacts written under {_RESULTS_DIR}")


if __name__ == "__main__":
    main()
