"""Plot the exactness experiments: spectrum checks (E4) and the universality construction (E5).

Usage, from the paper directory::

    python utils/plot_exactness.py --spectrum-run outdir/<e4 run> --universality-run outdir/<e5 run>

``e4_spectrum_checks.png`` (saved in the E4 run): for every check, the
largest Fourier coefficient outside the predicted spectrum, against the FFT
tolerance used to call a coefficient nonzero.
``e5_universality.png`` (saved in the E5 run): RMS error of the constructed
model ``f_m`` to ``sign(sin x)`` against ``m = K``, with the exact Parseval
truncation error. Both are also saved to ``results/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from _figures import load_completed_metrics, save_to_run_and_results, stamp_scope


def plot_spectrum(run_dir: Path) -> None:
    metrics = load_completed_metrics(run_dir, "spectrum")
    checks = metrics["checks"]
    names = [check["name"] for check in checks]
    outside = [max(check["largest_coefficient_outside"], 1e-18) for check in checks]
    colors = ["tab:green" if check["passed"] else "tab:red" for check in checks]
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.bar(range(len(checks)), outside, color=colors)
    axis.axhline(
        metrics["fft_tolerance"],
        color="black",
        ls="--",
        lw=1,
        label="FFT tolerance (nonzero threshold)",
    )
    axis.set_yscale("log")
    axis.set_ylim(1e-18, 1)
    axis.set_xticks(range(len(checks)))
    axis.set_xticklabels(names, rotation=70, ha="right", fontsize=7)
    axis.set_ylabel("largest |c_ω| outside predicted Ω")
    axis.set_title(
        f"E4: {sum(c['passed'] for c in checks)}/{len(checks)} checks with support = predicted spectrum "
        f"(green = passed; {metrics['random_draws']} random draws each)",
        fontsize=9,
    )
    axis.legend(fontsize=7)
    stamp_scope(
        figure, "exact check (analytic prediction, machine precision)", str(run_dir)
    )
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    save_to_run_and_results(figure, run_dir, "e4_spectrum_checks.png")


def plot_universality(run_dir: Path) -> None:
    metrics = load_completed_metrics(run_dir, "universality")
    rows = metrics["square_wave"]
    qubits = [row["n_qubits"] for row in rows]
    figure, axis = plt.subplots(figsize=(5.5, 4))
    axis.loglog(
        qubits,
        [row["rms_to_target"] for row in rows],
        "o",
        color="tab:blue",
        label="constructed model $f_m$ (App. C)",
    )
    axis.loglog(
        qubits,
        [row["parseval_tail_rms"] for row in rows],
        "k--",
        label="exact truncation error (Parseval)",
    )
    axis.set_xlabel("subsystems m (degree K = m)")
    axis.set_ylabel("RMS error to sign(sin x)")
    worst = max(record["max_abs_error"] for record in metrics["exact"])
    axis.set_title(
        f"E5: square-wave convergence; random series rebuilt to {worst:.0e}", fontsize=9
    )
    axis.legend(fontsize=7)
    stamp_scope(figure, "exact check (proof construction)", str(run_dir))
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    save_to_run_and_results(figure, run_dir, "e5_universality.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spectrum-run", type=Path, default=None)
    parser.add_argument("--universality-run", type=Path, default=None)
    args = parser.parse_args()
    if args.spectrum_run is None and args.universality_run is None:
        parser.error("give --spectrum-run and/or --universality-run")
    if args.spectrum_run is not None:
        plot_spectrum(args.spectrum_run)
    if args.universality_run is not None:
        plot_universality(args.universality_run)


if __name__ == "__main__":
    main()
