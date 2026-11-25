"""Regenerate every curated result of the reproduction from the stored runs.

Runs, in order: the forecast noise floor, the statistics, the comparisons with the
authors' run, with the photonic version and of all models together, the result
tables, then every figure. All inputs have fixed default locations: the four sweeps
under ``outdir/`` and the authors' release under the repository data root. Only the
single-seed run is passed, because its folder name carries a timestamp.

Usage (from the repository root)::

    python papers/QuaRK/utils/make_results.py --run papers/QuaRK/outdir/run_<timestamp>

Writes ``results/`` (and each figure also in its run or sweep folder).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

UTILS_DIR = Path(__file__).resolve().parent
PAPER_DIR = UTILS_DIR.parent
REPO_ROOT = PAPER_DIR.parents[1]
RESULTS_DIR = PAPER_DIR / "results"

RELEASE_RUN_RELPATH = (
    "QuaRK/authors_release_v1.0.0/storage/results/reg_sweep/c-shadows/full/"
    "N=10000__w=25__d=3__s=100/2026-02-02_14-02-13"
)
SWEEPS = ("main_comparison", "shot_noise_ablation", "scaling", "photonic_adaptation")

# Scope labels printed in the figure titles (result-scope rule of the reproduction).
SINGLE_SEED_LABEL = "preliminary: single seed 12345, identical to the authors' run"
PAPER_SETTINGS_LABEL = "paper-accurate settings, 5 seeds"
EXTENSION_LABEL = "extension, 5 seeds"


def build_commands(run: Path, outdir: Path, release_run: Path) -> list[list[str]]:
    """Return the commands to run, in order, as argument lists without the interpreter.

    Parameters
    ----------
    run : Path
        Single-seed run at the paper settings (seed 12345), behind Figs. 3–5.
    outdir : Path
        Folder holding ``sweep_<name>`` for the four sweeps.
    release_run : Path
        The authors' released run folder.

    Returns
    -------
    list of list of str
        One argument list per script call; the first item is the script path.
    """
    sweep = {name: str(outdir / f"sweep_{name}") for name in SWEEPS}
    run, release_run = str(run), str(release_run)

    def script(name: str) -> str:
        return str(UTILS_DIR / name)

    return [
        [
            script("forecast_noise_floor.py"),
            sweep["main_comparison"],
            "--candidate",
            "quark",
        ],
        [
            script("report_statistics.py"),
            "--main",
            sweep["main_comparison"],
            "--shot-noise",
            sweep["shot_noise_ablation"],
            "--scaling",
            sweep["scaling"],
            "--noise-floor",
            str(RESULTS_DIR / "forecast_noise_floor.json"),
        ],
        [script("compare_with_release.py"), run, "--release-run", release_run],
        [
            script("compare_photonic_with_quark.py"),
            "--photonic",
            sweep["photonic_adaptation"],
            "--main",
            sweep["main_comparison"],
            "--shot-noise",
            sweep["shot_noise_ablation"],
        ],
        [
            script("compare_fair_baselines.py"),
            "--main",
            sweep["main_comparison"],
            "--photonic",
            sweep["photonic_adaptation"],
            "--shot-noise",
            sweep["shot_noise_ablation"],
            "--scaling",
            sweep["scaling"],
            "--release-comparison",
            str(RESULTS_DIR / "release_comparison.json"),
        ],
        [
            script("build_result_tables.py"),
            "--run",
            run,
            "--release-run",
            release_run,
            "--main",
            sweep["main_comparison"],
            "--shot-noise",
            sweep["shot_noise_ablation"],
            "--scaling",
            sweep["scaling"],
            "--photonic",
            sweep["photonic_adaptation"],
        ],
        [
            script("plot_regularization_sweep.py"),
            run,
            "--label",
            SINGLE_SEED_LABEL,
            "--release-run",
            release_run,
        ],
        [script("plot_fit_examples.py"), run, "--label", SINGLE_SEED_LABEL],
        [
            script("plot_training_size_sweep.py"),
            run,
            "--label",
            SINGLE_SEED_LABEL,
            "--release-run",
            release_run,
        ],
        [
            script("plot_training_size_seeds.py"),
            sweep["main_comparison"],
            "--candidate",
            "quark",
            "--label",
            "paper-accurate settings, 5 seeds (12345 is the authors' data)",
        ],
        [
            script("plot_model_comparison.py"),
            sweep["main_comparison"],
            "--label",
            PAPER_SETTINGS_LABEL,
        ],
        [
            script("plot_model_comparison.py"),
            sweep["main_comparison"],
            "--label",
            "paper-accurate settings, 5 seeds; photonic: adaptation, simulated",
            "--photonic",
            sweep["photonic_adaptation"],
        ],
        [
            script("plot_shot_budget.py"),
            sweep["main_comparison"],
            sweep["shot_noise_ablation"],
            "--label",
            EXTENSION_LABEL,
        ],
        [
            script("plot_scaling.py"),
            sweep["main_comparison"],
            sweep["scaling"],
            "--label",
            EXTENSION_LABEL,
        ],
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run", type=Path, required=True, help="single-seed run behind Figs. 3–5"
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=PAPER_DIR / "outdir",
        help="folder holding the four sweeps",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="data root (default: DATA_DIR or <repo>/data)",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT))
    from runtime_lib.data_paths import resolve_data_root

    release_run = resolve_data_root(args.data_root) / RELEASE_RUN_RELPATH
    required = [args.run, release_run] + [
        args.outdir / f"sweep_{name}" for name in SWEEPS
    ]
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise SystemExit("missing input folders:\n  " + "\n  ".join(missing))

    for command in build_commands(
        args.run.resolve(), args.outdir.resolve(), release_run
    ):
        print("+", Path(command[0]).name, *command[1:], flush=True)
        subprocess.run([sys.executable, *command], check=True)


if __name__ == "__main__":
    main()
