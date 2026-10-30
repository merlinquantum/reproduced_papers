"""Compare a seed-12345 QuaRK run with the authors' released run behind Figs. 2-4.

Checks, task by task (current-point readout, exponential fading, Volterra):

* the tuned Matérn kernel (nu, xi) against ``model/meta.json``;
* Fig. 2: the training and test MSE over the 26 ridge values against ``reg_sweep.npz``
  (training MSE below ``ROUNDING_LEVEL`` is compared in absolute terms);
* Fig. 4: the test MSE over the 10 training sizes and the ridge used, with the ridge
  chosen on the test set (the authors' rule), against ``artifacts/train_size_sweep.pkl``
  (read with an unpickler that only accepts numpy arrays);
* the features, if the run saved them, against ``Phi_full`` (both rounded to float32,
  the precision of the saved features).

Usage (from the repository root)::

    python papers/QuaRK/utils/compare_with_release.py papers/QuaRK/outdir/<run> \\
        --release-run data/QuaRK/authors_release_v1.0.0/storage/results/reg_sweep/c-shadows/full/N=10000__w=25__d=3__s=100/2026-02-02_14-02-13

Writes ``results/release_comparison.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from plot_style import PAPER_TASKS, RESULTS_DIR, load_release, load_run

ROUNDING_LEVEL = 1e-10


def max_relative_difference(ours, theirs) -> float:
    ours, theirs = np.asarray(ours, dtype=float), np.asarray(theirs, dtype=float)
    return float(np.max(np.abs(ours - theirs) / np.abs(theirs)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--release-run", type=Path, required=True)
    args = parser.parse_args()

    metrics, arrays = load_run(args.run_dir)
    if metrics["dataset_meta"]["seed"] != 12345:
        raise SystemExit("the release is the authors' seed-12345 run")
    release = load_release(args.release_run)
    meta = json.loads((args.release_run / "model" / "meta.json").read_text())
    with np.load(args.release_run / "reg_sweep.npz", allow_pickle=False) as data:
        release_test_curve = data["mse_test"]
    sweep, sizes = arrays["regularization_sweep"], arrays["training_size_sweep"]
    if not np.array_equal(sweep["reg_grid"], release["reg_grid"]):
        raise SystemExit("ridge grids differ")
    if not np.array_equal(sizes["sizes"], release["sizes"]):
        raise SystemExit("training-size grids differ")

    report = {"run": args.run_dir.name, "tasks": {}}
    for i, task in enumerate(PAPER_TASKS):
        kernel = metrics["tasks"][task]["kernel"]
        theirs = meta["best_params_"][i]
        selection = metrics["tasks"][task]["selections"]["test"]
        above = release["mse_train"][i] > ROUNDING_LEVEL
        report["tasks"][task] = {
            "nu": {"ours": kernel["nu"], "release": theirs["nu"]},
            "xi": {"ours": kernel["xi"], "release": theirs["xi"]},
            "xi_relative_difference": abs(kernel["xi"] - theirs["xi"]) / theirs["xi"],
            # Below ROUNDING_LEVEL the training MSE is floating-point noise (interpolation
            # regime), so compare it in absolute terms there.
            "fig2_training_mse_max_relative_difference_above_rounding_level": (
                max_relative_difference(
                    sweep[f"{task}__mse_train"][above], release["mse_train"][i][above]
                )
            ),
            "fig2_training_mse_max_abs_difference_below_rounding_level": float(
                np.max(
                    np.abs(
                        sweep[f"{task}__mse_train"][~above]
                        - release["mse_train"][i][~above]
                    )
                )
            ),
            "fig2_test_mse_max_relative_difference": max_relative_difference(
                sweep[f"{task}__mse_test"], release_test_curve[i]
            ),
            "fig2_training_mse_at_largest_ridge": {
                "ours": float(sweep[f"{task}__mse_train"][-1]),
                "release": float(release["mse_train"][i][-1]),
            },
            "fig4_test_mse_max_relative_difference": max_relative_difference(
                sizes[f"{task}__test__mse_test"], release["size_mse_test"][i]
            ),
            "fig4_test_mse_first_last": {
                "ours": [
                    float(sizes[f"{task}__test__mse_test"][0]),
                    float(sizes[f"{task}__test__mse_test"][-1]),
                ],
                "release": [
                    float(release["size_mse_test"][i][0]),
                    float(release["size_mse_test"][i][-1]),
                ],
            },
            "fig4_ridge": {
                "ours": selection["reg"],
                "release": release["lambda_regs"][i],
            },
        }
    features = args.run_dir / "features.npz"
    if features.is_file():
        with np.load(features, allow_pickle=False) as data:
            (ours,) = (data[k] for k in data.files)
        with np.load(
            args.release_run / "model" / "arrays.npz", allow_pickle=False
        ) as data:
            theirs = data["Phi_full"].astype(np.float32)
        report["features_max_abs_difference_float32"] = float(
            np.max(np.abs(ours - theirs))
        )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "release_comparison.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
