"""Summarise the coefficient (E3), spectrum (E4) and universality (E5) runs.

Usage, from the paper directory::

    python utils/summarize_exact.py --run outdir/run_A --output results/e4_spectrum_summary

The experiment kind is read from the run's ``metrics.json``. Writes
``<output>.json`` (full precision) and ``<output>.md``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(run_dir: Path) -> dict:
    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    if status["status"] != "COMPLETED":
        raise ValueError(f"{run_dir} is not COMPLETED: {status['status']}")
    return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))


def _spectrum_markdown(metrics: dict, run_dir: Path) -> tuple[dict, str]:
    checks = metrics["checks"]
    passed = sum(check["passed"] for check in checks)
    largest_outside = max(check["largest_coefficient_outside"] for check in checks)
    summary = {
        "checks_passed": passed,
        "checks_total": len(checks),
        "largest_coefficient_outside_any_prediction": largest_outside,
        "bound_table": metrics["bound_table"],
        "all_passed": metrics["all_passed"],
    }
    lines = [
        f"Run `{run_dir}`: {passed}/{len(checks)} checks passed; largest coefficient "
        f"outside a predicted spectrum {largest_outside:.2e}; FFT tolerance "
        f"{metrics['fft_tolerance']:g}; {metrics['random_draws']} random draws per check.",
        "",
        "| Check | Predicted spectrum | Observed support = prediction | Max outside | Passed |",
        "|---|---|---|---:|---|",
    ]
    for check in checks:
        predicted = (
            f"{len(check['predicted'])} frequencies, max {max(check['predicted'])}"
            if "predicted" in check
            else f"{check['predicted_size']} frequency pairs"
        )
        lines.append(
            f"| {check['name']} | {predicted} | {check['equal']} | "
            f"{check['largest_coefficient_outside']:.1e} | {check['passed']} |"
        )
    lines += [
        "",
        "| Dense case | d | L | K | Paper bound d^(2L)/2 − 1 | Distinct-sum bound m(m−1)/2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in metrics["bound_table"]:
        lines.append(
            f"| {row['case']} | {row['dimension']} | {row['layers']} | {row['spectrum_size']} | "
            f"{row['paper_bound']:g} | {row['distinct_sum_bound']} |"
        )
    return summary, "\n".join(lines)


def _setting_degree(metrics: dict, setting: dict) -> int:
    """Return a setting's predicted degree.

    Phase 4 metrics record it per setting. Phase 3 metrics (qubit-only)
    record one ``predicted_degree`` for the whole run.
    """
    if "predicted_degree" in setting:
        return int(setting["predicted_degree"])
    return int(metrics["predicted_degree"])


def _coefficient_markdown(metrics: dict, run_dir: Path) -> tuple[dict, str]:
    lines = [
        f"Run `{run_dir}`: {metrics['n_samples']} random parameter draws per setting, "
        f"seed {metrics['seed']}. Entry: Var(Re c_n) + Var(Im c_n); "
        f"'0' marks a coefficient with |c_n| <= {metrics['zero_tolerance']:g} in every draw.",
        "",
        "| Setting | Predicted degree | Params | "
        + " | ".join(f"c_{n}" for n in range(metrics["max_frequency"] + 1))
        + " |",
        "|---|---:|---:|" + "---:|" * (metrics["max_frequency"] + 1),
    ]
    for setting in metrics["settings"]:
        cells = []
        for entry in setting["per_frequency"]:
            if entry["fraction_zero"] == 1.0:
                cells.append("0")
            else:
                cells.append(f"{entry['var_total']:.2e}")
        lines.append(
            f"| {setting['label']} | {_setting_degree(metrics, setting)} | {setting['n_parameters']} | "
            + " | ".join(cells)
            + " |"
        )
    summary = {"settings": metrics["settings"]}
    return summary, "\n".join(lines)


def _universality_markdown(metrics: dict, run_dir: Path) -> tuple[dict, str]:
    worst_exact = max(record["max_abs_error"] for record in metrics["exact"])
    worst_relative = max(
        abs(row["rms_to_target"] - row["parseval_tail_rms"]) / row["parseval_tail_rms"]
        for row in metrics["square_wave"]
    )
    summary = {
        "worst_exact_reconstruction_error": worst_exact,
        "worst_relative_gap_to_parseval": worst_relative,
        "exact": metrics["exact"],
        "square_wave": metrics["square_wave"],
    }
    lines = [
        f"Run `{run_dir}`: family {metrics['generator_family']}; "
        f"{metrics['evaluation_points']} evaluation points.",
        "",
        f"Exact reconstruction of random truncated series: worst max |f_m − g| = {worst_exact:.1e}.",
        "",
        "| Degree K | Qubits m | max abs error |",
        "|---:|---:|---:|",
    ]
    for record in metrics["exact"]:
        lines.append(
            f"| {record['degree']} | {record['n_qubits']} | {record['max_abs_error']:.1e} |"
        )
    lines += [
        "",
        f"Square wave sign(sin x): worst relative gap between the measured RMS error and "
        f"the Parseval truncation error = {worst_relative:.1e}.",
        "",
        "| m = K | RMS error to sign(sin x) | Parseval truncation error | max |f_m − truncation| |",
        "|---:|---:|---:|---:|",
    ]
    for row in metrics["square_wave"]:
        lines.append(
            f"| {row['n_qubits']} | {row['rms_to_target']:.4e} | {row['parseval_tail_rms']:.4e} | "
            f"{row['max_abs_error_to_truncation']:.1e} |"
        )
    return summary, "\n".join(lines)


def _photonic_spectrum_markdown(metrics: dict, run_dir: Path) -> tuple[dict, str]:
    checks = metrics["checks"]
    passed = sum(check["passed"] for check in checks)
    lines = [
        f"Run `{run_dir}` (MerLin adaptation): {passed}/{len(checks)} checks behave as predicted; "
        f"{metrics['random_draws']} random draws, {metrics['grid_points']}-point grid, FFT tolerance "
        f"{metrics['fft_tolerance']:g}. FOCK cases must be band-limited to the photon-number prediction; "
        "UNBUNCHED (postselected, renormalised) is expected not to be.",
        "",
        "| Check | Space | n | Modes | Encoded modes | L | Predicted degree | Observed degree | Max outside | Passed |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for check in checks:
        lines.append(
            f"| {check['name']} | {check['computation_space']} | {check['n_photons']} | {check['n_modes']} | "
            f"{check['encoded_modes']} | {check['encoding_layers']} | {max(check['predicted'])} | "
            f"{max(check['observed_union'])} | {check['largest_coefficient_outside']:.1e} | {check['passed']} |"
        )
    return {
        "checks_passed": passed,
        "checks_total": len(checks),
        "checks": checks,
    }, "\n".join(lines)


def _coefficient_reach_markdown(metrics: dict, run_dir: Path) -> tuple[dict, str]:
    lines = [
        f"Run `{run_dir}` (diagnostic): largest |c_k| found by maximisation (Adam lr {metrics['lr']:g}, "
        f"{metrics['steps']} steps, {len(metrics['init_seeds'])} random inits). A lower bound on the "
        "reachable value, and only a necessary condition for fitting: the other coefficients must also match.",
        "",
        "| Case | Kind | Model degree | k | Best \\|c_k\\| | Median \\|c_k\\| | Target \\|c_k\\| | Reached |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for case in metrics["cases"]:
        lines.append(
            f"| {case['name']} | {case['kind']} | {case['model_degree']} | {case['frequency']} | "
            f"{case['best_abs_coefficient']:.4f} | {case['median_abs_coefficient']:.4f} | "
            f"{case['target_abs_coefficient']:.4f} | {case['target_reached']} |"
        )
    return {"cases": metrics["cases"]}, "\n".join(lines)


_RENDERERS = {
    "spectrum": _spectrum_markdown,
    "coefficients": _coefficient_markdown,
    "universality": _universality_markdown,
    "photonic_spectrum": _photonic_spectrum_markdown,
    "coefficient_reach": _coefficient_reach_markdown,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, type=Path, help="Run directory")
    parser.add_argument(
        "--output", required=True, type=Path, help="Output path without suffix"
    )
    args = parser.parse_args()
    metrics = _load(args.run)
    renderer = _RENDERERS[metrics["experiment"]]
    summary, markdown = renderer(metrics, args.run)
    summary = {"run_dir": str(args.run), "experiment": metrics["experiment"], **summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    args.output.with_suffix(".md").write_text(markdown + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
