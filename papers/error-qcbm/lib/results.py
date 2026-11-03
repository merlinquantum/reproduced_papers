"""Validate, summarize and redraw the curated Figure 2 data without raw runs."""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

CONDITIONS = ("lossless", "postselection", "recycling")
BELENOS = (
    "belenos_postselection",
    "belenos_recycling",
    "lossless_partial",
    "lossless_ideal",
    "belenos_loss_recycling",
    "belenos_overlap084_recycling",
    "lossless_overlap084",
)
PANEL_F = BELENOS[1:]
SCHEMA = "error_qcbm.figure2.v1"


def _read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def load_result_bundle(directory: Path) -> dict:
    """Load and validate the immutable historical result bundle.

    Parameters
    ----------
    directory : pathlib.Path
        Curated results folder containing the manifest and its three CSV inputs.

    Returns
    -------
    dict
        Manifest, typed run records, per-run histories and financial histogram.
        This does not import a sampler, access a dataset or restore checkpoints.

    Raises
    ------
    ValueError
        If hashes, inventory, history grids, retry lineage or selection disagree.
    """
    directory = Path(directory)
    manifest = json.loads(
        (directory / "figure2_manifest.json").read_text(encoding="utf-8")
    )
    if manifest["schema"] != SCHEMA:
        raise ValueError("Unsupported Figure 2 bundle schema")
    for name, expected in manifest["input_sha256"].items():
        if Path(name).name != name:
            raise ValueError("Result inputs must be plain filenames")
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Result input SHA256 mismatch: {name}")
    runs = _read_csv(directory / "runs.csv")
    by_id = {}
    for row in runs:
        for name in (
            "seed",
            "completed_steps",
            "last_recorded_step",
            "shots",
            "committed_objective_calls",
        ):
            row[name] = int(row[name])
        for name in (
            "transmittance",
            "indistinguishability",
            "calibrated_gain",
            "last_finite_loss",
            "last_exact_lossless_same_I",
        ):
            row[name] = float(row[name])
        if row["run_id"] in by_id:
            raise ValueError("Duplicate result run identifier")
        by_id[row["run_id"]] = row
    if len(runs) != 263 or Counter(row["status"] for row in runs) != {
        "complete": 260,
        "failed": 3,
    }:
        raise ValueError("Expected 263 attempts, retaining three original failures")
    groups = defaultdict(list)
    for row in runs:
        groups[row["experiment"], row["condition"], row["attempt"]].append(row)
    expected_groups = {
        (experiment, condition, "original")
        for experiment in ("gaussian", "financial")
        for condition in CONDITIONS
    }
    expected_groups |= {("belenos", condition, "original") for condition in BELENOS}
    expected_groups |= {
        ("belenos", condition, "sampling_retry")
        for condition in (
            "belenos_recycling",
            "belenos_loss_recycling",
            "belenos_overlap084_recycling",
        )
    }
    if set(groups) != expected_groups:
        raise ValueError("Missing or unexpected experiment/condition/attempt group")
    for (experiment, _condition, attempt), rows in groups.items():
        expected_seeds = [8] if attempt == "sampling_retry" else list(range(20))
        if sorted(row["seed"] for row in rows) != expected_seeds:
            raise ValueError("Each original group must retain all 20 planned seeds")
        shots = {"gaussian": 200000, "financial": 60000, "belenos": 50000000}[
            experiment
        ]
        metric = "tvd" if experiment == "financial" else "kl"
        if any(row["shots"] != shots or row["loss"] != metric for row in rows):
            raise ValueError("Result shot budget or loss convention disagrees")
    histories = defaultdict(list)
    for row in _read_csv(directory / "curves.csv"):
        if row["run_id"] not in by_id:
            raise ValueError("Curve references an unknown run")
        record = {
            "update": int(row["update"]),
            "finite_shot_loss": float(row["finite_shot_loss"]),
            "exact_lossless_same_I": float(row["exact_lossless_same_I"]),
            "objective_calls": int(row["objective_calls"]),
        }
        if not np.isfinite(
            [record["finite_shot_loss"], record["exact_lossless_same_I"]]
        ).all():
            raise ValueError("Curve contains a nonfinite metric")
        histories[row["run_id"]].append(record)
    if set(histories) != set(by_id):
        raise ValueError("Every original failure and retry must retain its history")
    for row in runs:
        history = histories[row["run_id"]]
        expected_grid = (
            list(range(0, 1001, 10))
            if row["status"] == "complete"
            else list(range(0, 681, 10))
        )
        if [h["update"] for h in history] != expected_grid:
            raise ValueError("Recorded history grid is incomplete or duplicated")
        last = history[-1]
        if (
            last["update"] != row["last_recorded_step"]
            or last["finite_shot_loss"] != row["last_finite_loss"]
            or last["exact_lossless_same_I"] != row["last_exact_lossless_same_I"]
        ):
            raise ValueError("Run endpoint disagrees with finite-shot/exact history")
        if row["status"] == "failed":
            if (
                row["attempt"] != "original"
                or row["seed"] != 8
                or row["completed_steps"] != 683
                or row["committed_objective_calls"] != 1437
            ):
                raise ValueError("Unexpected original failure provenance")
        elif row["completed_steps"] != 1000 or row["committed_objective_calls"] != 2103:
            raise ValueError("Completed run has the wrong update/call count")
        if row["attempt"] == "sampling_retry":
            parent = by_id.get(row["parent_run_id"])
            if (
                parent is None
                or parent["status"] != "failed"
                or parent["condition"] != row["condition"]
                or row["branch_step"] != "683"
            ):
                raise ValueError("Sampling retry must reference its failed original")
            if (
                row["calibrated_gain"] != parent["calibrated_gain"]
                or history[:69] != histories[parent["run_id"]]
            ):
                raise ValueError("Retry must preserve inherited gain and history")
    histogram = _read_csv(directory / "financial_histogram.csv")
    if [int(row["bin"]) for row in histogram] != list(range(495)):
        raise ValueError("Financial histogram must retain all 495 model bins")
    h = {
        name: np.array([float(row[name]) for row in histogram])
        for name in (
            "left",
            "right",
            "target_count",
            "available_qcbm_count",
            "displayed_qcbm_count",
        )
    }
    for name in ("target_count", "available_qcbm_count", "displayed_qcbm_count"):
        if (
            not np.isfinite(h[name]).all()
            or (h[name] < 0).any()
            or not np.equal(h[name], np.floor(h[name])).all()
        ):
            raise ValueError("Histogram input must contain nonnegative integer counts")
        h[name] = h[name].astype(np.int64)
    if (
        not np.array_equal(h["left"][1:], h["right"][:-1])
        or not (h["right"] > h["left"]).all()
    ):
        raise ValueError("Histogram edges are not contiguous")
    if (
        h["target_count"].sum() != 5123
        or h["displayed_qcbm_count"].sum() != 5123
        or h["available_qcbm_count"].sum() != 39219
        or (h["displayed_qcbm_count"] > h["available_qcbm_count"]).any()
    ):
        raise ValueError("Panel d must use equal-size subsets of stored native counts")
    bundle = {
        "manifest": manifest,
        "runs": runs,
        "by_id": by_id,
        "histories": dict(histories),
        "histogram": h,
    }
    selected = choose_panel_f(bundle)
    if (
        manifest["panels"]["f"]["seed"] != selected
        or manifest["panels"]["e"]["seeds"] != [0, 1, 2]
        or manifest["panels"]["d"]["seed"] != 0
    ):
        raise ValueError("Manifest violates the fixed panel seed policy")
    if manifest["panels"]["f"]["conditions"] != list(PANEL_F):
        raise ValueError("Panel f must retain six matched simulation conditions")
    return bundle


def select_runs(
    bundle: dict, experiment: str, condition: str, *, supplemented: bool = False
) -> list[dict]:
    """Select complete original runs or an explicitly retry-supplemented set.

    Parameters
    ----------
    bundle : dict
        Validated data bundle.
    experiment, condition : str
        Exact group identifiers.
    supplemented : bool
        Include separate retry for a failed seed, preserving original rows in
        inventory. Default value is False.

    Returns
    -------
    list of dict
        One complete trajectory per seed, sorted by seed.
    """
    rows = [
        row
        for row in bundle["runs"]
        if row["experiment"] == experiment
        and row["condition"] == condition
        and row["status"] == "complete"
        and (row["attempt"] == "original" or supplemented)
    ]
    if len({row["seed"] for row in rows}) != len(rows):
        raise ValueError("Cannot pool duplicate original/retry initializations")
    return sorted(rows, key=lambda row: row["seed"])


def finite_curves(bundle: dict, rows: list[dict]) -> np.ndarray:
    """Return recorded finite-shot losses after updates 10..1000.

    Parameters
    ----------
    bundle : dict
        Validated data bundle.
    rows : list of dict
        Complete run records to plot or aggregate.

    Returns
    -------
    numpy.ndarray
        Shape ``(runs,100)``; the additional initial report is excluded.
    """
    if not rows or any(row["status"] != "complete" for row in rows):
        raise ValueError("Plot statistics require complete trajectories")
    return np.asarray(
        [
            [h["finite_shot_loss"] for h in bundle["histories"][row["run_id"]]][1:]
            for row in rows
        ]
    )


def choose_panel_f(bundle: dict) -> int:
    """Choose the original I=.729 recycling best seed, excluding retries.

    Parameters
    ----------
    bundle : dict
        Data bundle with run endpoints.

    Returns
    -------
    int
        Seed with minimum final finite-shot KL; seed breaks exact ties.
    """
    rows = select_runs(bundle, "belenos", "belenos_recycling")
    return min(rows, key=lambda row: (row["last_finite_loss"], row["seed"]))["seed"]


def summarize_results(bundle: dict) -> dict:
    """Calculate population SD, paired comparisons and selection from CSV data.

    Parameters
    ----------
    bundle : dict
        Validated result bundle.

    Returns
    -------
    dict
        Original and supplementary statistics; exact diagnostics stay separate.
    """
    summary = {
        "schema": SCHEMA,
        "attempts": len(bundle["runs"]),
        "original_attempts": 260,
        "original_failed": 3,
        "separate_retries": 3,
        "metric_scope": "finite-shot training/reporting; exact lossless-at-same-I is a separate diagnostic",
        "std_ddof": 0,
        "groups": {},
        "comparisons": {},
    }
    for experiment, conditions in (
        ("gaussian", CONDITIONS),
        ("financial", CONDITIONS),
        ("belenos", BELENOS),
    ):
        for condition in conditions:
            rows = select_runs(bundle, experiment, condition)
            values = np.array([row["last_finite_loss"] for row in rows])
            exact = np.array([row["last_exact_lossless_same_I"] for row in rows])
            original = [
                row
                for row in bundle["runs"]
                if row["experiment"] == experiment
                and row["condition"] == condition
                and row["attempt"] == "original"
            ]
            group = {
                "planned_original": 20,
                "complete_original": len(rows),
                "failed_original": 20 - len(rows),
                "original_complete_seeds": [row["seed"] for row in rows],
                "finite_mean": float(values.mean()),
                "finite_std_ddof0": float(values.std(ddof=0)),
                "exact_diagnostic_mean": float(exact.mean()),
                "exact_diagnostic_std_ddof0": float(exact.std(ddof=0)),
                "best5_seeds_by_final_finite_loss": [
                    row["seed"]
                    for row in sorted(
                        rows, key=lambda row: (row["last_finite_loss"], row["seed"])
                    )[:5]
                ],
                "failed_seed_last_recorded_is_not_final": [
                    row["seed"] for row in original if row["status"] == "failed"
                ],
            }
            supplemented = select_runs(bundle, experiment, condition, supplemented=True)
            if len(supplemented) != len(rows):
                v = np.array([row["last_finite_loss"] for row in supplemented])
                group["retry_supplemented"] = {
                    "complete_initializations": len(supplemented),
                    "finite_mean": float(v.mean()),
                    "finite_std_ddof0": float(v.std()),
                    "scope": "Includes an explicitly changed sampling stream; original failure retained",
                }
            summary["groups"][f"{experiment}/{condition}"] = group
    comparisons = [
        ("gaussian", "recycling", "postselection"),
        ("financial", "recycling", "postselection"),
        ("belenos", "belenos_recycling", "belenos_postselection"),
        ("belenos", "belenos_overlap084_recycling", "belenos_recycling"),
        ("belenos", "belenos_overlap084_recycling", "belenos_loss_recycling"),
        ("belenos", "lossless_overlap084", "lossless_partial"),
        ("belenos", "lossless_overlap084", "lossless_ideal"),
    ]
    for experiment, treatment, reference in comparisons:
        a = {row["seed"]: row for row in select_runs(bundle, experiment, treatment)}
        b = {row["seed"]: row for row in select_runs(bundle, experiment, reference)}
        seeds = sorted(a.keys() & b.keys())
        delta = np.array(
            [b[s]["last_finite_loss"] - a[s]["last_finite_loss"] for s in seeds]
        )
        exact_delta = np.array(
            [
                b[s]["last_exact_lossless_same_I"] - a[s]["last_exact_lossless_same_I"]
                for s in seeds
            ]
        )
        summary["comparisons"][f"{experiment}/{treatment}_vs_{reference}"] = {
            "seeds": seeds,
            "n": len(seeds),
            "finite_treatment_wins": int((delta > 0).sum()),
            "finite_mean_reference_minus_treatment": float(delta.mean()),
            "exact_diagnostic_treatment_wins": int((exact_delta > 0).sum()),
            "exact_diagnostic_mean_reference_minus_treatment": float(
                exact_delta.mean()
            ),
            "scope": "Paired complete originals only; separate retries excluded",
        }
    seed = choose_panel_f(bundle)
    summary["panel_f"] = {
        "seed": seed,
        "selection": "Lowest final finite-shot KL of original noisy I=.729 recycling among 19 complete originals; seed retained across controls",
        "final_finite_kl": {
            name: next(
                row["last_finite_loss"]
                for row in select_runs(bundle, "belenos", name)
                if row["seed"] == seed
            )
            for name in PANEL_F
        },
    }
    return summary


def _make_panel_figures(bundle: dict, parent=None) -> dict:
    """Build Figure 2 counterparts from curated values with no new sampling.

    Parameters
    ----------
    bundle : dict
        Validated CSV/manifest bundle.
    parent : matplotlib.figure.Figure or None, optional
        Draw the three panel pairs into subfigures of this parent when supplied.
        Default is None, preserving the independent pair layouts.

    Returns
    -------
    dict
        Matplotlib figures keyed by their established stems. Caller owns closing.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(1, 101)
    colors = {"lossless": "#00b000", "postselection": "#ff8000", "recycling": "#0080ff"}
    labels = {
        "lossless": "Lossless",
        "postselection": "Lossy",
        "recycling": "Lossy with recycling mitigation",
    }
    row_figures = parent.subfigures(3, 1, hspace=0.02) if parent is not None else None
    if row_figures is None:
        gaussian, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    else:
        gaussian = row_figures[0]
        axes = gaussian.subplots(1, 2)
    for name in CONDITIONS:
        rows = select_runs(bundle, "gaussian", name)
        a = finite_curves(bundle, rows)
        mean, std = a.mean(axis=0), a.std(axis=0)
        axes[0].plot(x, mean, color=colors[name], label=labels[name])
        axes[0].fill_between(
            x, np.maximum(0, mean - std), mean + std, color=colors[name], alpha=0.15
        )
        if name != "postselection":
            best = sorted(rows, key=lambda row: (row["last_finite_loss"], row["seed"]))[
                :5
            ]
            for i, curve in enumerate(finite_curves(bundle, best)):
                axes[1].plot(
                    x,
                    curve,
                    color=colors[name],
                    alpha=0.75,
                    label=labels[name] if i == 0 else None,
                )
    for axis, letter in zip(axes, ("a", "b"), strict=True):
        axis.set(xlabel="Iteration Number", ylabel="KL", xlim=(0, 100))
        axis.tick_params(direction="in", top=True, right=True)
        axis.legend(fontsize=8)
        axis.text(
            0.5,
            -0.23,
            rf"({letter}) $n=4,\ m=12,\ N=200000,$" + "\n" + r"$\eta=0.8$",
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=12,
        )
    if row_figures is None:
        financial, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    else:
        financial = row_figures[1]
        axes = financial.subplots(1, 2)
    financial.supxlabel(
        "Source: European Central Bank (ECB), EXR.D.USD.EUR.SP00.A (2004–2023).\n"
        "Log returns and histograms calculated by the reproduction authors.",
        fontsize=8,
    )
    for name in CONDITIONS:
        a = finite_curves(bundle, select_runs(bundle, "financial", name))
        mean, std = a.mean(axis=0), a.std(axis=0)
        axes[0].plot(x, mean, color=colors[name], label=labels[name])
        axes[0].fill_between(
            x, np.maximum(0, mean - std), mean + std, color=colors[name], alpha=0.16
        )
    h = bundle["histogram"]
    edges = np.r_[h["left"][::5], h["right"][-1]]
    axes[1].stairs(
        h["target_count"].reshape(-1, 5).sum(axis=1),
        edges,
        color="black",
        lw=0.9,
        label="Target samples",
    )
    axes[1].stairs(
        h["displayed_qcbm_count"].reshape(-1, 5).sum(axis=1),
        edges,
        color="#00b000",
        lw=0.9,
        label="QCBM samples",
    )
    axes[0].set(xlabel="Iteration Number", ylabel="TVD", ylim=(0, 1), xlim=(0, 100))
    axes[1].set(xlabel="Daily log returns EUR/USD", ylabel="Sample count")
    for axis, letter in zip(axes, ("c", "d"), strict=True):
        axis.text(
            0.5,
            -0.23,
            rf"({letter}) $n=4,\ m=12,\ N=60000,$" + "\n" + r"$\eta=0.8$",
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=12,
        )
        axis.tick_params(direction="in", top=True, right=True)
        axis.legend(fontsize=8)
    if row_figures is None:
        belenos, axes = plt.subplots(1, 2, figsize=(12, 5.2), layout="constrained")
    else:
        belenos = row_figures[2]
        axes = belenos.subplots(1, 2)
    color = dict(
        zip(
            BELENOS,
            (
                "#ff8000",
                "#0080ff",
                "#00a800",
                "#aa44ff",
                "#222222",
                "#c6385b",
                "#009caa",
            ),
            strict=True,
        )
    )
    label = dict(
        zip(
            BELENOS,
            (
                "Lossy",
                "Lossy with recycling mitigation (I=0.729)",
                "Lossless (I=0.729)",
                "Lossless, indistinguishable (I=1)",
                "Lossy with recycling mitigation (I=1)",
                "Lossy with recycling mitigation (I=0.84)",
                "Lossless (I=0.84)",
            ),
            strict=True,
        )
    )
    for name in BELENOS[:2]:
        rows = [
            row
            for row in select_runs(bundle, "belenos", name)
            if row["seed"] in (0, 1, 2)
        ]
        for i, curve in enumerate(finite_curves(bundle, rows)):
            axes[0].plot(
                x,
                curve,
                color=color[name],
                lw=0.85,
                label=label[name] if i == 0 else None,
            )
    seed = choose_panel_f(bundle)
    for name in PANEL_F:
        rows = [
            row for row in select_runs(bundle, "belenos", name) if row["seed"] == seed
        ]
        axes[1].plot(
            x, finite_curves(bundle, rows)[0], color=color[name], label=label[name]
        )
    for axis, letter in zip(axes, ("e", "f"), strict=True):
        axis.set(xlabel="Iteration Number", ylabel="KL", xlim=(0, 100))
        axis.tick_params(direction="in", top=True, right=True)
        axis.legend(fontsize=7.5)
        axis.text(
            0.5,
            -0.18,
            rf"({letter}) $n=4,\ m=12,\ N=5\times10^7,$"
            + "\n"
            + (r"$\eta=0.972$" if letter == "f" else r"$\eta=0.972,\ I=0.729$"),
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=10,
        )
    belenos.suptitle("Belenos-characterized MerLin simulation", fontsize=12)
    return {
        "gaussian_fig2ab": gaussian,
        "financial_fig2cd": financial,
        "belenos_fig2ef_original": belenos,
    }


def make_combined_figure(bundle: dict):
    """Arrange the unchanged six panels into one paper-style Figure 2.

    Parameters
    ----------
    bundle : dict
        Validated CSV/manifest bundle used by the independent panel pairs.

    Returns
    -------
    matplotlib.figure.Figure
        Three rows and two columns, with the financial source credit retained.
        The caller owns closing the figure.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(12, 15.6), layout="constrained")
    _make_panel_figures(bundle, parent=figure)
    return figure


def make_figures(bundle: dict) -> dict:
    """Build the panel-pair exports and their combined six-panel Figure 2.

    Parameters
    ----------
    bundle : dict
        Validated CSV/manifest bundle; no new sampling or training is performed.

    Returns
    -------
    dict
        Figures keyed by the three existing stems and ``figure2_combined``.
        The caller owns closing every figure.
    """
    figures = _make_panel_figures(bundle)
    figures["figure2_combined"] = make_combined_figure(bundle)
    return figures


def run_result_plotting(cfg: dict, run_dir: Path) -> None:
    """Validate public results and save plots/summary through the shared runtime.

    Parameters
    ----------
    cfg : dict
        ``results.directory`` is explicit, absolute or relative to this paper.
    run_dir : pathlib.Path
        New timestamped output directory; committed figures are not overwritten.
    """
    import matplotlib.pyplot as plt

    from .training import write_json

    directory = Path(cfg["results"]["directory"])
    if not directory.is_absolute():
        directory = Path(__file__).resolve().parents[1] / directory
    bundle = load_result_bundle(directory)
    summary = summarize_results(bundle)
    for name, figure in make_figures(bundle).items():
        for suffix in ("png", "svg"):
            figure.savefig(run_dir / f"{name}.{suffix}", dpi=180)
        plt.close(figure)
    write_json(run_dir / "summary.json", summary)
    write_json(
        run_dir / "metrics.json",
        {
            "status": "complete",
            "task": "plot_results",
            "attempts": len(bundle["runs"]),
            "new_photonic_shots": 0,
            "qpu_jobs_submitted": 0,
            "input_sha256": bundle["manifest"]["input_sha256"],
        },
    )
