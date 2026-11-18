"""Phase 4 summary: MerLin photonic runs against the Phase 3 qubit QELM, with the plan criteria.

Usage (from the paper folder)::

    python utils/summarize_merlin.py \
        --landsat outdir/merlin_landsat_m16/run_A:outdir/landsat_n16_standard/run_B \
                  outdir/merlin_landsat_m24/run_C:outdir/landsat_n24_standard/run_D \
        --narma outdir/merlin_narma_m16/run_E:outdir/narma_n16/run_F ... \
        --concentration outdir/merlin_concentration/run_G

Each ``--landsat``/``--narma`` item is ``photonic_run:qubit_run``. Writes
``results/merlin_summary.{json,md}`` and ``results/merlin_{landsat,narma,concentration}.png``.
The pass rules are the ones declared in LOG.md ("Phase 4 Adaptation Plan") before any run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PAPER_DIR = Path(__file__).resolve().parents[1]
LANDSAT_TOLERANCE = 0.05
NARMA_RELATIVE_TOLERANCE = 0.10
PAIRED_DIFFERENCES = (
    ("w2_et_scaled", "w2_click_scaled"),
    ("w1_et_scaled", "w1_click_scaled"),
    ("w2_etcut_scaled", "w2_et_scaled"),
    ("w2_click_scaled", "w2_click_unit"),
    ("w2_et_scaled", "w2_et_unit"),
    ("w2_click_unit", "w1_click_unit"),
)


def _metrics(run_dir: Path) -> dict:
    status = json.loads((run_dir / "run_status.json").read_text())
    if status["status"] != "COMPLETED":
        raise ValueError(f"{run_dir} is {status['status']}")
    return json.loads((run_dir / status["metrics_path"]).read_text())


def _pair(text: str) -> tuple[Path, Path]:
    photonic, qubit = text.split(":")
    return Path(photonic), Path(qubit)


def _headline(records: list[dict]) -> dict[str, list[dict]]:
    by_readout: dict[str, list[dict]] = {}
    for record in records:
        if record["analysis"] == "readouts":
            by_readout.setdefault(record["readout"], []).append(record)
    return by_readout


def _paired(by_readout: dict, first: str, second: str) -> dict:
    key = lambda r: (r["split_seed"], r["reservoir_seed"], r["shot_seed"])  # noqa: E731
    a = {key(r): r["f1_macro"] for r in by_readout[first]}
    b = {key(r): r["f1_macro"] for r in by_readout[second]}
    if set(a) != set(b):
        raise ValueError(f"{first} and {second} have different seed sets")
    diff = np.array([a[k] - b[k] for k in sorted(a)])
    return {
        "mean": float(diff.mean()),
        "stderr": float(diff.std(ddof=1) / np.sqrt(diff.size)),
        "n": int(diff.size),
        "a_better": int((diff > 0).sum()),
        "b_better": int((diff < 0).sum()),
    }


def _curve(summary: dict, field: str) -> tuple[list[float], list[float], list[float]]:
    points = []
    for key, value in summary.items():
        level = float(key.split(f"{field}=")[1])
        points.append((level, value["f1_macro_mean"], value["f1_macro_std"]))
    points.sort()
    return [p[0] for p in points], [p[1] for p in points], [p[2] for p in points]


def summarise_landsat(pairs: list[tuple[Path, Path]], outdir: Path) -> dict:
    result = {}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for photonic_run, qubit_run in pairs:
        photonic, qubit = _metrics(photonic_run), _metrics(qubit_run)
        size = f"{photonic['hardware']['n_modes']}/{photonic['hardware']['n_photons']}"
        readouts = {
            key.split("=")[1]: value
            for key, value in photonic["summary"]["readouts"].items()
        }
        qubit_readouts = {
            key.split("=")[1]: value
            for key, value in qubit["summary"]["readouts"].items()
        }
        best_name = max(
            ("w2_et_scaled", "w2_et_unit"), key=lambda n: readouts[n]["f1_macro_mean"]
        )
        qubit_reference = max(
            qubit_readouts["w2_et_scaled"]["f1_macro_mean"],
            qubit_readouts["w2_et_unit"]["f1_macro_mean"],
        )
        gap = readouts[best_name]["f1_macro_mean"] - qubit_reference
        by_readout = _headline(photonic["records"])
        cutoff = photonic["summary"]["cutoff_sweep"]
        retained = {
            key: value.get("n_eigentasks_retained_mean")
            for key, value in cutoff.items()
            if key.startswith("readout=w2_etcut_scaled")
        }
        result[size] = {
            "photonic_run": str(photonic_run),
            "qubit_run": str(qubit_run),
            "hardware": photonic["hardware"],
            "readouts": {
                name: {
                    "f1_macro_mean": value["f1_macro_mean"],
                    "f1_macro_std": value["f1_macro_std"],
                    "n_features": value["n_features_mean"],
                }
                for name, value in sorted(readouts.items())
            },
            "qubit_readouts": {
                name: value["f1_macro_mean"]
                for name, value in sorted(qubit_readouts.items())
            },
            "criterion": {
                "rule": f"best photonic w2 ET within {LANDSAT_TOLERANCE} F1 of qubit w2 ET",
                "photonic_best": best_name,
                "photonic_f1": readouts[best_name]["f1_macro_mean"],
                "qubit_f1": qubit_reference,
                "difference": gap,
                "preserved": abs(gap) <= LANDSAT_TOLERANCE,
            },
            "paired_differences": {
                f"{a} - {b}": _paired(by_readout, a, b) for a, b in PAIRED_DIFFERENCES
            },
            "cutoff_retained_w2_etcut_scaled": retained,
            "shots_curve": dict(
                zip(
                    *_curve(photonic["summary"]["shots_sweep"], "shots")[:2],
                    strict=True,
                )
            ),
            "qubit_shots_curve": dict(
                zip(*_curve(qubit["summary"]["shots_sweep"], "shots")[:2], strict=True)
            ),
            "train_size_curve": dict(
                zip(
                    *_curve(photonic["summary"]["train_size_sweep"], "train_size")[:2],
                    strict=True,
                )
            ),
        }
        for ax, analysis, field in (
            (axes[1], "shots_sweep", "shots"),
            (axes[2], "train_size_sweep", "train_size"),
        ):
            for metrics, label, style in (
                (photonic, f"photonic {size}", "-o"),
                (qubit, f"qubit N={size.split('/')[0]}", "--s"),
            ):
                x, y, err = _curve(metrics["summary"][analysis], field)
                ax.errorbar(x, y, yerr=err, fmt=style, capsize=2, label=label)
            ax.set(xscale="log", xlabel=field.replace("_", " "), ylabel="F1 macro")
    names = [
        "w1_click_unit",
        "w1_et_scaled",
        "w2_click_unit",
        "w2_click_scaled",
        "w2_et_scaled",
        "w2_etcut_scaled",
    ]
    width = 0.8 / len(pairs)
    for index, (size, entry) in enumerate(result.items()):
        axes[0].bar(
            np.arange(len(names)) + index * width,
            [entry["readouts"][n]["f1_macro_mean"] for n in names],
            width,
            yerr=[entry["readouts"][n]["f1_macro_std"] for n in names],
            capsize=2,
            label=f"photonic {size} (qubit w2 ET {entry['criterion']['qubit_f1']:.3f})",
        )
    axes[0].set_xticks(
        np.arange(len(names)) + width / 2, names, rotation=30, ha="right"
    )
    axes[0].set(ylim=(0.5, 0.9), ylabel="F1 macro (45 evaluations)", title="Readouts")
    axes[1].set_title("F1 vs shots (w1 click / w1 Pauli)")
    axes[2].set_title("F1 vs training size")
    for ax in axes:
        ax.legend(fontsize=7)
    fig.suptitle(
        "Landsat: MerLin photonic reservoir vs qubit QELM (z-score inputs; partial)"
    )
    fig.savefig(outdir / "merlin_landsat.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return result


def summarise_narma(pairs: list[tuple[Path, Path]], outdir: Path) -> dict:
    result = {}
    fig, axes = plt.subplots(1, len(pairs), figsize=(6 * len(pairs), 4), sharey=True)
    for ax, (photonic_run, qubit_run) in zip(np.atleast_1d(axes), pairs, strict=True):
        photonic, qubit = _metrics(photonic_run), _metrics(qubit_run)
        window = photonic["window"]
        shots = photonic["summary"]["qelm|shots_1e5"]
        qubit_shots = qubit["summary"]["qelm|shadow_1e5_shots"]
        r2 = shots["r2_mean"]
        edge_drop = r2[window - 1] - r2[window]
        relative = (
            shots["cumulative_r2_mean"] - qubit_shots["cumulative_r2_mean"]
        ) / qubit_shots["cumulative_r2_mean"]
        size = f"{photonic['n_modes']}/{photonic['n_photons']}"
        result[size] = {
            "photonic_run": str(photonic_run),
            "qubit_run": str(qubit_run),
            "hardware": photonic["hardware"],
            "window": window,
            "cumulative_r2": {
                key: value["cumulative_r2_mean"]
                for key, value in photonic["summary"].items()
            },
            "qubit_cumulative_r2": {
                key: value["cumulative_r2_mean"]
                for key, value in qubit["summary"].items()
            },
            "criterion": {
                "rule": f"photonic 1e5-shot cumulative R^2 within {NARMA_RELATIVE_TOLERANCE:.0%} of qubit 1e5-shot, memory edge at n = L",
                "relative_difference": relative,
                "r2_at_L": r2[window - 1],
                "r2_at_L_plus_1": r2[window],
                "edge_drop": edge_drop,
                "preserved": abs(relative) <= NARMA_RELATIVE_TOLERANCE
                and edge_drop > 0.3,
            },
        }
        orders = photonic["orders"]
        for metrics, key, label, style in (
            (photonic, "qelm|shots_1e5", "photonic, 1e5 shots", "-o"),
            (photonic, "qelm|gaussian_std_10^-1.5", "photonic, Gaussian 10^-1.5", ":"),
            (qubit, "qelm|shadow_1e5_shots", "qubit QELM, 1e5 shots", "--s"),
            (photonic, "degree2_window|none", "ridge, degree-2 window", "-.^"),
            (photonic, "esn|none", "ESN-400", "-.d"),
        ):
            entry = metrics["summary"][key]
            ax.errorbar(
                orders,
                entry["r2_mean"],
                yerr=entry["r2_std"],
                fmt=style,
                ms=4,
                capsize=2,
                label=f"{label} (cum {entry['cumulative_r2_mean']:.2f})",
            )
        ax.axvline(window, color="k", linestyle=":", linewidth=1)
        ax.set(
            title=f"{size} modes/photons, L = {window}",
            xlabel="NARMA order n",
            ylim=(-0.2, 1.05),
        )
        ax.legend(fontsize=7, loc="lower left")
    np.atleast_1d(axes)[0].set_ylabel("Test R^2")
    fig.suptitle("NARMA: MerLin photonic reservoir vs qubit QELM (partial)")
    fig.savefig(outdir / "merlin_narma.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return result


def summarise_concentration(run_dir: Path, outdir: Path) -> dict:
    metrics = _metrics(run_dir)
    records = metrics["records"]
    largest = max(r["n_modes"] for r in records)
    deepest = max(r["n_layers"] for r in records)
    per_seed = {}
    for record in records:
        if record["n_modes"] == largest and record["n_layers"] == deepest:
            per_seed.setdefault(record["reservoir_seed"], {})[record["setting"]] = (
                record
            )
    ordering = {
        f"seed {seed}|{weight}": settings["table_I_mapped"][
            f"{weight}_variability_mean"
        ]
        > max(
            settings["balanced"][f"{weight}_variability_mean"],
            settings["haar"][f"{weight}_variability_mean"],
        )
        for seed, settings in per_seed.items()
        for weight in ("w1", "w2")
    }
    haar = sorted(
        (
            key
            for key in metrics["summary"]
            if key.startswith("haar|") and key.endswith(f"n_layers={deepest}")
        ),
        key=lambda key: int(key.split("modes=")[1].split("|")[0]),
    )
    haar_trend = {
        weight: [
            metrics["summary"][key][f"{weight}_variability_mean_avg"] for key in haar
        ]
        for weight in ("w1", "w2")
    }
    haar_decreasing = all(
        all(a > b for a, b in zip(values, values[1:])) for values in haar_trend.values()
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    styles = {"table_I_mapped": "-o", "balanced": "--s", "haar": ":^"}
    for ax, weight in zip(axes, ("w1", "w2"), strict=True):
        for setting, style in styles.items():
            sizes = sorted({r["n_modes"] for r in records if r["setting"] == setting})
            for layers, alpha in zip(
                sorted({r["n_layers"] for r in records}), (0.35, 0.65, 1.0), strict=True
            ):
                values, errors = [], []
                for modes in sizes:
                    key = next(
                        k
                        for k in metrics["summary"]
                        if k.startswith(f"{setting}|modes={modes}|")
                        and k.endswith(f"n_layers={layers}")
                    )
                    values.append(
                        metrics["summary"][key][f"{weight}_variability_mean_avg"]
                    )
                    errors.append(
                        metrics["summary"][key][f"{weight}_variability_mean_std"]
                    )
                ax.errorbar(
                    sizes,
                    values,
                    yerr=errors,
                    fmt=style,
                    alpha=alpha,
                    capsize=2,
                    label=f"{setting}, {layers} layers",
                )
        ax.set(
            yscale="log",
            xlabel="modes (photons = modes / 4)",
            title=f"{'per-mode clicks' if weight == 'w1' else 'neighbour coincidences'}",
        )
        ax.legend(fontsize=6, ncol=1)
    axes[0].set_ylabel("mean Var_u over u ~ Unif(-1, 1)")
    fig.suptitle("Concentration analogue: photonic output variability (partial)")
    fig.savefig(outdir / "merlin_concentration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {
        "run": str(run_dir),
        "hardware": metrics["hardware"],
        "summary": metrics["summary"],
        "criterion": {
            "rule": "at the deepest layer count and largest size, mapped Table I variability exceeds balanced and Haar for every seed (w1 and w2), and Haar variability decreases with mode count",
            "largest_modes": largest,
            "deepest_layers": deepest,
            "ordering_per_seed": ordering,
            "haar_trend_vs_modes": haar_trend,
            "haar_decreasing": haar_decreasing,
            "preserved": all(ordering.values()) and haar_decreasing,
        },
    }


def _markdown(summary: dict) -> str:
    lines = [
        "# Phase 4 MerLin summary (partial validation; simulated, ideal photonics)",
        "",
    ]
    lines += [
        "## Landsat (F1 macro, 45 evaluations)",
        "",
        "| Readout | " + " | ".join(f"photonic {s}" for s in summary["landsat"]) + " |",
        "|---|" + "---|" * len(summary["landsat"]),
    ]
    names = next(iter(summary["landsat"].values()))["readouts"]
    for name in names:
        lines.append(
            f"| {name} | "
            + " | ".join(
                f"{e['readouts'][name]['f1_macro_mean']:.3f} ± {e['readouts'][name]['f1_macro_std']:.3f}"
                for e in summary["landsat"].values()
            )
            + " |"
        )
    lines.append("")
    for size, entry in summary["landsat"].items():
        c = entry["criterion"]
        lines.append(
            f"- {size}: best {c['photonic_best']} {c['photonic_f1']:.3f} vs qubit w2 ET {c['qubit_f1']:.3f} (difference {c['difference']:+.3f}): **{'preserved' if c['preserved'] else 'not preserved'}**."
        )
        for label, d in entry["paired_differences"].items():
            lines.append(
                f"  - {label}: {d['mean']:+.4f} ± {d['stderr']:.4f} (n = {d['n']}, {d['a_better']}/{d['b_better']})"
            )
        lines.append(
            "  - shots curve (100/1e3/1e4/1e5): photonic "
            + " / ".join(f"{v:.3f}" for v in entry["shots_curve"].values())
            + "; qubit "
            + " / ".join(f"{v:.3f}" for v in entry["qubit_shots_curve"].values())
        )
        lines.append(
            "  - eigentasks retained by the cutoff (w2_etcut_scaled): "
            + ", ".join(
                f"{k.split('nsr_cutoff=')[1]}: {v:.1f}"
                for k, v in sorted(
                    entry["cutoff_retained_w2_etcut_scaled"].items(),
                    key=lambda kv: float(kv[0].split("nsr_cutoff=")[1]),
                )
            )
        )
    lines += ["", "## NARMA (cumulative test R^2 over n = 1..L)", ""]
    for size, entry in summary["narma"].items():
        c = entry["criterion"]
        cum = entry["cumulative_r2"]
        lines.append(
            f"- {size} (L = {entry['window']}): photonic 1e5 shots {cum['qelm|shots_1e5']:.2f}, exact {cum['qelm|exact']:.2f}, Gaussian {cum['qelm|gaussian_std_10^-1.5']:.2f}; qubit 1e5 shots {entry['qubit_cumulative_r2']['qelm|shadow_1e5_shots']:.2f}; degree-2 ridge {cum['degree2_window|none']:.2f}; raw ridge {cum['raw_window|none']:.2f}; ESN {cum['esn|none']:.2f}. Relative difference {c['relative_difference']:+.1%}, R^2 at L {c['r2_at_L']:.2f} → {c['r2_at_L_plus_1']:.2f}: **{'preserved' if c['preserved'] else 'not preserved'}**."
        )
    c = summary["concentration"]["criterion"]
    lines += [
        "",
        "## Concentration",
        "",
        f"- Ordering at {c['largest_modes']} modes, {c['deepest_layers']} layers (mapped > balanced, Haar), per seed and weight: {c['ordering_per_seed']}.",
        f"- Haar variability vs modes (w1): {', '.join(f'{v:.2e}' for v in c['haar_trend_vs_modes']['w1'])}; decreasing: {c['haar_decreasing']}.",
        f"- **{'preserved' if c['preserved'] else 'not preserved'}**.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--landsat", nargs="+", type=_pair, required=True)
    parser.add_argument("--narma", nargs="+", type=_pair, required=True)
    parser.add_argument("--concentration", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=PAPER_DIR / "results")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    summary = {
        "landsat": summarise_landsat(args.landsat, args.outdir),
        "narma": summarise_narma(args.narma, args.outdir),
        "concentration": summarise_concentration(args.concentration, args.outdir),
    }
    (args.outdir / "merlin_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    text = _markdown(summary)
    (args.outdir / "merlin_summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
