"""Render the curated results (tables, figures, RESULTS.md) from completed sweeps.

Inputs, under ``--outdir`` (default ``papers/QCHIPP/outdir``):

- ``comparison100/``, ``comparison_onehot/``, ``comparison_reference/``: sweeps
  written by ``utils/run_comparison.py`` (36, 12 and 12 runs).
- ``label01_binding/``, ``label01_immunogenic/``: single-seed runs of
  ``configs/*_source_zeroone.json``.

and ``results/report_citations.json``. Every run is re-verified against its
``run_status.json`` before it is aggregated. The supplied-affinity diagnostic
is recomputed from DataS1 (``lib/affinity.py``).

Example (from the repository root)::

    python papers/QCHIPP/utils/render_results.py
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
for import_root in (REPO_ROOT, PROJECT_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from lib.affinity import affinity_diagnostic  # noqa: E402
from lib.artifacts import sha256, write_json  # noqa: E402

LOGGER = logging.getLogger("qchipp.render")

SWEEPS = {"comparison100": 36, "comparison_onehot": 12, "comparison_reference": 12}
PAPER_F1 = {"binding": 0.73, "immunogenic": 0.67}
MODEL_ORDER = [
    "gate_source",
    "gate_corrected",
    "merlin_reference",
    "classical_reference",
    "logistic_onehot",
    "rf_onehot",
    "merlin",
    "classical",
    "logistic",
    "rf",
]
MODEL_LABELS = [
    "Source QCNN\n81",
    "BCE QCNN\n81",
    "MerLin ref7\n346",
    "Matched MLP\n350",
    "LR one-hot\n81",
    "RF one-hot\n300 trees",
    "Legacy6*\n256",
    "MLP\n250",
    "LR bits\n21",
    "RF bits\n300 trees",
]
MODEL_COLORS = [
    "#2266a5",
    "#5da5da",
    "#8a54a2",
    "#666666",
    "#457d61",
    "#70a286",
    "#c5a5d0",
    "#aaaaaa",
    "#8e8e8e",
    "#9e9e9e",
]


def resolve_run_dir(sweep_dir: Path, recorded: str) -> Path:
    """Locate a run recorded in a sweep summary.

    New sweeps record ``<candidate>/<run_YYYYMMDD-HHMMSS>`` relative to the
    sweep directory. Older sweeps recorded absolute container paths; their
    last two components have the same layout, so they resolve the same way.
    """
    recorded_path = Path(recorded)
    if not recorded_path.is_absolute():
        return sweep_dir / recorded_path
    return sweep_dir.joinpath(*recorded_path.parts[-2:])


def load_sweep(sweep_dir: Path, expected_runs: int) -> dict:
    """Load a complete sweep and verify every run against its own artifacts."""
    summary = json.loads((sweep_dir / "summary.json").read_text())
    if not (summary["completed_count"] == summary["planned_count"] == expected_runs):
        raise ValueError(f"{sweep_dir} is incomplete or has an unexpected run count")
    if summary["status"] != "COMPLETE COMPARISON":
        raise ValueError(f"{sweep_dir} status is {summary['status']}")
    code_fingerprints = set()
    for record in summary["runs"]:
        run_dir = resolve_run_dir(sweep_dir, record["run_dir"])
        metrics = json.loads((run_dir / "metrics.json").read_text())
        if metrics != {k: v for k, v in record.items() if k != "run_dir"}:
            raise ValueError(f"{run_dir} metrics differ from the sweep summary")
        state = json.loads((run_dir / "run_status.json").read_text())
        if state["status"] != "COMPLETED" or state["metrics_sha256"] != sha256(
            run_dir / "metrics.json"
        ):
            raise ValueError(f"{run_dir} is not a verified completed run")
        code_fingerprints.add(
            json.dumps(state["code"]["source_hashes"], sort_keys=True)
        )
        record["run_dir"] = run_dir.relative_to(sweep_dir.parent).as_posix()
    # All runs of one comparison must come from identical model code.
    if len(code_fingerprints) != 1:
        raise ValueError(f"{sweep_dir} mixes runs from different code versions")
    return summary


def load_label_sensitivity(outdir: Path) -> list[dict]:
    """Load the latest zero/one source-label sensitivity run for each task."""
    records = []
    for task in ("binding", "immunogenic"):
        run_dir = sorted((outdir / f"label01_{task}").glob("run_*"))[-1]
        if (
            json.loads((run_dir / "run_status.json").read_text())["status"]
            != "COMPLETED"
        ):
            raise ValueError(f"{run_dir} did not complete")
        metrics = json.loads((run_dir / "metrics.json").read_text())
        if metrics["training"]["objective"] != "source zero_one clipped scalar log2":
            raise ValueError(f"{run_dir} is not a zero/one source-label run")
        records.append({"run_dir": run_dir.relative_to(outdir).as_posix(), **metrics})
    return records


def finite_shot_rows(runs: list[dict]) -> list[dict]:
    """F1 mean and SD for analytic and 1000-shot evaluation of the MerLin models."""
    rows = []
    for task in ("binding", "immunogenic"):
        for family in ("merlin", "merlin_reference"):
            members = [r for r in runs if r["task"] == task and r["model"] == family]
            for shots in (0, 1000):
                values = [
                    r["test"]["f1"]
                    if shots == 0
                    else r["finite_shots"][str(shots)]["f1"]
                    for r in members
                ]
                rows.append(
                    {
                        "task": task,
                        "model": family,
                        "shots_per_cell": shots,
                        "shots_per_peptide": shots * 3,
                        "f1_mean": float(np.mean(values)),
                        "f1_sample_sd": float(np.std(values, ddof=1)),
                        "seeds": [r["seed"] for r in members],
                    }
                )
    return rows


def plot_performance(rows: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(
        1, 2, figsize=(18, 5), sharey=True, constrained_layout=True
    )
    titles = {
        "binding": "Binding: 150 train / 999 test",
        "immunogenic": "Binder-only immunogenicity: 300 train / 1000 test",
    }
    for ax, task in zip(axes, ("binding", "immunogenic")):
        selected = [
            next(r for r in rows if r["task"] == task and r["model"] == name)
            for name in MODEL_ORDER
        ]
        ax.bar(
            np.arange(len(MODEL_ORDER)),
            [r["f1"]["mean"] for r in selected],
            yerr=[r["f1"]["sample_sd"] for r in selected],
            color=MODEL_COLORS,
            capsize=4,
        )
        ax.set_xticks(np.arange(len(MODEL_ORDER)), MODEL_LABELS, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(titles[task])
        ax.set_ylabel("Held-out F1 (mean ± sample SD, 3 seeds)")
    fig.suptitle(
        "Release-split benchmark, 100 updates. *Legacy six-mode model has "
        "global-phase aliases",
        fontsize=12,
    )
    fig.savefig(output / "performance.png", dpi=180)
    fig.savefig(output / "performance.svg")
    plt.close(fig)


def plot_finite_shots(shots: list[dict], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for task, color in (("binding", "#8a54a2"), ("immunogenic", "#2266a5")):
        for family, style in (("merlin_reference", "-"), ("merlin", "--")):
            data = [s for s in shots if s["task"] == task and s["model"] == family]
            label = f"{task} {'ref7' if family == 'merlin_reference' else 'legacy6'}"
            ax.errorbar(
                [0, 1],
                [s["f1_mean"] for s in data],
                yerr=[s["f1_sample_sd"] for s in data],
                marker="o",
                capsize=4,
                color=color,
                linestyle=style,
                label=label,
            )
    ax.set_xticks([0, 1], ["Analytic", "1,000 shots/cell\n3,000 shots/peptide"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 mean ± sample SD (3 seeds)")
    ax.set_title("MerLin: ideal two-photon cells, shot noise only")
    ax.legend()
    fig.savefig(output / "finite_shots.png", dpi=180)
    fig.savefig(output / "finite_shots.svg")
    plt.close(fig)


def _display_backend(backend: str) -> str:
    # Runs recorded before the 2026-10 refactor wrote "MerLin0.4.1 ..." without
    # a space; the metadata is kept as recorded and only the display is fixed.
    return re.sub(r"MerLin(?=\d)", "MerLin ", backend)


def _parameter_cell(row: dict) -> str:
    total = row["parameters"]["total"]
    return "300 trees" if total is None else str(total)


def results_markdown(result: dict, citations: dict) -> str:
    rows, runs = result["rows"], result["runs"]
    lines = [
        "# Results: partial release-split reproduction",
        "",
        "All tables are derived from structured run artifacts; no model was "
        "selected using test results.",
        "",
        "## Task and model coverage",
        "",
        "| Task | Model | Runs | Parameters | F1 mean ± SD | AUROC mean ± SD |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for task in ("binding", "immunogenic"):
        for name in MODEL_ORDER:
            r = next(r for r in rows if r["task"] == task and r["model"] == name)
            lines.append(
                f"| {task} | {name} | {r['completed_runs']}/{r['expected_runs']} | "
                f"{_parameter_cell(r)} | {r['f1']['mean']:.4f} ± {r['f1']['sample_sd']:.4f} | "
                f"{r['auroc']['mean']:.4f} ± {r['auroc']['sample_sd']:.4f} |"
            )
    lines += [
        "",
        "## Comparison with the paper",
        "",
        "The paper reports QRAC-10 binding F1 0.73 and selected binder-only "
        "immunogenicity F1 0.67 [1]. This protocol uses the larger released test "
        "sets and 100 updates, not the exact historical experiment, so the deltas "
        "are descriptive only.",
        "",
        "| Task | Paper F1 | Local source F1 | Delta | Agreement |",
        "|---|---:|---:|---:|---|",
    ]
    agreement = {
        "binding": "Similar binding range, not a quantitative replication",
        "immunogenic": "Reported immunogenicity performance not matched",
    }
    for r in result["paper_deltas"]:
        lines.append(
            f"| {r['task']} | {r['paper_f1']:.2f} | {r['local_f1']:.4f} | "
            f"{r['delta']:+.4f} | {agreement[r['task']]} |"
        )
    lines += [
        "",
        "## Label-convention sensitivity (single seed 49)",
        "",
        "| Task | Zero/one source F1 | AUROC | Accuracy |",
        "|---|---:|---:|---:|",
    ]
    for r in result["label_sensitivity"]:
        lines.append(
            f"| {r['task']} | {r['test']['f1']:.4f} | {r['test']['auroc']:.4f} | "
            f"{r['test']['accuracy']:.4f} |"
        )
    lines += [
        "",
        "With 0/1 labels the source objective gives negative examples zero loss. "
        "The higher immunogenicity F1 in this run comes with AUROC below 0.5, so "
        "F1 alone is misleading. This does not establish which labels were used in "
        "every published experiment.",
        "",
        "## MerLin resource assumptions",
        "",
        "| Task / model | Space / detector | Photons / modes per cell | Input state | "
        "Postselection | Backend | Shots | Seeds | Mean wall seconds |",
        "|---|---|---|---|---|---|---|---|---:|",
    ]
    for task in ("binding", "immunogenic"):
        for family in ("merlin_reference", "merlin"):
            r = next(r for r in rows if r["model"] == family and r["task"] == task)
            meta = next(
                m["photonic"]
                for m in runs
                if m["model"] == family and m["task"] == task
            )
            lines.append(
                f"| {task} / {family} | FOCK / ideal PNR | 2 / {meta['n_modes']}, "
                f"3 cell executions | {meta['input_state']} | none | {_display_backend(meta['backend'])} | "
                f"analytic training; analytic and 1000/cell test | 49, 50, 51 | "
                f"{r['mean_wall_seconds']:.2f} |"
            )
    lines += [
        "",
        "## Finite-shot F1",
        "",
        "| Task / model | Shots per cell | Shots per peptide | F1 mean ± SD |",
        "|---|---:|---:|---:|",
    ]
    for s in result["finite_shots"]:
        lines.append(
            f"| {s['task']} / {s['model']} | {s['shots_per_cell']} | "
            f"{s['shots_per_peptide']} | {s['f1_mean']:.4f} ± {s['f1_sample_sd']:.4f} |"
        )
    lines += [
        "",
        "## Interpretation and limitations",
        "",
        "The seven-mode reference adaptation fixes the global-phase feature alias "
        "of the legacy six-mode model, but shows no clear advantage over its "
        "350-parameter matched MLP. Legacy six-mode results are kept as a labelled "
        "ablation. The 81-parameter one-hot logistic control is stronger than the "
        "81-parameter source QCNN on both tasks.",
        "",
        "This is not a hardware-speed or clinical-utility claim. Three seeds "
        "measure optimisation variation on one split, not independent-cohort "
        "uncertainty. One binding test row without assay truth is excluded; the "
        "released 1000-row test sets differ from the paper's 50/100 descriptions. "
        "Experimental assay truth follows the supplement legend [12].",
        "",
        "## Supplied affinity-rank diagnostic (no fitting)",
        "",
        "| Population | Test count | AUROC (negative BA rank) | F1 (rank < 2) |",
        "|---|---:|---:|---:|",
    ]
    for row in result["affinity_diagnostic"]["rows"]:
        lines.append(
            f"| {row['task']} | {row['test_count']} | {row['auroc']:.4f} | "
            f"{row['f1_rank_lt2']:.4f} |"
        )
    lines += [
        "",
        "This evaluates the supplied, externally trained affinity predictions; it "
        "is not a parameter-matched baseline. The ranking association changes "
        "sharply after conditioning on experimental binders.",
        "",
        "## Sources",
        "",
    ]
    for source in citations["sources"]:
        lines.append(f"[{source['id']}]: {source['url']}")
    return "\n".join(lines) + "\n"


def render_report(outdir: Path | None = None, output: Path | None = None) -> dict:
    """Aggregate the verified sweeps and write the curated results.

    Parameters
    ----------
    outdir : Path | None
        Directory holding the sweeps. Default value is ``papers/QCHIPP/outdir``.
    output : Path | None
        Destination for curated artifacts. Default value is
        ``papers/QCHIPP/results``.

    Returns
    -------
    dict
        The aggregated content of ``comparison_summary.json``.
    """
    outdir = Path(outdir) if outdir is not None else PROJECT_DIR / "outdir"
    output = Path(output) if output is not None else PROJECT_DIR / "results"
    output.mkdir(parents=True, exist_ok=True)

    groups = {name: load_sweep(outdir / name, count) for name, count in SWEEPS.items()}
    sensitivity = load_label_sensitivity(outdir)
    runs = [record for group in groups.values() for record in group["runs"]]
    if len({(r["task"], r["model"], r["seed"]) for r in runs}) != sum(SWEEPS.values()):
        raise ValueError("Duplicate task/model/seed runs across sweeps")
    if len({r["source_sha256"] for r in runs + sensitivity}) != 1:
        raise ValueError("Runs were produced from different DataS1 files")

    rows = [row for group in groups.values() for row in group["rows"]]
    source_f1 = {
        task: next(
            r["f1"]["mean"]
            for r in rows
            if r["model"] == "gate_source" and r["task"] == task
        )
        for task in PAPER_F1
    }
    affinity = affinity_diagnostic()
    if affinity["source_sha256"] != runs[0]["source_sha256"]:
        raise ValueError("Runs were produced from a different DataS1 file")
    result = {
        "scope": "partial release-split comparison, not paper-accurate replication",
        "total_runs": len(runs) + len(sensitivity),
        "main_count": len(groups["comparison100"]["runs"]),
        "onehot_count": len(groups["comparison_onehot"]["runs"]),
        "reference_count": len(groups["comparison_reference"]["runs"]),
        "rows": rows,
        "runs": runs,
        "label_sensitivity": sensitivity,
        "finite_shots": finite_shot_rows(runs),
        "sources": [
            {
                "path": f"{name}/summary.json",
                "sha256": sha256(outdir / name / "summary.json"),
            }
            for name in SWEEPS
        ],
        "paper_deltas": [
            {
                "task": task,
                "paper_f1": paper_f1,
                "local_f1": source_f1[task],
                "delta": source_f1[task] - paper_f1,
            }
            for task, paper_f1 in PAPER_F1.items()
        ],
        "affinity_diagnostic": affinity,
    }
    write_json(output / "comparison_summary.json", result)
    write_json(output / "affinity_diagnostic.json", affinity)
    plot_performance(rows, output)
    plot_finite_shots(result["finite_shots"], output)
    citations = json.loads(
        (PROJECT_DIR / "results" / "report_citations.json").read_text()
    )
    (output / "RESULTS.md").write_text(
        results_markdown(result, citations), encoding="utf-8"
    )
    LOGGER.info("Rendered %d verified runs into %s", result["total_runs"], output)
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", type=Path, default=PROJECT_DIR / "outdir")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "results")
    args = parser.parse_args(argv)
    render_report(args.outdir, args.output)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    main()
