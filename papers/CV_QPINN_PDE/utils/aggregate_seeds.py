"""Aggregate seed sweep + ablation summaries from outdir/run_*/summary.json.

Groups runs by (experiment, n_params, cutoff, use_nested_loss,
pretrain_epochs, epochs), de-duplicates repeated seeds within a group,
and reports mean ± std on RMSE / MAE / L_inf / NMSE / wall-clock. Reads
only the JSON artefacts produced by `lib/runner.py`, not the per-run logs.

Outputs follow the outdir-vs-results policy (see utils/curate_results.py):
the full per-run markdown table is a generated raw artefact and goes to
outdir/; the compact per-group aggregate JSON is the curated record and
goes to results/.

Usage:

    python utils/aggregate_seeds.py [--outdir outdir]
                                   [--out outdir/seed_summary.md]
                                   [--json-out results/seed_aggregate.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def _key(summary: dict) -> tuple:
    cfg = summary.get("cfg", {})
    train = cfg.get("training", {})
    model = cfg.get("model", {})
    return (
        summary.get("experiment", "?"),
        int(summary.get("n_params", -1)),
        int(model.get("cutoff", -1)),
        bool(train.get("use_nested_loss", False)),
        int(train.get("pretrain_epochs", 0)),
        int(train.get("epochs", -1)),
    )


def _label(key: tuple) -> str:
    exp, params, cutoff, nested, pretrain, epochs = key
    bits = [exp, f"params={params}"]
    if cutoff > 0:
        bits.append(f"cutoff={cutoff}")
    bits.append("nested" if nested else "consistency")
    bits.append(f"ep={pretrain}+{epochs}" if pretrain else f"ep={epochs}")
    return ", ".join(bits)


def _mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], float("nan")
    return statistics.mean(values), statistics.stdev(values)


PROJECT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=PROJECT / "outdir")
    ap.add_argument("--out", type=Path, default=PROJECT / "outdir" / "seed_summary.md")
    ap.add_argument(
        "--json-out", type=Path, default=PROJECT / "results" / "seed_aggregate.json"
    )
    args = ap.parse_args()

    groups: dict[tuple, list[dict]] = defaultdict(list)
    # The launcher uses per-seed subdirectories (e.g. `heat_qpinn_seed42_d41d8c`)
    # and the runtime nests the actual run under a timestamped folder. Collect
    # both layouts so the aggregator works no matter which outdir scheme the
    # user invoked.
    candidates = (
        list(args.outdir.glob("run_*/summary.json"))
        + list(args.outdir.glob("*/run_*/summary.json"))
        + list(args.outdir.glob("*/summary.json"))
    )
    for summary_path in sorted(set(candidates)):
        try:
            summary = json.loads(summary_path.read_text())
        except Exception as exc:
            print(f"skip {summary_path}: {exc}")
            continue
        groups[_key(summary)].append({"path": summary_path, **summary})

    # Relaunched sweeps leave duplicate runs of the same seed in outdir (e.g.
    # both `heat_pinn_seed42` and `heat_pinn_seed42_<hash>` layouts). Keep one
    # run per (group, seed) — the first in path order — so mean/std are over
    # distinct seeds.
    for key, runs in groups.items():
        seen: set = set()
        unique = []
        for r in runs:
            seed = r.get("cfg", {}).get("seed")
            if seed in seen:
                continue
            seen.add(seed)
            unique.append(r)
        groups[key] = unique

    md_lines: list[str] = []
    md_lines.append("# Seed sweep and ablation summary\n")
    md_lines.append(
        f"Generated from `{args.outdir}` "
        f"covering {sum(len(v) for v in groups.values())} runs "
        f"across {len(groups)} configurations.\n"
    )
    md_lines.append(
        "| Configuration | Seeds | RMSE mean ± std | MAE mean ± std | L∞ mean ± std | Wall time (s) |"
    )
    md_lines.append("|---|---:|---:|---:|---:|---:|")
    for key in sorted(groups):
        runs = groups[key]
        seeds = [r.get("cfg", {}).get("seed", "?") for r in runs]
        rmses = [r["metrics"]["rmse"] for r in runs]
        maes = [r["metrics"]["mae"] for r in runs]
        linfs = [r["metrics"]["l_inf"] for r in runs]
        walls = [r.get("wall_time_sec", 0.0) for r in runs]
        rm, rs = _mean_std(rmses)
        ma, mas = _mean_std(maes)
        lm, ls = _mean_std(linfs)
        wm, ws = _mean_std(walls)
        seed_str = ", ".join(str(s) for s in seeds)
        md_lines.append(
            f"| {_label(key)} | {len(runs)} ({seed_str}) "
            f"| {rm:.3e} ± {rs:.1e} | {ma:.3e} ± {mas:.1e} "
            f"| {lm:.3e} ± {ls:.1e} | {wm:.0f} ± {ws:.0f} |"
        )

    md_lines.append("\n## Per-run details\n")
    for key in sorted(groups):
        md_lines.append(f"### {_label(key)}\n")
        md_lines.append("| Seed | RMSE | MAE | L∞ | NMSE | Wall (s) | Run dir |")
        md_lines.append("|---:|---:|---:|---:|---:|---:|---|")
        for r in groups[key]:
            metrics = r["metrics"]
            md_lines.append(
                f"| {r.get('cfg', {}).get('seed', '?')} "
                f"| {metrics['rmse']:.3e} | {metrics['mae']:.3e} "
                f"| {metrics['l_inf']:.3e} | {metrics['nmse']:.3e} "
                f"| {r.get('wall_time_sec', 0):.0f} | `{r['path'].parent.name}` |"
            )
        md_lines.append("")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"wrote {args.out}")

    json_groups = []
    for key in sorted(groups):
        exp, params, cutoff, nested, pretrain, epochs = key
        runs = groups[key]
        entry: dict = {
            "experiment": exp,
            "n_params": params,
            "cutoff": cutoff if cutoff > 0 else None,
            "loss": "nested" if nested else "consistency",
            "pretrain_epochs": pretrain,
            "epochs": epochs,
            "seeds": [r.get("cfg", {}).get("seed") for r in runs],
            "source_runs": [r["path"].parent.name for r in runs],
        }
        for metric in ("rmse", "mae", "l_inf", "nmse"):
            values = [r["metrics"][metric] for r in runs]
            mean, std = _mean_std(values)
            entry[metric] = {
                "values": values,
                "mean": mean,
                "std": None if len(values) == 1 else std,
            }
        entry["wall_time_sec"] = [r.get("wall_time_sec", 0.0) for r in runs]
        json_groups.append(entry)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(
            {"generated_from": str(args.outdir), "groups": json_groups}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
