"""Aggregate seed sweep + ablation summaries from outdir/run_*/summary.json.

Groups runs by (experiment, n_params, cutoff, use_nested_loss) and reports
mean ± std on RMSE / MAE / L_inf / NMSE / wall-clock. Reads only the JSON
artefacts produced by `lib/runner.py`, not the per-run logs.

Usage:

    python utils/aggregate_seeds.py [--outdir outdir]
                                   [--out results/seed_summary.md]
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
    )


def _label(key: tuple) -> str:
    exp, params, cutoff, nested = key
    bits = [exp, f"params={params}"]
    if cutoff > 0:
        bits.append(f"cutoff={cutoff}")
    bits.append("nested" if nested else "consistency")
    return ", ".join(bits)


def _mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], float("nan")
    return statistics.mean(values), statistics.stdev(values)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path,
                    default=Path("papers/CV_QPINN_PDE/outdir"))
    ap.add_argument("--out", type=Path,
                    default=Path("papers/CV_QPINN_PDE/results/seed_summary.md"))
    args = ap.parse_args()

    groups: dict[tuple, list[dict]] = defaultdict(list)
    # The launcher uses per-seed subdirectories (e.g. `heat_qpinn_seed42_d41d8c`)
    # and the runtime nests the actual run under a timestamped folder. Collect
    # both layouts so the aggregator works no matter which outdir scheme the
    # user invoked.
    candidates = (list(args.outdir.glob("run_*/summary.json"))
                  + list(args.outdir.glob("*/run_*/summary.json"))
                  + list(args.outdir.glob("*/summary.json")))
    for summary_path in sorted(set(candidates)):
        try:
            summary = json.loads(summary_path.read_text())
        except Exception as exc:
            print(f"skip {summary_path}: {exc}")
            continue
        groups[_key(summary)].append({"path": summary_path, **summary})

    md_lines: list[str] = []
    md_lines.append("# Seed sweep and ablation summary\n")
    md_lines.append(f"Generated from `{args.outdir}` "
                    f"covering {sum(len(v) for v in groups.values())} runs "
                    f"across {len(groups)} configurations.\n")
    md_lines.append("| Configuration | Seeds | RMSE mean ± std | MAE mean ± std | L∞ mean ± std | Wall time (s) |")
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
    args.out.write_text("\n".join(md_lines))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
