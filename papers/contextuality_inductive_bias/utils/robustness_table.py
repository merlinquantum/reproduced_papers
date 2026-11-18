"""Paired comparison over regenerated training sets (data-robustness sweep).

Usage (from the repository root)::

    python3 papers/contextuality_inductive_bias/utils/robustness_table.py \\
        --sweep-dir <fig8_data_robustness_dir> --out-dir papers/contextuality_inductive_bias/results

Each model class is trained once (init seed 0) on each generated training set,
so the data seed is the replicate. For every biased quantum model and every
other class the script reports the per-data-seed final test KL, the mean
paired difference (biased - other) with a paired-t 95% interval, the paired
t-test p-value and the number of data seeds on which the biased model is lower.
With 3 data seeds the tests have low power; the sign count is the robust
summary. Writes ``fig8_data_robustness.{csv,md,json}``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep_io import describe, load_sweep  # noqa: E402


def build(sweep_dir: Path, out_dir: Path) -> dict:
    """Write the tables and return the parsed results."""
    per_model: dict[str, dict[int, float]] = {}
    for record in load_sweep(sweep_dir):
        if record.candidate["train_source"] != "generated":
            raise ValueError(f"{record.run_dir}: expected a generated training set")
        name = describe(record.candidate)
        seeds = per_model.setdefault(name, {})
        data_seed = int(record.candidate["train_seed"])
        if data_seed in seeds:
            raise ValueError(f"Duplicate data seed {data_seed} for {name}")
        seeds[data_seed] = record.metrics["final"]["test_kl"]
    data_seeds = sorted(next(iter(per_model.values())))
    if any(sorted(v) != data_seeds for v in per_model.values()):
        raise ValueError("Model classes do not share the same data seeds")
    rows = []
    for biased in (m for m in per_model if m.startswith("quantum (biased)")):
        a = np.array([per_model[biased][s] for s in data_seeds])
        for other in (m for m in per_model if m != biased):
            b = np.array([per_model[other][s] for s in data_seeds])
            diff = a - b
            half = (
                stats.t.ppf(0.975, diff.size - 1)
                * diff.std(ddof=1)
                / np.sqrt(diff.size)
            )
            rows.append(
                {
                    "biased": biased,
                    "other": other,
                    "data_seeds": " ".join(map(str, data_seeds)),
                    "biased_kl": " ".join(f"{v:.4e}" for v in a),
                    "other_kl": " ".join(f"{v:.4e}" for v in b),
                    "mean_diff": diff.mean(),
                    "ci95_low": diff.mean() - half,
                    "ci95_high": diff.mean() + half,
                    "paired_t_p": stats.ttest_rel(a, b).pvalue,
                    "biased_lower_count": int((diff < 0).sum()),
                    "n": int(diff.size),
                }
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "fig8_data_robustness.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "### Data robustness (protocol paper, init seed 0, data seeds "
        + ", ".join(map(str, data_seeds))
        + ")",
        "",
        "| Model | " + " | ".join(f"data seed {s}" for s in data_seeds) + " | mean |",
        "|---|" + "---:|" * (len(data_seeds) + 1),
    ]
    for name, seeds in per_model.items():
        values = [seeds[s] for s in data_seeds]
        lines.append(
            f"| {name} | "
            + " | ".join(f"{v:.3e}" for v in values)
            + f" | {np.mean(values):.3e} |"
        )
    lines += [
        "",
        "| Biased | Other | Mean paired diff | 95% CI (paired t) | paired t p | Biased lower |",
        "|---|---|---:|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['biased']} | {row['other']} | {row['mean_diff']:.2e} | [{row['ci95_low']:.2e}, {row['ci95_high']:.2e}] | "
            f"{row['paired_t_p']:.2g} | {row['biased_lower_count']}/{row['n']} |"
        )
    (out_dir / "fig8_data_robustness.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    result = {"sweep": str(sweep_dir), "per_model": per_model, "comparisons": rows}
    (out_dir / "fig8_data_robustness.json").write_text(
        json.dumps(result, indent=2, default=float) + "\n", encoding="utf-8"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build(args.sweep_dir, args.out_dir)
    print(f"{len(result['comparisons'])} paired comparisons -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
