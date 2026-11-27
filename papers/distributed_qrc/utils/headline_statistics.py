"""Headline statistics of the paper (MAE and RMSE reductions vs the paper baseline).

Paper rule (Sec. III-B; Tables IV-VII): for each configuration, one "best" model
is taken among the 8 quantum or hybrid cells, and both its MAE and its RMSE are
compared with the classical reservoir + classical (linear) ridge baseline as
``1 - best / baseline``. Average and peak are over the 8 configurations. The
paper's selection criterion is not stated; selecting the cell with the highest
*test* R2 reproduces the reported percentages exactly for SRSR, SRMR and MRMR
(MRSR's reported 25.6 / 30.0 % match a min-MAE choice instead; the max-R2 rule
gives 24.7 / 30.6 %), so max test R2 is used as the paper rule.

Two evaluations are produced for our sweeps:

* ``paper_rule``: exactly the rule above, applied to each seed separately and
  summarized as mean +/- std over seeds;
* ``eval_selected``: the "best" cell is selected on the evaluation-split RMSE
  instead of on test metrics (our predeclared protocol), then its test
  reductions are reported.

The same paper rule is applied to the paper's own tables as a check
(``results/data/paper_tables.json``). Writes ``results/data/headline_statistics.json``.

Usage (from the repository root; any subset of E1-E4)::

    python papers/distributed_qrc/utils/headline_statistics.py papers/distributed_qrc/outdir/sweep_E1 [...]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

PAPER_DIR = Path(__file__).resolve().parents[1]
BASELINE = ("classical", "linear")


def _reductions(cells: dict, metric: str, selector: str) -> dict:
    """cells: {(config, reservoir, readout): {split.metric: value}} for one run (or the paper)."""

    per_config = {}
    for config in dict.fromkeys(c for c, _, _ in cells):
        group = {(r, o): v for (c, r, o), v in cells.items() if c == config}
        base = group[BASELINE][f"test.{metric}"]
        candidates = {k: v for k, v in group.items() if k != BASELINE}
        if (
            selector == "test"
        ):  # paper rule: one model per configuration, highest test R2
            best_key = max(candidates, key=lambda k: candidates[k]["test.r2"])
        else:  # eval-selected: lowest eval RMSE, then report its test metrics
            best_key = min(candidates, key=lambda k: candidates[k]["eval.rmse"])
        best = candidates[best_key][f"test.{metric}"]
        per_config[config] = {
            "reduction": 1 - best / base,
            "best_cell": "|".join(best_key),
            "best": best,
            "baseline": base,
        }
    values = np.array([v["reduction"] for v in per_config.values()])
    return {
        "average": float(values.mean()),
        "peak": float(values.max()),
        "per_config": per_config,
    }


def _run_cells(metrics: dict) -> dict:
    cells = {}
    for cell in metrics["cells"]:
        key = (cell["configuration"], cell["reservoir_type"], cell["readout_type"])
        cells[key] = {
            f"{split}.{name}": value
            for split in ("eval", "test")
            for name, value in cell["metrics"][split].items()
        }
    return cells


def sweep_statistics(sweep_dir: Path) -> dict:
    status = json.loads((sweep_dir / "sweep_status.json").read_text())
    if status["status"] != "COMPLETED":
        raise ValueError(f"{sweep_dir} is {status['status']}")
    runs = [
        json.loads((sweep_dir / r["metrics_path"]).read_text()) for r in status["runs"]
    ]
    architecture = runs[0]["architecture"]
    out = {"architecture": architecture, "seeds": [r["seed"] for r in runs]}
    for rule, selector in (("paper_rule", "test"), ("eval_selected", "eval")):
        out[rule] = {}
        for metric in ("mae", "rmse"):
            per_seed = [_reductions(_run_cells(r), metric, selector) for r in runs]
            averages = np.array([p["average"] for p in per_seed])
            peaks = np.array([p["peak"] for p in per_seed])
            out[rule][metric] = {
                "average_mean": float(averages.mean()),
                "average_std": float(averages.std(ddof=1)),
                "peak_mean": float(peaks.mean()),
                "peak_std": float(peaks.std(ddof=1)),
                "per_seed": per_seed,
            }
    return out


def paper_statistics() -> dict:
    cells = json.loads(
        (PAPER_DIR / "results" / "data" / "paper_tables.json").read_text()
    )["cells"]
    out = {}
    for architecture in ("srsr", "mrsr", "srmr", "mrmr"):
        table = {
            (c["configuration"], c["reservoir_type"], c["readout_type"]): {
                "test.mae": c["mae"],
                "test.rmse": c["rmse"],
                "test.r2": c["r2"],
            }
            for c in cells
            if c["architecture"] == architecture
        }
        out[architecture] = {
            metric: _reductions(table, metric, "test") for metric in ("mae", "rmse")
        }
    return out


def main() -> int:
    result = {"paper": paper_statistics(), "ours": {}}
    for arg in sys.argv[1:]:
        stats = sweep_statistics(Path(arg))
        result["ours"][stats["architecture"]] = stats
    out = PAPER_DIR / "results" / "data" / "headline_statistics.json"
    result = {
        "paper_reference": "abstract; Tables IV-VII (best cells), computed from Tables C1-C12",
        **result,
    }
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")

    print(
        f"{'arch':5s} {'metric':5s} | {'paper avg':>9s} {'paper peak':>10s} | {'ours paper-rule avg':>21s} {'peak':>15s} | {'ours eval-selected avg':>23s} {'peak':>15s}"
    )
    for architecture, paper in result["paper"].items():
        for metric in ("mae", "rmse"):
            row = f"{architecture:5s} {metric:5s} | {100 * paper[metric]['average']:8.1f}% {100 * paper[metric]['peak']:9.1f}% |"
            ours = result["ours"].get(architecture)
            if ours:
                for rule in ("paper_rule", "eval_selected"):
                    s = ours[rule][metric]
                    row += f" {100 * s['average_mean']:7.1f}% ± {100 * s['average_std']:4.1f}  {100 * s['peak_mean']:6.1f}% ± {100 * s['peak_std']:4.1f} |"
            print(row)
    print(f"wrote {os.path.relpath(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
