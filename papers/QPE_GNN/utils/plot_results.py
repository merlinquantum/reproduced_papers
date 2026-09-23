"""Compare metrics across multiple run directories and produce summary tables.

Usage::

    python -m utils.plot_results

Reads every ``outdir/run_*/metrics.json`` produced by ``implementation.py``
and writes a comparison figure plus a markdown summary table under
``results/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def collect_runs() -> list[dict]:
    rows = []
    for run_dir in sorted((ROOT / "outdir").glob("run_*")):
        config_path = run_dir / "config_snapshot.json"
        metrics_path = run_dir / "metrics.json"
        if not config_path.exists() or not metrics_path.exists():
            continue
        cfg = json.loads(config_path.read_text())
        m = json.loads(metrics_path.read_text())
        rows.append(
            {
                "run_dir": run_dir.name,
                "config": cfg,
                "metrics": m,
            }
        )
    return rows


def main() -> None:
    rows = collect_runs()
    if not rows:
        print("no runs found in outdir/")
        return
    out_md = ROOT / "results" / "summary.md"
    with open(out_md, "w") as f:
        f.write("# Reproduction smoke-run summary\n\n")
        f.write(
            "| Run | Dataset | Model | Encoding | PE dim | Epochs | Params | Test metric | Wall (s) |\n"
        )
        f.write(
            "| --- | ------- | ----- | -------- | -----: | -----: | -----: | ---------- | -------: |\n"
        )
        for r in rows:
            cfg = r["config"]
            m = r["metrics"]
            tm = m["test_metrics"]
            if "acc" in tm:
                metric_str = f"acc={tm['acc']:.3f}"
            else:
                metric_str = f"mae={tm['mae']:.4f}"
            f.write(
                f"| {r['run_dir']} | {cfg['dataset']} | {cfg['model']} | "
                f"{cfg.get('encoding', '—')} | {cfg.get('pe_dim', '—')} | "
                f"{cfg.get('epochs', '—')} | {m['num_params']} | "
                f"{metric_str} | {m['wall_clock_s']:.1f} |\n"
            )
    print(f"wrote {out_md}")

    # Group runs by dataset and plot training curves.
    by_dataset: dict[str, list[dict]] = {}
    for r in rows:
        ds = r["config"]["dataset"]
        by_dataset.setdefault(ds, []).append(r)
    for ds, group in by_dataset.items():
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for r in group:
            hist = r["metrics"]["history"]
            cfg = r["config"]
            label = f"{cfg.get('model', '?')}/{cfg.get('encoding', '—')}"
            axes[0].plot(hist["train_loss"], label=f"{label} train", alpha=0.7)
            axes[0].plot(
                hist["val_loss"], label=f"{label} val", linestyle="--", alpha=0.7
            )
            if hist.get("train_acc"):
                axes[1].plot(hist["train_acc"], label=f"{label} train", alpha=0.7)
                axes[1].plot(
                    hist["val_acc"], label=f"{label} val", linestyle="--", alpha=0.7
                )
                axes[1].set_ylabel("accuracy")
            elif hist.get("train_mae"):
                axes[1].plot(hist["train_mae"], label=f"{label} train", alpha=0.7)
                axes[1].plot(
                    hist["val_mae"], label=f"{label} val", linestyle="--", alpha=0.7
                )
                axes[1].set_ylabel("MAE")
        for ax in axes:
            ax.set_xlabel("epoch")
            ax.legend(fontsize=7, loc="best")
        axes[0].set_title(f"{ds} — loss curves")
        axes[1].set_title(f"{ds} — task metric")
        fig.tight_layout()
        out = ROOT / "results" / "figures" / f"compare_{ds}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
