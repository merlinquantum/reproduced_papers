"""Bar chart of final test accuracy across CNN / HQ-CNN / MerLin models.

Reads multiple ``classification.json`` files from completed runs and emits a
side-by-side bar chart (mirrors Fig. 9 of the paper).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _label(payload: dict) -> str:
    model = payload.get("model", "?")
    dataset = payload.get("dataset", "?")
    return f"{model}\n({dataset})"


def plot_bars(run_dirs: list[Path], out_path: Path, title: str) -> Path:
    payloads = []
    for d in run_dirs:
        p = Path(d) / "classification.json"
        if not p.exists():
            raise FileNotFoundError(p)
        payloads.append(json.loads(p.read_text()))

    labels = [_label(p) for p in payloads]
    train_acc = [p["final"]["train_acc"] for p in payloads]
    test_acc = [p["final"]["test_acc"] for p in payloads]
    gap = [p["final"]["generalization_gap"] for p in payloads]

    x = np.arange(len(payloads))
    w = 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.bar(x - w / 2, train_acc, w, label="train", color="#9ecae1")
    ax1.bar(x + w / 2, test_acc, w, label="test", color="#3182bd")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Final accuracy")
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend()

    ax2.bar(x, gap, color="#e6550d")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("train_acc − test_acc")
    ax2.set_title("Generalization gap (lower = better)")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", action="append", required=True, type=Path,
                   help="Repeatable. Each path is a run_* directory.")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--title", default="CNN vs HQ-CNN vs MerLin photonic")
    args = p.parse_args()
    out = plot_bars(args.run_dir, args.out, args.title)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
