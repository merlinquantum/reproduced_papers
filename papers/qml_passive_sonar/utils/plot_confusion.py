"""Plot confusion matrices for classification runs.

Reads ``classification.json`` from each run directory and emits a grid of
confusion-matrix heatmaps (one per model).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _plot_one(ax, cm: np.ndarray, classes: list[str], title: str) -> None:
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=8)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=9)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(int(cm[i, j])),
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black",
                    fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.04)


def plot_confusion(run_dirs: list[Path], out_path: Path) -> Path:
    payloads = [json.loads((Path(d) / "classification.json").read_text())
                for d in run_dirs]
    n = len(payloads)
    fig, axes = plt.subplots(1, n, figsize=(4 * n + 0.5, 4.2), squeeze=False)
    for i, p in enumerate(payloads):
        cm = np.asarray(p["confusion_matrix"], dtype=int)
        classes = p["class_names"]
        title = f"{p['model']} ({p['dataset']})\nacc={p['final']['test_acc']:.2%}"
        _plot_one(axes[0][i], cm, classes, title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", action="append", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    out = plot_confusion(args.run_dir, args.out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
