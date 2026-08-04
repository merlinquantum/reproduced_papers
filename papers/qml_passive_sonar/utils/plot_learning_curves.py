"""Plot per-epoch train/test learning curves for one or more runs.

Each run directory must contain ``classification.json`` with an
``epochs: [{train_loss, train_acc, test_loss, test_acc}, ...]`` array.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_curves(run_dirs: list[Path], out_path: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for d in run_dirs:
        p = json.loads((Path(d) / "classification.json").read_text())
        epochs = list(range(1, len(p["epochs"]) + 1))
        train_acc = [e["train_acc"] for e in p["epochs"]]
        test_acc = [e["test_acc"] for e in p["epochs"]]
        train_loss = [e["train_loss"] for e in p["epochs"]]
        test_loss = [e["test_loss"] for e in p["epochs"]]
        label = f"{p['model']} ({p['dataset']})"
        ax1.plot(epochs, train_acc, ls="--", label=f"{label} train", alpha=0.7)
        ax1.plot(epochs, test_acc, label=f"{label} test")
        ax2.plot(epochs, train_loss, ls="--", alpha=0.7)
        ax2.plot(epochs, test_loss, label=label)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Accuracy")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=7)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Loss")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=7)

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
    out = plot_curves(args.run_dir, args.out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
