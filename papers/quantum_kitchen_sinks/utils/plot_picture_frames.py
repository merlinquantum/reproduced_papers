"""Visualise the picture-frames dataset and the QKS decision regions.

Usage:
    python utils/plot_picture_frames.py --out results/picture_frames_data.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make `lib.*` and `runtime_lib.*` importable when this script is invoked from anywhere.
HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent
REPO_ROOT = PROJECT_DIR.parent.parent
for p in (REPO_ROOT, PROJECT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from lib.data import load_picture_frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    X_tr, y_tr, _, _ = load_picture_frames(n_train=args.n, n_test=0, seed=0)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    for cls, marker, color in [(0, "o", "C0"), (1, "x", "C1")]:
        ax.scatter(
            X_tr[y_tr == cls, 0],
            X_tr[y_tr == cls, 1],
            marker=marker,
            color=color,
            s=8,
            alpha=0.6,
            label=f"class {cls}",
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.legend(loc="upper right")
    ax.set_title("Synthetic picture-frames dataset")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
