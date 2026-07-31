"""Sanity-check the autoencoder: plot real digits vs their reconstructions.

The whole LatentQGAN pipeline hinges on a well-trained decoder: if the
autoencoder only reconstructs dark blobs, *every* generated image will be
black regardless of how good the generator is. Use this script to confirm
the cached AE reconstructs recognisable digits before trusting GAN samples.

Usage::

    python utils/plot_reconstructions.py --config configs/mnist_reduced.json --seed 0 --digit 0

Writes ``results/ae_reconstructions.png`` (top row: real, bottom: reconstructed).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

HERE = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(HERE))

from lib.autoencoder import Autoencoder  # noqa: E402
from lib.data import load_mnist, subset_by_class  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mnist_reduced.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--digit", type=int, default=0)
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    cfg = json.loads((HERE / args.config).read_text())
    T, N, NA = int(cfg.get("T", 5)), int(cfg.get("N", 4)), int(cfg.get("NA", 1))

    cache_dir = HERE / "outdir" / "_ae_cache"
    key = (f"ae_seed{args.seed}_size{cfg.get('ae_data_size', 'all')}"
           f"_ep{cfg.get('ae_epochs', 1)}_bs{cfg.get('ae_batch_size', 20)}"
           f"_lr{cfg.get('ae_lr', 0.05)}.pt")
    cache_path = cache_dir / key
    if not cache_path.exists():
        raise SystemExit(
            f"No cached AE at {cache_path}.\n"
            f"Run e.g.  python implementation.py --config {args.config} --seed {args.seed}  first.")

    ae = Autoencoder(T=T, NG=N - NA)
    ae.load_state_dict(torch.load(cache_path, map_location="cpu"))
    ae.eval()

    imgs, labels = load_mnist(cfg, train=True)
    d = subset_by_class(imgs, labels, args.digit, None)[: args.n]
    with torch.no_grad():
        recon = ae(d)
    mse = torch.nn.functional.mse_loss(recon, d).item()
    print(f"AE reconstruction MSE on digit {args.digit}: {mse:.4f} "
          f"(recon pixel max={recon.max():.3f}; a healthy AE reaches ~0.02 "
          "and max close to 1.0)")

    fig, ax = plt.subplots(2, args.n, figsize=(args.n, 2.2))
    for i in range(args.n):
        ax[0, i].imshow(d[i, 0], cmap="gray", vmin=0, vmax=1)
        ax[0, i].axis("off")
        ax[1, i].imshow(recon[i, 0], cmap="gray", vmin=0, vmax=1)
        ax[1, i].axis("off")
    ax[0, 0].set_title("real", fontsize=8, loc="left")
    ax[1, 0].set_title("reconstructed", fontsize=8, loc="left")
    fig.suptitle(f"AE reconstruction (digit {args.digit}, MSE={mse:.3f})", fontsize=9)
    fig.tight_layout()
    out = HERE / "results" / "ae_reconstructions.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
