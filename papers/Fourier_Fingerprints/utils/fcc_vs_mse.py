"""Test the paper's Section 3.1 claim on photonic circuits.

For every encoding, each circuit topology is trained on the same set of random
Fourier targets, and the resulting mean squared error is compared against the
circuit's Fourier coefficient correlation. The paper reports that lower FCC goes
with lower MSE for gate-model ansatzes; this asks whether the same holds in
linear optics.

Run from the paper directory::

    python utils/fcc_vs_mse.py

Writes ``results/fcc_vs_mse.json`` and ``results/fcc_vs_mse.png``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from lib.fourier import (  # noqa: E402
    AVAILABLE_CIRCUITS,
    N_POINTS_1D,
    N_SAMPLES,
    SCALE_FACTORS,
    PhotonicSpectralModel,
    compute_fingerprint,
)
from lib.learning import (  # noqa: E402
    SpectralRegressor,
    random_fourier_target,
    reachable_max_frequency,
    train_regressor,
)

ENCODING_MARKERS = {"linear": "o", "exponential": "s", "balanced": "^"}
CIRCUIT_COLORS = {
    "circuit_0": "#1f77b4",
    "circuit_1": "#d62728",
    "circuit_2": "#2ca02c",
    "circuit_3": "#9467bd",
}


def spearman(a, b) -> float:
    """Rank correlation, computed without pulling in SciPy."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.size < 2:
        return float("nan")
    rank_a = np.argsort(np.argsort(a))
    rank_b = np.argsort(np.argsort(b))
    if rank_a.std() == 0 or rank_b.std() == 0:
        return float("nan")
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def run_study(seeds, epochs, lr, fingerprint_samples):
    records = []
    for encoding in SCALE_FACTORS[1]:
        max_frequency = reachable_max_frequency(encoding)
        n_points = max(N_POINTS_1D, 2 * max_frequency + 2)
        # Same targets for every circuit under this encoding, so the comparison
        # between topologies is paired rather than confounded by target draw.
        targets = [
            random_fourier_target(max_frequency, n_points, seed=seed) for seed in seeds
        ]
        print(
            f"\n=== {encoding}: frequencies 0..{max_frequency}, "
            f"{n_points} grid points, {len(seeds)} targets"
        )

        for circuit_name, circuit_index in AVAILABLE_CIRCUITS.items():
            torch.manual_seed(0)
            reference = PhotonicSpectralModel(
                dimension=1, encoding=encoding, circuit_index=circuit_index
            )
            _, fcc, labels, _ = compute_fingerprint(
                reference, n_samples=fingerprint_samples
            )

            losses = []
            for seed, target in zip(seeds, targets):
                torch.manual_seed(seed)
                np.random.seed(seed)
                model = SpectralRegressor(
                    encoding=encoding, circuit_index=circuit_index
                )
                best, _ = train_regressor(model, target, epochs=epochs, lr=lr)
                losses.append(best)

            record = {
                "encoding": encoding,
                "circuit": circuit_name,
                "fcc": fcc,
                "n_active_frequencies": len(labels),
                "max_frequency": max_frequency,
                "mse_mean": float(np.mean(losses)),
                "mse_std": float(np.std(losses)),
                "mse_per_seed": losses,
                "n_trainable_params": sum(
                    p.numel()
                    for p in SpectralRegressor(
                        encoding=encoding, circuit_index=circuit_index
                    ).parameters()
                    if p.requires_grad
                ),
            }
            records.append(record)
            print(
                f"  {circuit_name:10s} FCC={fcc:.4f}  "
                f"MSE={record['mse_mean']:.4f} ± {record['mse_std']:.4f}"
            )
    return records


def plot_study(records, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for record in records:
        ax.errorbar(
            record["mse_mean"],
            record["fcc"],
            xerr=record["mse_std"],
            marker=ENCODING_MARKERS.get(record["encoding"], "o"),
            color=CIRCUIT_COLORS.get(record["circuit"], "gray"),
            capsize=3,
            markersize=9,
            linestyle="none",
        )
    for circuit, color in CIRCUIT_COLORS.items():
        ax.plot([], [], "o", color=color, label=circuit)
    for encoding, marker in ENCODING_MARKERS.items():
        ax.plot([], [], marker, color="black", label=encoding, linestyle="none")
    ax.set_xlabel("Mean squared error")
    ax.set_ylabel("Fourier coefficient correlation")
    ax.set_title("FCC against learning error (1D random Fourier series)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--fingerprint-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--results-dir", type=Path, default=PROJECT_DIR / "results")
    args = parser.parse_args(argv)

    seeds = list(range(args.seeds))
    args.results_dir.mkdir(parents=True, exist_ok=True)
    records = run_study(seeds, args.epochs, args.lr, args.fingerprint_samples)

    per_encoding = {}
    for encoding in sorted({r["encoding"] for r in records}):
        subset = [r for r in records if r["encoding"] == encoding]
        per_encoding[encoding] = spearman(
            [r["fcc"] for r in subset], [r["mse_mean"] for r in subset]
        )
    pooled = spearman([r["fcc"] for r in records], [r["mse_mean"] for r in records])

    payload = {
        "seeds": seeds,
        "epochs": args.epochs,
        "lr": args.lr,
        "fingerprint_samples": args.fingerprint_samples,
        "spearman_fcc_vs_mse": {"per_encoding": per_encoding, "pooled": pooled},
        "records": records,
    }
    out_json = args.results_dir / "fcc_vs_mse.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out_json}")

    print("\nSpearman(FCC, MSE):")
    for encoding, value in per_encoding.items():
        print(f"  {encoding:12s} {value:+.3f}   (n=4 circuits)")
    print(f"  {'pooled':12s} {pooled:+.3f}   (n={len(records)})")

    plot_study(records, args.results_dir / "fcc_vs_mse.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
