"""Sweep every dimension, encoding and circuit and curate the FCC results.

Writes ``results/fcc_summary.json`` with one record per configuration, plus a
grouped bar chart of the FCC scores and a chart of the number of active
frequencies. Run from the paper directory::

    python utils/summarize_fcc.py

The sweep is deterministic for a fixed ``--seed``: each configuration reseeds
before sampling, so records can be regenerated independently of one another.
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
    N_SAMPLES,
    SCALE_FACTORS,
    PhotonicSpectralModel,
    compute_fingerprint,
)

CIRCUIT_COLORS = {
    "circuit_0": "#1f77b4",
    "circuit_1": "#d62728",
    "circuit_2": "#2ca02c",
    "circuit_3": "#9467bd",
}


def sweep(seed: int, n_samples: int) -> list[dict]:
    """Compute the fingerprint for every (dimension, encoding, circuit)."""
    records = []
    for dimension in sorted(SCALE_FACTORS):
        for encoding in SCALE_FACTORS[dimension]:
            for circuit_name, circuit_index in AVAILABLE_CIRCUITS.items():
                torch.manual_seed(seed)
                np.random.seed(seed)
                model = PhotonicSpectralModel(
                    dimension=dimension,
                    encoding=encoding,
                    circuit_index=circuit_index,
                )
                fingerprint, fcc, labels, _ = compute_fingerprint(
                    model, n_samples=n_samples
                )
                records.append(
                    {
                        "dimension": dimension,
                        "encoding": encoding,
                        "circuit": circuit_name,
                        "fcc": fcc,
                        "n_active_frequencies": len(labels),
                        "n_photons": model.n_photons,
                        "n_modes": model.n_modes,
                        "n_trainable_params": sum(
                            p.numel() for p in model.parameters() if p.requires_grad
                        ),
                        "fock_dim": int(model.mode0_mask.shape[0]),
                    }
                )
                print(
                    f"{dimension}D {encoding:12s} {circuit_name:10s} "
                    f"FCC={fcc:.4f}  frequencies={len(labels)}"
                )
    return records


def _grouped_bar(records, value_key, ylabel, title, out_path):
    dimensions = sorted({r["dimension"] for r in records})
    fig, axes = plt.subplots(
        1, len(dimensions), figsize=(6 * len(dimensions), 4), squeeze=False
    )
    for ax, dimension in zip(axes[0], dimensions):
        subset = [r for r in records if r["dimension"] == dimension]
        encodings = sorted({r["encoding"] for r in subset})
        circuits = sorted({r["circuit"] for r in subset})
        positions = np.arange(len(encodings))
        width = 0.8 / len(circuits)
        for offset, circuit in enumerate(circuits):
            values = [
                next(
                    r[value_key]
                    for r in subset
                    if r["encoding"] == encoding and r["circuit"] == circuit
                )
                for encoding in encodings
            ]
            ax.bar(
                positions + offset * width - 0.4 + width / 2,
                values,
                width,
                label=circuit,
                color=CIRCUIT_COLORS.get(circuit, "gray"),
            )
        ax.set_xticks(positions)
        ax.set_xticklabels(encodings)
        ax.set_xlabel("encoding")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} ({dimension}D)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_DIR / "results",
        help="Directory to write the curated summary and figures into.",
    )
    args = parser.parse_args(argv)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    records = sweep(args.seed, args.n_samples)

    payload = {
        "seed": args.seed,
        "n_samples": args.n_samples,
        "records": records,
    }
    summary_path = args.results_dir / "fcc_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}")

    _grouped_bar(
        records,
        "fcc",
        "FCC",
        "Fourier coefficient correlation",
        args.results_dir / "fcc_by_encoding.png",
    )
    # 1D only: in 2D the active count is fixed at 25 by the |w1|+|w2| <= n_omega
    # cutoff, so every bar would be identical by construction.
    _grouped_bar(
        [r for r in records if r["dimension"] == 1],
        "n_active_frequencies",
        "active frequencies",
        "Accessible frequencies",
        args.results_dir / "active_frequencies.png",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
