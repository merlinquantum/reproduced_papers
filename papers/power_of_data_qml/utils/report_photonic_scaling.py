"""Backend scaling of the MerLin photonic banks: time and memory vs Fock dimension.

Usage (from the paper folder)::

    python utils/report_photonic_scaling.py --probe outdir/run_20261019-160710 \
        --study-run outdir/run_20261017-141820 --study-run outdir/run_20261019-164643 \
        --output results/photonic_scaling

Two sources:

* the scaling probe (experiment ``photonic_scaling``): one 800-point bank per
  point, with the fidelity Gram and labels, peak device and host memory;
* the study runs: ``simulation_seconds`` of every (seed, n, embedding) unit,
  which includes the bank, labels and Gram, so the probe can be checked against
  the production runs on both machines.

Writes ``<output>.md``, ``<output>.json`` and ``<output>.png``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
sys.path[:0] = [str(REPO_ROOT), str(PROJECT_DIR)]

from lib import photonic_merlin  # noqa: E402

COLORS = {"PA": "#2a78d6", "PB": "#eb6834"}


def study_unit_timings(run_dir: Path) -> list[dict]:
    """Mean simulation seconds per (embedding, n, photons) of a study run."""
    config = json.loads((run_dir / "config_snapshot.json").read_text())
    cap = config.get("photonic", {}).get("max_photons")
    machine = (
        "L4 GPU, float32" if config["device"].startswith("cuda") else "M5 CPU, float64"
    )
    grouped = {}
    for line in (run_dir / "records.jsonl").read_text().splitlines():
        record = json.loads(line)
        embedding = record.get("embedding")
        if embedding not in photonic_merlin.PHOTONIC_EMBEDDINGS:
            continue
        n_modes = photonic_merlin.mode_count(embedding, record["n"])
        n_photons = photonic_merlin.photon_count(n_modes, cap)
        key = (embedding, record["n"], n_photons)
        grouped.setdefault(key, []).append(record)
    rows = []
    for (embedding, n, n_photons), records in sorted(grouped.items()):
        n_modes = photonic_merlin.mode_count(embedding, n)
        rows.append(
            {
                "source": run_dir.name,
                "machine": machine,
                "embedding": embedding,
                "n": n,
                "n_modes": n_modes,
                "n_photons": n_photons,
                "fock_dimension": photonic_merlin.fock_dimension(n_modes, n_photons),
                "units": len(records),
                "fidelity_kernel": all(r["quantum_kernel_available"] for r in records),
                "simulation_seconds_mean": statistics.mean(
                    r["simulation_seconds"] for r in records
                ),
                "peak_device_mb_max": max(
                    (
                        r["simulation_peak_device_mb"]
                        for r in records
                        if "simulation_peak_device_mb" in r
                    ),
                    default=None,
                ),
            }
        )
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--study-run", action="append", default=[], type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    probe = json.loads((args.probe / "metrics.json").read_text())["points"]
    measured = [p for p in probe if p["status"] == "measured"]
    if len(measured) != len(probe):
        raise ValueError(f"{len(probe) - len(measured)} probe points were not measured")
    production = [row for run in args.study_run for row in study_unit_timings(run)]

    lines = [
        "# Photonic backend scaling (MerLin 0.2.3, FOCK, threshold detectors)",
        "",
        f"Probe `{args.probe.name}`: one 800-point bank per point (states, no-click features, "
        "labels, fidelity Gram), NVIDIA L4, float32. Peak device memory from "
        "`torch.cuda.max_memory_allocated`; peak host memory is the process maximum so far.",
        "",
        "| Embedding | n | Modes | Photons | Fock dim | Simulation (s) | Gram (s) | µs per Fock state | Peak device (MB) | Peak host (MB) | Batch points |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for p in sorted(measured, key=lambda p: (p["fock_dimension"], p["embedding"])):
        total = p["simulation_seconds"] + p["gram_seconds"]
        lines.append(
            f"| {p['embedding']} | {p['n']} | {p['n_modes']} | {p['n_photons']} | "
            f"{p['fock_dimension']:,} | {p['simulation_seconds']:.1f} | {p['gram_seconds']:.1f} | "
            f"{1e6 * total / p['fock_dimension']:.0f} | {p['peak_device_mb']:,.0f} | "
            f"{p['peak_host_mb']:,.0f} | {p['batch_points']} |"
        )
    lines += [
        "",
        "## Production runs (mean `simulation_seconds` per unit: bank + labels + Gram)",
        "",
        "| Run | Machine | Embedding | n | Photons | Fock dim | Units | Fidelity kernel | Simulation (s) | Peak device (MB) |",
        "|---|---|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for r in production:
        peak = (
            "—"
            if r["peak_device_mb_max"] is None
            else f"{r['peak_device_mb_max']:,.0f}"
        )
        lines.append(
            f"| `{r['source']}` | {r['machine']} | {r['embedding']} | {r['n']} | {r['n_photons']} | "
            f"{r['fock_dimension']:,} | {r['units']} | {'yes' if r['fidelity_kernel'] else 'no'} | "
            f"{r['simulation_seconds_mean']:.1f} | {peak} |"
        )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for embedding in photonic_merlin.PHOTONIC_EMBEDDINGS:
        points = sorted(
            (p for p in measured if p["embedding"] == embedding),
            key=lambda p: p["fock_dimension"],
        )
        dims = [p["fock_dimension"] for p in points]
        axes[0].plot(
            dims,
            [p["simulation_seconds"] + p["gram_seconds"] for p in points],
            "o-",
            color=COLORS[embedding],
            label=f"{embedding}, L4 probe",
        )
        axes[1].plot(
            dims,
            [p["peak_device_mb"] / 1024 for p in points],
            "o-",
            color=COLORS[embedding],
            label=f"{embedding}, device",
        )
        axes[1].plot(
            dims,
            [p["n_photons"] and 800 * p["fock_dimension"] * 8 / 2**30 for p in points],
            ":",
            color=COLORS[embedding],
            label=f"{embedding}, stored states (host)",
        )
        cpu = sorted(
            (
                r
                for r in production
                if r["embedding"] == embedding
                and r["machine"].startswith("M5")
                and r["fidelity_kernel"]
            ),
            key=lambda r: r["fock_dimension"],
        )
        axes[0].plot(
            [r["fock_dimension"] for r in cpu],
            [r["simulation_seconds_mean"] for r in cpu],
            "s--",
            color=COLORS[embedding],
            markerfacecolor="white",
            label=f"{embedding}, M5 CPU (study units)",
        )
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Fock dimension C(m+k−1, k)")
        ax.grid(True, which="both", color="#e4e3df", linewidth=0.6)
        ax.legend(fontsize=7, frameon=False)
    axes[0].set_ylabel("seconds per 800-point bank")
    axes[1].set_ylabel("GiB")
    axes[1].axhline(22.5, color="#52514e", linewidth=0.8)
    axes[1].text(2e3, 23.5, "L4 memory", fontsize=7)
    fig.suptitle(
        "MerLin photonic banks: time and memory vs Fock dimension (simulated, noise-free)",
        fontsize=10,
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=150, facecolor="white")
    lines += ["", f"![scaling]({args.output.with_suffix('.png').name})"]
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    args.output.with_suffix(".json").write_text(
        json.dumps({"probe": probe, "production": production}, indent=2) + "\n"
    )
    print(f"wrote {args.output.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
