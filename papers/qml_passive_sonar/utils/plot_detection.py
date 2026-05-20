"""Plot SAV vs DEMON spectral traces (reproduction of Fig. 4 from the paper).

Reads ``sav_detection.json`` plus the cached per-class NPZ traces written by
the runner and emits a multi-panel matplotlib figure with one row per class
(SAV trace on the left, DEMON trace on the right) and detection markers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_sav_vs_demon(run_dir: Path, out_path: Path) -> Path:
    """Render the SAV vs DEMON figure from a completed sav_detection run."""
    run_dir = Path(run_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = json.loads((run_dir / "sav_detection.json").read_text())
    traces_path = run_dir / "sav_detection_traces.npz"
    traces = np.load(traces_path, allow_pickle=True)

    classes = [r["class"] for r in payload["sav"]]
    n = len(classes)
    fig, axes = plt.subplots(n, 2, figsize=(11, 2.4 * n), sharex=False, squeeze=False)
    for i, cls in enumerate(classes):
        sav_f = traces[f"sav_{cls}_freqs"]
        sav_v = traces[f"sav_{cls}_vstack"]
        sav_thr = float(payload["sav"][i]["threshold"])
        sav_peaks = np.asarray(payload["sav"][i]["peak_freqs_hz"], dtype=float)

        demon_f = traces[f"demon_{cls}_freqs"]
        demon_v = traces[f"demon_{cls}_spectrum"]
        demon_thr = float(payload["demon"][i]["threshold"])
        demon_peaks = np.asarray(payload["demon"][i]["peak_freqs_hz"], dtype=float)
        expected = np.asarray(payload["sav"][i]["expected_freqs_hz"], dtype=float)

        ax = axes[i][0]
        ax.plot(sav_f, sav_v, color="#1f77b4", lw=0.9, label="SAV V_stack(f)")
        ax.axhline(sav_thr, color="#d62728", ls="--", lw=0.8, label=f"thr={sav_thr:.2e}")
        for f0 in expected:
            ax.axvline(f0, color="#2ca02c", alpha=0.25, lw=1.5)
        if sav_peaks.size:
            ax.scatter(sav_peaks, np.interp(sav_peaks, sav_f, sav_v), color="#d62728", s=18, zorder=5)
        ax.set_title(f"SAV — class {cls}")
        ax.set_xlim(0, min(500.0, float(sav_f.max())))
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("V_stack")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=7)

        ax = axes[i][1]
        ax.plot(demon_f, demon_v, color="#ff7f0e", lw=0.9, label="DEMON spectrum")
        ax.axhline(demon_thr, color="#d62728", ls="--", lw=0.8, label=f"thr={demon_thr:.2e}")
        for f0 in expected:
            ax.axvline(f0, color="#2ca02c", alpha=0.25, lw=1.5)
        if demon_peaks.size:
            ax.scatter(demon_peaks, np.interp(demon_peaks, demon_f, demon_v), color="#d62728", s=18, zorder=5)
        ax.set_title(f"DEMON — class {cls}")
        ax.set_xlim(0, min(500.0, float(demon_f.max())))
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=7)

    fig.suptitle("SAV vs DEMON tonal detection (synthetic data)", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    out = plot_sav_vs_demon(args.run_dir, args.out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
