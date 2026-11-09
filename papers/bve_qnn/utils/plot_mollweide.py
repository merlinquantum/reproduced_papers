"""Standalone utility to regenerate the paper's Mollweide comparison figure.

Usage (from papers/bve_qnn/):
    python utils/plot_mollweide.py --hour 22
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def plot_mollweide(
    results_path: Path,
    target_hour: int,
    out_path: Path,
    data_path: Path | None = None,
) -> None:
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    exp1_results = np.load(results_path)
    psi_pred_training = exp1_results["psi_pred_training"]
    psi_qcl_training = exp1_results["psi_qcl_training"]
    training_hours = exp1_results["training_hours"]

    if "lat_downsampled" in exp1_results.files:
        lat_downsampled = exp1_results["lat_downsampled"]
        lon_downsampled = exp1_results["lon_downsampled"]
    elif data_path is not None and data_path.exists():
        sem_data = np.load(data_path)
        lat_downsampled = sem_data["lat_downsampled"]
        lon_downsampled = sem_data["lon_downsampled"]
    else:
        raise FileNotFoundError(
            "lat/lon grid not found in the results file. Re-run evaluation "
            "or pass --data pointing at the SEM dataset."
        )

    matching_indices = np.where(training_hours == target_hour)[0]
    if len(matching_indices) == 0:
        available_hours = ", ".join(str(int(hour)) for hour in training_hours)
        raise ValueError(
            f"Hour {target_hour} is not available in {results_path}. "
            f"Available hours: {available_hours}."
        )
    target_index = matching_indices[0]
    psi_sem_t = psi_qcl_training[target_index]
    psi_qnn_t = psi_pred_training[target_index]

    # Visual scaling only, to mimic the paper colorbar around [-1, 2].
    psi_min, psi_max = psi_sem_t.min(), psi_sem_t.max()
    psi_sem_plot = 3.0 * (psi_sem_t - psi_min) / (psi_max - psi_min) - 1.0
    psi_qnn_plot = 3.0 * (psi_qnn_t - psi_min) / (psi_max - psi_min) - 1.0

    lon_grid, lat_grid = np.meshgrid(lon_downsampled, lat_downsampled)
    projection = ccrs.Mollweide(central_longitude=0)
    data_crs = ccrs.PlateCarree()

    fig = plt.figure(figsize=(16, 7))
    ax_sem = fig.add_subplot(1, 2, 1, projection=projection)
    ax_qnn = fig.add_subplot(1, 2, 2, projection=projection)

    for ax, field, title in [
        (ax_sem, psi_sem_plot, "SEM"),
        (ax_qnn, psi_qnn_plot, "Quantum"),
    ]:
        mesh = ax.pcolormesh(
            lon_grid,
            lat_grid,
            field,
            transform=data_crs,
            cmap="nipy_spectral",
            vmin=-1.0,
            vmax=2.2,
            shading="auto",
        )
        ax.set_global()
        ax.gridlines(
            draw_labels=False, linewidth=0.8, color="black", alpha=0.7, linestyle=":"
        )
        ax.coastlines(linewidth=0.6, alpha=0.5)
        ax.set_title(title, fontsize=28, pad=18)

    cbar = fig.colorbar(
        mesh, ax=[ax_sem, ax_qnn], orientation="horizontal", fraction=0.055, pad=0.08
    )
    cbar.set_label(r"$\psi$", fontsize=30)
    cbar.ax.tick_params(labelsize=14)

    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    print(f"saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=Path("results/exp1_merlin_results.npz")
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Optional SEM npz used only if results lack lat/lon arrays",
    )
    parser.add_argument("--hour", type=int, default=22)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/exp1_real_data_comparison_t22_cartopy.png"),
    )
    args = parser.parse_args()

    plot_mollweide(args.results, args.hour, args.out, data_path=args.data)


if __name__ == "__main__":
    main()
