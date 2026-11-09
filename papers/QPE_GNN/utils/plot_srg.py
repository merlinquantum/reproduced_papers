"""Plot the Section 4.3 strongly-regular-graph experiment from one run.

Reads ``srg_metrics.json`` from a run folder of
``configs/srg_paper_original.json`` and draws:

* ``srg_distinguished_pairs.png``: the fraction of non-isomorphic pairs each
  method separates, per SRG family;
* ``srg_distances.png``: every non-isomorphic pair distance per method and
  family on a log scale, against the decision tolerance and the largest
  isomorphic-control distance.

The figures go to the run folder and to ``--output-dir``.

Usage (from ``papers/QPE_GNN``)::

    python utils/plot_srg.py --run-dir outdir/run_YYYYMMDD-HHMMSS --output-dir results/srg

The same script draws the Phase 4 photonic run (``configs/srg_photonic.json``)
when given its run folder, ``--output-dir results/srg_photonic`` and a
``--scope`` label naming the photonic assumptions.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[1]

METHOD_LABELS = {
    "two_particle_xy": "2-particle QW",
    "ising_p2_linked": "Ising p=2 (linked-cluster proxy)",
    "rrwp": "RRWP (classical)",
    # Phase 4 photonic adaptation (configs/srg_photonic.json).
    "photon2_threshold": "2 photons, threshold",
    "photon3_threshold": "3 photons, threshold",
    "ablation_photon1": "1 photon (ablation)",
    "ablation_photon2_pnr": "2 photons, PNR (ablation)",
    "ablation_photon2_distinguishable": "2 distinguishable (ablation)",
}
SCOPE = "paper-accurate graphs and protocol; Ising p=2 parameters unpublished (proxy)"


def plot_srg(run_dir: Path, output_dir: Path, scope: str = SCOPE) -> list[Path]:
    """Draw the SRG figures of one run.

    Parameters
    ----------
    run_dir : pathlib.Path
        Run folder holding ``srg_metrics.json``.
    output_dir : pathlib.Path
        Curated output folder (normally ``results/srg``).
    scope : str
        Result-scope label shown in every title. Default value is ``SCOPE``
        (the Phase 3 paper experiment).

    Returns
    -------
    list[pathlib.Path]
        Every figure file written.
    """
    metrics = json.loads((run_dir / "srg_metrics.json").read_text(encoding="utf-8"))
    families = list(metrics["families"])
    methods = [m for m in METHOD_LABELS if m in metrics["families"][families[0]]]
    tolerance = metrics["tolerance"]

    bar_figure, bar_axis = plt.subplots(figsize=(7, 4))
    width = 0.8 / len(methods)
    # Vertical count labels keep narrow bars readable when many methods share a family.
    crowded = len(methods) > 3
    for method_index, method in enumerate(methods):
        fractions = [
            metrics["families"][family][method]["distinguished_pairs"]
            / metrics["families"][family][method]["total_pairs"]
            for family in families
        ]
        positions = (
            np.arange(len(families)) + (method_index - (len(methods) - 1) / 2) * width
        )
        bars = bar_axis.bar(positions, fractions, width, label=METHOD_LABELS[method])
        for bar, family in zip(bars, families, strict=True):
            result = metrics["families"][family][method]
            bar_axis.annotate(
                f"{result['distinguished_pairs']}/{result['total_pairs']}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7 if crowded else 8,
                rotation=90 if crowded else 0,
            )
    wrapped_scope = textwrap.fill(f"[{scope}]", 80)
    bar_axis.set_xticks(np.arange(len(families)), families)
    bar_axis.set_ylim(0, 1.3 if crowded else 1.15)
    bar_axis.set_ylabel("fraction of non-isomorphic pairs distinguished")
    bar_axis.set_title(
        f"SRG distinguishability (Sec. 4.3)\n{wrapped_scope}", fontsize=9
    )
    bar_axis.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    bar_figure.tight_layout()

    distance_figure, distance_axes = plt.subplots(
        1, len(families), figsize=(5.5 * len(families), 5), squeeze=False
    )
    floor = 1e-18
    for axis_index, (axis, family) in enumerate(
        zip(distance_axes[0], families, strict=True)
    ):
        for method_index, method in enumerate(methods):
            result = metrics["families"][family][method]
            matrix = np.asarray(result["distance_matrix"], dtype=float)
            upper = matrix[np.triu_indices_from(matrix, k=1)]
            jitter = np.random.default_rng(method_index).uniform(
                -0.15, 0.15, upper.size
            )
            axis.scatter(method_index + jitter, np.maximum(upper, floor), s=8)
            axis.hlines(
                max(result["maximum_isomorphic_control_distance"], floor),
                method_index - 0.3,
                method_index + 0.3,
                colors="black",
                label="max isomorphic control" if method_index == 0 else None,
            )
        axis.axhline(tolerance, color="#C44E52", linestyle="--", label="tolerance")
        axis.set_yscale("log")
        axis.set_xticks(
            range(len(methods)),
            [METHOD_LABELS[m] for m in methods],
            fontsize=7,
            rotation=30,
            ha="right",
        )
        if axis_index == 0:
            axis.set_ylabel(
                f"pair distance (values below {floor:g} drawn at {floor:g})"
            )
        axis.set_title(family)
    handles, labels = distance_axes[0][0].get_legend_handles_labels()
    distance_figure.legend(handles, labels, fontsize=8, loc="lower center", ncol=2)
    distance_figure.suptitle(f"SRG pair distances\n{wrapped_scope}", fontsize=9)
    distance_figure.tight_layout(rect=(0, 0.06, 1, 1))

    written = []
    for folder in dict.fromkeys([output_dir, run_dir]):
        folder.mkdir(parents=True, exist_ok=True)
        for file_name, figure in (
            ("srg_distinguished_pairs.png", bar_figure),
            ("srg_distances.png", distance_figure),
        ):
            path = folder / file_name
            figure.savefig(path, dpi=150)
            written.append(path)
    plt.close(bar_figure)
    plt.close(distance_figure)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--scope",
        default=SCOPE,
        help="Result-scope label for the titles (default: the Phase 3 label)",
    )
    args = parser.parse_args(argv)

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else PROJECT_DIR / path

    for path in plot_srg(resolve(args.run_dir), resolve(args.output_dir), args.scope):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
