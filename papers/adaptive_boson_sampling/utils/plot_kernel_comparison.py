"""Plot simulated kernels next to the kernels digitised from Figs. 3e and 4.

Usage::

    python utils/plot_kernel_comparison.py outdir/run_<platform_kernel> ...

For each ``platform_kernel`` run, the figure shows the digitised paper panels,
the designed kernel (fixed pre-state, B2/B3 only) and the simulated instance with
the highest Pearson r against the paper's simulated panel. That instance is picked
for display only; the tables report every instance. Figures go to the run
directory and to ``results/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def instance_key(instance: dict) -> str:
    """Return the kernels.npz key of a metrics instance."""
    seed = instance["u0_seed"]
    return "fixed_mesh" if seed is None else f"u0_seed_{seed}"


def plot_run(run_dir: Path) -> Path:
    """Write ``kernel_comparison_<platform>_<strategy>.png`` for one run."""
    metrics = json.loads((run_dir / "metrics.json").read_text())
    kernels = np.load(run_dir / "kernels.npz")
    platform = metrics["platform"]
    instances = metrics["instances"]
    strategy = instances[0]["u0_selection"]["strategy"]

    panels: dict[str, np.ndarray] = {}
    for key in kernels.files:
        if key.startswith("paper_"):
            panels[key.removeprefix("paper_").replace("_", " ")] = kernels[key]
    if "designed" in kernels.files:
        panels["designed kernel (fixed pre-state)"] = kernels["designed"]

    paper_simulated = [
        name for name in instances[0]["vs_digitized"] if "experimental" not in name
    ]
    if paper_simulated:
        reference = paper_simulated[0]
        shown = max(
            instances, key=lambda inst: inst["vs_digitized"][reference]["pearson_r"]
        )
        r = shown["vs_digitized"][reference]["pearson_r"]
        title = f"ours: {instance_key(shown)} (r={r:.2f} vs paper)"
    else:
        shown = instances[0]
        title = f"ours: {instance_key(shown)}"
    panels[title] = np.asarray(shown["kernel"])

    count = len(panels)
    figure, axes = plt.subplots(1, count, figsize=(3.6 * count, 3.6), squeeze=False)
    for axis, (name, kernel) in zip(axes[0], panels.items()):
        image = axis.imshow(kernel, cmap="viridis", vmin=0.0, vmax=1.0)
        axis.set_title(name, fontsize=8)
        axis.set_xlabel("outcome index")
    axes[0][0].set_ylabel("outcome index")
    figure.colorbar(image, ax=axes[0].tolist(), shrink=0.8, label="kernel value")
    figure.suptitle(
        f"Platform {platform}, U_0 strategy {strategy} - {metrics['validity']}",
        fontsize=9,
    )
    # Phase 3 runs carry no "simulation" record and used the numpy reference;
    # MerLin runs (Phase 4) get the backend in the file name.
    backend = instances[0].get("simulation", {"backend": "numpy"})["backend"]
    suffix = "" if backend == "numpy" else f"_{backend}"
    name = f"kernel_comparison_{platform}_{strategy}{suffix}.png"
    RESULTS_DIR.mkdir(exist_ok=True)
    for target in (run_dir / name, RESULTS_DIR / name):
        figure.savefig(target, dpi=120, bbox_inches="tight")
    plt.close(figure)
    return RESULTS_DIR / name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    for run_dir in parser.parse_args().run_dirs:
        print(plot_run(run_dir))


if __name__ == "__main__":
    main()
