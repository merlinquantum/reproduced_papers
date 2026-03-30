from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


CONFIG_RE = re.compile(r"config_(\d+)(?:_input_([01]+))?$")


@dataclass
class ConfigPoint:
    setup: str
    config_id: str
    input_state: str
    run_count: int = 0
    point_count: int = 0
    similarity_sum: float = 0.0
    diversity_sum: float = 0.0
    selected_run: str = ""

    @property
    def mean_similarity(self) -> float:
        if self.point_count == 0:
            return float("nan")
        return self.similarity_sum / self.point_count

    @property
    def mean_diversity(self) -> float:
        if self.point_count == 0:
            return float("nan")
        return self.diversity_sum / self.point_count

    @property
    def label(self) -> str:
        if self.input_state:
            return f"{self.setup}:cfg{self.config_id}:{self.input_state}"
        return f"{self.setup}:cfg{self.config_id}"


def _config_info_from_ssim_path(path: Path) -> tuple[str, str, str] | None:
    # New layout:
    # .../<mode>/setup_<x>/config_<id>_input_<bits>/run_<n>/ssim_progress.csv
    # Legacy layout:
    # .../<mode>/config_<id>/run_<n>/ssim_progress.csv
    config_dir = path.parent.parent
    match = CONFIG_RE.match(config_dir.name)
    if not match:
        return None
    config_id = match.group(1)
    input_state = match.group(2) or ""
    setup_dir = config_dir.parent
    setup = setup_dir.name if setup_dir.name.startswith("setup_") else "legacy"
    return setup, config_id, input_state


def _read_similarity_diversity(path: Path, last_n: int) -> tuple[list[float], list[float]]:
    similarity: list[float] = []
    diversity: list[float] = []
    sim_idx: int | None = None
    div_idx: int | None = None

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            if row[0].startswith("#"):
                header = [cell.strip().lstrip("#").strip().lower() for cell in row]
                if "similarity" in header:
                    sim_idx = header.index("similarity")
                elif "ssim" in header:
                    sim_idx = header.index("ssim")
                if "diversity" in header:
                    div_idx = header.index("diversity")
                continue
            if row[0].strip().lower() == "iter":
                continue
            try:
                cur_sim = float(row[sim_idx]) if sim_idx is not None and sim_idx < len(row) else float(row[-1])
                cur_div = float(row[div_idx]) if div_idx is not None and div_idx < len(row) else float("nan")
            except (ValueError, IndexError):
                continue
            if cur_div == cur_div:  # NaN check
                similarity.append(cur_sim)
                diversity.append(cur_div)

    if last_n > 0:
        return similarity[-last_n:], diversity[-last_n:]
    return similarity, diversity


def collect_points(run_dir: Path, last_n: int) -> dict[tuple[str, str, str], ConfigPoint]:
    points: dict[tuple[str, str, str], ConfigPoint] = {}
    for ssim_file in sorted(run_dir.glob("**/ssim_progress.csv")):
        info = _config_info_from_ssim_path(ssim_file)
        if info is None:
            continue
        sim_values, div_values = _read_similarity_diversity(ssim_file, last_n)
        if not sim_values or not div_values:
            continue
        setup, config_id, input_state = info
        key = (setup, config_id, input_state)
        point = points.setdefault(
            key, ConfigPoint(setup=setup, config_id=config_id, input_state=input_state)
        )
        point.run_count += 1
        point.point_count += min(len(sim_values), len(div_values))
        point.similarity_sum += sum(sim_values)
        point.diversity_sum += sum(div_values)
    return points


def collect_points_best_run(run_dir: Path, last_n: int) -> dict[tuple[str, str, str], ConfigPoint]:
    run_stats: dict[tuple[str, str, str], list[tuple[str, float, float, int]]] = {}
    for ssim_file in sorted(run_dir.glob("**/ssim_progress.csv")):
        info = _config_info_from_ssim_path(ssim_file)
        if info is None:
            continue
        sim_values, div_values = _read_similarity_diversity(ssim_file, last_n)
        if not sim_values or not div_values:
            continue
        n = min(len(sim_values), len(div_values))
        if n == 0:
            continue
        key = info
        run_name = ssim_file.parent.name
        mean_sim = sum(sim_values[:n]) / n
        mean_div = sum(div_values[:n]) / n
        run_stats.setdefault(key, []).append((run_name, mean_sim, mean_div, n))

    points: dict[tuple[str, str, str], ConfigPoint] = {}
    for (setup, config_id, input_state), stats in run_stats.items():
        best_run = max(stats, key=lambda item: item[1])
        run_name, mean_sim, mean_div, n = best_run
        points[(setup, config_id, input_state)] = ConfigPoint(
            setup=setup,
            config_id=config_id,
            input_state=input_state,
            run_count=1,
            point_count=n,
            similarity_sum=mean_sim * n,
            diversity_sum=mean_div * n,
            selected_run=run_name,
        )
    return points


def _save_summary_csv(path: Path, rows: list[ConfigPoint]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "setup",
                "config_id",
                "input_state",
                "mean_similarity",
                "mean_diversity",
                "run_count",
                "point_count",
                "selected_run",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.setup,
                    row.config_id,
                    row.input_state,
                    f"{row.mean_similarity:.8f}",
                    f"{row.mean_diversity:.8f}",
                    row.run_count,
                    row.point_count,
                    row.selected_run,
                ]
            )


def plot_similarity_vs_diversity(
    points: list[ConfigPoint],
    out_path: Path,
    title: str,
    max_diversity: float | None,
    annotate_top_k: int,
) -> int:
    filtered = [
        p
        for p in points
        if p.point_count > 0
        and p.mean_similarity == p.mean_similarity
        and p.mean_diversity == p.mean_diversity
        and (max_diversity is None or p.mean_diversity <= max_diversity)
    ]
    if not filtered:
        return 0

    setup_order = sorted({p.setup for p in filtered})
    cmap = plt.get_cmap("tab10")
    color_map = {setup: cmap(i % 10) for i, setup in enumerate(setup_order)}

    fig, ax = plt.subplots(figsize=(9, 6))
    for setup in setup_order:
        subset = [p for p in filtered if p.setup == setup]
        ax.scatter(
            [p.mean_diversity for p in subset],
            [p.mean_similarity for p in subset],
            color=color_map[setup],
            s=55,
            alpha=0.85,
            label=setup,
        )

    best = max(filtered, key=lambda p: p.mean_similarity)
    ax.scatter(
        [best.mean_diversity],
        [best.mean_similarity],
        marker="*",
        s=180,
        color="black",
        label=f"best_ssim ({best.label})",
    )

    if annotate_top_k > 0:
        ranked = sorted(filtered, key=lambda p: p.mean_similarity, reverse=True)[:annotate_top_k]
        for p in ranked:
            ax.annotate(
                p.label,
                (p.mean_diversity, p.mean_similarity),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=8,
            )

    ax.set_title(title)
    ax.set_xlabel("Diversity of generated images")
    ax.set_ylabel("Similarity to real images (SSIM)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return len(filtered)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot mean SSIM similarity vs diversity from an experiment folder "
            "(aggregated across runs and last-N steps)."
        )
    )
    parser.add_argument(
        "experiment_dir",
        type=Path,
        help=(
            "Path to a result folder (for example: "
            "papers/photonic_QGAN/results/run_20260210-123359)."
        ),
    )
    parser.add_argument(
        "--last-n",
        type=int,
        default=10,
        help="Use only the last N points from each ssim_progress.csv (default: 10).",
    )
    parser.add_argument(
        "--max-diversity",
        type=float,
        default=0.5,
        help=(
            "Optional diversity cutoff for plotting, similar to the original notebook "
            "(default: 0.5). Use a negative value to disable."
        ),
    )
    parser.add_argument(
        "--annotate-top-k",
        type=int,
        default=5,
        help="Annotate top-K points by mean similarity on the plot (default: 5).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output image path (default: <experiment_dir>/ssim_vs_diversity.png).",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Optional CSV output with aggregated metrics per config.",
    )
    parser.add_argument(
        "--best",
        action="store_true",
        help="Use only the best run per config (by mean similarity on the selected last-N window).",
    )
    args = parser.parse_args()

    experiment_dir = args.experiment_dir
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment path not found: {experiment_dir}")

    points_map = (
        collect_points_best_run(experiment_dir, args.last_n)
        if args.best
        else collect_points(experiment_dir, args.last_n)
    )
    points = sorted(points_map.values(), key=lambda p: p.mean_similarity, reverse=True)
    if not points:
        print("No valid ssim_progress.csv files found in the provided experiment directory.")
        return

    out_path = args.out or (experiment_dir / "ssim_vs_diversity.png")
    csv_out = args.csv_out or (experiment_dir / "ssim_vs_diversity_summary.csv")
    max_div = None if args.max_diversity < 0 else args.max_diversity
    mode = "best_run" if args.best else "avg_runs"
    title = f"Similarity vs Diversity ({experiment_dir.name}, last_n={args.last_n}, mode={mode})"

    plotted_count = plot_similarity_vs_diversity(
        points=points,
        out_path=out_path,
        title=title,
        max_diversity=max_div,
        annotate_top_k=args.annotate_top_k,
    )
    _save_summary_csv(csv_out, points)

    print(f"Read configs: {len(points)}")
    print(f"Plotted points: {plotted_count}")
    print(f"Selection mode: {mode}")
    print(f"Plot saved to: {out_path}")
    print(f"Summary CSV: {csv_out}")


if __name__ == "__main__":
    main()
