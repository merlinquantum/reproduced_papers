from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

CONFIG_RE = re.compile(r"config_(\d+)(?:_input_([01]+))?$")


@dataclass
class ConfigStats:
    setup: str
    config_id: str
    input_state: str
    run_count: int = 0
    point_count: int = 0
    ssim_sum: float = 0.0
    selected_run: str = ""

    @property
    def mean_ssim(self) -> float:
        if self.point_count == 0:
            return float("nan")
        return self.ssim_sum / self.point_count


def _read_ssim_values(path: Path, last_n: int) -> list[float]:
    values: list[float] = []
    metric_idx: int | None = None
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            # Parse header emitted by np.savetxt(header=...).
            if row[0].startswith("#"):
                header = [cell.strip().lstrip("#").strip().lower() for cell in row]
                if "ssim" in header:
                    metric_idx = header.index("ssim")
                elif "similarity" in header:
                    metric_idx = header.index("similarity")
                continue
            if row[0] == "iter":
                continue
            try:
                if metric_idx is not None and metric_idx < len(row):
                    values.append(float(row[metric_idx]))
                else:
                    values.append(float(row[-1]))
            except (ValueError, IndexError):
                continue
    if last_n > 0:
        return values[-last_n:]
    return values


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


def collect_config_stats(
    run_dir: Path, last_n: int
) -> dict[tuple[str, str, str], ConfigStats]:
    stats: dict[tuple[str, str, str], ConfigStats] = {}
    for ssim_file in sorted(run_dir.glob("**/ssim_progress.csv")):
        info = _config_info_from_ssim_path(ssim_file)
        if info is None:
            continue
        setup, config_id, input_state = info
        values = _read_ssim_values(ssim_file, last_n)
        if not values:
            continue
        key = (setup, config_id, input_state)
        cfg = stats.setdefault(
            key, ConfigStats(setup=setup, config_id=config_id, input_state=input_state)
        )
        cfg.run_count += 1
        cfg.point_count += len(values)
        cfg.ssim_sum += sum(values)
    return stats


def collect_config_stats_best_run(
    run_dir: Path, last_n: int
) -> dict[tuple[str, str, str], ConfigStats]:
    run_stats: dict[tuple[str, str, str], list[tuple[str, float, int]]] = {}
    for ssim_file in sorted(run_dir.glob("**/ssim_progress.csv")):
        info = _config_info_from_ssim_path(ssim_file)
        if info is None:
            continue
        values = _read_ssim_values(ssim_file, last_n)
        if not values:
            continue
        n = len(values)
        run_name = ssim_file.parent.name
        mean_val = sum(values) / n
        run_stats.setdefault(info, []).append((run_name, mean_val, n))

    stats: dict[tuple[str, str, str], ConfigStats] = {}
    for (setup, config_id, input_state), entries in run_stats.items():
        run_name, best_mean, n = max(entries, key=lambda item: item[1])
        stats[(setup, config_id, input_state)] = ConfigStats(
            setup=setup,
            config_id=config_id,
            input_state=input_state,
            run_count=1,
            point_count=n,
            ssim_sum=best_mean * n,
            selected_run=run_name,
        )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank photonic_QGAN configs by average SSIM from a run folder."
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Path to a run directory (e.g. papers/photonic_QGAN/results/run_20260209-160101).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="Show only the top N configs (0 means all).",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Optional output CSV path for the ranking table.",
    )
    parser.add_argument(
        "--last-n",
        type=int,
        default=10,
        help="Use only the last N SSIM steps from each run file (default: 10).",
    )
    parser.add_argument(
        "--best",
        action="store_true",
        help="Use only the best run per config (by mean SSIM on the selected last-N window).",
    )
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    stats = (
        collect_config_stats_best_run(run_dir, args.last_n)
        if args.best
        else collect_config_stats(run_dir, args.last_n)
    )
    ranking = sorted(stats.values(), key=lambda item: item.mean_ssim, reverse=True)

    if args.top > 0:
        ranking = ranking[: args.top]

    if not ranking:
        print("No valid ssim_progress.csv files found.")
        return

    print("setup,config_id,input_state,mean_ssim,run_count,point_count,selected_run")
    for item in ranking:
        print(
            f"{item.setup},{item.config_id},{item.input_state},{item.mean_ssim:.6f},{item.run_count},{item.point_count},{item.selected_run}"
        )

    if args.csv_out is not None:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "setup",
                    "config_id",
                    "input_state",
                    "mean_ssim",
                    "run_count",
                    "point_count",
                    "selected_run",
                ]
            )
            for item in ranking:
                writer.writerow(
                    [
                        item.setup,
                        item.config_id,
                        item.input_state,
                        f"{item.mean_ssim:.6f}",
                        item.run_count,
                        item.point_count,
                        item.selected_run,
                    ]
                )


if __name__ == "__main__":
    main()
