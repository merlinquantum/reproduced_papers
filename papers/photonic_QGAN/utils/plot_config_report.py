from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

ITER_IMG_RE = re.compile(r"fake_iter_(\d+)\.png$")


def _find_run_root(config_dir: Path) -> Path | None:
    for parent in [config_dir] + list(config_dir.parents):
        if (parent / "config_snapshot.json").exists():
            return parent
    return None


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _list_run_dirs(config_dir: Path) -> list[Path]:
    runs = [p for p in sorted(config_dir.glob("run_*")) if p.is_dir()]
    return runs


def _ssim_column_index(ssim_csv: Path) -> int:
    try:
        with ssim_csv.open("r", encoding="utf-8") as handle:
            first = handle.readline().strip()
    except OSError:
        return -1
    if first.startswith("#"):
        header = [h.strip().lstrip("#").strip().lower() for h in first.split(",")]
        if "ssim" in header:
            return header.index("ssim")
        if "similarity" in header:
            return header.index("similarity")
    return -1


def _run_ssim_score(run_dir: Path, last_n: int = 10) -> float:
    ssim_csv = run_dir / "ssim_progress.csv"
    if not ssim_csv.exists():
        return float("-inf")
    idx = _ssim_column_index(ssim_csv)
    vals: list[float] = []
    try:
        with ssim_csv.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p.strip() for p in line.split(",")]
                if parts[0].lower() == "iter":
                    continue
                try:
                    if idx >= 0 and idx < len(parts):
                        vals.append(float(parts[idx]))
                    else:
                        vals.append(float(parts[-1]))
                except ValueError:
                    continue
    except OSError:
        return float("-inf")
    if not vals:
        return float("-inf")
    window = vals[-last_n:] if last_n > 0 else vals
    return float(np.mean(window))


def _choose_best_run(run_dirs: list[Path], last_n: int = 10) -> tuple[Path, float]:
    best_run = run_dirs[0]
    best_score = float("-inf")
    for run_dir in run_dirs:
        score = _run_ssim_score(run_dir, last_n=last_n)
        if score > best_score:
            best_score = score
            best_run = run_dir
    return best_run, best_score


def _aggregate_losses(
    run_dirs: list[Path],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    loss_runs: list[np.ndarray] = []
    for run_dir in run_dirs:
        loss_csv = run_dir / "loss_progress.csv"
        if not loss_csv.exists():
            continue
        losses = np.loadtxt(loss_csv, delimiter=",", comments="#")
        if losses.ndim == 1:
            losses = losses.reshape(1, -1)
        if losses.shape[1] < 2:
            continue
        loss_runs.append(losses[:, :2])
    if not loss_runs:
        raise FileNotFoundError("No loss_progress.csv found in selected run folders.")

    min_len = min(arr.shape[0] for arr in loss_runs)
    stacked = np.stack([arr[:min_len] for arr in loss_runs], axis=0)  # [R, T, 2]
    # D_loss is stored as errD_real + errD_fake; divide by 2 to get the mean of the
    # two terms, matching the original parse_results notebooks convention.
    d_mean = stacked[:, :, 0].mean(axis=0) / 2
    d_std = stacked[:, :, 0].std(axis=0) / 2
    g_mean = stacked[:, :, 1].mean(axis=0)
    g_std = stacked[:, :, 1].std(axis=0)
    iterations = np.arange(1, min_len + 1)
    return iterations, d_mean, d_std, g_mean, g_std


def _pick_five_iteration_images(run_dir: Path) -> list[tuple[int, Path]]:
    img_dir = run_dir / "generated_every_100"
    if not img_dir.exists():
        return []
    candidates: list[tuple[int, Path]] = []
    for p in sorted(img_dir.glob("fake_iter_*.png")):
        m = ITER_IMG_RE.match(p.name)
        if m:
            candidates.append((int(m.group(1)), p))
    if not candidates:
        return []

    max_iter = max(it for it, _ in candidates)
    targets = [
        max(1, int(round(max_iter * frac))) for frac in (0.2, 0.4, 0.6, 0.8, 1.0)
    ]
    picked: list[tuple[int, Path]] = []
    used = set()
    for target in targets:
        best_idx = None
        best_dist = None
        for idx, (it, _) in enumerate(candidates):
            if idx in used:
                continue
            dist = abs(it - target)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_idx = idx
        if best_idx is not None:
            used.add(best_idx)
            picked.append(candidates[best_idx])
    return sorted(picked, key=lambda x: x[0])


def _fallback_five_images_from_fake_progress(
    run_dir: Path, image_size: int
) -> list[np.ndarray]:
    fake_csv = run_dir / "fake_progress.csv"
    if not fake_csv.exists():
        return []
    data = np.loadtxt(fake_csv, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    n = data.shape[0]
    if n == 0:
        return []
    idx = np.linspace(0, n - 1, 5, dtype=int)
    pixels = image_size * image_size
    out: list[np.ndarray] = []
    for i in idx:
        row = data[i].reshape(-1)
        if row.size >= pixels:
            out.append(row[:pixels].reshape(image_size, image_size))
    return out


def _generate_from_saved_model(
    config_dir: Path,
    run_dir: Path,
    image_size: int,
    lossy: bool,
    use_clements: bool,
    sim: bool,
) -> list[np.ndarray]:
    import torch
    from lib.generators import PatchGenerator

    config = _load_json(config_dir / "config.json")
    state_path = run_dir / "G_params_progress.pt"
    if not state_path.exists():
        return []

    gen = PatchGenerator(
        image_size=image_size,
        gen_count=int(config["gen_count"]),
        gen_arch=list(config["arch"]),
        input_state=list(config["input_state"]),
        pnr=bool(config["pnr"]),
        lossy=lossy,
        remote_token=None,
        use_clements=use_clements,
        sim=sim,
    )
    state_dict = torch.load(state_path, map_location="cpu")
    gen.load_state_dict(state_dict)
    gen.eval()

    noise_dim = int(config["noise_dim"])
    with torch.no_grad():
        z = torch.normal(0, 2 * np.pi, (5, noise_dim))
        out = gen(z).detach().cpu().numpy()
    return [out[i].reshape(image_size, image_size) for i in range(min(5, out.shape[0]))]


def build_report(
    config_dir: Path,
    run_name: str | None,
    out_path: Path | None,
    use_saved_model: bool,
    ssim_last_n: int,
) -> Path:
    if run_name:
        run_dirs = [config_dir / run_name]
        if not run_dirs[0].exists():
            raise FileNotFoundError(f"Run folder not found: {run_dirs[0]}")
    else:
        run_dirs = _list_run_dirs(config_dir)
        if not run_dirs:
            raise FileNotFoundError(f"No run_* folders found in: {config_dir}")

    best_run_dir, best_run_ssim = _choose_best_run(run_dirs, last_n=ssim_last_n)

    run_root = _find_run_root(config_dir)
    snapshot = _load_json(run_root / "config_snapshot.json") if run_root else {}
    model_cfg = snapshot.get("model", {})
    ideal_cfg = snapshot.get("ideal", {})
    image_size = int(model_cfg.get("image_size", 8))
    lossy = bool(model_cfg.get("lossy", False))
    use_clements = bool(model_cfg.get("use_clements", False))
    sim = bool(model_cfg.get("sim", False))
    digit = ideal_cfg.get("digit", "unknown")

    iterations, d_mean, d_std, g_mean, g_std = _aggregate_losses(run_dirs)

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(
        2, 2, width_ratios=[1.9, 2.1], height_ratios=[1, 1], wspace=0.25, hspace=0.35
    )

    ax_loss = fig.add_subplot(gs[:, 0])
    ax_loss.plot(
        iterations, g_mean, color="tab:blue", label="G Loss (mean)", linewidth=1.7
    )
    ax_loss.fill_between(
        iterations, g_mean - g_std, g_mean + g_std, color="tab:blue", alpha=0.18
    )
    ax_loss.plot(
        iterations, d_mean, color="tab:orange", label="D Loss (mean)", linewidth=1.7
    )
    ax_loss.fill_between(
        iterations, d_mean - d_std, d_mean + d_std, color="tab:orange", alpha=0.18
    )
    ax_loss.set_title(f"Loss vs Iterations (avg across {len(run_dirs)} run(s))")
    ax_loss.set_xlabel("Iteration")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend(loc="best")

    top_gs = gs[0, 1].subgridspec(1, 5, wspace=0.05)
    picked = _pick_five_iteration_images(best_run_dir)
    if picked:
        for i, (it, p) in enumerate(picked):
            ax = fig.add_subplot(top_gs[0, i])
            ax.imshow(mpimg.imread(p), cmap="gray")
            ax.set_title(f"it={it}", fontsize=9)
            ax.axis("off")
    else:
        fallback_imgs = _fallback_five_images_from_fake_progress(
            best_run_dir, image_size
        )
        for i in range(5):
            ax = fig.add_subplot(top_gs[0, i])
            if i < len(fallback_imgs):
                ax.imshow(fallback_imgs[i], cmap="gray")
                ax.set_title("fallback", fontsize=9)
            ax.axis("off")
    fig.text(
        0.67,
        0.92,
        f"Image Evolution (best run: {best_run_dir.name}, SSIM={best_run_ssim:.4f})",
        ha="center",
        va="center",
        fontsize=11,
    )

    bottom_gs = gs[1, 1].subgridspec(1, 5, wspace=0.05)
    generated = []
    generated_title = f"Generated with Saved Model (target digit={digit})"
    if use_saved_model:
        try:
            generated = _generate_from_saved_model(
                config_dir=config_dir,
                run_dir=best_run_dir,
                image_size=image_size,
                lossy=lossy,
                use_clements=use_clements,
                sim=sim,
            )
        except Exception:
            generated = []
    if not generated:
        generated = _fallback_five_images_from_fake_progress(best_run_dir, image_size)
        if use_saved_model:
            generated_title = (
                f"Generated (fallback from fake_progress, target digit={digit})"
            )
        else:
            generated_title = f"Generated from fake_progress (target digit={digit})"

    for i in range(5):
        ax = fig.add_subplot(bottom_gs[0, i])
        if i < len(generated):
            ax.imshow(generated[i], cmap="gray")
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=9)
        ax.axis("off")
    fig.text(
        0.67,
        0.47,
        generated_title,
        ha="center",
        va="center",
        fontsize=11,
    )

    config_name = config_dir.name
    run_label = run_name if run_name else "all_runs"
    fig.suptitle(f"Photonic QGAN Report: {config_name} / {run_label}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if out_path is None:
        out_path = config_dir / f"{run_label}_report.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a summary plot for one photonic_QGAN config folder."
    )
    parser.add_argument(
        "config_dir",
        type=Path,
        help="Path like .../ideal/setup_a/config_0_input_1111",
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Optional specific run folder (e.g. run_1). If omitted, aggregate all run_*.",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Optional output image path."
    )
    parser.add_argument(
        "--no-model",
        action="store_true",
        help="Do not load saved model for lower-right panel; use fake_progress fallback.",
    )
    parser.add_argument(
        "--ssim-last-n",
        type=int,
        default=10,
        help="Use last N SSIM values to choose the best run (default: 10).",
    )
    args = parser.parse_args()

    out = build_report(
        args.config_dir,
        args.run,
        args.out,
        use_saved_model=not args.no_model,
        ssim_last_n=args.ssim_last_n,
    )
    print(f"Saved report: {out}")


if __name__ == "__main__":
    main()
