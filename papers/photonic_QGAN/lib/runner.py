from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

from loguru import logger
from tqdm.auto import tqdm

# Set up paths based on the runner location
RUNNER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RUNNER_DIR.parent  # photonic_QGAN/
REPO_ROOT = PROJECT_ROOT.parent.parent  # reproduced_papers/


def _resolve_digit_list(cfg: dict) -> list[int]:
    digits_cfg = cfg.get("digits", {})
    explicit = digits_cfg.get("digits")
    if explicit:
        return [int(d) for d in explicit]
    start = int(digits_cfg.get("digit_start", 1))
    end = int(digits_cfg.get("digit_end", 9))
    return list(range(start, end + 1))


def _coerce_str_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        return [chunk.strip() for chunk in text.split(",") if chunk.strip()]
    if isinstance(loaded, list):
        return [str(item) for item in loaded]
    return [str(loaded)]


def _coerce_int_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [int(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        return [int(chunk.strip()) for chunk in text.split(",") if chunk.strip()]
    if isinstance(loaded, list):
        return [int(item) for item in loaded]
    return [int(loaded)]


def _load_config_grid(path_value) -> list[dict]:
    if not path_value:
        return []
    path = Path(path_value)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "config_grid" in payload:
        payload = payload["config_grid"]
    if not isinstance(payload, list):
        raise ValueError("ideal.config_grid_path must point to a JSON list")
    return payload


def _default_ideal_grid() -> list[dict]:
    arch_grid_45modes = [
        {"noise_dim": 1, "arch": ["var", "enc[2]", "var"]},
        {"noise_dim": 1, "arch": ["var", "var", "enc[2]", "var", "var"]},
        {"noise_dim": 2, "arch": ["var", "enc[1]", "var", "enc[3]", "var"]},
        {
            "noise_dim": 2,
            "arch": ["var", "var", "enc[1]", "var", "var", "enc[3]", "var", "var"],
        },
    ]
    input_grid_4modes = [
        {"input_state": [1, 1, 1, 1], "gen_count": 2, "pnr": True},
        {"input_state": [1, 1, 1, 1], "gen_count": 4, "pnr": False},
        {"input_state": [1, 0, 1, 1], "gen_count": 4, "pnr": True},
    ]
    arch_grid_5modes = [
        {
            "noise_dim": 3,
            "arch": ["var", "enc[0]", "var", "enc[2]", "var", "enc[4]", "var"],
        },
    ]
    input_grid_5modes = [
        {"input_state": [0, 1, 0, 1, 0], "gen_count": 4, "pnr": False},
        {"input_state": [1, 0, 1, 0, 1], "gen_count": 2, "pnr": True},
    ]
    arch_grid_8modes = [
        {"noise_dim": 1, "arch": ["var", "enc[4]", "var"]},
        {"noise_dim": 1, "arch": ["var", "var", "enc[4]", "var", "var"]},
        {"noise_dim": 2, "arch": ["var", "enc[2]", "var", "enc[5]", "var"]},
        {
            "noise_dim": 2,
            "arch": ["var", "var", "enc[2]", "var", "var", "enc[5]", "var", "var"],
        },
        {
            "noise_dim": 3,
            "arch": ["var", "enc[1]", "var", "enc[4]", "var", "enc[6]", "var"],
        },
        {
            "noise_dim": 4,
            "arch": [
                "var",
                "enc[1]",
                "var",
                "enc[3]",
                "var",
                "enc[5]",
                "var",
                "enc[7]",
                "var",
            ],
        },
    ]
    input_grid_8modes = [
        {"input_state": [0, 0, 1, 0, 0, 1, 0, 0], "gen_count": 2, "pnr": False},
    ]

    config_grid: list[dict] = []
    for inp in input_grid_5modes:
        for arch in arch_grid_45modes + arch_grid_5modes:
            config = inp.copy()
            config.update(arch)
            config_grid.append(config)
    for inp in input_grid_4modes:
        for arch in arch_grid_45modes:
            config = inp.copy()
            config.update(arch)
            config_grid.append(config)
    for inp in input_grid_8modes:
        for arch in arch_grid_8modes:
            config = inp.copy()
            config.update(arch)
            config_grid.append(config)
    return config_grid


def _resolve_csv_path(
    csv_path: str | Path, data_root: str | Path | None, project_dir: Path
) -> Path:
    """Resolve CSV file path with multiple fallback strategies.

    Priority order:
    1. Absolute paths are returned as-is
    2. repo_root/data/<csv_path> (default data directory)
    3. data_root/<csv_path> if data_root is provided
    4. project_dir/<csv_path> (legacy fallback)
    """
    path = Path(csv_path)
    if path.is_absolute():
        return path

    candidate_paths: list[Path] = []

    # Priority 1: Default data directory at repo root
    candidate_paths.append((REPO_ROOT / "data" / path).resolve())

    # Priority 2: Explicit data_root if provided
    if data_root:
        candidate_paths.append((Path(data_root) / path).resolve())

    # Priority 3: Relative to project_dir (legacy fallback)
    candidate_paths.append((project_dir / path).resolve())

    for candidate in candidate_paths:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "CSV file not found; tried: "
        + ", ".join(str(candidate) for candidate in candidate_paths)
    )


def _prepare_dataset(
    csv_path: str | Path,
    label: int | None,
    batch_size: int,
    opt_iter_num: int,
    data_root: str | Path | None,
    project_dir: Path,
    log,
):
    import torch
    from torch.utils.data import RandomSampler
    from torchvision import transforms

    from papers.shared.photonic_QGAN.digits import DigitsDataset

    resolved_csv = _resolve_csv_path(csv_path, data_root, project_dir)
    dataset = DigitsDataset(
        csv_file=str(resolved_csv),
        label=label if label is not None else 0,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    sampler = RandomSampler(
        dataset, replacement=True, num_samples=batch_size * opt_iter_num
    )
    log.debug(
        "Prepared dataset csv_path={} label={} size={} batch_size={} opt_iter_num={}",
        resolved_csv,
        label,
        len(dataset),
        batch_size,
        opt_iter_num,
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, drop_last=True, sampler=sampler
    )


def _run_qgan(
    cfg: dict,
    run_dir: Path,
    run_cfg: dict,
    dataloader,
    write_to_disk: bool,
    lrD: float,
    opt_params: dict,
    log_every: int,
    show_progress: bool,
    log,
) -> None:
    import numpy as np
    import perceval as pcvl
    from lib.qgan import QGAN

    model_cfg = cfg.get("model", {})
    image_size = int(model_cfg.get("image_size", 8))
    batch_size = int(model_cfg.get("batch_size", 4))
    lossy = bool(model_cfg.get("lossy", False))
    remote_token = model_cfg.get("remote_token")
    use_clements = bool(model_cfg.get("use_clements", False))
    sim = bool(model_cfg.get("sim", False))

    gen_arch = run_cfg["arch"]
    noise_dim = int(run_cfg["noise_dim"])
    input_state = run_cfg["input_state"]
    pnr = bool(run_cfg["pnr"])
    gen_count = int(run_cfg["gen_count"])

    log.info(
        "Starting QGAN run image_size={} gen_count={} noise_dim={} batch_size={} pnr={} lossy={} use_clements={} sim={}",
        image_size,
        gen_count,
        noise_dim,
        batch_size,
        pnr,
        lossy,
        use_clements,
        sim,
    )
    log.info(
        "Generator config patches={} modes={} input_state={} arch={}",
        gen_count,
        len(input_state),
        input_state,
        gen_arch,
    )
    iterator = (
        tqdm(dataloader, desc="iter", leave=False) if show_progress else dataloader
    )

    def _log_progress(
        i,
        d_loss,
        g_loss,
        _g_params,
        _d_state,
        _fake_samples,
        _optG,
    ):
        if log_every > 0 and (i + 1) % log_every == 0:
            log.info("Iteration {} D_loss={} G_loss={}", i + 1, d_loss, g_loss)

    start = time.perf_counter()
    qgan = QGAN(
        image_size,
        gen_count,
        gen_arch,
        pcvl.BasicState(input_state),
        noise_dim,
        batch_size,
        pnr,
        lossy,
        remote_token=remote_token,
        use_clements=use_clements,
        sim=sim,
    )
    (
        D_loss_progress,
        G_loss_progress,
        G_params_progress,
        fake_data_progress,
    ) = qgan.fit(
        iterator,
        lrD,
        opt_params,
        silent=True,
        callback=_log_progress if log_every > 0 else None,
    )

    if write_to_disk:
        np.savetxt(
            run_dir / "fake_progress.csv",
            fake_data_progress,
            delimiter=",",
        )
        np.savetxt(
            run_dir / "loss_progress.csv",
            np.array(np.array([D_loss_progress, G_loss_progress]).transpose()),
            delimiter=",",
            header="D_loss, G_loss",
        )
        np.savetxt(
            run_dir / "G_params_progress.csv",
            np.array(G_params_progress),
            delimiter=",",
        )
        try:
            import matplotlib.pyplot as plt

            sample = np.array(fake_data_progress)[-1].reshape(image_size, image_size)
            plt.imshow(sample, cmap="gray")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(run_dir / "fake_progress_last.png", dpi=150)
            plt.close()
            log.debug(
                "Saved image preview under {}", run_dir / "fake_progress_last.png"
            )
        except Exception as exc:
            log.warning("Failed to save image preview: {}", exc)
        log.debug("Saved outputs under %s", run_dir)
    duration = time.perf_counter() - start
    log.info("Completed run at {} duration={:.2f}s", run_dir, duration)


def _write_config(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def train_and_evaluate(cfg, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(run_dir / "qgan.log", level="DEBUG")
    logger.debug(
        "Resolved config: {}",
        json.dumps(cfg, indent=2, default=str),
    )

    mode = cfg.get("run", {}).get("mode", "smoke")
    if mode == "smoke":
        artifact = run_dir / "done.txt"
        artifact.write_text(
            "Smoke run complete. Use --mode digits or --mode ideal for training.\n",
            encoding="utf-8",
        )
        logger.info("Wrote placeholder artifact: %s", artifact)
        return

    if mode not in {"digits", "ideal"}:
        raise ValueError(f"Unsupported run mode: {mode}")

    run_cfg = cfg.get("run", {})
    write_to_disk = bool(run_cfg.get("write_to_disk", True))
    runs = int(run_cfg.get("runs", 1))
    show_progress = bool(run_cfg.get("progress", False))
    log_every = int(run_cfg.get("log_every", 0))
    logger.info(
        "Run configuration mode={} runs={} write_to_disk={} progress={} log_every={}",
        mode,
        runs,
        write_to_disk,
        show_progress,
        log_every,
    )

    project_dir = run_dir.parent.parent.resolve()
    dataset_cfg = cfg.get("dataset", {})
    csv_path = dataset_cfg.get("csv_path", "photonic_QGAN/optdigits_csv.csv")

    # Resolve data_root: use config value, DATA_DIR env var, or default to repo_root/data
    import os

    data_root_cfg = cfg.get("data_root")
    if data_root_cfg:
        data_root = Path(data_root_cfg).expanduser()
    elif "DATA_DIR" in os.environ:
        data_root = Path(os.environ["DATA_DIR"]).expanduser()
    else:
        data_root = REPO_ROOT / "data"

    logger.info("Dataset csv_path={}", csv_path)

    training_cfg = cfg.get("training", {})
    if mode == "digits":
        train_cfg = training_cfg.get("digits", {})
    else:
        train_cfg = training_cfg.get("ideal", {})
    spsa_iter_num = int(train_cfg.get("spsa_iter_num", 0))
    opt_iter_num = int(train_cfg.get("opt_iter_num", 0))
    lrD = float(train_cfg.get("lrD", 0.002))
    if spsa_iter_num <= 0 or opt_iter_num <= 0:
        raise ValueError("Training iterations must be positive for digits/ideal modes.")

    opt_params = {"spsa_iter_num": spsa_iter_num, "opt_iter_num": opt_iter_num}
    spsa_a = train_cfg.get("spsa_a")
    spsa_k = train_cfg.get("spsa_k")
    if spsa_a is not None:
        opt_params["a"] = float(spsa_a)
    if spsa_k is not None:
        opt_params["k"] = int(spsa_k)
    logger.info(
        "Training params spsa_iter_num={} opt_iter_num={} lrD={}",
        spsa_iter_num,
        opt_iter_num,
        lrD,
    )
    if "a" in opt_params or "k" in opt_params:
        logger.info(
            "SPSA overrides a={} k={}",
            opt_params.get("a"),
            opt_params.get("k"),
        )

    if mode == "digits":
        digits_cfg = cfg.get("digits", {})
        digits = _resolve_digit_list(cfg)
        logger.info("Digits mode digits={}", digits)
        base_config = {
            "noise_dim": int(digits_cfg.get("noise_dim", 1)),
            "arch": _coerce_str_list(digits_cfg.get("arch")),
            "input_state": _coerce_int_list(digits_cfg.get("input_state")),
            "gen_count": int(digits_cfg.get("gen_count", 1)),
            "pnr": bool(digits_cfg.get("pnr", False)),
        }
        logger.debug("Digits base config: {}", json.dumps(base_config, indent=2))
        model_cfg = cfg.get("model", {})
        batch_size = int(model_cfg.get("batch_size", 4))
        logger.info("Model batch_size={}", batch_size)
        output_root = run_dir / "digits"
        if output_root.exists():
            shutil.rmtree(output_root)
        output_root.mkdir(parents=True, exist_ok=True)

        for digit in digits:
            dataloader = _prepare_dataset(
                csv_path,
                digit,
                batch_size,
                opt_iter_num,
                data_root,
                project_dir,
                logger,
            )
            config_path = output_root / f"config_{digit}"
            config_path.mkdir(parents=True, exist_ok=True)
            config_payload = base_config.copy()
            config_payload.update({"digit": digit})
            _write_config(config_path / "config.json", config_payload)
            logger.info("Digit {} output_dir={}", digit, config_path)

            run_num = 0
            attempt = 0
            while run_num < runs and attempt < 1000:
                attempt += 1
                run_num += 1
                save_path = config_path / f"run_{run_num}"
                save_path.mkdir(parents=True, exist_ok=True)
                logger.info("Digit {} run {}/{}", digit, run_num, runs)
                try:
                    _run_qgan(
                        cfg,
                        save_path,
                        base_config,
                        dataloader,
                        write_to_disk,
                        lrD,
                        opt_params,
                        log_every,
                        show_progress,
                        logger,
                    )
                except Exception as exc:
                    logger.exception("Run failed for digit {}: {}", digit, exc)
                    shutil.rmtree(save_path, ignore_errors=True)
                    run_num -= 1

        return

    ideal_cfg = cfg.get("ideal", {})
    if ideal_cfg.get("config_grid_path"):
        config_grid = _load_config_grid(ideal_cfg.get("config_grid_path"))
        logger.info(
            "Ideal mode config_grid_path={} size={}",
            ideal_cfg.get("config_grid_path"),
            len(config_grid),
        )
    elif ideal_cfg.get("config_grid"):
        config_grid = list(ideal_cfg["config_grid"])
        logger.info("Ideal mode config_grid_size={}", len(config_grid))
    else:
        config_grid = _default_ideal_grid()
        logger.info("Ideal mode default config_grid_size={}", len(config_grid))
    model_cfg = cfg.get("model", {})
    batch_size = int(model_cfg.get("batch_size", 4))
    logger.info("Model batch_size={}", batch_size)
    dataloader = _prepare_dataset(
        csv_path, None, batch_size, opt_iter_num, data_root, project_dir, logger
    )

    output_root = run_dir / "ideal"
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for config_num, config in enumerate(config_grid):
        config_path = output_root / f"config_{config_num}"
        config_path.mkdir(parents=True, exist_ok=True)
        _write_config(config_path / "config.json", config)
        logger.debug("Ideal config {}: {}", config_num, json.dumps(config, indent=2))

        run_num = 0
        attempt = 0
        while run_num < runs and attempt < 1000:
            attempt += 1
            run_num += 1
            save_path = config_path / f"run_{run_num}"
            save_path.mkdir(parents=True, exist_ok=True)
            logger.info("Ideal config {} run {}/{}", config_num, run_num, runs)
            try:
                _run_qgan(
                    cfg,
                    save_path,
                    config,
                    dataloader,
                    write_to_disk,
                    lrD,
                    opt_params,
                    log_every,
                    show_progress,
                    logger,
                )
            except Exception as exc:
                logger.exception("Run failed for config {}: {}", config_num, exc)
                shutil.rmtree(save_path, ignore_errors=True)
                run_num -= 1
